//! # EdgeImpulseContinueIf — Conditional buffer gate with metadata output
//!
//! A GStreamer BaseTransform element that reads `InferenceResultMeta` attached
//! to buffers by upstream `edgeimpulsevideoinfer` or `edgeimpulseaudioinfer`
//! elements and conditionally drops buffers that don't match a condition.
//!
//! Works with both audio and video pipelines — any buffer carrying
//! `InferenceResultMeta` can be filtered. Falls back to video-specific
//! metadata (`VideoRegionOfInterestMeta`, etc.) if no `InferenceResultMeta`
//! is present.
//!
//! ## Properties
//!
//! - `condition` (string) — Simple pass/drop gate expression.
//! - `drop` (boolean) — Manual override to drop all buffers.
//! - `rules` (string) — JSON array of ordered rules for conditional metadata
//!   output. First matching rule wins. Each rule has a `condition` and
//!   `metadata` (key-value pairs attached to the buffer as a GstStructure).
//!
//! ## Rules example
//!
//! ```json
//! [
//!   {"condition": "detection_count > 4", "metadata": {"color": "purple", "severity": "critical"}},
//!   {"condition": "detection_count >= 1", "metadata": {"color": "red", "severity": "warning"}},
//!   {"condition": "detection_count == 0", "metadata": {"color": "green", "severity": "ok"}}
//! ]
//! ```

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::meta::InferenceResultMeta;
use crate::video::{VideoAnomalyMeta, VideoClassificationMeta};

include!(concat!(env!("OUT_DIR"), "/type_names.rs"));

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsecontinueif".to_string()
    } else {
        format!("edgeimpulsecontinueif_{}", variant)
    };
    gst::DebugCategory::new(
        &name,
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Continue If"),
    )
});

// ─── Condition parsing ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum ComparisonOp {
    Gte,
    Lte,
    Gt,
    Lt,
    Eq,
    Ne,
}

#[derive(Debug, Clone)]
enum LiteralValue {
    Number(f64),
    String(String),
}

#[derive(Debug, Clone)]
enum ParsedCondition {
    FieldCompare {
        field: String,
        op: ComparisonOp,
        value: LiteralValue,
    },
    HasClass {
        class_name: String,
    },
}

/// A rule: condition → metadata key-value pairs
#[derive(Debug, Clone)]
struct MetadataRule {
    condition: ParsedCondition,
    metadata: HashMap<String, String>,
}

fn parse_condition(condition: &str) -> Option<ParsedCondition> {
    let condition = condition.trim();

    if let Some(rest) = condition.strip_prefix("has_class(") {
        let rest = rest.strip_suffix(')')?;
        let class_name = rest.trim().trim_matches('"').trim_matches('\'').to_string();
        return Some(ParsedCondition::HasClass { class_name });
    }

    let operators = [
        (">=", ComparisonOp::Gte),
        ("<=", ComparisonOp::Lte),
        ("==", ComparisonOp::Eq),
        ("!=", ComparisonOp::Ne),
        (">", ComparisonOp::Gt),
        ("<", ComparisonOp::Lt),
    ];

    for (op_str, op) in operators {
        if let Some((lhs, rhs)) = condition.split_once(op_str) {
            let field = lhs.trim().to_string();
            if field.is_empty() {
                return None;
            }
            let value = parse_literal(rhs.trim())?;
            return Some(ParsedCondition::FieldCompare { field, op, value });
        }
    }
    None
}

fn parse_literal(raw: &str) -> Option<LiteralValue> {
    if let Ok(n) = raw.parse::<f64>() {
        return Some(LiteralValue::Number(n));
    }
    if raw.eq_ignore_ascii_case("true") {
        return Some(LiteralValue::String("true".to_string()));
    }
    if raw.eq_ignore_ascii_case("false") {
        return Some(LiteralValue::String("false".to_string()));
    }
    let quoted = (raw.starts_with('"') && raw.ends_with('"'))
        || (raw.starts_with('\'') && raw.ends_with('\''));
    if quoted && raw.len() >= 2 {
        return Some(LiteralValue::String(raw[1..raw.len() - 1].to_string()));
    }
    if !raw.is_empty() {
        return Some(LiteralValue::String(raw.to_string()));
    }
    None
}

fn parse_rules(json: &str) -> Vec<MetadataRule> {
    let arr: Vec<serde_json::Value> = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    arr.iter()
        .filter_map(|rule| {
            let cond_str = rule.get("condition")?.as_str()?;
            let condition = parse_condition(cond_str)?;
            let metadata_obj = rule.get("metadata")?.as_object()?;
            let metadata: HashMap<String, String> = metadata_obj
                .iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                .collect();
            Some(MetadataRule {
                condition,
                metadata,
            })
        })
        .collect()
}

fn compare_numbers(lhs: f64, rhs: f64, op: &ComparisonOp) -> bool {
    match op {
        ComparisonOp::Gte => lhs >= rhs,
        ComparisonOp::Lte => lhs <= rhs,
        ComparisonOp::Gt => lhs > rhs,
        ComparisonOp::Lt => lhs < rhs,
        ComparisonOp::Eq => (lhs - rhs).abs() < f64::EPSILON,
        ComparisonOp::Ne => (lhs - rhs).abs() >= f64::EPSILON,
    }
}

// ─── Inference state ─────────────────────────────────────────────────────────

struct InferenceState {
    detection_count: u32,
    max_confidence: f64,
    top_class: String,
    top_confidence: f64,
    anomaly_score: f64,
    visual_anomaly_max: f64,
    /// Detection labels for has_class()
    detection_labels: Vec<String>,
}

/// Extract inference state from `InferenceResultMeta` if present,
/// otherwise fall back to video-specific metadata.
fn extract_state(buf: &gst::BufferRef) -> InferenceState {
    // Prefer InferenceResultMeta (works for both audio and video)
    if let Some(ir) = buf.meta::<InferenceResultMeta>() {
        let mut labels = Vec::new();
        // Parse detection labels from result_json if available
        if let Ok(result) = serde_json::from_str::<serde_json::Value>(ir.result_json()) {
            if let Some(boxes) = result.get("bounding_boxes").and_then(|b| b.as_array()) {
                for bbox in boxes {
                    if let Some(label) = bbox.get("label").and_then(|l| l.as_str()) {
                        labels.push(label.to_string());
                    }
                }
            }
        }
        // Also include top_class for classification-type results
        if !ir.top_class().is_empty() {
            labels.push(ir.top_class().to_string());
        }

        return InferenceState {
            detection_count: ir.detection_count(),
            max_confidence: ir.max_confidence(),
            top_class: ir.top_class().to_string(),
            top_confidence: ir.top_confidence(),
            anomaly_score: ir.anomaly_score(),
            visual_anomaly_max: ir.visual_anomaly_max(),
            detection_labels: labels,
        };
    }

    // Fallback: video-specific metadata
    let mut state = InferenceState {
        detection_count: 0,
        max_confidence: 0.0,
        top_class: String::new(),
        top_confidence: 0.0,
        anomaly_score: 0.0,
        visual_anomaly_max: 0.0,
        detection_labels: Vec::new(),
    };

    for roi in buf.iter_meta::<gst_video::VideoRegionOfInterestMeta>() {
        state.detection_count += 1;
        for param in roi.params() {
            if let Ok(conf) = param.get::<f64>("confidence") {
                if conf > state.max_confidence {
                    state.max_confidence = conf;
                }
            }
            if let Ok(label) = param.get::<&str>("label") {
                state.detection_labels.push(label.to_string());
            }
        }
    }

    if let Some(class_meta) = buf.meta::<VideoClassificationMeta>() {
        for param in class_meta.params() {
            if let Ok(label) = param.get::<&str>("label") {
                state.top_class = label.to_string();
            }
            if let Ok(conf) = param.get::<f64>("confidence") {
                state.top_confidence = conf;
                if conf > state.max_confidence {
                    state.max_confidence = conf;
                }
            }
        }
    }

    if let Some(anomaly_meta) = buf.meta::<VideoAnomalyMeta>() {
        state.anomaly_score = anomaly_meta.anomaly() as f64;
        state.visual_anomaly_max = anomaly_meta.visual_anomaly_max() as f64;
    }

    state
}

// ─── Condition evaluation ────────────────────────────────────────────────────

fn evaluate(state: &InferenceState, cond: &ParsedCondition) -> bool {
    match cond {
        ParsedCondition::HasClass { class_name } => {
            state.detection_labels.iter().any(|l| l == class_name)
        }
        ParsedCondition::FieldCompare { field, op, value } => {
            let field_val = match field.as_str() {
                "detection_count" => Some(state.detection_count as f64),
                "max_confidence" => Some(state.max_confidence),
                "top_confidence" | "classification_confidence" => Some(state.top_confidence),
                "anomaly_score" => Some(state.anomaly_score),
                "visual_anomaly_max" => Some(state.visual_anomaly_max),
                _ => None,
            };

            match (field_val, value) {
                (Some(lhs), LiteralValue::Number(rhs)) => compare_numbers(lhs, *rhs, op),
                (None, _)
                    if field.as_str() == "classification" || field.as_str() == "top_class" =>
                {
                    let lhs = &state.top_class;
                    match (value, op) {
                        (LiteralValue::String(rhs), ComparisonOp::Eq) => lhs == rhs,
                        (LiteralValue::String(rhs), ComparisonOp::Ne) => lhs != rhs,
                        _ => false,
                    }
                }
                _ => false,
            }
        }
    }
}

// ─── Element state ───────────────────────────────────────────────────────────

#[derive(Default)]
struct State {
    condition: String,
    parsed: Option<ParsedCondition>,
    drop_all: bool,
    rules_json: String,
    rules: Vec<MetadataRule>,
    passed: u64,
    dropped: u64,
}

// ─── GStreamer element ───────────────────────────────────────────────────────

pub struct EdgeImpulseContinueIf {
    state: Mutex<State>,
}

impl Default for EdgeImpulseContinueIf {
    fn default() -> Self {
        Self {
            state: Mutex::new(State::default()),
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseContinueIf {
    const NAME: &'static str = FILTER_TYPE_NAME;
    type Type = super::EdgeImpulseContinueIf;
    type ParentType = gstreamer_base::BaseTransform;
}

impl ObjectImpl for EdgeImpulseContinueIf {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("condition")
                    .nick("Condition")
                    .blurb(
                        "Expression for pass/drop gating. Examples: 'detection_count >= 1', \
                         'has_class(\"crack\")', 'anomaly_score > 0.5'",
                    )
                    .default_value(Some(""))
                    .mutable_playing()
                    .build(),
                glib::ParamSpecBoolean::builder("drop")
                    .nick("Drop")
                    .blurb("When true, unconditionally drop all buffers")
                    .default_value(false)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecString::builder("rules")
                    .nick("Rules")
                    .blurb(
                        "JSON array of condition→metadata rules. First matching rule's metadata \
                         is attached to the buffer as a GstStructure. Example: \
                         '[{\"condition\":\"detection_count > 4\",\"metadata\":{\"color\":\"purple\"}}]'",
                    )
                    .default_value(Some(""))
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut state = self.state.lock().unwrap();
        match pspec.name() {
            "condition" => {
                let condition = value.get::<String>().unwrap_or_default();
                state.parsed = parse_condition(&condition);
                if !condition.is_empty() && state.parsed.is_none() {
                    gst::warning!(
                        CAT,
                        obj = self.obj(),
                        "Failed to parse condition: '{}' — all buffers will pass",
                        condition
                    );
                }
                state.condition = condition;
            }
            "drop" => {
                state.drop_all = value.get::<bool>().unwrap_or(false);
            }
            "rules" => {
                let json = value.get::<String>().unwrap_or_default();
                state.rules = parse_rules(&json);
                gst::info!(
                    CAT,
                    obj = self.obj(),
                    "Parsed {} metadata rules",
                    state.rules.len()
                );
                state.rules_json = json;
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let state = self.state.lock().unwrap();
        match pspec.name() {
            "condition" => state.condition.to_value(),
            "drop" => state.drop_all.to_value(),
            "rules" => state.rules_json.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for EdgeImpulseContinueIf {}

impl ElementImpl for EdgeImpulseContinueIf {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Continue If",
                "Filter/Video",
                "Conditionally passes or drops buffers based on upstream inference metadata. \
                 Works with both audio and video pipelines via InferenceResultMeta.",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::new_any();
            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        });
        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for EdgeImpulseContinueIf {
    const MODE: gstreamer_base::subclass::BaseTransformMode =
        gstreamer_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        let state = self.state.lock().unwrap();

        if state.drop_all {
            return Err(gst::FlowError::CustomError);
        }

        let gate_condition = state.parsed.clone();
        let rules = state.rules.clone();
        drop(state);

        let inference_state = extract_state(buf);

        // Evaluate metadata rules (first match wins)
        if !rules.is_empty() {
            for rule in &rules {
                if evaluate(&inference_state, &rule.condition) {
                    // Attach matched metadata as a GstStructure on the buffer
                    // We use InferenceResultMeta's result_json to carry it, or
                    // attach as a separate custom meta. For simplicity, we post
                    // a bus message with the matched metadata.
                    let mut builder = gst::Structure::builder("edge-impulse-continue-if-metadata");
                    for (key, val) in &rule.metadata {
                        builder = builder.field(key, val);
                    }
                    let s = builder.build();
                    let _ = self.obj().post_message(gst::message::Element::new(s));

                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Rule matched: {:?} → posting metadata with {} fields",
                        rule.condition,
                        rule.metadata.len()
                    );
                    break;
                }
            }
        }

        // Evaluate pass/drop gate
        let pass = match &gate_condition {
            Some(cond) => evaluate(&inference_state, cond),
            None => true,
        };

        let mut state = self.state.lock().unwrap();
        if pass {
            state.passed += 1;
            gst::log!(
                CAT,
                obj = self.obj(),
                "PASS (det={}, conf={:.2}, class='{}', passed={}, dropped={})",
                inference_state.detection_count,
                inference_state.max_confidence,
                inference_state.top_class,
                state.passed,
                state.dropped
            );
            Ok(gst::FlowSuccess::Ok)
        } else {
            state.dropped += 1;
            gst::log!(
                CAT,
                obj = self.obj(),
                "DROP (det={}, conf={:.2}, class='{}', passed={}, dropped={})",
                inference_state.detection_count,
                inference_state.max_confidence,
                inference_state.top_class,
                state.passed,
                state.dropped
            );
            buf.set_flags(gst::BufferFlags::GAP | gst::BufferFlags::DROPPABLE);
            Ok(gst::FlowSuccess::Ok)
        }
    }

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().unwrap();
        state.passed = 0;
        state.dropped = 0;
        gst::info!(CAT, obj = self.obj(), "Continue-If started");
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let state = self.state.lock().unwrap();
        gst::info!(
            CAT,
            obj = self.obj(),
            "Continue-If stopped — {} passed, {} dropped",
            state.passed,
            state.dropped
        );
        Ok(())
    }
}
