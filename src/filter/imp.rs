//! # EdgeImpulseContinueIf — Conditional video gate based on inference metadata
//!
//! A GStreamer BaseTransform element that reads inference metadata attached to
//! video buffers by upstream `edgeimpulsevideoinfer` elements and conditionally
//! drops buffers that don't match a user-specified condition.
//!
//! ## Supported metadata
//!
//! - `gst_video::VideoRegionOfInterestMeta` — object detection bounding boxes
//! - `VideoClassificationMeta` — classification label + confidence
//! - `VideoAnomalyMeta` — anomaly scores
//!
//! ## Properties
//!
//! - `condition` (string) — Expression evaluated per buffer.
//!   Examples: `detection_count >= 1`, `max_confidence > 0.8`,
//!             `has_class("crack")`, `anomaly_score > 0.5`
//! - `drop` (boolean) — When true, all buffers are dropped unconditionally (manual override).
//!
//! ## Condition variables
//!
//! | Variable              | Type    | Source                        |
//! |-----------------------|---------|-------------------------------|
//! | `detection_count`     | number  | ROI meta count                |
//! | `max_confidence`      | number  | Highest confidence across ROI |
//! | `has_class("name")`   | boolean | Any ROI with matching label   |
//! | `classification`      | string  | Top classification label      |
//! | `classification_confidence` | number | Top classification score |
//! | `anomaly_score`       | number  | Anomaly meta score            |
//! | `visual_anomaly_max`  | number  | Visual anomaly max score      |

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;

use crate::video::{VideoAnomalyMeta, VideoClassificationMeta};

// Include generated type names for variant-specific builds
include!(concat!(env!("OUT_DIR"), "/type_names.rs"));

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsecontinueif".to_string()
    } else {
        format!("edgeimpulsecontinueif_{}", variant)
    };
    gst::DebugCategory::new(&name, gst::DebugColorFlags::empty(), Some("Edge Impulse Continue If"))
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
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone)]
enum ParsedCondition {
    /// Field comparison: `detection_count >= 1`
    FieldCompare {
        field: String,
        op: ComparisonOp,
        value: LiteralValue,
    },
    /// Function call: `has_class("crack")`
    HasClass { class_name: String },
}

fn parse_condition(condition: &str) -> Option<ParsedCondition> {
    let condition = condition.trim();

    // has_class("name") or has_class('name')
    if let Some(rest) = condition.strip_prefix("has_class(") {
        let rest = rest.strip_suffix(')')?;
        let class_name = rest
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
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
        return Some(LiteralValue::Bool(true));
    }
    if raw.eq_ignore_ascii_case("false") {
        return Some(LiteralValue::Bool(false));
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

// ─── Metadata extraction ─────────────────────────────────────────────────────

/// Extracted inference state from a buffer's metadata.
struct BufferInferenceState {
    detection_count: u32,
    max_confidence: f64,
    detection_labels: Vec<String>,
    classification_label: Option<String>,
    classification_confidence: f64,
    anomaly_score: f64,
    visual_anomaly_max: f64,
}

fn extract_inference_state(buf: &gst::BufferRef) -> BufferInferenceState {
    let mut state = BufferInferenceState {
        detection_count: 0,
        max_confidence: 0.0,
        detection_labels: Vec::new(),
        classification_label: None,
        classification_confidence: 0.0,
        anomaly_score: 0.0,
        visual_anomaly_max: 0.0,
    };

    // 1. Object detection — standard GStreamer ROI meta
    for roi in buf.iter_meta::<gst_video::VideoRegionOfInterestMeta>() {
        state.detection_count += 1;

        // Extract label and confidence from detection params
        // (same pattern as overlay/imp.rs)
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

    // 2. Classification — custom meta
    if let Some(class_meta) = buf.meta::<VideoClassificationMeta>() {
        for param in class_meta.params() {
            if let Ok(label) = param.get::<&str>("label") {
                state.classification_label = Some(label.to_string());
            }
            if let Ok(conf) = param.get::<f64>("confidence") {
                state.classification_confidence = conf;
            }
        }
    }

    // 3. Anomaly — custom meta
    if let Some(anomaly_meta) = buf.meta::<VideoAnomalyMeta>() {
        state.anomaly_score = anomaly_meta.anomaly() as f64;
        state.visual_anomaly_max = anomaly_meta.visual_anomaly_max() as f64;
    }

    state
}

// ─── Condition evaluation ────────────────────────────────────────────────────

fn evaluate_condition(state: &BufferInferenceState, condition: &ParsedCondition) -> bool {
    match condition {
        ParsedCondition::HasClass { class_name } => {
            state.detection_labels.iter().any(|l| l == class_name)
                || state
                    .classification_label
                    .as_ref()
                    .map(|l| l == class_name)
                    .unwrap_or(false)
        }
        ParsedCondition::FieldCompare { field, op, value } => {
            let field_value = match field.as_str() {
                "detection_count" => Some(state.detection_count as f64),
                "max_confidence" => Some(state.max_confidence),
                "classification_confidence" => Some(state.classification_confidence),
                "anomaly_score" => Some(state.anomaly_score),
                "visual_anomaly_max" => Some(state.visual_anomaly_max),
                _ => None,
            };

            match (field_value, value) {
                (Some(lhs), LiteralValue::Number(rhs)) => compare_numbers(lhs, *rhs, op),
                (None, _) => {
                    // String field comparison (e.g., classification == "crack")
                    if field.as_str() == "classification" {
                        let lhs = state.classification_label.as_deref().unwrap_or("");
                        match (value, op) {
                            (LiteralValue::String(rhs), ComparisonOp::Eq) => lhs == rhs,
                            (LiteralValue::String(rhs), ComparisonOp::Ne) => lhs != rhs,
                            _ => false,
                        }
                    } else {
                        false
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
    /// Counters for debug logging
    passed: u64,
    dropped: u64,
}

// ─── GStreamer element implementation ─────────────────────────────────────────

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
    // Use a unique type name per variant to avoid conflicts
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
                        "Expression evaluated per buffer to decide pass/drop. \
                         Examples: 'detection_count >= 1', 'has_class(\"crack\")', \
                         'anomaly_score > 0.5'",
                    )
                    .default_value(Some(""))
                    .mutable_playing()
                    .build(),
                glib::ParamSpecBoolean::builder("drop")
                    .nick("Drop")
                    .blurb("When true, unconditionally drop all buffers (manual override)")
                    .default_value(false)
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
                        "Failed to parse condition: '{}' — all buffers will pass through",
                        condition
                    );
                } else if let Some(ref parsed) = state.parsed {
                    gst::info!(
                        CAT,
                        obj = self.obj(),
                        "Condition set: '{}' → {:?}",
                        condition,
                        parsed
                    );
                }
                state.condition = condition;
            }
            "drop" => {
                state.drop_all = value.get::<bool>().unwrap_or(false);
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let state = self.state.lock().unwrap();
        match pspec.name() {
            "condition" => state.condition.to_value(),
            "drop" => state.drop_all.to_value(),
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
                "Conditionally passes or drops video buffers based on upstream inference metadata",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            // Accept any video format — we never touch the pixel data
            let caps = gst::Caps::new_any();

            vec![
                gst::PadTemplate::new("sink", gst::PadDirection::Sink, gst::PadPresence::Always, &caps)
                    .unwrap(),
                gst::PadTemplate::new("src", gst::PadDirection::Src, gst::PadPresence::Always, &caps)
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

        // Manual override: drop everything
        if state.drop_all {
            gst::log!(CAT, obj = self.obj(), "Dropping buffer (manual override)");
            return Err(gst::FlowError::CustomError);
        }

        // No condition set → pass everything through
        let parsed = match &state.parsed {
            Some(p) => p.clone(),
            None => return Ok(gst::FlowSuccess::Ok),
        };

        // Release lock before reading metadata (avoid holding across FFI)
        drop(state);

        let inference_state = extract_inference_state(buf);
        let pass = evaluate_condition(&inference_state, &parsed);

        let mut state = self.state.lock().unwrap();
        if pass {
            state.passed += 1;
            gst::log!(
                CAT,
                obj = self.obj(),
                "Buffer PASSED (detections={}, max_conf={:.2}, passed={}, dropped={})",
                inference_state.detection_count,
                inference_state.max_confidence,
                state.passed,
                state.dropped
            );
            Ok(gst::FlowSuccess::Ok)
        } else {
            state.dropped += 1;
            gst::log!(
                CAT,
                obj = self.obj(),
                "Buffer DROPPED (detections={}, max_conf={:.2}, passed={}, dropped={})",
                inference_state.detection_count,
                inference_state.max_confidence,
                state.passed,
                state.dropped
            );
            // Return a non-fatal "skip" by marking the buffer as a GAP.
            // This tells downstream that no meaningful data is in this buffer,
            // but doesn't cause a pipeline error.
            buf.set_flags(gst::BufferFlags::GAP | gst::BufferFlags::DROPPABLE);
            Ok(gst::FlowSuccess::Ok)
        }
    }

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().unwrap();
        state.passed = 0;
        state.dropped = 0;
        gst::info!(CAT, obj = self.obj(), "Filter started");
        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let state = self.state.lock().unwrap();
        gst::info!(
            CAT,
            obj = self.obj(),
            "Filter stopped — {} passed, {} dropped",
            state.passed,
            state.dropped
        );
        Ok(())
    }
}
