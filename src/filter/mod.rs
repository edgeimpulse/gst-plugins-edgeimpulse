//! # EdgeImpulseContinueIf — Conditional buffer gate
//!
//! A GStreamer element that passes or drops buffers based on upstream inference
//! metadata. Works with both audio and video pipelines — any buffer carrying
//! [`InferenceResultMeta`](crate::meta::InferenceResultMeta) can be filtered.
//!
//! ## Properties
//!
//! | Property | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `condition` | `String` | `""` | Expression evaluated per buffer (`""` = pass all) |
//! | `drop` | `bool` | `false` | Manual override to unconditionally drop all buffers |
//! | `rules` | `String` | `""` | JSON array of ordered rules for conditional metadata output |
//!
//! ## Condition variables
//!
//! These are extracted from [`InferenceResultMeta`](crate::meta::InferenceResultMeta)
//! (or video-specific metadata as fallback):
//!
//! | Variable | Type | Description |
//! |----------|------|-------------|
//! | `detection_count` | number | Number of detected objects |
//! | `max_confidence` | number | Highest confidence across all detections |
//! | `has_class("name")` | function | True if any detection/classification matches the label |
//! | `classification` | string | Top classification label |
//! | `classification_confidence` | number | Top classification confidence |
//! | `anomaly_score` | number | Anomaly score |
//! | `visual_anomaly_max` | number | Visual anomaly max score |
//!
//! Supported operators: `>=`, `<=`, `>`, `<`, `==`, `!=`
//!
//! ## Rules
//!
//! The `rules` property accepts a JSON array of ordered rules. The first matching
//! rule wins. Each rule has a `condition` and `metadata` (key-value pairs posted
//! as a `edge-impulse-continue-if-metadata` bus message):
//!
//! ```json
//! [
//!   {"condition": "detection_count > 4",  "metadata": {"severity": "critical"}},
//!   {"condition": "detection_count >= 1", "metadata": {"severity": "warning"}},
//!   {"condition": "detection_count == 0", "metadata": {"severity": "ok"}}
//! ]
//! ```
//!
//! ## Example
//!
//! ```text
//! edgeimpulsevideoinfer ! edgeimpulsecontinueif condition="detection_count >= 1" ! ...
//! ```

mod imp;

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;

// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct EdgeImpulseContinueIf(ObjectSubclass<imp::EdgeImpulseContinueIf>)
        @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
}

unsafe impl Send for EdgeImpulseContinueIf {}
unsafe impl Sync for EdgeImpulseContinueIf {}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsecontinueif".to_string()
    } else {
        format!("edgeimpulsecontinueif_{}", variant)
    };
    gst::Element::register(
        Some(plugin),
        &name,
        gst::Rank::NONE,
        EdgeImpulseContinueIf::static_type(),
    )
}
