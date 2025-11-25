//! EdgeImpulseSink GStreamer Element
//!
//! This sink element uploads audio or video buffers to Edge Impulse using the ingestion API.
//!
//! # Features
//! - Supports both audio (WAV) and video (PNG) ingestion.
//! - Batches and uploads buffers at a configurable interval (`upload-interval-ms` property).
//! - Exposes properties for API key, HMAC key, label, category, and upload interval.
//! - Posts custom messages to the GStreamer bus:
//!   - `edge-impulse-ingestion-result`: Sent when a sample is successfully ingested. Contains filename, media type, length, label, and category.
//!   - `edge-impulse-ingestion-error`: Sent when ingestion fails. Contains filename, media type, error message, label, and category.
//!
//! # Usage
//! Register the element as `edgeimpulsesink` in your GStreamer pipeline. Example:
//!
//! ```
//! ... ! edgeimpulsesink api-key="<API_KEY>" upload-interval-ms=1000 category=training
//! ```
//!
//! # Properties
//! - `api-key` (String, required): Edge Impulse API key.
//! - `hmac-key` (String, optional): Optional HMAC key for signing requests.
//! - `label` (String, optional): Optional label for the sample.
//! - `category` (String, default: "training"): Category for the sample (training, testing, anomaly).
//! - `upload-interval-ms` (u32, default: 0): Minimum interval in milliseconds between uploads (0 = every buffer).
//!
//! # Bus Messages
//! Listen for `Element` messages on the GStreamer bus to receive ingestion results and errors.
//!
//! # Example
//! See `examples/audio_ingestion.rs` for a full pipeline and bus message handling example.

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer_base as gst_base;

mod imp;

glib::wrapper! {
    pub struct EdgeImpulseSink(ObjectSubclass<imp::EdgeImpulseSink>)
        @extends gst_base::BaseSink, gst::Element, gst::Object;
}

unsafe impl Send for EdgeImpulseSink {}
unsafe impl Sync for EdgeImpulseSink {}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsesink".to_string()
    } else {
        format!("edgeimpulsesink_{}", variant)
    };
    gst::Element::register(
        Some(plugin),
        &name,
        gst::Rank::NONE,
        EdgeImpulseSink::static_type(),
    )
}
