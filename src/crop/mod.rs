//! # EdgeImpulseCrop — Dynamic per-detection crop element
//!
//! A 1-to-N GStreamer element that reads bounding box metadata from upstream
//! [`edgeimpulsevideoinfer`](crate::video) and pushes one cropped buffer per
//! detection downstream.
//!
//! ## Properties
//!
//! | Property | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `padding` | `i32` | `0` | Extra pixels around each bounding box (clamped to frame bounds) |
//! | `target-width` | `i32` | `0` | Resize crops to this width (`0` = keep natural size) |
//! | `target-height` | `i32` | `0` | Resize crops to this height (`0` = keep natural size) |
//!
//! ## Metadata
//!
//! Each cropped buffer carries a [`CropOriginMeta`](meta::CropOriginMeta)
//! recording where the crop came from in the original frame. This lets
//! downstream elements map classification results back to full-frame
//! coordinates:
//!
//! ```text
//! Original frame (1920x1080)       Cropped buffer (96x96)
//! ┌────────────────────────┐       ┌──────────┐
//! │                        │       │CropOrigin│
//! │   ┌──────┐             │  ──►  │ x=100    │
//! │   │detect│             │       │ y=200    │
//! │   └──────┘             │       │ w=150    │
//! │                        │       │ h=150    │
//! └────────────────────────┘       └──────────┘
//! ```
//!
//! ## Behaviour
//!
//! - If no detections are present the full frame is passed through unchanged.
//! - Setting fixed `target-width`/`target-height` avoids GStreamer caps
//!   renegotiation between crops of different sizes.
//!
//! ## Example
//!
//! ```text
//! edgeimpulsevideoinfer ! edgeimpulsecrop padding=10 target-width=96 target-height=96 ! ...
//! ```

mod imp;
pub mod meta;

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;

// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct EdgeImpulseCrop(ObjectSubclass<imp::EdgeImpulseCrop>)
        @extends gstreamer::Element, gstreamer::Object;
}

unsafe impl Send for EdgeImpulseCrop {}
unsafe impl Sync for EdgeImpulseCrop {}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsecrop".to_string()
    } else {
        format!("edgeimpulsecrop_{}", variant)
    };
    gst::Element::register(
        Some(plugin),
        &name,
        gst::Rank::NONE,
        EdgeImpulseCrop::static_type(),
    )
}

// Re-export for downstream consumers
#[allow(unused_imports)]
pub use meta::CropOriginMeta;
