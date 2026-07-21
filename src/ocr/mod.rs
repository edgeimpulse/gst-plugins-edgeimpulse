//! # EdgeImpulseOcr — OCR text reader element
//!
//! Runs OCR on RGB frames and attaches recognized text as standard
//! `VideoRegionOfInterestMeta` (label = text) so `edgeimpulseoverlay` renders
//! it, and optionally posts an `ocr` element message on the bus.

mod backend;
mod ctc;
mod decode;
mod imp;
mod recognizer;
mod shaping;
mod stabilize;
mod stabilize_by_id;

#[cfg(feature = "ocr")]
mod ocrs_backend;

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;

glib::wrapper! {
    pub struct EdgeImpulseOcr(ObjectSubclass<imp::EdgeImpulseOcr>)
        @extends gstreamer_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulseocr".to_string()
    } else {
        format!("edgeimpulseocr_{}", variant)
    };
    gst::Element::register(
        Some(plugin),
        &name,
        gst::Rank::NONE,
        EdgeImpulseOcr::static_type(),
    )
}
