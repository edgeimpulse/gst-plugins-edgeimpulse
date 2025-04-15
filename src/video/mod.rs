mod imp;
mod meta;

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;

// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct EdgeImpulseVideoInfer(ObjectSubclass<imp::EdgeImpulseVideoInfer>)
        @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
}

// GStreamer elements need to be thread-safe. For the private implementation
// this is automatically enforced but for the public wrapper type we need
// to specify this manually.
unsafe impl Send for EdgeImpulseVideoInfer {}
unsafe impl Sync for EdgeImpulseVideoInfer {}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "edgeimpulsevideoinfer",
        gst::Rank::NONE,
        EdgeImpulseVideoInfer::static_type(),
    )
}

pub use meta::{VideoAnomalyMeta, VideoClassificationMeta};
