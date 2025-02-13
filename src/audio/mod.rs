use gstreamer as gst;
use glib::prelude::*;

mod imp;

// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct EdgeImpulseInfer(ObjectSubclass<imp::EdgeImpulseInfer>)
        @extends gst::Element, gst::Object;
}

// GStreamer elements need to be thread-safe. For the private implementation
// this is automatically enforced but for the public wrapper type we need
// to specify this manually.
unsafe impl Send for EdgeImpulseInfer {}
unsafe impl Sync for EdgeImpulseInfer {}

// Register the type for our element
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "edgeimpulseinfer",
        gst::Rank::NONE,
        EdgeImpulseInfer::static_type(),
    )
}