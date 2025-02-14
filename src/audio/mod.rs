use glib::prelude::*;
use gstreamer as gst;

mod imp;

// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct EdgeImpulseAudioInfer(ObjectSubclass<imp::EdgeImpulseAudioInfer>)
        @extends gst::Element, gst::Object;
}

// GStreamer elements need to be thread-safe. For the private implementation
// this is automatically enforced but for the public wrapper type we need
// to specify this manually.
unsafe impl Send for EdgeImpulseAudioInfer {}
unsafe impl Sync for EdgeImpulseAudioInfer {}

// Register the type for our element
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "edgeimpulseaudioinfer",
        gst::Rank::NONE,
        EdgeImpulseAudioInfer::static_type(),
    )
}
