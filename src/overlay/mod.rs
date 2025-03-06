use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
mod imp;

// Wrapper type registration
glib::wrapper! {
    pub struct EdgeImpulseOverlay(ObjectSubclass<imp::EdgeImpulseOverlay>)
        @extends gst_video::VideoFilter, gst::Element, gst::Object;
}

// Plugin registration
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "edgeimpulseoverlay",
        gst::Rank::NONE,
        EdgeImpulseOverlay::static_type(),
    )
}
