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
    gst::Element::register(
        Some(plugin),
        "edgeimpulsesink",
        gst::Rank::NONE,
        EdgeImpulseSink::static_type(),
    )
}
