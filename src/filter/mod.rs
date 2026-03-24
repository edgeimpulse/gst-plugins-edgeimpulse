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
