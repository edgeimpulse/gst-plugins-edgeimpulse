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
