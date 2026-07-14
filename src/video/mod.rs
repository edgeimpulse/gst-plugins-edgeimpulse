#[cfg(feature = "inference")]
mod imp;
pub(crate) mod meta;

#[cfg(feature = "inference")]
use gstreamer as gst;
#[cfg(feature = "inference")]
use gstreamer::glib;
#[cfg(feature = "inference")]
use gstreamer::prelude::*;

// The public Rust wrapper type for our element
#[cfg(feature = "inference")]
glib::wrapper! {
    pub struct EdgeImpulseVideoInfer(ObjectSubclass<imp::EdgeImpulseVideoInfer>)
        @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
}

// GStreamer elements need to be thread-safe. For the private implementation
// this is automatically enforced but for the public wrapper type we need
// to specify this manually.
#[cfg(feature = "inference")]
unsafe impl Send for EdgeImpulseVideoInfer {}
#[cfg(feature = "inference")]
unsafe impl Sync for EdgeImpulseVideoInfer {}

#[cfg(feature = "inference")]
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsevideoinfer".to_string()
    } else {
        format!("edgeimpulsevideoinfer_{}", variant)
    };
    gst::Element::register(
        Some(plugin),
        &name,
        gst::Rank::NONE,
        EdgeImpulseVideoInfer::static_type(),
    )
}

pub use meta::{VideoAnomalyMeta, VideoClassificationMeta};
