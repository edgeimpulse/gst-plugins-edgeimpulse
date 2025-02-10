use gstreamer as gst;
use gstreamer::glib;

mod edgeimpulse;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    edgeimpulse::register(plugin)
}

gst::plugin_define!(
    edgeimpulse,
    "Edge Impulse Inference Plugin",
    plugin_init,
    env!("CARGO_PKG_VERSION"),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    "https://github.com/edgeimpulse/gst-plugin-edgeimpulse",
    env!("BUILD_REL_DATE")
);