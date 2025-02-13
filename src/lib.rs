use gstreamer as gst;
use gstreamer::glib;

mod common;
mod video;
mod audio;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    video::register(plugin)?;
    audio::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    edgeimpulse,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    "https://github.com/edgeimpulse/gst-plugin-edgeimpulse",
    env!("BUILD_REL_DATE")
);