use gstreamer as gst;
use gstreamer::glib;

mod audio;
mod common;
mod overlay;
pub mod sink;
mod video;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    video::register(plugin)?;
    audio::register(plugin)?;
    overlay::register(plugin)?;
    sink::register(plugin)?;
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
    "https://github.com/edgeimpulse/gst-plugins-edgeimpulse",
    env!("BUILD_REL_DATE")
);
