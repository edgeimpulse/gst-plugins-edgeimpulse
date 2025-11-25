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

// Include the dynamically generated plugin definition from build.rs
// This ensures the registration function name matches what GStreamer expects
include!(concat!(env!("OUT_DIR"), "/plugin_define.rs"));
