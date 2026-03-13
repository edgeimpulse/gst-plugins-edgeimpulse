//! Integration test for NV12 pipeline support.
//!
//! Tests that edgeimpulsevideoinfer can receive NV12 system-memory buffers
//! and process them without errors.
//!
//! Note: This test does not load an actual model — it verifies that the element
//! correctly accepts NV12 caps, maps the buffer, and passes it through without
//! crashing. Model-dependent tests require the FFI runtime.

use gstreamer as gst;
use gstreamer::prelude::*;

fn init() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().unwrap();
        // Register our plugin
        let plugin_path = std::env::var("GST_PLUGIN_PATH")
            .unwrap_or_else(|_| "target/debug".to_string());
        let registry = gst::Registry::get();
        // Scan for our plugin
        registry.scan_path(std::path::Path::new(&plugin_path));
    });
}

#[test]
fn test_nv12_passthrough_pipeline() {
    init();

    // Create a simple pipeline: videotestsrc (NV12) → edgeimpulsevideoinfer → fakesink
    let pipeline = gst::parse::launch(
        "videotestsrc num-buffers=5 ! video/x-raw,format=NV12,width=320,height=240 ! \
         edgeimpulsevideoinfer ! fakesink sync=false",
    )
    .expect("Failed to create pipeline");

    let pipeline = pipeline
        .dynamic_cast::<gst::Pipeline>()
        .expect("Not a pipeline");

    pipeline.set_state(gst::State::Playing).expect("Failed to set pipeline to Playing");

    let bus = pipeline.bus().expect("Pipeline has no bus");
    let mut eos = false;
    let mut error = false;

    for msg in bus.iter_timed(gst::ClockTime::from_seconds(10)) {
        match msg.view() {
            gst::MessageView::Eos(_) => {
                eos = true;
                break;
            }
            gst::MessageView::Error(err) => {
                eprintln!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                error = true;
                break;
            }
            _ => {}
        }
    }

    pipeline.set_state(gst::State::Null).expect("Failed to set pipeline to Null");

    assert!(eos, "Pipeline did not reach EOS");
    assert!(!error, "Pipeline encountered an error");
}

#[test]
fn test_nv12_caps_negotiation() {
    init();

    // Verify that the element accepts NV12 caps
    let factory = gst::ElementFactory::find("edgeimpulsevideoinfer")
        .expect("edgeimpulsevideoinfer not found — check GST_PLUGIN_PATH");

    let element = factory.create().build().expect("Failed to create element");

    let sink_pad = element.static_pad("sink").expect("No sink pad");
    let sink_caps = sink_pad.pad_template_caps();

    // Check that NV12 is in the accepted formats
    let nv12_caps = gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .field("width", 640i32)
        .field("height", 480i32)
        .build();

    assert!(
        sink_caps.can_intersect(&nv12_caps),
        "Sink pad should accept NV12 caps. Sink caps: {:?}",
        sink_caps
    );
}

#[test]
fn test_gbm_caps_negotiation() {
    init();

    let factory = gst::ElementFactory::find("edgeimpulsevideoinfer")
        .expect("edgeimpulsevideoinfer not found");

    let element = factory.create().build().expect("Failed to create element");

    let sink_pad = element.static_pad("sink").expect("No sink pad");
    let sink_caps = sink_pad.pad_template_caps();

    // Check that memory:GBM NV12 caps are accepted
    let mut gbm_caps = gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .field("width", 640i32)
        .field("height", 480i32)
        .build();
    {
        let caps_ref = gbm_caps.make_mut();
        caps_ref.set_features(0, Some(gst::CapsFeatures::new(["memory:GBM"])));
    }

    assert!(
        sink_caps.can_intersect(&gbm_caps),
        "Sink pad should accept memory:GBM NV12 caps. Sink caps: {:?}",
        sink_caps
    );
}

#[test]
fn test_rgb_still_works() {
    init();

    // Ensure existing RGB pipelines are not broken
    let pipeline = gst::parse::launch(
        "videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=320,height=240 ! \
         edgeimpulsevideoinfer ! fakesink sync=false",
    )
    .expect("Failed to create pipeline");

    let pipeline = pipeline.dynamic_cast::<gst::Pipeline>().unwrap();

    pipeline.set_state(gst::State::Playing).unwrap();

    let bus = pipeline.bus().unwrap();
    let mut eos = false;

    for msg in bus.iter_timed(gst::ClockTime::from_seconds(10)) {
        match msg.view() {
            gst::MessageView::Eos(_) => {
                eos = true;
                break;
            }
            gst::MessageView::Error(err) => {
                panic!(
                    "Error from {:?}: {}",
                    err.src().map(|s| s.path_string()),
                    err.error()
                );
            }
            _ => {}
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();
    assert!(eos, "Pipeline did not reach EOS");
}
