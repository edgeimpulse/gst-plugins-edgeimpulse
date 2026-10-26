//! Integration tests for the edgeimpulsecrop element.
//!
//! Detections are attached by a pad probe instead of an inference element, so
//! these tests need no model. Run with the plugin on GST_PLUGIN_PATH, like
//! tests/ocr.rs:
//!   GST_PLUGIN_PATH="$(pwd)/target/release" cargo test --release --test crop
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use std::sync::{Arc, Mutex};

fn init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| gst::init().expect("gst init"));
}

fn crop_element_name() -> String {
    let variant = env!("PLUGIN_VARIANT");
    if variant.is_empty() {
        "edgeimpulsecrop".into()
    } else {
        format!("edgeimpulsecrop_{variant}")
    }
}

/// Run four 64x64 frames through the crop, attaching one detection to every
/// other frame, and return how many buffers reached the sink.
fn crop_output_count(crop_props: &str) -> usize {
    init();
    if gst::ElementFactory::find(&crop_element_name()).is_none() {
        panic!(
            "edgeimpulsecrop not found — build the plugin and set \
             GST_PLUGIN_PATH=\"$(pwd)/target/release\" (see the Run command)"
        );
    }

    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=4 ! video/x-raw,format=RGB,width=64,height=64 ! \
         identity name=tag ! {} {crop_props} ! appsink name=sink",
        crop_element_name()
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();

    let tag = pipeline.by_name("tag").unwrap();
    let frame = Arc::new(Mutex::new(0usize));
    tag.static_pad("src")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, move |_, info| {
            let mut frame = frame.lock().unwrap();
            if *frame % 2 == 0 {
                if let Some(buffer) = info.buffer_mut() {
                    let buffer = buffer.make_mut();
                    gst_video::VideoRegionOfInterestMeta::add(buffer, "object", (8, 8, 16, 16));
                }
            }
            *frame += 1;
            gst::PadProbeReturn::Ok
        });

    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gstreamer_app::AppSink>()
        .unwrap();
    let count = Arc::new(Mutex::new(0usize));
    let c2 = count.clone();
    sink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |s| {
                let _ = s.pull_sample().unwrap();
                *c2.lock().unwrap() += 1;
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    pipeline.set_state(gst::State::Playing).unwrap();
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(10)) {
        use gst::MessageView::*;
        match msg.view() {
            Eos(..) => break,
            Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
    let n = *count.lock().unwrap();
    n
}

#[test]
fn passes_frames_without_detections_through_by_default() {
    // Two crops plus two full frames.
    assert_eq!(crop_output_count(""), 4);
}

#[test]
fn drop_empty_drops_frames_without_detections() {
    // Only the two crops.
    assert_eq!(crop_output_count("drop-empty=true"), 2);
}

#[test]
fn drop_empty_property_round_trips() {
    init();
    let e = gst::ElementFactory::make(&crop_element_name())
        .build()
        .unwrap();
    assert!(!e.property::<bool>("drop-empty"));
    e.set_property("drop-empty", true);
    assert!(e.property::<bool>("drop-empty"));
}
