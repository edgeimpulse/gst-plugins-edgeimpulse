//! Integration tests for the edgeimpulseocr element.
use gstreamer as gst;
use gstreamer::prelude::*;
use std::sync::{Arc, Mutex};

fn init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| gst::init().expect("gst init"));
}

fn ocr_element_name() -> String {
    let variant = env!("PLUGIN_VARIANT");
    if variant.is_empty() {
        "edgeimpulseocr".into()
    } else {
        format!("edgeimpulseocr_{variant}")
    }
}

#[test]
fn passes_buffers_through_unchanged() {
    init();
    // The plugin is a cdylib GStreamer loads from GST_PLUGIN_PATH (see the Run
    // command below), exactly like tests/e2e.rs. Integration tests do NOT link
    // the crate (a cdylib has no rlib artifact), so there is nothing to register
    // in-process — the element must be discoverable on GST_PLUGIN_PATH.
    if gst::ElementFactory::find(&ocr_element_name()).is_none() {
        panic!(
            "edgeimpulseocr not found — build the plugin and set \
             GST_PLUGIN_PATH=\"$(pwd)/target/release\" (see the Run command)"
        );
    }

    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=64,height=64 ! \
         videoconvert ! {} ! appsink name=sink",
        ocr_element_name()
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();

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
    assert_eq!(*count.lock().unwrap(), 3);
}

#[test]
fn exposes_configurable_properties() {
    init();
    let e = gst::ElementFactory::make(&ocr_element_name())
        .build()
        .unwrap();
    e.set_property("backend", "tesseract");
    e.set_property("min-confidence", 0.5f64);
    e.set_property("max-text-length", 32u32);
    e.set_property("post-message", false);
    e.set_property("interval", 5u32);
    assert_eq!(e.property::<String>("backend"), "tesseract");
    assert_eq!(e.property::<f64>("min-confidence"), 0.5);
    assert_eq!(e.property::<u32>("max-text-length"), 32);
    assert!(!e.property::<bool>("post-message"));
    assert_eq!(e.property::<u32>("interval"), 5);
}

#[test]
fn noop_backend_attaches_no_metas() {
    init();
    let got_meta = Arc::new(Mutex::new(false));
    let gm = got_meta.clone();
    // `edge-impulse` currently maps to the no-op backend (see `build_backend`),
    // giving a deterministic "recognizes nothing" path — this checks the element
    // runs in a real pipeline and attaches no ROI metas when there is no result.
    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=2 ! video/x-raw,format=RGB,width=80,height=48 ! \
         videoconvert ! {} backend=edge-impulse interval=1 ! appsink name=sink",
        ocr_element_name()
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gstreamer_app::AppSink>()
        .unwrap();
    sink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |s| {
                let sample = s.pull_sample().unwrap();
                if let Some(buf) = sample.buffer() {
                    if buf
                        .iter_meta::<gstreamer_video::VideoRegionOfInterestMeta>()
                        .count()
                        > 0
                    {
                        *gm.lock().unwrap() = true;
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    pipeline.set_state(gst::State::Playing).unwrap();
    for msg in pipeline
        .bus()
        .unwrap()
        .iter_timed(gst::ClockTime::from_seconds(10))
    {
        use gst::MessageView::*;
        match msg.view() {
            Eos(..) => break,
            Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
    assert!(
        !*got_meta.lock().unwrap(),
        "Noop backend must not attach metas"
    );
}

/// End-to-end check of the ocrs recognition path. Recognition runs on a worker
/// thread, so results land on a *later* output buffer than the frame that
/// triggered them; feeding the same text frame on a loop (`imagefreeze`) lets
/// the worker catch up and attach a `VideoRegionOfInterestMeta` whose `label`
/// carries the recognized text. Skipped unless `OCR_MODELS_DIR` (containing
/// `text-detection.rten` / `text-recognition.rten`) and `OCR_TEST_IMAGE` (an
/// image with legible text) are both set, so CI — which has no models — stays
/// green.
#[test]
fn worker_recognizes_text_and_attaches_results() {
    init();
    let (Ok(models), Ok(image)) = (
        std::env::var("OCR_MODELS_DIR"),
        std::env::var("OCR_TEST_IMAGE"),
    ) else {
        eprintln!("skipping: set OCR_MODELS_DIR and OCR_TEST_IMAGE to run");
        return;
    };
    let found = Arc::new(Mutex::new(Vec::<String>::new()));
    let f = found.clone();
    let pipeline = gst::parse::launch(&format!(
        "filesrc location={image} ! decodebin ! imagefreeze ! videoconvert ! \
         video/x-raw,format=RGB ! {elem} backend=ocrs interval=1 \
         detection-model={models}/text-detection.rten \
         recognition-model={models}/text-recognition.rten ! \
         appsink name=sink max-buffers=2 drop=true",
        elem = ocr_element_name(),
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gstreamer_app::AppSink>()
        .unwrap();
    sink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |s| {
                let sample = s.pull_sample().unwrap();
                if let Some(buf) = sample.buffer() {
                    for m in buf.iter_meta::<gstreamer_video::VideoRegionOfInterestMeta>() {
                        if let Some(p) = m.params().find(|p| p.name() == "detection") {
                            if let Ok(l) = p.get::<String>("label") {
                                f.lock().unwrap().push(l);
                            }
                        }
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    pipeline.set_state(gst::State::Playing).unwrap();
    let start = std::time::Instant::now();
    while found.lock().unwrap().is_empty() && start.elapsed().as_secs() < 20 {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    pipeline.set_state(gst::State::Null).unwrap();
    let labels = found.lock().unwrap();
    eprintln!("recognized labels: {labels:?}");
    assert!(
        !labels.is_empty(),
        "worker never attached a recognized line within 20s (did the plugin \
         build with --features ocrs, and are the .rten models in OCR_MODELS_DIR?)"
    );
}
