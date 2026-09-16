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
fn edge_impulse_without_upstream_detections_attaches_no_metas() {
    init();
    let got_meta = Arc::new(Mutex::new(false));
    let gm = got_meta.clone();
    // With no upstream detections on the buffers (plain videotestsrc), the
    // edge-impulse backend has nothing to decode, so it must attach no ROI metas.
    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=2 ! video/x-raw,format=RGB,width=80,height=48 ! \
         videoconvert ! {} backend=edge-impulse-characters interval=1 ! appsink name=sink",
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
        "edge-impulse must not attach metas when there are no upstream detections"
    );
}

#[test]
fn edge_impulse_decodes_upstream_detections() {
    init();
    if gst::ElementFactory::find(&ocr_element_name()).is_none() {
        panic!(
            "edgeimpulseocr not found — build the plugin and set \
             GST_PLUGIN_PATH=\"$(pwd)/target/debug\""
        );
    }

    // Labels of ROI metas seen on output buffers, read from their `detection`
    // param (matching how downstream consumers read them).
    let out_labels = Arc::new(Mutex::new(Vec::<String>::new()));
    let ol = out_labels.clone();

    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=80,height=48 ! \
         videoconvert ! {elem} name=ocr backend=edge-impulse-characters interval=1 ! \
         appsink name=sink",
        elem = ocr_element_name(),
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();

    // Mimic an upstream edgeimpulsevideoinfer: inject two character detections
    // ("A" at x=0, "B" at x=20, same row) on the ocr element's sink pad.
    let ocr = pipeline.by_name("ocr").unwrap();
    let sinkpad = ocr.static_pad("sink").unwrap();
    sinkpad.add_probe(gst::PadProbeType::BUFFER, |_pad, info| {
        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
            let buf = buffer.make_mut();
            for (label, x, conf) in [("A", 0u32, 0.9f64), ("B", 20u32, 0.8f64)] {
                let mut roi =
                    gstreamer_video::VideoRegionOfInterestMeta::add(buf, label, (x, 10, 10, 20));
                roi.add_param(
                    gst::Structure::builder("detection")
                        .field("label", label)
                        .field("confidence", conf)
                        .build(),
                );
            }
        }
        gst::PadProbeReturn::Ok
    });

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
                                ol.lock().unwrap().push(l);
                            }
                        }
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    let ocr_texts = Arc::new(Mutex::new(Vec::<String>::new()));
    let ocr_frame_dims = Arc::new(Mutex::new(Vec::<(i32, i32)>::new()));
    let ot = ocr_texts.clone();
    let ofd = ocr_frame_dims.clone();
    pipeline.set_state(gst::State::Playing).unwrap();
    for msg in pipeline
        .bus()
        .unwrap()
        .iter_timed(gst::ClockTime::from_seconds(10))
    {
        use gst::MessageView::*;
        match msg.view() {
            Element(e) => {
                if let Some(st) = e.structure() {
                    if st.name() == "ocr" {
                        ot.lock().unwrap().push(st.get::<String>("text").unwrap());
                        ofd.lock().unwrap().push((
                            st.get::<i32>("frame_width").unwrap(),
                            st.get::<i32>("frame_height").unwrap(),
                        ));
                    }
                }
            }
            Eos(..) => break,
            Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();

    assert_eq!(
        *ocr_texts.lock().unwrap(),
        vec!["AB".to_string()],
        "expected exactly one ocr message with the assembled text"
    );
    assert_eq!(
        *ocr_frame_dims.lock().unwrap(),
        vec![(80, 48)],
        "ocr message dimensions must match negotiated caps"
    );
    assert_eq!(
        *out_labels.lock().unwrap(),
        vec!["AB".to_string()],
        "character ROIs must be consumed and replaced by a single line ROI"
    );
}

#[test]
fn edge_impulse_interval_throttles_ocr_messages() {
    init();
    if gst::ElementFactory::find(&ocr_element_name()).is_none() {
        panic!(
            "edgeimpulseocr not found — build the plugin and set \
             GST_PLUGIN_PATH=\"$(pwd)/target/debug\""
        );
    }

    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=80,height=48 ! \
         videoconvert ! {elem} name=ocr backend=edge-impulse-characters interval=5 ! \
         appsink name=sink",
        elem = ocr_element_name(),
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();

    // Inject one decodable detection on every buffer.
    let ocr = pipeline.by_name("ocr").unwrap();
    let sinkpad = ocr.static_pad("sink").unwrap();
    sinkpad.add_probe(gst::PadProbeType::BUFFER, |_pad, info| {
        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
            let buf = buffer.make_mut();
            let mut roi =
                gstreamer_video::VideoRegionOfInterestMeta::add(buf, "X", (0, 10, 10, 20));
            roi.add_param(
                gst::Structure::builder("detection")
                    .field("label", "X")
                    .field("confidence", 0.9f64)
                    .build(),
            );
        }
        gst::PadProbeReturn::Ok
    });

    // Drain output buffers so the pipeline runs to EOS.
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gstreamer_app::AppSink>()
        .unwrap();
    sink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |s| {
                let _ = s.pull_sample().unwrap();
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    let ocr_count = Arc::new(Mutex::new(0usize));
    let oc = ocr_count.clone();
    pipeline.set_state(gst::State::Playing).unwrap();
    for msg in pipeline
        .bus()
        .unwrap()
        .iter_timed(gst::ClockTime::from_seconds(10))
    {
        use gst::MessageView::*;
        match msg.view() {
            Element(e) => {
                if e.structure().map_or(false, |st| st.name() == "ocr") {
                    *oc.lock().unwrap() += 1;
                }
            }
            Eos(..) => break,
            Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();

    assert_eq!(
        *ocr_count.lock().unwrap(),
        1,
        "interval=5 over 5 buffers must post exactly one ocr message"
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
