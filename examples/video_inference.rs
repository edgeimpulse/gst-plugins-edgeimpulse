//! Video Classification Example using edgeimpulseinfer GStreamer plugin
//!
//! This example demonstrates how to use the Edge Impulse GStreamer plugin to perform
//! video classification using a trained model.
//!
//! Usage:
//!   cargo run --example video_inference -- --model <path_to_model>
//!
//! Environment setup:
//! export GST_PLUGIN_PATH="target/debug:$GST_PLUGIN_PATH"

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use serde_json;
use std::error::Error;

/// Command line parameters for the video classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct VideoClassifyParams {
    /// Path to Edge Impulse model file
    #[arg(short, long)]
    model: String,

    /// Video format (RGB, RGBA, BGR, BGRA)
    #[arg(short, long, default_value = "RGB")]
    format: String,

    /// Input width
    #[arg(short = 'W', long)]
    width: i32,

    /// Input height
    #[arg(short = 'H', long)]
    height: i32,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,

    /// Model block thresholds in format 'blockId.type=value' (e.g., '5.min_score=0.6')
    #[clap(long)]
    threshold: Vec<String>,
}

// macOS specific run loop handling
#[cfg(not(target_os = "macos"))]
fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    main()
}

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)]
fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    use std::{
        ffi::c_void,
        sync::mpsc::{channel, Sender},
        thread,
    };

    use cocoa::{
        appkit::{NSApplication, NSWindow},
        base::id,
        delegate,
    };
    use objc::{
        msg_send,
        runtime::{Object, Sel},
        sel, sel_impl,
    };

    unsafe {
        let app = cocoa::appkit::NSApp();
        let (send, recv) = channel::<()>();

        extern "C" fn on_finish_launching(this: &Object, _cmd: Sel, _notification: id) {
            let send = unsafe {
                let send_pointer = *this.get_ivar::<*const c_void>("send");
                let boxed = Box::from_raw(send_pointer as *mut Sender<()>);
                *boxed
            };
            send.send(()).unwrap();
        }

        let delegate = delegate!("AppDelegate", {
            app: id = app,
            send: *const c_void = Box::into_raw(Box::new(send)) as *const c_void,
            (applicationDidFinishLaunching:) => on_finish_launching as extern "C" fn(&Object, Sel, id)
        });
        app.setDelegate_(delegate);

        let t = thread::spawn(move || {
            recv.recv().unwrap();
            let res = main();
            let app = cocoa::appkit::NSApp();
            app.stop_(cocoa::base::nil);
            let event = cocoa::appkit::NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                cocoa::base::nil,
                cocoa::appkit::NSEventType::NSApplicationDefined,
                cocoa::foundation::NSPoint { x: 0.0, y: 0.0 },
                cocoa::appkit::NSEventModifierFlags::empty(),
                0.0,
                0,
                cocoa::base::nil,
                cocoa::appkit::NSEventSubtype::NSApplicationActivatedEventType,
                0,
                0,
            );
            app.postEvent_atStart_(event, cocoa::base::YES);
            res
        });

        app.run();
        t.join().unwrap()
    }
}

fn create_pipeline(args: &VideoClassifyParams) -> Result<gst::Pipeline, Box<dyn Error>> {
    // Initialize GStreamer
    gst::init()?;

    // Create pipeline
    let pipeline = gst::Pipeline::new();

    // Create elements
    let src = gst::ElementFactory::make("avfvideosrc")
        .build()
        .expect("Could not create avfvideosrc element.");

    let queue1 = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 2u32)
        .property_from_str("leaky", "downstream")
        .build()
        .expect("Could not create queue element.");

    let videoconvert1 = gst::ElementFactory::make("videoconvert")
        .property("n-threads", 4u32)
        .build()
        .expect("Could not create videoconvert element.");

    let videoscale1 = gst::ElementFactory::make("videoscale")
        .build()
        .expect("Could not create videoscale element.");

    let caps1 = gst::ElementFactory::make("capsfilter")
        .build()
        .expect("Could not create capsfilter element.");

    let queue2 = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 2u32)
        .property_from_str("leaky", "downstream")
        .build()
        .expect("Could not create queue element.");

    let mut classifier_factory = if args.debug {
        gst::ElementFactory::make("edgeimpulsevideoinfer")
            .property("model-path-with-debug", &args.model)
    } else {
        gst::ElementFactory::make("edgeimpulsevideoinfer").property("model-path", &args.model)
    };

    // Set thresholds if provided
    for threshold in &args.threshold {
        classifier_factory = classifier_factory.property("threshold", threshold);
    }

    let classifier = classifier_factory
        .build()
        .expect("Could not create edgeimpulsevideoinfer element.");

    let queue3 = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 2u32)
        .property_from_str("leaky", "downstream")
        .build()
        .expect("Could not create queue element.");

    let overlay = gst::ElementFactory::make("edgeimpulseoverlay")
        .build()
        .expect("Could not create edgeimpulseoverlay element.");

    let videoscale2 = gst::ElementFactory::make("videoscale")
        .build()
        .expect("Could not create videoscale element.");

    let caps2 = gst::ElementFactory::make("capsfilter")
        .build()
        .expect("Could not create capsfilter element.");

    let videoconvert2 = gst::ElementFactory::make("videoconvert")
        .property("n-threads", 4u32)
        .build()
        .expect("Could not create videoconvert element.");

    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()
        .expect("Could not create autovideosink element.");

    // Set caps using provided dimensions
    let caps1_struct = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", args.width)
        .field("height", args.height)
        .build();
    caps1.set_property("caps", &caps1_struct);

    let caps2_struct = gst::Caps::builder("video/x-raw")
        .field("width", 480i32)
        .field("height", 480i32)
        .build();
    caps2.set_property("caps", &caps2_struct);

    // Add elements to the pipeline
    pipeline.add_many(&[
        &src,
        &queue1,
        &videoconvert1,
        &videoscale1,
        &caps1,
        &queue2,
        &classifier,
        &queue3,
        &overlay,
        &videoscale2,
        &caps2,
        &videoconvert2,
        &sink,
    ])?;

    // Link the elements
    gst::Element::link_many(&[
        &src,
        &queue1,
        &videoconvert1,
        &videoscale1,
        &caps1,
        &queue2,
        &classifier,
        &queue3,
        &overlay,
        &videoscale2,
        &caps2,
        &videoconvert2,
        &sink,
    ])?;

    // Add debug probe to check ROI metadata
    let overlay_sink_pad = overlay.static_pad("sink").unwrap();
    overlay_sink_pad.add_probe(gst::PadProbeType::BUFFER, move |_, probe_info| {
        if let Some(buffer) = probe_info.buffer() {
            let rois: Vec<_> = buffer
                .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
                .collect();
            println!("Number of ROIs on buffer: {}", rois.len());

            for roi in rois {
                let (x, y, w, h) = roi.rect();
                println!("ROI: {} at ({}, {}, {}, {})", roi.roi_type(), x, y, w, h);

                // Iterate through all parameters
                for param in roi.params() {
                    println!("ROI param: {:?}", param);
                }
            }
        }
        gst::PadProbeReturn::Ok
    });

    Ok(pipeline)
}

fn example_main() -> Result<(), Box<dyn Error>> {
    let args = VideoClassifyParams::parse();

    let pipeline = create_pipeline(&args)?;

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Element(element) => {
                let structure = element.structure().unwrap();
                if structure.name() == "edge-impulse-video-inference-result" {
                    println!("Inference result: {:?}", structure);
                    if let Ok(result) = structure.get::<String>("result") {
                        // Parse the JSON string
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                            // Handle object detection (array of objects)
                            if let Some(boxes) = json["bounding_boxes"].as_array() {
                                println!("Object detection results:");
                                for bbox in boxes {
                                    if let (Some(label), Some(value)) =
                                        (bbox["label"].as_str(), bbox["value"].as_f64())
                                    {
                                        println!(
                                            "  {}: {:.1}% (x: {}, y: {}, w: {}, h: {})",
                                            label,
                                            value * 100.0,
                                            bbox["x"].as_f64().unwrap_or(0.0),
                                            bbox["y"].as_f64().unwrap_or(0.0),
                                            bbox["width"].as_f64().unwrap_or(0.0),
                                            bbox["height"].as_f64().unwrap_or(0.0)
                                        );
                                    }
                                }
                            }
                            // Handle visual anomaly detection (object with grid: array of objects)
                            else if let Some(anomaly) = json["anomaly"].as_f64() {
                                println!("Anomaly score: {:.2}%", anomaly * 100.0);
                                if let Some(classification) = json["classification"].as_object() {
                                    for (label, value) in classification {
                                        if let Some(conf) = value.as_f64() {
                                            println!("  Class: {} ({:.1}%)", label, conf * 100.0);
                                        }
                                    }
                                }
                                if let Some(grid) = json["visual_anomaly_grid"].as_array() {
                                    println!("Visual anomaly grid ({} cells):", grid.len());
                                    for cell in grid {
                                        let x = cell["x"].as_u64().unwrap_or(0);
                                        let y = cell["y"].as_u64().unwrap_or(0);
                                        let width = cell["width"].as_u64().unwrap_or(0);
                                        let height = cell["height"].as_u64().unwrap_or(0);
                                        let score = cell
                                            .get("score")
                                            .and_then(|v| v.as_f64())
                                            .or_else(|| cell.get("value").and_then(|v| v.as_f64()));
                                        if let Some(score) = score {
                                            println!(
                                                "    Cell at ({}, {}) size {}x{}: score {:.2}%",
                                                x, y, width, height, score
                                            );
                                        } else {
                                            println!(
                                                "    Cell at ({}, {}) size {}x{}: score N/A",
                                                x, y, width, height
                                            );
                                        }
                                    }
                                }
                            }
                            // Handle classification (object only)
                            else if let Some(classification) = json["classification"].as_object()
                            {
                                println!("Classification results:");
                                for (label, value) in classification {
                                    if let Some(conf) = value.as_f64() {
                                        println!("  {}: {:.1}%", label, conf * 100.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            MessageView::Error(err) => {
                println!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            MessageView::Eos(..) => break,
            _ => (),
        }
    }

    pipeline.set_state(gst::State::Null)?;
    println!("Pipeline stopped");

    Ok(())
}

fn main() {
    run(|| example_main().unwrap());
}
