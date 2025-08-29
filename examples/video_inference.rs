#![allow(deprecated)]

//! Video Classification Example using edgeimpulseinfer GStreamer plugin
//!
//! This example demonstrates how to use the Edge Impulse GStreamer plugin to perform
//! video classification using a trained model with performance optimizations.
//!
//! Usage:
//!   # EIM mode (requires model path):
//!   cargo run --example video_inference -- --model <path_to_model>
//!
//!   # FFI mode (no model path needed):
//!   cargo run --example video_inference
//!
//! Environment setup:
//! export GST_PLUGIN_PATH="target/debug:$GST_PLUGIN_PATH"

use clap::Parser;
use edge_impulse_runner::ffi::ModelMetadata;
use gstreamer as gst;
use gstreamer::prelude::*;
use serde_json;
use std::error::Error;
use std::thread;
use std::time::{Duration, Instant};

/// Command line parameters for the video classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct VideoClassifyParams {
    /// Path to Edge Impulse model file (optional for FFI mode)
    #[arg(short, long)]
    model: Option<String>,

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

    /// Enable performance monitoring
    #[arg(long)]
    perf: bool,



    /// Stroke width for bounding boxes
    #[arg(long, default_value = "2")]
    stroke_width: i32,

    /// Text color in hex format (e.g., 0xFFFFFF for white)
    #[arg(long, default_value = "0xFFFFFF")]
    text_color: String,

    /// Background color in hex format (e.g., 0x000000 for black)
    #[arg(long, default_value = "0x000000")]
    background_color: String,

    /// Text scale ratio (0.1 to 5.0). Values > 1.0 make text larger, < 1.0 make text smaller
    #[arg(long, default_value = "1.0")]
    text_scale_ratio: f64,
}

// Performance tracking structure
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    frame_count: u64,
    total_inference_time: Duration,
    start_time: Instant,
    last_fps_time: Instant,
    fps_samples: Vec<f64>,
    inference_samples: Vec<Duration>,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            frame_count: 0,
            total_inference_time: Duration::ZERO,
            start_time: Instant::now(),
            last_fps_time: Instant::now(),
            fps_samples: Vec::new(),
            inference_samples: Vec::new(),
        }
    }

    fn update(&mut self, inference_time_ms: u32) {
        self.frame_count += 1;
        let inference_time = Duration::from_millis(inference_time_ms as u64);
        self.total_inference_time += inference_time;
        self.inference_samples.push(inference_time);

        // Keep only last 100 samples for rolling average
        if self.inference_samples.len() > 100 {
            self.inference_samples.remove(0);
        }

        // Calculate FPS every second
        let now = Instant::now();
        if now.duration_since(self.last_fps_time) >= Duration::from_secs(1) {
            let fps = self.frame_count as f64 / now.duration_since(self.start_time).as_secs_f64();
            self.fps_samples.push(fps);

            // Keep only last 10 FPS samples
            if self.fps_samples.len() > 10 {
                self.fps_samples.remove(0);
            }

            self.last_fps_time = now;
        }
    }

    fn print_summary(&self) {
        let total_time = self.start_time.elapsed();
        let avg_fps = if !self.fps_samples.is_empty() {
            self.fps_samples.iter().sum::<f64>() / self.fps_samples.len() as f64
        } else {
            0.0
        };

        let avg_inference_time = if !self.inference_samples.is_empty() {
            self.inference_samples.iter().sum::<Duration>() / self.inference_samples.len() as u32
        } else {
            Duration::ZERO
        };

        let min_inference_time = self
            .inference_samples
            .iter()
            .min()
            .unwrap_or(&Duration::ZERO);
        let max_inference_time = self
            .inference_samples
            .iter()
            .max()
            .unwrap_or(&Duration::ZERO);

        println!("\n📊 PERFORMANCE SUMMARY:");
        println!("   Total frames processed: {}", self.frame_count);
        println!("   Total runtime: {:.2}s", total_time.as_secs_f64());
        println!("   Average FPS: {:.2}", avg_fps);
        println!(
            "   Average inference time: {:.2}ms",
            avg_inference_time.as_millis()
        );
        println!(
            "   Min inference time: {:.2}ms",
            min_inference_time.as_millis()
        );
        println!(
            "   Max inference time: {:.2}ms",
            max_inference_time.as_millis()
        );
        println!(
            "   Total inference time: {:.2}s",
            self.total_inference_time.as_secs_f64()
        );
        println!(
            "   Inference efficiency: {:.1}%",
            (self.total_inference_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
        );
    }
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

    // Check for FFI mode
    let is_ffi_mode = args.model.is_none();
    if is_ffi_mode {
        println!("✅ FFI Mode: No model path provided, will use FFI backend");
    } else {
        println!(
            "🔧 EIM Mode: Using model path: {}",
            args.model.as_ref().unwrap()
        );
    }

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

    // Create classifier based on mode
    let mut classifier_factory = gst::ElementFactory::make("edgeimpulsevideoinfer");

    if is_ffi_mode {
        if args.debug {
            classifier_factory = classifier_factory.property("debug", true);
            println!("🔧 Setting debug mode for FFI inference");
        }
    } else {
        if args.debug {
            classifier_factory =
                classifier_factory.property("model-path-with-debug", args.model.as_ref().unwrap());
            println!(
                "🔧 Setting model-path-with-debug: {}",
                args.model.as_ref().unwrap()
            );
        } else {
            classifier_factory =
                classifier_factory.property("model-path", args.model.as_ref().unwrap());
            println!("🔧 Setting model-path: {}", args.model.as_ref().unwrap());
        }
    }

    // Set thresholds if provided
    for threshold in &args.threshold {
        classifier_factory = classifier_factory.property("threshold", threshold);
    }

    let classifier = classifier_factory
        .build()
        .expect("Could not create edgeimpulsevideoinfer element.");

    // Print model metadata when available
    if let Some(path) = classifier.property::<Option<String>>("model-path") {
        println!("📁 Model path: {}", path);
    }

    // Try to get model parameters after a short delay to allow model loading
    let classifier_clone = classifier.clone();
    thread::spawn(move || {
        thread::sleep(std::time::Duration::from_millis(500));
        let debug_enabled = classifier_clone.property::<bool>("debug");
        println!("🔧 Debug mode: {}", debug_enabled);
    });

    let queue3 = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 2u32)
        .property_from_str("leaky", "downstream")
        .build()
        .expect("Could not create queue element.");

    let overlay = gst::ElementFactory::make("edgeimpulseoverlay")
        .property("stroke-width", &args.stroke_width)
        .property(
            "text-color",
            &u32::from_str_radix(&args.text_color[2..], 16).unwrap_or(0xFFFFFF),
        )
        .property(
            "background-color",
            &u32::from_str_radix(&args.background_color[2..], 16).unwrap_or(0x000000),
        )
        .property("text-scale-ratio", &args.text_scale_ratio)
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

    Ok(pipeline)
}

fn example_main() -> Result<(), Box<dyn Error>> {
    let args = VideoClassifyParams::parse();

    let pipeline = create_pipeline(&args)?;

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    let mut perf_metrics = PerformanceMetrics::new();

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Element(element) => {
                let structure = element.structure().unwrap();

                // Print model loaded message
                if structure.name() == "edge-impulse-model-loaded" {
                    println!("✅ Model loaded successfully!");
                    if let Ok(model_type) = structure.get::<String>("model-type") {
                        println!("📊 Model type: {}", model_type);
                    }
                    if let Ok(input_width) = structure.get::<u32>("input-width") {
                        println!("📏 Input width: {}", input_width);
                    }
                    if let Ok(input_height) = structure.get::<u32>("input-height") {
                        println!("📏 Input height: {}", input_height);
                    }
                    if let Ok(channel_count) = structure.get::<u32>("channel-count") {
                        println!("🎨 Channel count: {}", channel_count);
                    }
                    if let Ok(has_anomaly) = structure.get::<bool>("has-anomaly") {
                        println!("🔍 Has anomaly detection: {}", has_anomaly);
                    }
                }

                // Print model info from inference results (for FFI mode)
                if structure.name() == "edge-impulse-video-inference-result"
                    && perf_metrics.frame_count == 1
                {
                    println!("🔍 Model Info (from first inference):");
                    if let Ok(result_type) = structure.get::<String>("type") {
                        println!("   📊 Result type: {}", result_type);
                    }

                    // Get project ID from model metadata
                    let metadata = ModelMetadata::get();
                    println!("   🆔 Project ID: {}", metadata.project_id);
                    println!("   📋 Project Name: {}", metadata.project_name);
                    println!("   👤 Project Owner: {}", metadata.project_owner);
                    println!("   🏷️  Deploy Version: {}", metadata.deploy_version);

                    if let Ok(result) = structure.get::<String>("result") {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                            if let Some(boxes) = json["bounding_boxes"].as_array() {
                                println!("   🎯 Object detection model");
                                println!("   📦 Bounding boxes: {} objects", boxes.len());
                            } else if json.get("anomaly").is_some() {
                                println!("   🔍 Anomaly detection model");
                            } else if let Some(classification) = json["classification"].as_object()
                            {
                                println!("   🏷️  Classification model");
                                println!("   📋 Classes: {}", classification.len());
                            }
                        }
                    }
                }

                if structure.name() == "edge-impulse-video-inference-result" {
                    // Extract timing information for performance monitoring
                    let timing_ms = structure.get::<u32>("timing_ms").unwrap_or(0);
                    perf_metrics.update(timing_ms);

                    // Always print performance info
                    let current_fps = if !perf_metrics.fps_samples.is_empty() {
                        perf_metrics.fps_samples.last().unwrap()
                    } else {
                        &0.0
                    };
                    let avg_inference = if !perf_metrics.inference_samples.is_empty() {
                        perf_metrics
                            .inference_samples
                            .iter()
                            .rev()
                            .take(10)
                            .sum::<Duration>()
                            / perf_metrics.inference_samples.iter().rev().take(10).count() as u32
                    } else {
                        Duration::ZERO
                    };

                    println!("🎯 Frame #{:3} | FPS: {:4.1} | Inference: {:3}ms | Avg: {:4.1}ms | Confidence: {:.1}%",
                        perf_metrics.frame_count,
                        current_fps,
                        timing_ms,
                        avg_inference.as_millis(),
                        if let Ok(result) = structure.get::<String>("result") {
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                                if let Some(boxes) = json["bounding_boxes"].as_array() {
                                    if let Some(first_box) = boxes.first() {
                                        first_box["value"].as_f64().unwrap_or(0.0) * 100.0
                                    } else { 0.0 }
                                } else { 0.0 }
                            } else { 0.0 }
                        } else { 0.0 }
                    );

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

    perf_metrics.print_summary();
    Ok(())
}

fn main() {
    run(|| example_main().unwrap());
}
