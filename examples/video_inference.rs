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
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use serde_json;
use std::error::Error;
use std::time::{Duration, Instant};

/// Command line parameters for the video classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct VideoClassifyParams {
    /// Path to Edge Impulse model file (required for EIM mode, ignored for FFI mode)
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

    /// Target FPS (0 for unlimited)
    #[arg(long, default_value = "30")]
    fps: u32,

    /// Enable performance monitoring
    #[arg(long)]
    perf: bool,

    /// Queue size for buffering (default: 4)
    #[arg(long, default_value = "4")]
    queue_size: u32,
}

// Performance tracking structure
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    frame_count: u64,
    total_inference_time: Duration,
    _total_pipeline_time: Duration,
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
            _total_pipeline_time: Duration::ZERO,
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

        let min_inference_time = self.inference_samples.iter().min().unwrap_or(&Duration::ZERO);
        let max_inference_time = self.inference_samples.iter().max().unwrap_or(&Duration::ZERO);

        println!("\n📊 PERFORMANCE SUMMARY:");
        println!("   Total frames processed: {}", self.frame_count);
        println!("   Total runtime: {:.2}s", total_time.as_secs_f64());
        println!("   Average FPS: {:.2}", avg_fps);
        println!("   Average inference time: {:.2}ms", avg_inference_time.as_millis());
        println!("   Min inference time: {:.2}ms", min_inference_time.as_millis());
        println!("   Max inference time: {:.2}ms", max_inference_time.as_millis());
        println!("   Total inference time: {:.2}s", self.total_inference_time.as_secs_f64());
        println!("   Inference efficiency: {:.1}%",
            (self.total_inference_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
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
#[allow(deprecated)]
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

    println!("[DEBUG] Creating full pipeline with Edge Impulse inference");

    // Create pipeline with performance optimizations
    let pipeline = gst::Pipeline::new();

    // Create elements with performance optimizations
    let src = gst::ElementFactory::make("avfvideosrc")
        .property("device-index", 0i32) // Use first camera
        .build()
        .expect("Could not create avfvideosrc element.");

    // Optimized queue with larger buffer
    let queue1 = gst::ElementFactory::make("queue")
        .property("max-size-buffers", args.queue_size as u32)
        .property("max-size-bytes", 0u32)
        .property("max-size-time", 0u64)
        .build()
        .expect("Could not create queue element.");

    // Optimized video conversion with threading
    let videoconvert1 = gst::ElementFactory::make("videoconvert")
        .property("n-threads", 8u32) // Use 8 threads for conversion
        .build()
        .expect("Could not create videoconvert element.");

    // Video scaling
    let videoscale1 = gst::ElementFactory::make("videoscale")
        .build()
        .expect("Could not create videoscale element.");

    // Caps filter for model input - only after conversion/scaling
    let caps1 = gst::ElementFactory::make("capsfilter")
        .property("caps", gst::Caps::builder("video/x-raw")
            .field("width", args.width as i32)
            .field("height", args.height as i32)
            .field("format", "RGB")
            .build())
        .build()
        .expect("Could not create capsfilter element.");

    // Queue for inference
    let queue2 = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 4u32)
        .build()
        .expect("Could not create queue element.");

    // Edge Impulse classifier
    let mut classifier_factory = gst::ElementFactory::make("edgeimpulsevideoinfer");

    // Set model path if provided (for EIM mode) or let FFI mode use statically linked model
    if let Some(ref model_path) = args.model {
        if args.debug {
            classifier_factory = classifier_factory.property("model-path-with-debug", model_path);
            println!("🔧 Setting model-path-with-debug: {}", model_path);
        } else {
            classifier_factory = classifier_factory.property("model-path", model_path);
            println!("🔧 Setting model-path: {}", model_path);
        }
    } else {
        println!("🔧 No model path provided - will use FFI mode with lazy loading");
    }

    // Set debug mode for FFI mode if requested
    if args.debug && args.model.is_none() {
        classifier_factory = classifier_factory.property("debug", true);
        println!("🔧 Setting debug mode for FFI inference");
    }

    // Set thresholds if provided
    for threshold in &args.threshold {
        classifier_factory = classifier_factory.property("threshold", threshold);
        println!("🔧 Setting threshold: {}", threshold);
    }

    let classifier = classifier_factory
        .build()
        .expect("Could not create edgeimpulsevideoinfer element.");

    // Queue for overlay
    let queue3 = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 4u32)
        .build()
        .expect("Could not create queue element.");

    // Video overlay for bounding boxes
    let overlay = gst::ElementFactory::make("edgeimpulseoverlay")
        .build()
        .expect("Could not create edgeimpulseoverlay element.");

    // Video scaling for display
    let videoscale2 = gst::ElementFactory::make("videoscale")
        .build()
        .expect("Could not create videoscale element.");

    // Caps filter for display
    let caps2 = gst::ElementFactory::make("capsfilter")
        .property("caps", gst::Caps::builder("video/x-raw")
            .field("width", 640i32)
            .field("height", 480i32)
            .build())
        .build()
        .expect("Could not create capsfilter element.");

    // Video conversion for display
    let videoconvert2 = gst::ElementFactory::make("videoconvert")
        .build()
        .expect("Could not create videoconvert element.");

    // Optimized sink with async processing
    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false) // Disable sync for better performance
        .build()
        .expect("Could not create autovideosink element.");

    // Add elements to the pipeline
    pipeline.add_many(&[
        &src,
        &queue1,
        &videoconvert1,
        &videoscale1,
        &caps1, // Caps filter after conversion/scaling
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

    // Check if we're in FFI mode and provide guidance
    if args.model.is_none() {
        let ei_project_id = std::env::var("EI_PROJECT_ID").ok();
        let ei_api_key = std::env::var("EI_API_KEY").ok();

        if ei_project_id.is_none() || ei_api_key.is_none() {
            println!("⚠️  FFI Mode: No model path provided, but environment variables are not set.");
            println!("   To use FFI mode, set the following environment variables:");
            println!("   export EI_PROJECT_ID=your_project_id");
            println!("   export EI_API_KEY=your_api_key");
            println!("   Then run: cargo run --example video_inference -- -W 224 -H 224 --debug");
            println!();
            println!("   Alternatively, use EIM mode with a model file:");
            println!("   cargo run --example video_inference -- --model /path/to/model.eim -W 224 -H 224 --debug");
            println!();
            println!("   Current environment:");
            println!("   EI_PROJECT_ID: {}", ei_project_id.as_deref().unwrap_or("not set"));
            println!("   EI_API_KEY: {}", if ei_api_key.is_some() { "set" } else { "not set" });
            println!();
            println!("   Continuing with FFI mode (model will be loaded lazily if env vars are set)...");
        } else {
            println!("✅ FFI Mode: Environment variables are set, model will be loaded automatically.");
        }
    } else {
        println!("✅ EIM Mode: Using model file: {}", args.model.as_ref().unwrap());
    }

    // Print performance configuration
    if args.perf {
        println!("🚀 Performance mode enabled:");
        println!("   Target FPS: {}", args.fps);
        println!("   Queue size: {}", args.queue_size);
        println!("   Threads: 8 (optimized)");
        println!("   Sync: disabled");
        println!("   Async: enabled");
    }

    let pipeline = create_pipeline(&args)?;

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    let mut perf_metrics = PerformanceMetrics::new();

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Element(element) => {
                let structure = element.structure().unwrap();
                if structure.name() == "edge-impulse-video-inference-result" {
                    // Extract timing information
                    let timing_ms = structure.get::<u32>("timing_ms").unwrap_or(0);
                    perf_metrics.update(timing_ms);

                    // Print performance info if enabled
                    if args.perf {
                        let current_fps = if !perf_metrics.fps_samples.is_empty() {
                            perf_metrics.fps_samples.last().unwrap()
                        } else {
                            &0.0
                        };
                        let avg_inference = if !perf_metrics.inference_samples.is_empty() {
                            perf_metrics.inference_samples.iter()
                                .rev()
                                .take(10)
                                .sum::<Duration>() / perf_metrics.inference_samples.iter().rev().take(10).count() as u32
                        } else {
                            Duration::ZERO
                        };

                        println!("📊 Frame #{} | FPS: {:.1} | Inference: {}ms | Avg: {:.1}ms",
                            perf_metrics.frame_count, current_fps, timing_ms, avg_inference.as_millis());
                    }

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

    // Print performance summary
    if args.perf {
        perf_metrics.print_summary();
    }

    Ok(())
}

fn main() {
    run(|| example_main().unwrap());
}
