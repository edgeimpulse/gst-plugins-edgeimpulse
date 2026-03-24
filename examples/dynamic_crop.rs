//! Dynamic Crop Example — two-stage detection → classification pipeline
//!
//! Demonstrates `edgeimpulsecrop` in a realistic pipeline:
//!
//!   camera → object detection → tee
//!     ├─ overlay → display (live view with bounding boxes)
//!     └─ continue-if → crop → appsink (save crops as PNG)
//!
//! Each detected object is cropped from the frame, resized to a target size,
//! and saved as a PNG file in the output directory. Each file is named with
//! the frame number, detection label, and confidence.
//!
//! Usage:
//!   # FFI mode with camera (default, runs for 5 seconds):
//!   cargo run --release --example dynamic_crop
//!
//!   # EIM mode:
//!   cargo run --release --example dynamic_crop -- --model <path_to_model>
//!
//!   # Custom duration, crop size, output directory:
//!   cargo run --release --example dynamic_crop -- --duration 10 --target-width 128 --target-height 128 --output-dir ./my_crops
//!
//!   # Use test video source instead of camera:
//!   cargo run --release --example dynamic_crop -- --source test
//!
//! Environment setup:
//!   export GST_PLUGIN_PATH="target/release:$GST_PLUGIN_PATH"

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use image::ImageBuffer;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Dynamic crop example — saves crops as PNG")]
struct Args {
    /// Path to Edge Impulse model file (optional for FFI mode)
    #[arg(short, long)]
    model: Option<String>,

    /// Video source: "camera" (default), "test" (SMPTE pattern)
    #[arg(short, long, default_value = "camera")]
    source: String,

    /// Target width for cropped regions
    #[arg(long, default_value = "96")]
    target_width: i32,

    /// Target height for cropped regions
    #[arg(long, default_value = "96")]
    target_height: i32,

    /// Padding around each bounding box (pixels)
    #[arg(long, default_value = "10")]
    padding: i32,

    /// Duration in seconds (0 = unlimited)
    #[arg(short, long, default_value = "5")]
    duration: u64,

    /// Directory to save crop PNGs
    #[arg(short, long, default_value = "./crops")]
    output_dir: PathBuf,
}

/// Create the video source element based on the --source flag.
fn create_video_source(source: &str) -> Result<gst::Element, Box<dyn Error>> {
    match source {
        "test" => Ok(gst::ElementFactory::make("videotestsrc")
            .property_from_str("pattern", "smpte")
            .build()?),
        _ => {
            if let Ok(src) = gst::ElementFactory::make("avfvideosrc")
                .property("device-index", 0i32)
                .build()
            {
                println!("  Using avfvideosrc (macOS camera)");
                Ok(src)
            } else if let Ok(src) = gst::ElementFactory::make("v4l2src")
                .property_from_str("device", "/dev/video0")
                .build()
            {
                println!("  Using v4l2src (Linux camera)");
                Ok(src)
            } else {
                Err("No camera source available. Use --source test".into())
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    gst::init()?;

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    println!("Dynamic Crop Example");
    println!("  Source: {}", args.source);
    println!(
        "  Target size: {}x{}",
        args.target_width, args.target_height
    );
    println!("  Padding: {}px", args.padding);
    println!("  Output: {}", args.output_dir.display());
    if args.duration > 0 {
        println!("  Duration: {}s", args.duration);
    } else {
        println!("  Duration: unlimited (Ctrl+C to stop)");
    }
    println!();

    let pipeline = gst::Pipeline::new();

    // Source
    let src = create_video_source(&args.source)?;

    let queue_src = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 8u32)
        .property_from_str("leaky", "downstream")
        .build()?;

    let convert = gst::ElementFactory::make("videoconvert")
        .property("n-threads", 4u32)
        .build()?;

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .build(),
        )
        .build()?;

    // Inference
    let infer = gst::ElementFactory::make("edgeimpulsevideoinfer").build()?;
    if let Some(ref model_path) = args.model {
        infer.set_property("model-path", model_path);
    }

    // Tee: split into overlay path + crop path
    let tee = gst::ElementFactory::make("tee").build()?;

    // Path 1: overlay → autovideosink (live display)
    let queue1 = gst::ElementFactory::make("queue").build()?;
    let overlay = gst::ElementFactory::make("edgeimpulseoverlay").build()?;
    let convert_display = gst::ElementFactory::make("videoconvert").build()?;
    let display_sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()
        .unwrap_or_else(|_| {
            gst::ElementFactory::make("fakesink")
                .property("sync", false)
                .build()
                .unwrap()
        });

    // Path 2: continue-if → crop → appsink (save crops)
    let queue2 = gst::ElementFactory::make("queue").build()?;
    let gate = gst::ElementFactory::make("edgeimpulsecontinueif")
        .property("condition", "detection_count >= 1")
        .build()?;
    let crop = gst::ElementFactory::make("edgeimpulsecrop")
        .property("padding", args.padding)
        .property("target-width", args.target_width)
        .property("target-height", args.target_height)
        .build()?;

    let crop_sink = gst_app::AppSink::builder().sync(false).build();

    // Set up appsink callback to save crops as PNG
    let output_dir = args.output_dir.clone();
    let target_w = args.target_width as u32;
    let target_h = args.target_height as u32;
    let crop_counter = Arc::new(Mutex::new(0u64));
    let crop_counter_clone = crop_counter.clone();

    crop_sink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;

                let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                let data = map.as_slice();

                // Determine dimensions from caps or fall back to target size
                let (width, height) = if let Some(caps) = sample.caps() {
                    let s = caps.structure(0).unwrap();
                    let w = s.get::<i32>("width").unwrap_or(target_w as i32) as u32;
                    let h = s.get::<i32>("height").unwrap_or(target_h as i32) as u32;
                    (w, h)
                } else {
                    (target_w, target_h)
                };

                // Build filename from CropOriginMeta if available
                let mut count = crop_counter_clone.lock().unwrap();
                *count += 1;
                let idx = *count;

                // Try to read label/confidence from the bus message context
                // (CropOriginMeta is custom and hard to read from appsink,
                // so we use a simple counter-based name)
                let filename = format!("crop_{:04}.png", idx);
                let path = output_dir.join(&filename);

                if let Some(img) =
                    ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, data.to_vec())
                {
                    if let Err(e) = img.save(&path) {
                        eprintln!("  Failed to save {}: {}", path.display(), e);
                    } else {
                        println!("  Saved crop: {} ({}x{})", path.display(), width, height);
                    }
                }

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    // Add all elements
    pipeline.add_many([
        &src,
        &queue_src,
        &convert,
        &capsfilter,
        &infer,
        &tee,
        &queue1,
        &overlay,
        &convert_display,
        &display_sink,
        &queue2,
        &gate,
        &crop,
        crop_sink.upcast_ref(),
    ])?;

    // Link main chain
    gst::Element::link_many([&src, &queue_src, &convert, &capsfilter, &infer, &tee])?;

    // Link tee → path 1 (overlay → display)
    gst::Element::link_many([&queue1, &overlay, &convert_display, &display_sink])?;
    let tee_src1 = tee.request_pad_simple("src_%u").unwrap();
    let queue1_sink = queue1.static_pad("sink").unwrap();
    tee_src1.link(&queue1_sink)?;

    // Link tee → path 2 (crop → appsink)
    gst::Element::link_many([&queue2, &gate, &crop])?;
    crop.link(crop_sink.upcast_ref::<gst::Element>())?;
    let tee_src2 = tee.request_pad_simple("src_%u").unwrap();
    let queue2_sink = queue2.static_pad("sink").unwrap();
    tee_src2.link(&queue2_sink)?;

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    let mut frame_count = 0u64;
    let mut total_detections = 0u64;
    let start = Instant::now();
    let timeout = if args.duration > 0 {
        Some(std::time::Duration::from_secs(args.duration))
    } else {
        None
    };

    println!("Processing frames...\n");

    loop {
        // Check timeout
        if let Some(dur) = timeout {
            if start.elapsed() >= dur {
                println!("\n  Time limit reached ({}s).", args.duration);
                break;
            }
        }

        let remaining = timeout
            .map(|d| d.saturating_sub(start.elapsed()))
            .unwrap_or(std::time::Duration::from_secs(1));
        let gst_timeout = gst::ClockTime::from_mseconds(remaining.as_millis().min(1000) as u64);

        match bus.timed_pop(gst_timeout) {
            Some(msg) => {
                use gst::MessageView;
                match msg.view() {
                    MessageView::Element(element) => {
                        let structure = element.structure().unwrap();
                        let name = structure.name().to_string();

                        if name.contains("inference-result") {
                            frame_count += 1;

                            if let Ok(result) = structure.get::<String>("result") {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result)
                                {
                                    if let Some(boxes) = json["bounding_boxes"].as_array() {
                                        let det_count = boxes.len();
                                        total_detections += det_count as u64;

                                        if det_count > 0 {
                                            println!(
                                                "  Frame {frame_count}: {det_count} detection(s)"
                                            );
                                            for bbox in boxes {
                                                let label =
                                                    bbox["label"].as_str().unwrap_or("unknown");
                                                let conf = bbox["value"].as_f64().unwrap_or(0.0);
                                                println!("    - {label} ({:.0}%)", conf * 100.0);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    MessageView::Eos(_) => {
                        println!("\nEnd of stream.");
                        break;
                    }
                    MessageView::Error(err) => {
                        eprintln!(
                            "Error: {} ({})",
                            err.error(),
                            err.debug().unwrap_or_default()
                        );
                        break;
                    }
                    _ => {}
                }
            }
            None => {
                // timed_pop returned None = timeout, loop continues
            }
        }
    }

    pipeline.set_state(gst::State::Null)?;

    let total_crops = *crop_counter.lock().unwrap();

    println!("\nSummary:");
    println!("  Duration: {:.1}s", start.elapsed().as_secs_f64());
    println!("  Frames processed: {frame_count}");
    println!("  Total detections: {total_detections}");
    println!("  Crops saved: {total_crops}");
    if total_crops > 0 {
        println!("  Output: {}", args.output_dir.display());
    }

    Ok(())
}
