//! Dynamic Crop Example — two-stage detection → classification pipeline
//!
//! Demonstrates `edgeimpulsecrop` in a realistic pipeline:
//!
//!   camera → object detection → tee
//!     ├─ overlay → display (live view with bounding boxes)
//!     └─ continue-if → crop → fakesink (per-crop output)
//!
//! Each detected object is cropped from the frame, resized to a target size,
//! and pushed downstream as a separate buffer with `CropOriginMeta` attached.
//!
//! In a real deployment, the cropped buffers would feed into a second
//! `edgeimpulsevideoinfer` element running a classification model.
//!
//! Usage:
//!   # FFI mode with camera (default):
//!   cargo run --release --example dynamic_crop
//!
//!   # EIM mode:
//!   cargo run --release --example dynamic_crop -- --model <path_to_model>
//!
//!   # Use test video source instead of camera:
//!   cargo run --release --example dynamic_crop -- --source test
//!
//!   # Custom crop size and padding:
//!   cargo run --release --example dynamic_crop -- --target-width 128 --target-height 128 --padding 20
//!
//! Environment setup:
//!   export GST_PLUGIN_PATH="target/release:$GST_PLUGIN_PATH"

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use std::error::Error;

#[derive(Parser, Debug)]
#[command(author, version, about = "Dynamic crop example")]
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

    /// Number of frames to process (0 = unlimited, default for camera)
    #[arg(short, long, default_value = "0")]
    num_frames: u32,
}

/// Create the video source element based on the --source flag.
/// Tries avfvideosrc (macOS) first, falls back to v4l2src (Linux).
fn create_video_source(source: &str, num_frames: u32) -> Result<gst::Element, Box<dyn Error>> {
    match source {
        "test" => {
            let n = if num_frames == 0 { 30 } else { num_frames };
            Ok(gst::ElementFactory::make("videotestsrc")
                .property("num-buffers", n as i32)
                .property_from_str("pattern", "smpte")
                .build()?)
        }
        _ => {
            // Try macOS camera first, then Linux
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
                Err("No camera source available. Install GStreamer camera plugins or use --source test".into())
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    gst::init()?;

    println!("Dynamic Crop Example");
    println!("  Source: {}", args.source);
    println!(
        "  Target size: {}x{}",
        args.target_width, args.target_height
    );
    println!("  Padding: {}px", args.padding);
    if args.num_frames > 0 {
        println!("  Frames: {}", args.num_frames);
    } else {
        println!("  Frames: unlimited (Ctrl+C to stop)");
    }
    println!();

    let pipeline = gst::Pipeline::new();

    // Source
    let src = create_video_source(&args.source, args.num_frames)?;

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640i32)
                .field("height", 480i32)
                .field("framerate", gst::Fraction::new(15, 1))
                .build(),
        )
        .build()?;

    let convert = gst::ElementFactory::make("videoconvert").build()?;

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
            // Fall back to fakesink if no display available (headless/CI)
            gst::ElementFactory::make("fakesink")
                .property("sync", false)
                .build()
                .unwrap()
        });

    // Path 2: continue-if → crop → fakesink (crop output)
    let queue2 = gst::ElementFactory::make("queue").build()?;
    let gate = gst::ElementFactory::make("edgeimpulsecontinueif")
        .property("condition", "detection_count >= 1")
        .build()?;
    let crop = gst::ElementFactory::make("edgeimpulsecrop")
        .property("padding", args.padding)
        .property("target-width", args.target_width)
        .property("target-height", args.target_height)
        .build()?;
    let crop_sink = gst::ElementFactory::make("fakesink")
        .property("sync", false)
        .build()?;

    // Add all elements
    pipeline.add_many([
        &src,
        &capsfilter,
        &convert,
        &infer,
        &tee,
        &queue1,
        &overlay,
        &convert_display,
        &display_sink,
        &queue2,
        &gate,
        &crop,
        &crop_sink,
    ])?;

    // Link main chain
    gst::Element::link_many([&src, &capsfilter, &convert, &infer, &tee])?;

    // Link tee → path 1 (overlay → display)
    gst::Element::link_many([&queue1, &overlay, &convert_display, &display_sink])?;
    let tee_src1 = tee.request_pad_simple("src_%u").unwrap();
    let queue1_sink = queue1.static_pad("sink").unwrap();
    tee_src1.link(&queue1_sink)?;

    // Link tee → path 2 (crop)
    gst::Element::link_many([&queue2, &gate, &crop, &crop_sink])?;
    let tee_src2 = tee.request_pad_simple("src_%u").unwrap();
    let queue2_sink = queue2.static_pad("sink").unwrap();
    tee_src2.link(&queue2_sink)?;

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    let mut frame_count = 0u64;
    let mut total_detections = 0u64;
    let mut total_crops = 0u64;

    println!("Processing frames...\n");

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Element(element) => {
                let structure = element.structure().unwrap();
                let name = structure.name().to_string();

                if name.contains("inference-result") {
                    frame_count += 1;

                    if let Ok(result) = structure.get::<String>("result") {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                            if let Some(boxes) = json["bounding_boxes"].as_array() {
                                let det_count = boxes.len();
                                total_detections += det_count as u64;
                                total_crops += det_count as u64;

                                if det_count > 0 {
                                    println!(
                                        "  Frame {frame_count}: {det_count} detection(s) → {det_count} crop(s) at {}x{}",
                                        args.target_width, args.target_height
                                    );
                                    for bbox in boxes {
                                        let label = bbox["label"].as_str().unwrap_or("unknown");
                                        let conf = bbox["value"].as_f64().unwrap_or(0.0);
                                        let x = bbox["x"].as_u64().unwrap_or(0);
                                        let y = bbox["y"].as_u64().unwrap_or(0);
                                        let w = bbox["width"].as_u64().unwrap_or(0);
                                        let h = bbox["height"].as_u64().unwrap_or(0);
                                        println!("    - {label} ({conf:.0}%) at ({x},{y}) {w}x{h}");
                                    }
                                } else if frame_count <= 5 || frame_count % 30 == 0 {
                                    // Only print "no detections" for first few frames and periodically
                                    println!("  Frame {frame_count}: no detections");
                                }
                            } else {
                                println!("  Frame {frame_count}: classification result (no crops)");
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

    pipeline.set_state(gst::State::Null)?;

    println!("\nSummary:");
    println!("  Frames processed: {frame_count}");
    println!("  Total detections: {total_detections}");
    println!("  Total crops produced: {total_crops}");
    println!(
        "  Crop size: {}x{} (padding: {}px)",
        args.target_width, args.target_height, args.padding
    );

    Ok(())
}
