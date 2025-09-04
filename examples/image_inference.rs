#![allow(deprecated)]

//! Image Classification Example using edgeimpulseinfer GStreamer plugin
//!
//! This example demonstrates how to use the Edge Impulse GStreamer plugin to perform
//! image classification using a trained model on a single image file.
//!
//! The edgeimpulsevideoinfer element automatically handles frame resizing to match model
//! input requirements and scales detection results back to the original resolution.
//!
//! Usage:
//!   # EIM mode (requires model path):
//!   cargo run --example image_inference -- --model <path_to_model> --image <path_to_image>
//!
//!   # FFI mode (no model path needed):
//!   cargo run --example image_inference -- --image <path_to_image>
//!
//!
//! Environment setup:
//! export GST_PLUGIN_PATH="target/debug:$GST_PLUGIN_PATH"

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use serde_json;
use std::error::Error;
use std::path::Path;

/// Command line parameters for the image classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct ImageClassifyParams {
    /// Path to Edge Impulse model file (optional for FFI mode)
    #[arg(short, long)]
    model: Option<String>,

    /// Path to input image file
    #[arg(short, long)]
    image: String,

    /// Output image file path (optional, will save with overlay if provided)
    #[arg(short, long)]
    output: Option<String>,

    /// Video format (RGB, RGBA, BGR, BGRA)
    #[arg(short, long, default_value = "RGB")]
    format: String,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,

    /// Model block thresholds in format 'blockId.type=value' (e.g., '5.min_score=0.6')
    #[clap(long)]
    threshold: Vec<String>,

    /// Stroke width for bounding boxes
    #[arg(long, default_value = "2")]
    stroke_width: i32,

    /// Text color in hex format (e.g., 0xFFFFFF for white)
    #[arg(long, default_value = "0xFFFFFF")]
    text_color: String,
}

fn create_pipeline(args: &ImageClassifyParams) -> Result<gst::Pipeline, Box<dyn Error>> {
    // Initialize GStreamer
    gst::init()?;

    // Validate input image exists
    if !Path::new(&args.image).exists() {
        return Err(format!("Input image file not found: {}", args.image).into());
    }

    // Create pipeline
    let pipeline = gst::Pipeline::new();

    // Create elements
    let source = gst::ElementFactory::make("filesrc")
        .name("source")
        .property("location", &args.image)
        .build()?;

    let decodebin = gst::ElementFactory::make("decodebin")
        .name("decodebin")
        .build()?;

    let videoconvert = gst::ElementFactory::make("videoconvert")
        .name("videoconvert")
        .build()?;

    let videoscale = gst::ElementFactory::make("videoscale")
        .name("videoscale")
        .build()?;

    let caps_filter = gst::ElementFactory::make("capsfilter")
        .name("caps_filter")
        .property(
            "caps",
            &gst::Caps::builder("video/x-raw")
                .field("format", &args.format)
                .field("width", &args.width)
                .field("height", &args.height)
                .build(),
        )
        .build()?;

    let inference = gst::ElementFactory::make("edgeimpulsevideoinfer")
        .name("inference")
        .build()?;

    let overlay = gst::ElementFactory::make("edgeimpulseoverlay")
        .name("overlay")
        .property("stroke-width", &args.stroke_width)
        .property(
            "text-color",
            &u32::from_str_radix(&args.text_color[2..], 16)?,
        )
        .build()?;

    // Add elements to pipeline (excluding overlay and sink which will be added conditionally)
    pipeline.add_many(&[
        &source,
        &decodebin,
        &videoconvert,
        &videoscale,
        &caps_filter,
        &inference,
    ])?;

    // Link elements (except decodebin which needs dynamic linking)
    videoconvert.link(&videoscale)?;
    videoscale.link(&caps_filter)?;
    caps_filter.link(&inference)?;

    // Connect decodebin pad-added signal
    let videoconvert_clone = videoconvert.clone();
    decodebin.connect_pad_added(move |_, pad| {
        let sink_pad = videoconvert_clone.static_pad("sink").unwrap();

        if !sink_pad.is_linked() {
            if let Err(err) = pad.link(&sink_pad) {
                eprintln!("Failed to link decodebin to videoconvert: {}", err);
            }
        }
    });

    // Link source to decodebin
    source.link(&decodebin)?;

    // Add overlay and sink based on whether output is specified
    if let Some(output_path) = &args.output {
        let encoder = gst::ElementFactory::make("pngenc")
            .name("encoder")
            .build()?;

        let sink = gst::ElementFactory::make("filesink")
            .name("sink")
            .property("location", output_path)
            .build()?;

        pipeline.add_many(&[&overlay, &encoder, &sink])?;
        inference.link(&overlay)?;
        overlay.link(&encoder)?;
        encoder.link(&sink)?;
    } else {
        // Display the image with overlay
        let sink = gst::ElementFactory::make("autovideosink")
            .name("sink")
            .property("sync", &false)
            .build()?;

        pipeline.add_many(&[&overlay, &sink])?;
        inference.link(&overlay)?;
        overlay.link(&sink)?;
    }

    // Set model path if provided
    if let Some(model_path) = &args.model {
        inference.set_property("model-path", model_path);
        println!("🔧 EIM Mode: Using model at {}", model_path);
    } else {
        println!("✅ FFI Mode: No model path provided, will use FFI backend");
    }

    // Set thresholds if provided
    for threshold in &args.threshold {
        if let Some((block_id, threshold_type, value)) = parse_threshold(threshold) {
            let property_name = format!("threshold-{}-{}", block_id, threshold_type);
            inference.set_property(&property_name, value);
        }
    }

    Ok(pipeline)
}

fn parse_threshold(threshold: &str) -> Option<(String, String, f64)> {
    let parts: Vec<&str> = threshold.split('.').collect();
    if parts.len() == 3 {
        let block_id = parts[0].to_string();
        let threshold_type = parts[1].to_string();
        if let Ok(value) = parts[2].parse::<f64>() {
            return Some((block_id, threshold_type, value));
        }
    }
    None
}

fn example_main() -> Result<(), Box<dyn Error>> {
    let args = ImageClassifyParams::parse();

    if args.debug {
        println!("🔧 Debug mode enabled");
    }

    println!("🚀 Starting Edge Impulse Image Inference");
    println!("📁 Input image: {}", args.image);
    if let Some(output) = &args.output {
        println!("💾 Output image: {}", output);
    }
    println!("📐 Image dimensions: {}x{}", args.width, args.height);
    println!("🎨 Format: {}", args.format);
    println!("🔧 Debug mode: {}", args.debug);

    let pipeline = create_pipeline(&args)?;

    // Start playing
    println!("▶️  Setting pipeline state to Playing...");
    pipeline.set_state(gst::State::Playing)?;

    // Set up bus monitoring
    let bus = pipeline.bus().unwrap();

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Eos(..) => {
                println!("✅ End of stream reached");
                break;
            }
            MessageView::Error(err) => {
                eprintln!(
                    "❌ Error from {:?}: {} ({})",
                    err.src().map(|s| s.name()),
                    err.error(),
                    err.debug().unwrap_or_else(|| glib::GString::from(""))
                );
                break;
            }
            MessageView::StateChanged(state) => {
                if state.src().map(|s| s.name()) == Some("pipeline0".into()) {
                    println!(
                        "🔄 State changed from {:?} to {:?}",
                        state.old(),
                        state.current()
                    );
                }
            }
            MessageView::Element(element) => {
                let structure = element.structure().unwrap();

                if structure.name() == "edge-impulse-video-inference-result" {
                    if let Ok(result) = structure.get::<String>("result") {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                            println!(
                                "🧠 Inference result: {}",
                                serde_json::to_string_pretty(&json).unwrap()
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Clean up
    pipeline.set_state(gst::State::Null)?;

    println!("✅ Image inference completed successfully!");
    Ok(())
}

fn main() {
    if let Err(e) = example_main() {
        eprintln!("❌ Error: {}", e);
        std::process::exit(1);
    }
}
