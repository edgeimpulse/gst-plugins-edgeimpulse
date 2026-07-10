//! OCR text-recognition example using the `edgeimpulseocr` GStreamer element.
//!
//! Reads a single image, runs text recognition with the built-in pure-Rust
//! `ocrs` backend, and prints every `ocr` message the element posts on the bus.
//!
//! Recognition runs on a worker thread, so results for a still image land a few
//! frames after it enters the element. The pipeline therefore uses `imagefreeze`
//! to keep feeding the frame until the first result arrives (mirroring how a
//! live camera would supply a continuous stream).
//!
//! Build with the `ocrs` feature (which embeds the default detection and
//! recognition models) and disable the default `ffi` backend so no Edge Impulse
//! model needs to be compiled in:
//!
//! ```sh
//! export GST_PLUGIN_PATH="$(pwd)/target/release:$GST_PLUGIN_PATH"
//! cargo run --release --no-default-features --features "eim ocrs" \
//!     --example ocr_inference -- --image path/to/text.png
//! ```
//!
//! To use your own models instead of the embedded ones, pass their paths with
//! `--detection-model` and `--recognition-model`.

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use std::collections::HashSet;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

/// Command line parameters for the OCR example.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct OcrParams {
    /// Path to the input image file
    #[arg(short, long)]
    image: String,

    /// Path to a text-detection model (optional; uses the embedded model if unset)
    #[arg(long, default_value = "")]
    detection_model: String,

    /// Path to a text-recognition model (optional; uses the embedded model if unset)
    #[arg(long, default_value = "")]
    recognition_model: String,

    /// Seconds to wait for the first recognition result before giving up
    #[arg(long, default_value = "30")]
    timeout: u64,
}

fn build_pipeline(args: &OcrParams) -> Result<gst::Pipeline, Box<dyn Error>> {
    gst::init()?;

    if !Path::new(&args.image).exists() {
        return Err(format!("Input image file not found: {}", args.image).into());
    }

    let pipeline = gst::Pipeline::new();

    let source = gst::ElementFactory::make("filesrc")
        .name("source")
        .property("location", &args.image)
        .build()?;
    let decodebin = gst::ElementFactory::make("decodebin")
        .name("decodebin")
        .build()?;
    // Repeat the single decoded frame so the async worker keeps receiving it
    // until it produces a result.
    let imagefreeze = gst::ElementFactory::make("imagefreeze")
        .name("imagefreeze")
        .build()?;
    let videoconvert = gst::ElementFactory::make("videoconvert")
        .name("videoconvert")
        .build()?;
    let caps_filter = gst::ElementFactory::make("capsfilter")
        .name("caps_filter")
        .property(
            "caps",
            gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .build(),
        )
        .build()?;

    let ocr = gst::ElementFactory::make("edgeimpulseocr")
        .name("ocr")
        .property("backend", "ocrs")
        .build()?;
    if !args.detection_model.is_empty() {
        ocr.set_property("detection-model", &args.detection_model);
    }
    if !args.recognition_model.is_empty() {
        ocr.set_property("recognition-model", &args.recognition_model);
    }

    let sink = gst::ElementFactory::make("fakesink")
        .name("sink")
        .property("sync", false)
        .build()?;

    pipeline.add_many([
        &source,
        &decodebin,
        &imagefreeze,
        &videoconvert,
        &caps_filter,
        &ocr,
        &sink,
    ])?;

    // decodebin exposes its source pad dynamically once it detects the format.
    imagefreeze.link(&videoconvert)?;
    videoconvert.link(&caps_filter)?;
    caps_filter.link(&ocr)?;
    ocr.link(&sink)?;

    let imagefreeze_sink = imagefreeze.clone();
    decodebin.connect_pad_added(move |_, pad| {
        let sink_pad = imagefreeze_sink.static_pad("sink").unwrap();
        if !sink_pad.is_linked() {
            if let Err(err) = pad.link(&sink_pad) {
                eprintln!("Failed to link decodebin to imagefreeze: {err}");
            }
        }
    });
    source.link(&decodebin)?;

    Ok(pipeline)
}

fn example_main() -> Result<(), Box<dyn Error>> {
    let args = OcrParams::parse();

    println!("🚀 Starting Edge Impulse OCR inference");
    println!("📁 Input image: {}", args.image);

    let pipeline = build_pipeline(&args)?;
    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    let start = Instant::now();
    let hard_timeout = Duration::from_secs(args.timeout);
    // Once the first result arrives, wait a short grace period to collect the
    // remaining lines from the same frame, then stop (imagefreeze never ends).
    let grace = Duration::from_secs(1);
    let mut first_result_at: Option<Instant> = None;
    let mut seen = HashSet::new();

    loop {
        if let Some(t) = first_result_at {
            if t.elapsed() > grace {
                break;
            }
        } else if start.elapsed() > hard_timeout {
            eprintln!("⚠️  No text recognized within {}s", args.timeout);
            break;
        }

        let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(200)) else {
            continue;
        };
        use gst::MessageView;
        match msg.view() {
            MessageView::Error(err) => {
                eprintln!(
                    "❌ Error from {:?}: {} ({})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug().unwrap_or_default()
                );
                break;
            }
            MessageView::Element(element) => {
                let Some(s) = element.structure() else {
                    continue;
                };
                if s.name() != "ocr" {
                    continue;
                }
                let text = s.get::<String>("text").unwrap_or_default();
                // The same frozen frame is recognized repeatedly; only report
                // each distinct string once.
                if !seen.insert(text.clone()) {
                    continue;
                }
                let confidence = s.get::<f64>("confidence").unwrap_or(0.0);
                let x = s.get::<i32>("x").unwrap_or(0);
                let y = s.get::<i32>("y").unwrap_or(0);
                let width = s.get::<i32>("width").unwrap_or(0);
                let height = s.get::<i32>("height").unwrap_or(0);
                println!(
                    "📖 OCR: {text:?} (confidence {confidence:.2}) at [{x}, {y}, {width}x{height}]"
                );
                if first_result_at.is_none() {
                    first_result_at = Some(Instant::now());
                }
            }
            _ => {}
        }
    }

    pipeline.set_state(gst::State::Null)?;

    if seen.is_empty() {
        println!("✅ OCR inference finished (no text found)");
    } else {
        println!(
            "✅ OCR inference finished ({} line(s) recognized)",
            seen.len()
        );
    }
    Ok(())
}

fn main() {
    if let Err(e) = example_main() {
        eprintln!("❌ Error: {e}");
        std::process::exit(1);
    }
}
