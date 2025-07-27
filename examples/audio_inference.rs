//! Real-time Audio Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform audio
//! classification using GStreamer. It can process audio from either a microphone
//! input or an audio file.
//!
//! Usage:
//!   # FFI mode (default, no model path needed):
//!   cargo run --example audio_inference [OPTIONS]
//!
//!   # EIM mode (legacy, requires model path):
//!   cargo run --example audio_inference -- --model <path_to_model> [OPTIONS]
//!
//! Optional arguments:
//!   --model <path>         Path to the Edge Impulse model file (.eim) - EIM mode only
//!   --audio <path>         Path to input audio file (if not specified, uses microphone)
//!   --threshold <float>    Confidence threshold (0.0 to 1.0) for showing results (default: 0.8)
//!
//! Example with microphone (FFI mode):
//!   cargo run --example audio_inference
//!
//! Example with audio file (FFI mode):
//!   cargo run --example audio_inference -- --audio input.wav
//!
//! Example with microphone (EIM mode):
//!   cargo run --example audio_inference -- --model model.eim
//!
//! Example with audio file (EIM mode):
//!   cargo run --example audio_inference -- --model model.eim --audio input.wav

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use serde_json;
use std::path::Path;

/// Command line parameters for the real-time audio classification example
#[derive(Parser, Debug)]
struct AudioClassifyParams {
    /// Path to the Edge Impulse model file (.eim) - EIM mode only (legacy)
    #[clap(short, long)]
    model: Option<String>,

    /// Optional path to input audio file
    #[clap(short, long)]
    audio: Option<String>,

    /// Model block thresholds in format 'blockId.type=value' (e.g., '5.min_score=0.6')
    #[clap(long)]
    threshold: Vec<String>,
}

fn create_pipeline(
    model_path: Option<&Path>,
    audio_path: Option<&Path>,
    thresholds: &[String],
) -> Result<gst::Pipeline, Box<dyn std::error::Error>> {
    let pipeline = gst::Pipeline::new();

    // Create source element based on whether we have an audio file or using mic
    let source = if let Some(path) = audio_path {
        let filesrc = gst::ElementFactory::make("filesrc").build()?;
        let wavparse = gst::ElementFactory::make("wavparse").build()?;
        filesrc.set_property("location", path.to_str().unwrap());
        pipeline.add_many(&[&filesrc, &wavparse])?;
        gst::Element::link_many(&[&filesrc, &wavparse])?;
        wavparse.upcast()
    } else {
        let autoaudiosrc = gst::ElementFactory::make("autoaudiosrc").build()?;
        pipeline.add(&autoaudiosrc)?;
        autoaudiosrc.upcast()
    };

    // Create pipeline elements
    let audioconvert1 = gst::ElementFactory::make("audioconvert").build()?;
    let audioresample1 = gst::ElementFactory::make("audioresample").build()?;
    let capsfilter1 = gst::ElementFactory::make("capsfilter").build()?;
    let mut edgeimpulseinfer_factory = gst::ElementFactory::make("edgeimpulseaudioinfer");

    // Set model path if provided (EIM mode)
    if let Some(path) = model_path {
        edgeimpulseinfer_factory =
            edgeimpulseinfer_factory.property("model-path", path.to_str().unwrap());
    }

    // Set thresholds if provided
    for threshold in thresholds {
        edgeimpulseinfer_factory = edgeimpulseinfer_factory.property("threshold", threshold);
    }

    let edgeimpulseinfer = edgeimpulseinfer_factory.build()?;
    let audioconvert2 = gst::ElementFactory::make("audioconvert").build()?;
    let audioresample2 = gst::ElementFactory::make("audioresample").build()?;
    let capsfilter2 = gst::ElementFactory::make("capsfilter").build()?;
    let sink = gst::ElementFactory::make("fakesink").build()?;

    // Configure caps - the Edge Impulse element expects S16LE mono audio at 16kHz
    let caps1 = gst::Caps::builder("audio/x-raw")
        .field("format", "S16LE")
        .field("channels", 1)
        .field("rate", 16000)
        .field("layout", "interleaved")
        .build();
    capsfilter1.set_property("caps", &caps1);

    // Configure output caps for the sink - standard audio format
    let caps2 = gst::Caps::builder("audio/x-raw")
        .field("format", "F32LE")
        .field("channels", 2)
        .field("rate", 44100)
        .build();
    capsfilter2.set_property("caps", &caps2);

    // Add elements to pipeline
    pipeline.add_many(&[
        &audioconvert1,
        &audioresample1,
        &capsfilter1,
        &edgeimpulseinfer,
        &audioconvert2,
        &audioresample2,
        &capsfilter2,
        &sink,
    ])?;

    // Link elements
    gst::Element::link_many(&[
        &source,
        &audioconvert1,
        &audioresample1,
        &capsfilter1,
        &edgeimpulseinfer,
        &audioconvert2,
        &audioresample2,
        &capsfilter2,
        &sink,
    ])?;

    Ok(pipeline)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GStreamer
    gst::init()?;
    println!("GStreamer initialized");

    // Parse command line arguments
    let params = AudioClassifyParams::parse();

    // Validate audio file if provided
    let audio_path = params.audio.as_deref().map(|p| {
        let path = Path::new(p);
        if !path.exists() {
            panic!("Audio file does not exist: {}", p);
        }
        println!("Using audio file: {}", p);
        path
    });

    if audio_path.is_none() {
        println!("No audio file specified, using microphone input");
    }

    // Create pipeline using model path and audio source
    let model_path = params.model.as_deref().map(Path::new);
    let pipeline = create_pipeline(model_path, audio_path, &params.threshold)?;

    // Start playing
    println!("Setting pipeline state to Playing...");
    pipeline.set_state(gst::State::Playing)?;
    println!("Pipeline is playing...");

    // Wait for messages on the pipeline's bus
    let bus = pipeline.bus().unwrap();
    println!("Listening for pipeline messages...");

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Eos(..) => {
                println!("End of stream");
                break;
            }
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null)?;
                return Err(format!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                )
                .into());
            }
            MessageView::StateChanged(state) => {
                println!(
                    "State changed from {:?} to {:?}",
                    state.old(),
                    state.current()
                );
            }
            MessageView::Element(element) => {
                let structure = element.structure();
                if let Some(s) = structure {
                    if s.name() == "edge-impulse-audio-inference-result" {
                        if let Ok(result) = s.get::<String>("result") {
                            // Parse the JSON string
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                                if let Some(classifications) = json["classification"].as_object() {
                                    if let Some((top_label, top_conf)) = classifications
                                        .iter()
                                        .filter_map(|(label, value)| {
                                            value.as_f64().map(|v| (label, v))
                                        })
                                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                                    {
                                        // Assign a color per label (simple hash to color)
                                        let colors =
                                            [31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96];
                                        let color_idx = (top_label
                                            .bytes()
                                            .fold(0u8, |acc, b| acc.wrapping_add(b))
                                            as usize)
                                            % colors.len();
                                        let color_code = colors[color_idx];
                                        println!(
                                            "\x1b[1;{}mDetected {} ({:.1}%)\x1b[0m",
                                            color_code,
                                            top_label,
                                            top_conf * 100.0
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => (),
        }
    }

    // Clean up
    println!("Cleaning up...");
    pipeline.set_state(gst::State::Null)?;

    Ok(())
}
