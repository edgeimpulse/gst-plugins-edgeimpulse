//! Real-time Audio Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform audio
//! classification using GStreamer. It can process audio from either a microphone
//! input or an audio file.
//!
//! Usage:
//!   cargo run --example audio_classify -- --model <path_to_model> [OPTIONS]
//!
//! Required arguments:
//!   --model <path>         Path to the Edge Impulse model file (.eim)
//!
//! Optional arguments:
//!   --audio <path>         Path to input audio file (if not specified, uses microphone)
//!   --frequency <hz>       Sample rate in Hz (default: 16000)
//!
//! Example with microphone:
//!   cargo run --example audio_classify -- --model model.eim
//!
//! Example with audio file:
//!   cargo run --example audio_classify -- --model model.eim --audio input.wav

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use std::path::Path;

/// Command line parameters for the real-time audio classification example
#[derive(Parser, Debug)]
struct AudioClassifyParams {
    /// Path to the Edge Impulse model file (.eim)
    #[clap(short, long)]
    model: String,

    /// Optional path to input audio file
    #[clap(short, long)]
    audio: Option<String>,

    /// Sample rate in Hz (default: 16000)
    #[clap(short, long, default_value = "16000")]
    frequency: i32,
}

fn create_pipeline(model_path: &Path, audio_path: &Path) -> Result<gst::Pipeline, Box<dyn std::error::Error>> {
    let pipeline = gst::Pipeline::new();

    // Create elements
    let filesrc = gst::ElementFactory::make("filesrc").build()?;
    let wavparse = gst::ElementFactory::make("wavparse").build()?;
    let audioconvert = gst::ElementFactory::make("audioconvert").build()?;
    let audioresample = gst::ElementFactory::make("audioresample").build()?;
    let edgeimpulseinfer = gst::ElementFactory::make("edgeimpulseinfer").build()?;
    let sink = gst::ElementFactory::make("autoaudiosink").build()?;

    // Set properties
    filesrc.set_property("location", audio_path.to_str().unwrap());
    edgeimpulseinfer.set_property("model-path", model_path.to_str().unwrap());

    // Add elements to pipeline
    pipeline.add_many(&[
        &filesrc,
        &wavparse,
        &audioconvert,
        &audioresample,
        &edgeimpulseinfer,
        &sink,
    ])?;

    // Link elements
    gst::Element::link_many(&[
        &filesrc,
        &wavparse,
        &audioconvert,
        &audioresample,
        &edgeimpulseinfer,
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
    if let Some(audio_path) = &params.audio {
        let path = Path::new(audio_path);
        if !path.exists() {
            return Err(format!("Audio file does not exist: {}", audio_path).into());
        }
        println!("Audio file exists: {}", audio_path);
    }

    // Create pipeline using model path and frequency
    let pipeline = create_pipeline(
        &Path::new(&params.model),
        params.audio.as_deref().map(|s| Path::new(s)).unwrap_or_else(|| Path::new(""))
    )?;

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
            _ => (),
        }
    }

    // Clean up
    println!("Cleaning up...");
    pipeline.set_state(gst::State::Null)?;

    Ok(())
}