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
//!   --threshold <float>    Confidence threshold (0.0 to 1.0) for showing results (default: 0.8)
//!
//! Example with microphone:
//!   cargo run --example audio_classify -- --model model.eim
//!
//! Example with audio file:
//!   cargo run --example audio_classify -- --model model.eim --audio input.wav

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use serde_json;
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

    /// Confidence threshold (0.0 to 1.0) for showing results
    #[clap(short, long, default_value = "0.8")]
    threshold: f32,
}

fn create_pipeline(
    model_path: &Path,
    audio_path: Option<&Path>,
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
    let capsfilter1 = gst::ElementFactory::make("capsfilter").build()?;
    let audioconvert1 = gst::ElementFactory::make("audioconvert").build()?;
    let audioresample1 = gst::ElementFactory::make("audioresample").build()?;
    let capsfilter2 = gst::ElementFactory::make("capsfilter").build()?;
    let edgeimpulseinfer = gst::ElementFactory::make("edgeimpulseaudioinfer").build()?;
    let audioconvert2 = gst::ElementFactory::make("audioconvert").build()?;
    let audioresample2 = gst::ElementFactory::make("audioresample").build()?;
    let capsfilter3 = gst::ElementFactory::make("capsfilter").build()?;
    let sink = gst::ElementFactory::make("autoaudiosink").build()?;

    // Configure caps
    let caps1 = gst::Caps::builder("audio/x-raw")
        .field("format", "F32LE")
        .build();
    capsfilter1.set_property("caps", &caps1);

    let caps2 = gst::Caps::builder("audio/x-raw")
        .field("format", "S16LE")
        .field("channels", 1)
        .field("rate", 16000)
        .field("layout", "interleaved")
        .build();
    capsfilter2.set_property("caps", &caps2);

    let caps3 = gst::Caps::builder("audio/x-raw")
        .field("format", "F32LE")
        .field("channels", 2)
        .field("rate", 44100)
        .build();
    capsfilter3.set_property("caps", &caps3);

    edgeimpulseinfer.set_property("model-path", model_path.to_str().unwrap());

    // Add elements to pipeline
    pipeline.add_many(&[
        &capsfilter1,
        &audioconvert1,
        &audioresample1,
        &capsfilter2,
        &edgeimpulseinfer,
        &audioconvert2,
        &audioresample2,
        &capsfilter3,
        &sink,
    ])?;

    // Link elements
    gst::Element::link_many(&[
        &source,
        &capsfilter1,
        &audioconvert1,
        &audioresample1,
        &capsfilter2,
        &edgeimpulseinfer,
        &audioconvert2,
        &audioresample2,
        &capsfilter3,
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
    let pipeline = create_pipeline(&Path::new(&params.model), audio_path)?;

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
                    if s.name() == "edge-impulse-inference-result" {
                        if let Ok(result) = s.get::<String>("result") {
                            // Parse the JSON string
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                                if let Some(classifications) = json["classification"].as_object() {
                                    for (label, value) in classifications {
                                        if let Some(confidence) = value.as_f64() {
                                            // Only show detections with confidence > threshold
                                            if confidence > params.threshold as f64 {
                                                println!(
                                                    "Detected {} with confidence {:.1}%",
                                                    label,
                                                    confidence * 100.0
                                                );
                                            }
                                        }
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
