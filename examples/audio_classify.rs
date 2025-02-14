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

    // Create remaining elements
    let audioconvert = gst::ElementFactory::make("audioconvert").build()?;
    let audioresample = gst::ElementFactory::make("audioresample").build()?;
    let capsfilter = gst::ElementFactory::make("capsfilter").build()?;
    let audiobuffer = gst::ElementFactory::make("queue").build()?;
    let edgeimpulseinfer = gst::ElementFactory::make("edgeimpulseinfer").build()?;
    let sink = gst::ElementFactory::make("autoaudiosink").build()?;

    // Configure audio format
    let caps = gst::Caps::builder("audio/x-raw")
        .field("format", "S16LE")
        .field("rate", 16000)
        .field("channels", 1)
        .build();
    capsfilter.set_property("caps", &caps);

    // Configure buffer size to match model's expected input
    audiobuffer.set_property("max-size-buffers", 1u32);
    audiobuffer.set_property("max-size-bytes", (16000 * 2) as u32); // 1 second of S16LE audio

    edgeimpulseinfer.set_property("model-path", model_path.to_str().unwrap());

    // Add remaining elements to pipeline
    pipeline.add_many(&[
        &audioconvert,
        &audioresample,
        &capsfilter,
        &audiobuffer,
        &edgeimpulseinfer,
        &sink,
    ])?;

    // Link elements
    gst::Element::link_many(&[
        &source,
        &audioconvert,
        &audioresample,
        &capsfilter,
        &audiobuffer,
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
                    println!("Got element message with name: {}", s.name());

                    if s.name() == "edge-impulse-inference-result" {
                        println!("Got inference result message");
                        println!("Message structure: {:?}", s);

                        // Try to get the result string first
                        if let Ok(result_str) = s.get::<String>("result") {
                            println!("Raw result string: {}", result_str);
                        }

                        // Extract classification results
                        if let Ok(classifications) = s.get::<gst::Structure>("classification") {
                            // Find the class with highest confidence
                            let mut max_confidence = 0.0f32;
                            let mut max_class = "";

                            for field in classifications.fields() {
                                if let Ok(confidence) = classifications.get::<f32>(field.as_str()) {
                                    if confidence > max_confidence {
                                        max_confidence = confidence;
                                        max_class = field.as_str();
                                    }
                                }
                            }

                            if max_confidence > 0.0 {
                                println!(
                                    "Detected: {} ({:.1}%)",
                                    max_class,
                                    max_confidence * 100.0
                                );
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
