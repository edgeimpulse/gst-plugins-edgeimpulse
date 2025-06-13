// Audio Ingestion Example for Edge Impulse
//
// This example demonstrates how to use the custom GStreamer sink element `edgeimpulsesink`
// to ingest audio samples directly to Edge Impulse using their ingestion API. The pipeline
// records audio from your default microphone, converts it to the required format, and sends
// it to Edge Impulse in batches (configurable interval).
//
// Usage:
//   cargo run --release --example audio_ingestion -- --api-key <YOUR_API_KEY> [--upload-interval-ms <interval>]
//
// Arguments:
//   --api-key                Your Edge Impulse API key (required)
//   --upload-interval-ms     Interval in milliseconds to batch and upload samples (default: 1000)
//
// The example listens for custom bus messages from the `edgeimpulsesink` element and prints
// details about each successfully ingested sample (filename, media type, length, label, category).
//
// Requirements:
//   - You must have GStreamer and the gst-plugin-edgeimpulse plugin built and available.
//   - The API key must be valid for your Edge Impulse project.
//
// Example:
//   cargo run --release --example audio_ingestion -- --api-key ei_xxx_yourkeyhere

use clap::Parser;
use gstreamer as gst;
use gstreamer::parse;
use gstreamer::prelude::*;

/// Audio ingestion example for Edge Impulse
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Edge Impulse API key
    #[arg(long)]
    api_key: String,

    /// Upload interval in milliseconds
    #[arg(long, default_value_t = 1000)]
    upload_interval_ms: u32,
}

fn main() {
    gst::init().unwrap();
    let args = Args::parse();

    let pipeline_str = format!(
        "autoaudiosrc ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channels=1,rate=16000 ! edgeimpulsesink api-key=\"{}\" upload-interval-ms={} category=training",
        args.api_key, args.upload_interval_ms
    );

    let pipeline = match parse::launch(&pipeline_str) {
        Ok(p) => p.dynamic_cast::<gst::Pipeline>().unwrap(),
        Err(e) => {
            eprintln!("Failed to launch pipeline: {}", e);
            std::process::exit(-1);
        }
    };
    let bus = pipeline.bus().unwrap();
    pipeline.set_state(gst::State::Playing).unwrap();
    println!("Audio ingestion started. Press Ctrl+C to stop.");
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                eprintln!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            MessageView::Element(element) => {
                if let Some(s) = element.structure() {
                    if s.name() == "edge-impulse-ingestion-result" {
                        let filename = s.get::<&str>("filename").unwrap_or("");
                        let media_type = s.get::<&str>("media_type").unwrap_or("");
                        let length = s.get::<u64>("length").unwrap_or(0);
                        let label = s.get::<Option<&str>>("label").unwrap_or(None);
                        let category = s.get::<&str>("category").unwrap_or("");
                        println!(
                            "✅ Sample ingested: file='{}', media_type='{}', length={}ms, label={:?}, category='{}'",
                            filename, media_type, length, label, category
                        );
                    } else if s.name() == "edge-impulse-ingestion-error" {
                        let filename = s.get::<&str>("filename").unwrap_or("");
                        let media_type = s.get::<&str>("media_type").unwrap_or("");
                        let error = s.get::<&str>("error").unwrap_or("");
                        let label = s.get::<Option<&str>>("label").unwrap_or(None);
                        let category = s.get::<&str>("category").unwrap_or("");
                        eprintln!(
                            "❌ Ingestion error: file='{}', media_type='{}', error='{}', label={:?}, category='{}'",
                            filename, media_type, error, label, category
                        );
                    }
                }
            }
            _ => (),
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
}
