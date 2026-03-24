//! Continue-If Example — conditional gating based on inference results
//!
//! Demonstrates how to use `edgeimpulsecontinueif` to gate a video pipeline
//! based on inference metadata. Only buffers matching the condition pass
//! through; others are marked as GAP and dropped.
//!
//! The example also shows the `rules` property for conditional metadata output:
//! different bus messages are emitted depending on detection count.
//!
//! Usage:
//!   # FFI mode (model compiled in):
//!   cargo run --example continue_if
//!
//!   # EIM mode:
//!   cargo run --example continue_if -- --model <path_to_model>
//!
//!   # Custom condition:
//!   cargo run --example continue_if -- --condition "max_confidence > 0.9"
//!
//! Environment setup:
//!   export GST_PLUGIN_PATH="target/debug:$GST_PLUGIN_PATH"

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use std::error::Error;

#[derive(Parser, Debug)]
#[command(author, version, about = "Continue-If gate example")]
struct Args {
    /// Path to Edge Impulse model file (optional for FFI mode)
    #[arg(short, long)]
    model: Option<String>,

    /// Gate condition expression
    #[arg(short, long, default_value = "detection_count >= 1")]
    condition: String,

    /// Number of frames to process (0 = unlimited)
    #[arg(short, long, default_value = "50")]
    num_frames: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    gst::init()?;

    println!("Continue-If Example");
    println!("  Condition: {}", args.condition);
    println!();

    // Build pipeline: videotestsrc → infer → continue-if → fakesink
    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("videotestsrc")
        .property("num-buffers", args.num_frames as i32)
        .property_from_str("pattern", "smpte")
        .build()?;

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 320i32)
                .field("height", 320i32)
                .field("framerate", gst::Fraction::new(5, 1))
                .build(),
        )
        .build()?;

    let convert = gst::ElementFactory::make("videoconvert").build()?;

    let infer = gst::ElementFactory::make("edgeimpulsevideoinfer").build()?;
    if let Some(ref model_path) = args.model {
        infer.set_property("model-path", model_path);
    }

    let gate = gst::ElementFactory::make("edgeimpulsecontinueif")
        .property("condition", &args.condition)
        .build()?;

    // Set rules for conditional metadata output
    let rules = r#"[
        {"condition": "detection_count > 4", "metadata": {"severity": "critical", "color": "purple"}},
        {"condition": "detection_count >= 1", "metadata": {"severity": "warning", "color": "red"}},
        {"condition": "detection_count == 0", "metadata": {"severity": "ok", "color": "green"}}
    ]"#;
    gate.set_property("rules", rules);

    let sink = gst::ElementFactory::make("fakesink").build()?;

    pipeline.add_many([&src, &capsfilter, &convert, &infer, &gate, &sink])?;
    gst::Element::link_many([&src, &capsfilter, &convert, &infer, &gate, &sink])?;

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    let mut passed = 0u64;
    let mut total = 0u64;

    println!("Processing frames...\n");

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Element(element) => {
                let structure = element.structure().unwrap();
                let name = structure.name().to_string();

                if name.contains("inference-result") {
                    total += 1;
                    if let Ok(result) = structure.get::<String>("result") {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result) {
                            let det_count = json["bounding_boxes"]
                                .as_array()
                                .map(|b| b.len())
                                .unwrap_or(0);
                            print!("  Frame {total}: {det_count} detection(s)");
                        }
                    }
                }

                if name == "edge-impulse-continue-if-metadata" {
                    passed += 1;
                    let severity = structure.get::<String>("severity").unwrap_or_default();
                    let color = structure.get::<String>("color").unwrap_or_default();
                    println!(" → severity={severity}, color={color}");
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
    println!("  Total frames: {total}");
    println!("  Passed gate:  {passed}");
    println!("  Dropped:      {}", total.saturating_sub(passed));

    Ok(())
}
