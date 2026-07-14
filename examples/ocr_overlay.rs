#![allow(deprecated)]
#![allow(unexpected_cfgs)]

//! Live webcam OCR with bounding-box overlay.
//!
//! Captures the default camera, runs the self-contained `ocrs` OCR backend
//! (its detection + recognition models are embedded in the plugin at build
//! time), draws a box and the recognized text over every detected line, and
//! shows the result in a window. Recognized lines are also printed to stdout.
//!
//! Pipeline:
//!   camera -> queue -> videoconvert -> capsfilter(RGB)
//!          -> edgeimpulseocr backend=ocrs
//!          -> queue -> edgeimpulseoverlay -> videoconvert -> autovideosink
//!
//! Build the plugin with the `ocr` and `presentation` capabilities and point
//! GStreamer at the freshly built plugin so it can find the elements by name:
//!
//!   cargo build --release --no-default-features --features ocr,presentation
//!   export GST_PLUGIN_PATH="$(pwd)/target/release:$GST_PLUGIN_PATH"
//!   cargo run  --release --no-default-features --features ocr,presentation \
//!       --example ocr_overlay
//!
//! On macOS the terminal is prompted for camera access on first run. Close the
//! window or press Ctrl-C to stop.

use gstreamer as gst;
use gstreamer::prelude::*;
use std::error::Error;

use clap::Parser;

/// Live webcam OCR with a tunable overlay.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// OCR backend. "ocrs" (embedded, pure Rust) runs OCR on the raw frames and
    /// is the only backend this webcam pipeline supports. "edge-impulse" decodes
    /// per-character detections from an upstream `edgeimpulsevideoinfer` element,
    /// which this example does not add — so it produces no text here.
    #[arg(long, default_value = "ocrs")]
    backend: String,

    /// Drop recognized lines whose confidence is below this value (0.0..1.0).
    #[arg(long, default_value = "0.0")]
    min_confidence: f64,

    /// Truncate recognized text to at most this many characters.
    #[arg(long, default_value = "256")]
    max_text_length: u32,

    /// Run OCR every Nth frame (1 = every frame).
    #[arg(long, default_value = "1")]
    interval: u32,

    /// Override the detection model (.rten); empty uses the embedded model.
    #[arg(long, default_value = "")]
    detection_model: String,

    /// Override the recognition model (.rten); empty uses the embedded model.
    #[arg(long, default_value = "")]
    recognition_model: String,

    /// Overlay bounding-box stroke width in pixels.
    #[arg(long, default_value = "2")]
    stroke_width: i32,

    /// Overlay text scale ratio (0.1..5.0; >1 larger, <1 smaller).
    #[arg(long, default_value = "1.0")]
    text_scale_ratio: f64,

    /// Hide the recognized-text labels (draw boxes only).
    #[arg(long)]
    no_labels: bool,

    /// Overlay text color, hex RRGGBB or 0xRRGGBB.
    #[arg(long, default_value = "0xFFFFFF")]
    text_color: String,

    /// Overlay label background color, hex RRGGBB or 0xRRGGBB.
    #[arg(long, default_value = "0x000000")]
    background_color: String,
}

// macOS shows the video window from the Cocoa main loop, so the pipeline runs
// on a worker thread while `NSApplication` owns the main thread. On other
// platforms `run` just calls the closure directly.
#[cfg(not(target_os = "macos"))]
fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    main()
}

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)]
fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    use std::{
        ffi::c_void,
        sync::mpsc::{channel, Sender},
        thread,
    };

    use cocoa::{
        appkit::{NSApplication, NSWindow},
        base::id,
        delegate,
    };
    use objc::{
        msg_send,
        runtime::{Object, Sel},
        sel, sel_impl,
    };

    unsafe {
        let app = cocoa::appkit::NSApp();
        let (send, recv) = channel::<()>();

        extern "C" fn on_finish_launching(this: &Object, _cmd: Sel, _notification: id) {
            let send = unsafe {
                let send_pointer = *this.get_ivar::<*const c_void>("send");
                let boxed = Box::from_raw(send_pointer as *mut Sender<()>);
                *boxed
            };
            send.send(()).unwrap();
        }

        let delegate = delegate!("AppDelegate", {
            app: id = app,
            send: *const c_void = Box::into_raw(Box::new(send)) as *const c_void,
            (applicationDidFinishLaunching:) => on_finish_launching as extern "C" fn(&Object, Sel, id)
        });
        app.setDelegate_(delegate);

        let t = thread::spawn(move || {
            recv.recv().unwrap();
            let res = main();
            let app = cocoa::appkit::NSApp();
            app.stop_(cocoa::base::nil);
            let event = cocoa::appkit::NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                cocoa::base::nil,
                cocoa::appkit::NSEventType::NSApplicationDefined,
                cocoa::foundation::NSPoint { x: 0.0, y: 0.0 },
                cocoa::appkit::NSEventModifierFlags::empty(),
                0.0,
                0,
                cocoa::base::nil,
                cocoa::appkit::NSEventSubtype::NSApplicationActivatedEventType,
                0,
                0,
            );
            app.postEvent_atStart_(event, cocoa::base::YES);
            res
        });

        app.run();
        t.join().unwrap()
    }
}

fn create_pipeline(args: &Args) -> Result<gst::Pipeline, Box<dyn Error>> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    // Camera source: avfvideosrc on macOS, autovideosrc elsewhere.
    #[cfg(target_os = "macos")]
    let source = gst::ElementFactory::make("avfvideosrc")
        .property("device-index", 0i32)
        .build()?;
    #[cfg(not(target_os = "macos"))]
    let source = gst::ElementFactory::make("autovideosrc").build()?;

    let queue_in = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 4u32)
        .property_from_str("leaky", "downstream")
        .build()?;

    let convert_in = gst::ElementFactory::make("videoconvert").build()?;

    // edgeimpulseocr requires RGB input.
    let caps_filter = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .build(),
        )
        .build()?;

    let mut ocr_factory = gst::ElementFactory::make("edgeimpulseocr")
        .property("backend", &args.backend)
        .property("min-confidence", args.min_confidence)
        .property("max-text-length", args.max_text_length)
        .property("interval", args.interval);
    if !args.detection_model.is_empty() {
        ocr_factory = ocr_factory.property("detection-model", &args.detection_model);
    }
    if !args.recognition_model.is_empty() {
        ocr_factory = ocr_factory.property("recognition-model", &args.recognition_model);
    }
    let ocr = ocr_factory.build()?;

    let queue_out = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 4u32)
        .property_from_str("leaky", "downstream")
        .build()?;

    let text_color =
        u32::from_str_radix(args.text_color.trim_start_matches("0x"), 16).unwrap_or(0xFF_FF_FF);
    let background_color = u32::from_str_radix(args.background_color.trim_start_matches("0x"), 16)
        .unwrap_or(0x00_00_00);
    let mut overlay_factory = gst::ElementFactory::make("edgeimpulseoverlay")
        .property("stroke-width", args.stroke_width)
        .property("text-color", text_color)
        .property("background-color", background_color)
        .property("text-scale-ratio", args.text_scale_ratio);
    if args.no_labels {
        overlay_factory = overlay_factory.property("show-labels", false);
    }
    let overlay = overlay_factory.build()?;

    let convert_out = gst::ElementFactory::make("videoconvert").build()?;

    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    let elements = [
        &source,
        &queue_in,
        &convert_in,
        &caps_filter,
        &ocr,
        &queue_out,
        &overlay,
        &convert_out,
        &sink,
    ];
    pipeline.add_many(elements)?;
    gst::Element::link_many(elements)?;

    Ok(pipeline)
}

fn example_main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    println!("🚀 Starting webcam OCR overlay");
    println!(
        "⚙️  backend={}  min-confidence={:.2}  max-text-length={}  interval={}",
        args.backend, args.min_confidence, args.max_text_length, args.interval
    );
    println!(
        "⚙️  overlay: stroke-width={}  text-scale-ratio={}  labels={}  text-color={}  background-color={}",
        args.stroke_width,
        args.text_scale_ratio,
        !args.no_labels,
        args.text_color,
        args.background_color
    );
    println!("ℹ️  Watch the per-line confidence below to pick a --min-confidence.");

    let pipeline = create_pipeline(&args)?;
    pipeline.set_state(gst::State::Playing)?;
    println!("▶️  Pipeline playing — hold some text up to the camera. Ctrl-C to stop.");

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Element(element) => {
                if let Some(s) = element.structure() {
                    if s.name() == "ocr" {
                        let text = s.get::<String>("text").unwrap_or_default();
                        let confidence = s.get::<f64>("confidence").unwrap_or(0.0);
                        let x = s.get::<i32>("x").unwrap_or(0);
                        let y = s.get::<i32>("y").unwrap_or(0);
                        let width = s.get::<i32>("width").unwrap_or(0);
                        let height = s.get::<i32>("height").unwrap_or(0);
                        println!(
                            "📖 OCR: {text:?} (confidence {confidence:.2}) at [{x}, {y}, {width}x{height}]"
                        );
                    }
                }
            }
            MessageView::Error(err) => {
                eprintln!(
                    "❌ Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            MessageView::Eos(..) => {
                println!("End of stream");
                break;
            }
            _ => (),
        }
    }

    pipeline.set_state(gst::State::Null)?;
    println!("Pipeline stopped");
    Ok(())
}

fn main() {
    run(|| {
        if let Err(e) = example_main() {
            eprintln!("❌ Error: {e}");
            std::process::exit(1);
        }
    });
}
