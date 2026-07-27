#![allow(deprecated)]
#![allow(unexpected_cfgs)]

//! Live webcam (or still-image) text recognition with the **FFI recognizer**
//! backend (`edgeimpulseocr backend=edge-impulse-recognizer`).
//!
//! Unlike `ocr_overlay.rs` (which uses the self-contained `ocrs` backend), this
//! example exercises the real two-stage crop -> recognizer path that runs on
//! device. The recognizer needs a [`CropOriginMeta`], which `edgeimpulsecrop`
//! only attaches for upstream detections. Since a bare camera frame carries no
//! detector output, this example synthesizes a **full-frame**
//! `VideoRegionOfInterestMeta` on every buffer via a pad probe; `edgeimpulsecrop`
//! turns it into a model-sized crop (default 320x48) plus the `CropOriginMeta`
//! the recognizer consumes. Point the camera at a line of (uppercase) text.
//!
//! The recognizer model must be baked into this build (FREEFORM_OUTPUT=1). Build
//! with the validated native-macOS full-tflite recipe and point GStreamer at the
//! freshly built plugin:
//!
//! ```sh
//! cd gst-plugin-edgeimpulse
//! # Only when switching the baked model/engine:
//! bash ../edge-impulse-ffi-rs/clean-model.sh
//! export GST_PLUGIN_PATH="$(pwd)/target/release"
//! DYLD_FALLBACK_LIBRARY_PATH=/tmp/bz2lib:/usr/local/lib:/usr/lib \
//! EI_MODEL=../ei-local-model TARGET_MAC_ARM64=1 USE_FULL_TFLITE=1 \
//! cargo run --release --no-default-features --features "ffi ocr" \
//!     --example webcam_text_recognizer -- --normalize upper-alnum --preview
//! ```
//!
//! Headless / no camera: feed a still image instead of the camera. It must be a
//! tight line of text at (or near) the model input size:
//!
//! ```sh
//! cargo run --release --no-default-features --features "ffi ocr" \
//!     --example webcam_text_recognizer -- --image /tmp/ocr_ABC123.png
//! ```
//!
//! Recognized lines are printed to stdout. Close the preview window or press
//! Ctrl-C to stop.

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use std::error::Error;

/// Live webcam / still-image test harness for the FFI text recognizer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Recognizer model input width (the crop is resized to this).
    #[arg(long, default_value = "320")]
    width: u32,

    /// Recognizer model input height (the crop is resized to this).
    #[arg(long, default_value = "48")]
    height: u32,

    /// Camera index for `avfvideosrc` (macOS). Ignored with `--image`.
    #[arg(long, default_value = "0")]
    device_index: i32,

    /// Use a still image instead of the camera (path to a decodable file).
    #[arg(long, default_value = "")]
    image: String,

    /// Post-decode normalization: none | upper | upper-alnum.
    #[arg(long, default_value = "upper-alnum")]
    normalize: String,

    /// Drop recognized lines whose confidence is below this value (0.0..1.0).
    #[arg(long, default_value = "0.0")]
    min_confidence: f64,

    /// Show the live camera feed in a window (helps you frame the text).
    #[arg(long)]
    preview: bool,
}

// macOS shows the video window from the Cocoa main loop, so the pipeline runs on
// a worker thread while `NSApplication` owns the main thread. On other platforms
// `run` just calls the closure directly. Only used when `--preview` is set.
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

/// Camera source fragment: `avfvideosrc` on macOS, `autovideosrc` elsewhere.
#[cfg(target_os = "macos")]
fn camera_source(device_index: i32) -> String {
    format!("avfvideosrc device-index={device_index}")
}
#[cfg(not(target_os = "macos"))]
fn camera_source(_device_index: i32) -> String {
    "autovideosrc".to_string()
}

/// Build the `gst-launch`-style pipeline description for the given arguments.
fn pipeline_description(args: &Args) -> String {
    // Source: still image (decodebin + imagefreeze) or live camera.
    let source = if args.image.is_empty() {
        camera_source(args.device_index)
    } else {
        format!(
            "filesrc location=\"{}\" ! decodebin ! imagefreeze",
            args.image
        )
    };

    // Front-end: convert to the RGB the OCR element requires.
    let front = "queue max-size-buffers=4 leaky=downstream ! videoconvert ! video/x-raw,format=RGB";

    // Crop fits the reading-zone ROI (attached by the probe) to the model size
    // preserving aspect ratio (resize-mode=fit-longest, matching how the model
    // was trained) and attaches CropOriginMeta for the recognizer.
    let crop = format!(
        "edgeimpulsecrop name=crop target-width={w} target-height={h} resize-mode=fit-longest",
        w = args.width,
        h = args.height,
    );

    // Recognizer: decodes text from the cropped band and posts an `ocr` message.
    let ocr = format!(
        "edgeimpulseocr backend=edge-impulse-recognizer normalize={n} \
         min-confidence={c} interval=1 post-message=true ! fakesink sync=false",
        n = args.normalize,
        c = args.min_confidence,
    );

    if args.preview {
        // Tee AFTER the crop so the preview window shows the exact model input
        // (the central band, fit to the model size) upscaled — align text to it.
        let pw = args.width * 2;
        let ph = args.height * 2;
        format!(
            "{source} ! {front} ! {crop} ! tee name=t \
             t. ! queue max-size-buffers=4 leaky=downstream ! {ocr} \
             t. ! queue max-size-buffers=4 leaky=downstream ! videoscale \
             ! video/x-raw,width={pw},height={ph} ! videoconvert ! autovideosink sync=false"
        )
    } else {
        format!("{source} ! {front} ! {crop} ! {ocr}")
    }
}

/// Attach a central reading-zone ROI to every buffer entering the crop, so the
/// crop -> recognizer path runs without an upstream detector. The zone is a
/// horizontal band whose aspect ratio matches the model input (target_w:target_h),
/// centered vertically. A band keeps the text large and legible instead of
/// squeezing the whole (often vertical) camera frame into 320x48.
fn install_reading_zone_probe(
    pipeline: &gst::Pipeline,
    target: (u32, u32),
    fallback: (u32, u32),
) -> Result<(), Box<dyn Error>> {
    let (target_w, target_h) = target;
    let crop = pipeline
        .by_name("crop")
        .ok_or("pipeline is missing the 'crop' element")?;
    let sink_pad = crop
        .static_pad("sink")
        .ok_or("edgeimpulsecrop has no sink pad")?;

    sink_pad.add_probe(gst::PadProbeType::BUFFER, move |pad, info| {
        // Frame size from the negotiated caps; fall back to the model size.
        let (w, h) = pad
            .current_caps()
            .and_then(|caps| gst_video::VideoInfo::from_caps(&caps).ok())
            .map(|vi| (vi.width(), vi.height()))
            .unwrap_or(fallback);

        // Reading band: full width, height set so the band matches the model's
        // aspect ratio, centered vertically.
        let band_h =
            ((w as u64 * target_h as u64) / target_w.max(1) as u64).clamp(1, h as u64) as u32;
        let y_off = h.saturating_sub(band_h) / 2;

        if let Some(buffer) = info.buffer_mut() {
            let buffer = buffer.make_mut();
            let mut roi =
                gst_video::VideoRegionOfInterestMeta::add(buffer, "text", (0, y_off, w, band_h));
            roi.add_param(
                gst::Structure::builder("detection")
                    .field("label", "text")
                    .field("confidence", 1.0_f64)
                    .field("object_id", 0_u64)
                    .build(),
            );
        }

        gst::PadProbeReturn::Ok
    });

    Ok(())
}

fn example_main(args: Args) -> Result<(), Box<dyn Error>> {
    gst::init()?;

    if !args.image.is_empty() && !std::path::Path::new(&args.image).exists() {
        return Err(format!("--image file not found: {}", args.image).into());
    }

    println!("🚀 webcam text recognizer (backend=edge-impulse-recognizer)");
    println!(
        "⚙️  model-size={}x{}  normalize={}  min-confidence={:.2}  source={}",
        args.width,
        args.height,
        args.normalize,
        args.min_confidence,
        if args.image.is_empty() {
            "camera".to_string()
        } else {
            args.image.clone()
        }
    );

    let description = pipeline_description(&args);

    let pipeline = gst::parse::launch_full(&description, None, gst::ParseFlags::empty())?
        .downcast::<gst::Pipeline>()
        .map_err(|_| "parsed pipeline is not a gst::Pipeline")?;

    install_reading_zone_probe(
        &pipeline,
        (args.width, args.height),
        (args.width, args.height),
    )?;

    pipeline.set_state(gst::State::Playing)?;
    println!(
        "▶️  playing — hold an uppercase code in the central horizontal band. Ctrl-C to stop."
    );

    let bus = pipeline.bus().ok_or("pipeline has no bus")?;
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
                            "📖 {text:?} (confidence {confidence:.2}) at [{x}, {y}, {width}x{height}]"
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
    let args = Args::parse();
    let preview = args.preview;
    let dispatch = move || {
        if let Err(e) = example_main(args) {
            eprintln!("❌ Error: {e}");
            std::process::exit(1);
        }
    };

    // A preview window needs the Cocoa main loop on macOS; without it, run
    // directly so headless/`--image` use needs no window server.
    if preview {
        run(dispatch);
    } else {
        dispatch();
    }
}
