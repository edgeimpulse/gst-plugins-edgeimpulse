//! Video Classification Example using edgeimpulseinfer GStreamer plugin
//!
//! This example demonstrates how to use the Edge Impulse GStreamer plugin to perform
//! video classification using a trained model.
//!
//! Usage:
//!   cargo run --example video_classify -- --model <path_to_model> [--debug]
//!
//! Environment setup:
//! export GST_PLUGIN_PATH="target/debug:$GST_PLUGIN_PATH"

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use std::error::Error;
use std::path::Path;

/// Command line parameters for the video classification example
#[derive(Parser, Debug)]
struct VideoClassifyParams {
    /// Path to the Edge Impulse model file
    #[arg(short, long)]
    model: String,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,

    /// Confidence threshold (0.0 to 1.0) for showing results
    #[clap(short, long, default_value = "0.8")]
    threshold: f32,
}

// macOS specific run loop handling
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
            (applicationDidFinishLaunching:) => on_finish_launching as extern fn(&Object, Sel, id)
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

fn create_pipeline(model_path: &Path) -> Result<gst::Pipeline, Box<dyn Error>> {
    // Get absolute path to model
    let absolute_model_path = model_path.canonicalize()?;
    println!("Using model at: {}", absolute_model_path.display());

    // Create a new pipeline
    let pipeline = gst::Pipeline::new();

    // Create elements
    let src = gst::ElementFactory::make("avfvideosrc")
        .property("device-index", 0i32)
        .build()?;

    let videoscale = gst::ElementFactory::make("videoscale").build()?;
    let videoconvert = gst::ElementFactory::make("videoconvert").build()?;
    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            gst::Caps::builder("video/x-raw")
                .field("width", 1280i32)
                .field("height", 720i32)
                .build(),
        )
        .build()?;

    let tee = gst::ElementFactory::make("tee").build()?;

    // Preview branch
    let queue_preview = gst::ElementFactory::make("queue").build()?;
    let convert_preview = gst::ElementFactory::make("videoconvert").build()?;
    let sink_preview = gst::ElementFactory::make("autovideosink").build()?;

    // Inference branch
    let queue_infer = gst::ElementFactory::make("queue").build()?;
    let convert_infer = gst::ElementFactory::make("videoconvert").build()?;
    let inferencer = gst::ElementFactory::make("edgeimpulseinfer")
        .property("model-path", absolute_model_path.to_str().unwrap())
        .build()?;

    // Add elements to pipeline
    pipeline.add_many(&[
        &src, &videoscale, &videoconvert, &capsfilter, &tee,
        &queue_preview, &convert_preview, &sink_preview,
        &queue_infer, &convert_infer, &inferencer,
    ])?;

    // Link the main pipeline elements
    gst::Element::link_many(&[
        &src, &videoscale, &videoconvert, &capsfilter, &tee,
    ])?;

    // Link preview branch
    gst::Element::link_many(&[
        &queue_preview, &convert_preview, &sink_preview,
    ])?;

    // Link inference branch
    gst::Element::link_many(&[
        &queue_infer, &convert_infer, &inferencer,
    ])?;

    // Link tee to both branches
    let tee_src_pad_template = tee.pad_template("src_%u").unwrap();
    let tee_preview_pad = tee.request_pad(&tee_src_pad_template, None, None)
        .ok_or("Failed to get tee preview pad")?;
    let tee_infer_pad = tee.request_pad(&tee_src_pad_template, None, None)
        .ok_or("Failed to get tee inference pad")?;
    let queue_preview_pad = queue_preview.static_pad("sink")
        .ok_or("Failed to get queue preview sink pad")?;
    let queue_infer_pad = queue_infer.static_pad("sink")
        .ok_or("Failed to get queue inference sink pad")?;

    tee_preview_pad.link(&queue_preview_pad)?;
    tee_infer_pad.link(&queue_infer_pad)?;

    Ok(pipeline)
}

fn example_main() -> Result<(), Box<dyn Error>> {
    let args = VideoClassifyParams::parse();

    // Initialize GStreamer
    gst::init()?;

    // Create and build the pipeline
    let pipeline = create_pipeline(Path::new(&args.model))?;

    // Start playing
    pipeline.set_state(gst::State::Playing)?;
    println!("Playing... (Ctrl+C to stop)");

    // Wait until error or EOS
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                println!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            MessageView::StateChanged(state) => {
                if args.debug {
                    println!(
                        "State changed from {:?}: {:?} -> {:?}",
                        state.src().map(|s| s.path_string()),
                        state.old(),
                        state.current()
                    );
                }
            }
            _ => (),
        }
    }

    // Cleanup
    pipeline.set_state(gst::State::Null)?;
    println!("Pipeline stopped");

    Ok(())
}

fn main() {
    #[cfg(target_os = "macos")]
    run(|| example_main().unwrap());

    #[cfg(not(target_os = "macos"))]
    println!("This example only works on macOS");
}