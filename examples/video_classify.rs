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
use edge_impulse_runner::{EimModel, ModelParameters};
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

fn create_pipeline(model_path: &Path, model_params: &ModelParameters) -> Result<gst::Pipeline, Box<dyn Error>> {
    // Get absolute path to model
    let absolute_model_path = model_path.canonicalize()?;
    println!("Using model at: {}", absolute_model_path.display());

    // Create a new pipeline
    let pipeline = gst::Pipeline::new();

    // Create elements
    let src = gst::ElementFactory::make("avfvideosrc")
        .build()?;

    let convert = gst::ElementFactory::make("videoconvert")
        .build()?;

    let scale = gst::ElementFactory::make("videoscale")
        .build()?;

    let caps = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            gst::Caps::builder("video/x-raw")
                .field("width", model_params.image_input_width as i32)
                .field("height", model_params.image_input_height as i32)
                .field("format", "RGB")
                .build()
        )
        .build()?;

    let tee = gst::ElementFactory::make("tee")
        .build()?;

    // Preview branch elements
    let queue_preview = gst::ElementFactory::make("queue")
        .build()?;
    let preview_convert = gst::ElementFactory::make("videoconvert")
        .build()?;
    let preview_sink = gst::ElementFactory::make("autovideosink")
        .build()?;

    // Inference branch elements
    let queue_inference = gst::ElementFactory::make("queue")
        .build()?;
    let inference_convert = gst::ElementFactory::make("videoconvert")
        .build()?;
    let inference = gst::ElementFactory::make("edgeimpulseinfer")
        .property("model-path", absolute_model_path.to_str().unwrap())
        .build()?;

    // Add elements to pipeline
    pipeline.add_many(&[
        &src,
        &convert,
        &scale,
        &caps,
        &tee,
        &queue_preview,
        &preview_convert,
        &preview_sink,
        &queue_inference,
        &inference_convert,
        &inference,
    ])?;

    // Link the main pipeline elements
    gst::Element::link_many(&[&src, &convert, &scale, &caps, &tee])?;

    // Link preview branch
    gst::Element::link_many(&[&queue_preview, &preview_convert, &preview_sink])?;

    // Link inference branch
    gst::Element::link_many(&[&queue_inference, &inference_convert, &inference])?;

    // Link tee pads
    let tee_src_pad_template = tee.pad_template("src_%u").unwrap();

    // Link preview branch
    let tee_preview_pad = tee.request_pad(&tee_src_pad_template, None, None).unwrap();
    let queue_preview_sink_pad = queue_preview.static_pad("sink").unwrap();
    tee_preview_pad.link(&queue_preview_sink_pad)?;

    // Link inference branch
    let tee_inference_pad = tee.request_pad(&tee_src_pad_template, None, None).unwrap();
    let queue_inference_sink_pad = queue_inference.static_pad("sink").unwrap();
    tee_inference_pad.link(&queue_inference_sink_pad)?;

    Ok(pipeline)
}

fn example_main() -> Result<(), Box<dyn Error>> {
    // Initialize GStreamer
    gst::init()?;

    // Parse command line arguments
    let params = VideoClassifyParams::parse();

    // Load the model and get its parameters
    let model = EimModel::new_with_debug(&params.model, params.debug)?;
    let model_params = model.parameters()?;

    println!(
        "Model expects {}x{} input with {} channels",
        model_params.image_input_width,
        model_params.image_input_height,
        model_params.image_channel_count
    );

    // Create pipeline using model parameters and path
    let pipeline = create_pipeline(&Path::new(&params.model), &model_params)?;

    // Start playing
    pipeline.set_state(gst::State::Playing)?;
    println!("Pipeline is playing...");

    // Wait for messages on the pipeline's bus
    let bus = pipeline.bus().unwrap();
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
            _ => (),
        }
    }

    // Cleanup
    pipeline.set_state(gst::State::Null)?;
    println!("Pipeline stopped");

    Ok(())
}

fn main() {
    run(|| example_main().unwrap());
}