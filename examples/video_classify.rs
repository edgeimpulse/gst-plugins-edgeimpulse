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

/// Command line parameters for the video classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct VideoClassifyParams {
    /// Path to Edge Impulse model file
    #[arg(short, long)]
    model: String,

    /// Video format (RGB, RGBA, BGR, BGRA)
    #[arg(short, long, default_value = "RGB")]
    format: String,

    /// Input width
    #[arg(short = 'W', long, default_value_t = 96)]
    width: i32,

    /// Input height
    #[arg(short = 'H', long, default_value_t = 96)]
    height: i32,

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

fn create_pipeline(args: &VideoClassifyParams) -> Result<gst::Pipeline, Box<dyn Error>> {
    // Initialize GStreamer
    gst::init()?;

    // Create pipeline
    let pipeline = gst::Pipeline::new();

    // Create elements
    let src = gst::ElementFactory::make("avfvideosrc")
        .build()
        .expect("Could not create avfvideosrc element.");
    let videoconvert0 = gst::ElementFactory::make("videoconvert")
        .build()
        .expect("Could not create videoconvert element.");
    let videoscale = gst::ElementFactory::make("videoscale")
        .build()
        .expect("Could not create videoscale element.");
    let caps1 = gst::ElementFactory::make("capsfilter")
        .build()
        .expect("Could not create capsfilter element.");
    let classifier = gst::ElementFactory::make("edgeimpulseinfer")
        .build()
        .expect("Could not create edgeimpulseinfer element.");
    let videoconvert1 = gst::ElementFactory::make("videoconvert")
        .build()
        .expect("Could not create videoconvert element.");
    let sink = gst::ElementFactory::make("autovideosink")
        .build()
        .expect("Could not create autovideosink element.");

    // Set caps for the classifier input
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", 96i32)
        .field("height", 96i32)
        .build();
    caps1.set_property("caps", &caps);

    // Set the model path
    classifier.set_property("model-path", args.model.to_string());

    // Add elements to the pipeline
    pipeline.add_many(&[&src, &videoconvert0, &videoscale, &caps1, &classifier, &videoconvert1, &sink])?;

    // Link the elements
    gst::Element::link_many(&[&src, &videoconvert0, &videoscale, &caps1])?;
    gst::Element::link_many(&[&classifier, &videoconvert1, &sink])?;

    // Link caps1 to classifier with compatible caps
    caps1.link(&classifier)?;

    Ok(pipeline)
}

fn example_main() -> Result<(), Box<dyn Error>> {
    let args = VideoClassifyParams::parse();

    let pipeline = create_pipeline(&args)?;

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null)?;
                eprintln!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            MessageView::Eos(..) => break,
            MessageView::Element(element) => {
                if let Some(s) = element.structure() {
                    if s.name() == "edge-impulse-inference-result" {
                        println!("Inference result: {:#?}", s);
                    }
                }
            }
            _ => (),
        }
    }

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