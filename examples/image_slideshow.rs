use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use tempfile::TempDir;

use anyhow::{anyhow, Result};
use clap::Parser;
use gstreamer::prelude::*;
use gstreamer::{Element, ElementFactory, MessageView, Pipeline, State};
use image::io::Reader as ImageReader;
use image::ImageFormat;

/// A GStreamer-based image slideshow that runs inference on images from a folder.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Edge Impulse model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to the folder containing images
    #[arg(short, long)]
    folder: PathBuf,

    /// Maximum number of images to process (default: 100)
    #[arg(short = 'n', long, default_value = "100")]
    max_images: usize,

    /// Input width
    #[arg(short = 'W', long)]
    width: i32,

    /// Input height
    #[arg(short = 'H', long)]
    height: i32,

    /// Slideshow framerate (images per second, default: 1)
    #[arg(long, default_value = "1")]
    framerate: i32,
}

/// Copies and converts images from the source folder to a temporary directory as JPEGs named image_N.jpg.
fn copy_and_rename_images(
    source_folder: &Path,
    temp_dir: &TempDir,
    max_images: usize,
) -> Result<PathBuf> {
    let mut image_files: Vec<_> = fs::read_dir(source_folder)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    let ext = ext.to_lowercase();
                    ext == "jpg" || ext == "jpeg" || ext == "png"
                })
                .unwrap_or(false)
        })
        .collect();

    image_files.sort_by_key(|entry| entry.path());
    image_files.truncate(max_images);

    for (i, entry) in image_files.iter().enumerate() {
        let old_path = entry.path();
        let new_name = format!("image_{}.jpg", i + 1);
        let new_path = temp_dir.path().join(new_name);
        println!("Converting {:?} -> {:?}", old_path, new_path);
        let img = ImageReader::open(&old_path)?
            .with_guessed_format()?
            .decode()?;
        img.save_with_format(&new_path, ImageFormat::Jpeg)?;
    }

    Ok(temp_dir.path().to_path_buf())
}

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
    use cocoa::appkit::NSApplication;
    use cocoa::base::id;
    use cocoa::delegate;
    use objc::runtime::{Object, Sel};
    use objc::{msg_send, sel, sel_impl};
    use std::{
        ffi::c_void,
        sync::mpsc::{channel, Sender},
        thread,
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
        let _: () = msg_send![app, setDelegate: delegate];

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

/// Main function: builds and runs the GStreamer image slideshow pipeline.
fn example_main() -> Result<()> {
    let args = Args::parse();

    if !args.model.exists() {
        return Err(anyhow!("Model file does not exist: {:?}", args.model));
    }
    if !args.folder.exists() || !args.folder.is_dir() {
        return Err(anyhow!(
            "Image folder does not exist or is not a directory: {:?}",
            args.folder
        ));
    }

    let temp_dir = TempDir::new()?;
    let images_folder = copy_and_rename_images(&args.folder, &temp_dir, args.max_images)?;

    gstreamer::init()?;

    let pipeline = Pipeline::new();
    let filesrc = ElementFactory::make("multifilesrc")
        .property(
            "location",
            images_folder.join("image_%d.jpg").to_str().unwrap(),
        )
        .property("start-index", 1i32)
        .property("loop", true)
        .property("caps", &gstreamer::Caps::builder("image/jpeg").build())
        .property("blocksize", 0u32)
        .build()?;
    let decodebin = ElementFactory::make("decodebin").build()?;
    let videoconvert1 = ElementFactory::make("videoconvert").build()?;
    let queue = ElementFactory::make("queue")
        .property_from_str("leaky", "downstream")
        .property("max-size-buffers", 1u32)
        .property("max-size-time", 30000000000u64)
        .build()?;
    let videoscale = ElementFactory::make("videoscale").build()?;
    let videorate = ElementFactory::make("videorate")
        .property("max-rate", args.framerate)
        .build()?;
    let capsfilter_gray = ElementFactory::make("capsfilter").build()?;
    let edgeimpulse = ElementFactory::make("edgeimpulsevideoinfer")
        .property("model-path", args.model.to_str().unwrap())
        .build()?;
    let videoconvert2 = ElementFactory::make("videoconvert").build()?;
    let capsfilter_rgb = ElementFactory::make("capsfilter").build()?;
    let overlay = ElementFactory::make("edgeimpulseoverlay").build()?;
    let autovideosink = ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    // Set caps for GRAY8 before inference, including framerate to control slideshow speed
    let caps_gray = gstreamer::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .field("width", args.width)
        .field("height", args.height)
        .field("framerate", &gstreamer::Fraction::new(args.framerate, 1))
        .build();
    capsfilter_gray.set_property("caps", &caps_gray);

    let caps_rgb = gstreamer::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", args.width)
        .field("height", args.height)
        .build();
    capsfilter_rgb.set_property("caps", &caps_rgb);

    pipeline.add_many([
        &filesrc,
        &decodebin,
        &videoconvert1,
        &queue,
        &videoscale,
        &videorate,
        &capsfilter_gray,
        &edgeimpulse,
        &videoconvert2,
        &capsfilter_rgb,
        &overlay,
        &autovideosink,
    ])?;

    filesrc.link(&decodebin)?;

    let videoconvert1_clone = videoconvert1.clone();
    decodebin.connect_pad_added(move |_dbin, src_pad| {
        let sink_pad = videoconvert1_clone
            .static_pad("sink")
            .expect("Failed to get sink pad from videoconvert1");
        if sink_pad.is_linked() {
            return;
        }
        let src_pad_caps = src_pad
            .current_caps()
            .expect("Failed to get caps from decodebin src pad");
        let src_pad_struct = src_pad_caps
            .structure(0)
            .expect("Failed to get structure from caps");
        if src_pad_struct.name().starts_with("video/") {
            src_pad
                .link(&sink_pad)
                .expect("Failed to link decodebin to videoconvert1");
        }
    });

    Element::link_many([
        &videoconvert1,
        &queue,
        &videoscale,
        &videorate,
        &capsfilter_gray,
        &edgeimpulse,
        &videoconvert2,
        &capsfilter_rgb,
        &overlay,
        &autovideosink,
    ])?;

    pipeline.set_state(State::Playing)?;

    let main_loop = glib::MainLoop::new(None, false);
    let bus = pipeline
        .bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    let _ = bus.add_watch(move |_, msg| {
        match msg.view() {
            MessageView::Eos(..) => {
                println!("End of stream");
                process::exit(0);
            }
            MessageView::Error(err) => {
                println!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                process::exit(1);
            }
            _ => (),
        };
        glib::ControlFlow::Continue
    })?;

    main_loop.run();
    pipeline.set_state(State::Null)?;
    Ok(())
}

fn main() {
    run(|| example_main().unwrap());
}
