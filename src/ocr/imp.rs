use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base as gst_base;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::VideoFrameExt;
use gstreamer_video::VideoFrameRef;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::ocr::backend::{Backend, NoopBackend, OcrBackend, OcrLine};
use crate::ocr::shaping::{attach_results, build_ocr_message, filter_and_truncate};

include!(concat!(env!("OUT_DIR"), "/type_names.rs"));

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulseocr",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse OCR"),
    )
});

#[derive(Debug, Clone)]
pub struct Settings {
    pub backend: String,
    pub detection_model: String,
    pub recognition_model: String,
    pub min_confidence: f64,
    pub max_text_length: u32,
    pub post_message: bool,
    pub interval: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            backend: "ocrs".into(),
            detection_model: String::new(),
            recognition_model: String::new(),
            min_confidence: 0.0,
            max_text_length: 256,
            post_message: true,
            interval: 1,
        }
    }
}

/// Recognition results shared from the worker thread to the streaming thread.
/// `generation` bumps on every new result so the streaming thread can post an
/// `ocr` message once per recognition rather than once per output frame.
#[derive(Default)]
struct Latest {
    generation: u64,
    lines: Vec<OcrLine>,
    /// PTS (ms) of the frame these lines were recognized from, carried so
    /// `ocr` messages timestamp the recognized frame, not the output frame.
    pts_ms: i64,
}

/// A frame handed to the worker for recognition.
struct FrameJob {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    pts_ms: i64,
}

struct State {
    frame_tx: SyncSender<FrameJob>,
    worker: Option<JoinHandle<()>>,
    latest: Arc<Mutex<Latest>>,
    frame_count: u64,
    last_posted_generation: u64,
}

#[derive(Default)]
pub struct EdgeImpulseOcr {
    pub(crate) settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
    info: Mutex<Option<gst_video::VideoInfo>>,
    /// Set once the worker thread is observed gone, so we warn only once
    /// instead of on every subsequently dropped frame.
    worker_gone_logged: AtomicBool,
    /// Frames seen by the edge-impulse path, used to post an `ocr` message only
    /// once per `interval` frames (reset in `start`).
    ei_frame_count: AtomicU64,
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseOcr {
    const NAME: &'static str = OCR_TYPE_NAME;
    type Type = super::EdgeImpulseOcr;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for EdgeImpulseOcr {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("backend")
                    .nick("Backend")
                    .blurb("OCR backend implementation to use")
                    .default_value(Some("ocrs"))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("detection-model")
                    .nick("Detection Model")
                    .blurb("Path to the OCR text detection model file")
                    .default_value(Some(""))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("recognition-model")
                    .nick("Recognition Model")
                    .blurb("Path to the OCR text recognition model file")
                    .default_value(Some(""))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecDouble::builder("min-confidence")
                    .nick("Minimum Confidence")
                    .blurb("Minimum confidence threshold for OCR text results")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(0.0)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecUInt::builder("max-text-length")
                    .nick("Maximum Text Length")
                    .blurb("Maximum number of characters to recognize per text region")
                    .minimum(1)
                    .maximum(u32::MAX)
                    .default_value(256)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecBoolean::builder("post-message")
                    .nick("Post Message")
                    .blurb("Post OCR results on the GStreamer bus")
                    .default_value(true)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecUInt::builder("interval")
                    .nick("Interval")
                    .blurb("Process one frame every N input frames")
                    .minimum(1)
                    .maximum(u32::MAX)
                    .default_value(1)
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings.lock().unwrap();
        match pspec.name() {
            "backend" => settings.backend = value.get().unwrap(),
            "detection-model" => settings.detection_model = value.get().unwrap(),
            "recognition-model" => settings.recognition_model = value.get().unwrap(),
            "min-confidence" => settings.min_confidence = value.get().unwrap(),
            "max-text-length" => settings.max_text_length = value.get().unwrap(),
            "post-message" => settings.post_message = value.get().unwrap(),
            "interval" => settings.interval = value.get().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings.lock().unwrap();
        match pspec.name() {
            "backend" => settings.backend.to_value(),
            "detection-model" => settings.detection_model.to_value(),
            "recognition-model" => settings.recognition_model.to_value(),
            "min-confidence" => settings.min_confidence.to_value(),
            "max-text-length" => settings.max_text_length.to_value(),
            "post-message" => settings.post_message.to_value(),
            "interval" => settings.interval.to_value(),
            _ => unimplemented!(),
        }
    }
}
impl GstObjectImpl for EdgeImpulseOcr {}

impl ElementImpl for EdgeImpulseOcr {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static M: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse OCR",
                "Filter/Analyzer/Video",
                "Reads text from video frames and attaches it as ROI metadata",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*M)
    }
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static T: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::new(1, i32::MAX))
                .field("height", gst::IntRange::new(1, i32::MAX))
                .build();
            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        });
        T.as_slice()
    }
}

impl EdgeImpulseOcr {
    fn build_backend(settings: &Settings) -> Box<dyn OcrBackend> {
        match Backend::parse(&settings.backend) {
            Some(Backend::Ocrs) => Self::build_ocrs(settings),
            _ => Box::new(NoopBackend),
        }
    }

    #[cfg(feature = "ocrs")]
    fn build_ocrs(settings: &Settings) -> Box<dyn OcrBackend> {
        // Each model loads from its explicit path when set, else from the
        // embedded default (see OcrsBackend::new), so partial configuration
        // still yields a working engine rather than a silent no-op.
        match crate::ocr::ocrs_backend::OcrsBackend::new(
            &settings.detection_model,
            &settings.recognition_model,
        ) {
            Ok(b) => Box::new(b),
            Err(e) => {
                gst::error!(
                    CAT,
                    "Failed to build ocrs backend: {e}; falling back to no-op"
                );
                Box::new(NoopBackend)
            }
        }
    }

    #[cfg(not(feature = "ocrs"))]
    fn build_ocrs(_settings: &Settings) -> Box<dyn OcrBackend> {
        Box::new(NoopBackend)
    }

    /// Decode the character detections an upstream `edgeimpulsevideoinfer`
    /// attached to this buffer into lines of text: consume the per-character ROI
    /// metas, assemble them into lines, attach one line ROI, and post one `ocr`
    /// message per line. Runs inline (no worker) because there is no model.
    fn transform_ip_edge_impulse(
        &self,
        buf: &mut gst::BufferRef,
        min_confidence: f64,
        max_text_length: u32,
        post_message: bool,
        interval: u32,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let pts_ms = buf.pts().map(|t| t.mseconds() as i64).unwrap_or(0);
        let lines = crate::ocr::decode::process_buffer(
            buf,
            min_confidence as f32,
            max_text_length as usize,
        );
        // Detections are consumed and lines attached every frame (so the overlay
        // stays stable); `interval` only throttles how often we *post* messages.
        let n = self.ei_frame_count.fetch_add(1, Ordering::Relaxed) + 1;
        let due = n % (interval.max(1) as u64) == 0;
        if post_message && due {
            for line in &lines {
                let s = build_ocr_message(line, pts_ms);
                let _ = self
                    .obj()
                    .post_message(gst::message::Element::builder(s).src(&*self.obj()).build());
            }
        }
        Ok(gst::FlowSuccess::Ok)
    }
}

impl BaseTransformImpl for EdgeImpulseOcr {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings.lock().unwrap().clone();
        // The edge-impulse backend decodes upstream detection metas inline in
        // transform_ip; it evaluates no model and needs no worker thread.
        if matches!(
            Backend::parse(&settings.backend),
            Some(Backend::EdgeImpulse)
        ) {
            self.ei_frame_count.store(0, Ordering::Relaxed);
            return self.parent_start();
        }
        self.worker_gone_logged.store(false, Ordering::Relaxed);
        let latest = Arc::new(Mutex::new(Latest::default()));
        let latest_worker = latest.clone();
        // Bound of 1: at most one frame waits while another is recognized;
        // newer frames are dropped (see `try_send` in transform_ip) so a slow
        // backend can never build an unbounded backlog.
        let (frame_tx, frame_rx) = sync_channel::<FrameJob>(1);
        // Build the backend inside the worker so model loading never blocks the
        // streaming thread, and recognition runs entirely off it.
        let worker = std::thread::spawn(move || {
            let mut backend = Self::build_backend(&settings);
            while let Ok(job) = frame_rx.recv() {
                match backend.recognize(&job.rgb, job.width, job.height) {
                    Ok(lines) => {
                        let mut latest = latest_worker.lock().unwrap();
                        latest.generation = latest.generation.wrapping_add(1);
                        latest.lines = lines;
                        latest.pts_ms = job.pts_ms;
                    }
                    Err(e) => gst::warning!(CAT, "OCR worker recognition failed: {e}"),
                }
            }
        });
        *self.state.lock().unwrap() = Some(State {
            frame_tx,
            worker: Some(worker),
            latest,
            frame_count: 0,
            last_posted_generation: 0,
        });
        self.parent_start()
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        // Take the state out (releasing the lock immediately), then drop the
        // sender so the worker's recv() returns Err and the thread exits, and
        // join it. The worker never locks `state`, so this cannot deadlock.
        let taken = self.state.lock().unwrap().take();
        if let Some(State {
            frame_tx, worker, ..
        }) = taken
        {
            drop(frame_tx);
            if let Some(handle) = worker {
                let _ = handle.join();
            }
        }
        *self.info.lock().unwrap() = None;
        self.parent_stop()
    }

    fn set_caps(&self, incaps: &gst::Caps, _outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let info = gst_video::VideoInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse OCR input caps"))?;
        *self.info.lock().unwrap() = Some(info);
        Ok(())
    }

    fn unit_size(&self, caps: &gst::Caps) -> Option<usize> {
        gst_video::VideoInfo::from_caps(caps)
            .ok()
            .map(|info| info.size())
    }

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        let (backend, interval, min_confidence, max_text_length, post_message) = {
            let settings = self.settings.lock().unwrap();
            (
                settings.backend.clone(),
                // max(1) keeps the modulo below divide-by-zero-safe independent
                // of the GObject minimum, so a direct Settings construction
                // (e.g. in a unit test) can never panic here.
                settings.interval.max(1),
                settings.min_confidence,
                settings.max_text_length,
                settings.post_message,
            )
        };

        // The edge-impulse backend decodes upstream detection metas synchronously
        // (no pixels, no worker); handle it before the caps/worker path below.
        if matches!(Backend::parse(&backend), Some(Backend::EdgeImpulse)) {
            return self.transform_ip_edge_impulse(
                buf,
                min_confidence,
                max_text_length,
                post_message,
                interval,
            );
        }

        let info = self.info.lock().unwrap().clone();
        let Some(info) = info else {
            return Ok(gst::FlowSuccess::Ok);
        };

        // Under the state lock (kept cheap): bump the frame counter, grab a
        // sender if this frame is due for recognition, snapshot the latest
        // results + their source PTS, and decide whether they are new enough to
        // post. The frame copy and message posting happen after the lock.
        let (sender, raw_lines, result_pts_ms, post_new) = {
            let mut guard = self.state.lock().unwrap();
            let Some(state) = guard.as_mut() else {
                return Ok(gst::FlowSuccess::Ok);
            };
            state.frame_count += 1;
            let sender = if state.frame_count % interval as u64 == 0 {
                Some(state.frame_tx.clone())
            } else {
                None
            };
            let (lines, generation, pts_ms) = {
                let latest = state.latest.lock().unwrap();
                (latest.lines.clone(), latest.generation, latest.pts_ms)
            };
            let post_new = post_message && generation != state.last_posted_generation;
            if post_new {
                state.last_posted_generation = generation;
            }
            (sender, lines, pts_ms, post_new)
        };

        // Hand the current frame to the worker, dropping it if the worker is
        // busy (or gone) so recognition never stalls the streaming thread.
        if let Some(sender) = sender {
            let pts_ms = buf.pts().map(|t| t.mseconds() as i64).unwrap_or(0);
            let job = {
                let frame = VideoFrameRef::from_buffer_ref_readable(buf, &info)
                    .map_err(|_| gst::FlowError::Error)?;
                let width = frame.width();
                let height = frame.height();
                let stride = frame.plane_stride()[0] as usize;
                let row_bytes = width as usize * 3;
                let src = frame.plane_data(0).map_err(|_| gst::FlowError::Error)?;
                let mut rgb = Vec::with_capacity(row_bytes * height as usize);
                for row in 0..height as usize {
                    let start = row * stride;
                    rgb.extend_from_slice(&src[start..start + row_bytes]);
                }
                FrameJob {
                    rgb,
                    width,
                    height,
                    pts_ms,
                }
            };
            if let Err(std::sync::mpsc::TrySendError::Disconnected(_)) = sender.try_send(job) {
                // The worker only disconnects if it panicked; warn once so a
                // wedged element that silently drops every frame from here on
                // is at least diagnosable.
                if !self.worker_gone_logged.swap(true, Ordering::Relaxed) {
                    gst::error!(
                        CAT,
                        obj = self.obj(),
                        "OCR worker thread has exited; recognition stopped"
                    );
                }
            }
        }

        // Filter on the streaming thread so `min-confidence` / `max-text-length`
        // stay live-settable (the worker stores raw results); this is cheap.
        let lines = filter_and_truncate(raw_lines, min_confidence as f32, max_text_length as usize);

        // Timestamp the frame the text was recognized from (carried through the
        // worker), not the current output frame which arrives later.
        if post_new {
            for line in &lines {
                let s = build_ocr_message(line, result_pts_ms);
                let _ = self
                    .obj()
                    .post_message(gst::message::Element::builder(s).src(&*self.obj()).build());
            }
        }
        attach_results(buf, &lines);
        Ok(gst::FlowSuccess::Ok)
    }
}
