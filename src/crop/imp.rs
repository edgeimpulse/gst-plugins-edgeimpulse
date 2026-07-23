//! # EdgeImpulseCrop — Dynamic crop element
//!
//! Reads bounding box metadata from upstream `edgeimpulsevideoinfer` and pushes
//! one cropped buffer per detection downstream.  Each crop carries a
//! [`CropOriginMeta`] so classification results can be mapped back to the
//! original frame coordinates.
//!
//! This is a 1-to-N element: one input buffer may produce N output buffers
//! (one per detected object).  If no detections are present the full frame
//! is passed through unchanged.

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use image::imageops::FilterType;
use image::{ImageBuffer, RgbImage};
use once_cell::sync::Lazy;
use std::sync::Mutex;

use super::meta::CropOriginMeta;

include!(concat!(env!("OUT_DIR"), "/type_names.rs"));

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsecrop".to_string()
    } else {
        format!("edgeimpulsecrop_{}", variant)
    };
    gst::DebugCategory::new(
        &name,
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Crop"),
    )
});

// ─── Detection extracted from ROI metadata ───────────────────────────────────

struct Detection {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    label: String,
    confidence: f64,
    object_id: u64,
}

// ─── Element state ───────────────────────────────────────────────────────────

struct State {
    /// Extra pixels around each bounding box
    padding: i32,
    /// Target width for output crops (0 = use crop's natural size)
    target_width: i32,
    /// Target height for output crops (0 = use crop's natural size)
    target_height: i32,
    /// How crops are fitted to the target dimensions
    resize_mode: ResizeMode,
    /// Input video info (set during caps negotiation)
    video_info: Option<gst_video::VideoInfo>,
    /// Whether we've set output caps yet
    src_caps_set: bool,
    /// Segment event deferred until after caps are set on src pad
    pending_segment: Option<gst::Event>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            padding: 0,
            target_width: 0,
            target_height: 0,
            resize_mode: ResizeMode::Squash,
            video_info: None,
            src_caps_set: false,
            pending_segment: None,
        }
    }
}

// ─── Element implementation ──────────────────────────────────────────────────

pub struct EdgeImpulseCrop {
    sink_pad: gst::Pad,
    src_pad: gst::Pad,
    state: Mutex<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseCrop {
    const NAME: &'static str = CROP_TYPE_NAME;
    type Type = super::EdgeImpulseCrop;
    type ParentType = gst::Element;

    fn with_class(klass: &Self::Class) -> Self {
        let sink_pad = gst::Pad::builder_from_template(&klass.pad_template("sink").unwrap())
            .chain_function(|pad, parent, buffer| {
                EdgeImpulseCrop::catch_panic_pad_function(
                    parent,
                    || Err(gst::FlowError::Error),
                    |crop| crop.sink_chain(pad, buffer),
                )
            })
            .event_function(|pad, parent, event| {
                EdgeImpulseCrop::catch_panic_pad_function(
                    parent,
                    || false,
                    |crop| crop.sink_event(pad, event),
                )
            })
            .build();

        let src_pad = gst::Pad::builder_from_template(&klass.pad_template("src").unwrap()).build();

        Self {
            sink_pad,
            src_pad,
            state: Mutex::new(State::default()),
        }
    }
}

impl ObjectImpl for EdgeImpulseCrop {
    fn constructed(&self) {
        self.parent_constructed();
        self.obj().add_pad(&self.sink_pad).unwrap();
        self.obj().add_pad(&self.src_pad).unwrap();
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecInt::builder("padding")
                    .nick("Padding")
                    .blurb("Extra pixels around each bounding box crop")
                    .minimum(0)
                    .maximum(1000)
                    .default_value(0)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecInt::builder("target-width")
                    .nick("Target Width")
                    .blurb(
                        "Resize all crops to this width (0 = keep natural crop size). \
                         Setting a fixed target avoids caps renegotiation per-crop.",
                    )
                    .minimum(0)
                    .maximum(4096)
                    .default_value(0)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecInt::builder("target-height")
                    .nick("Target Height")
                    .blurb(
                        "Resize all crops to this height (0 = keep natural crop size). \
                         Setting a fixed target avoids caps renegotiation per-crop.",
                    )
                    .minimum(0)
                    .maximum(4096)
                    .default_value(0)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("resize-mode")
                    .nick("Resize Mode")
                    .blurb(
                        "How crops are fitted to target-width/target-height: \
                         'squash' stretches ignoring aspect ratio (default); \
                         'fit-longest' preserves aspect ratio and zero-pads \
                         (matches Edge Impulse FIT_LONGEST model preprocessing).",
                    )
                    .default_value(Some("squash"))
                    .mutable_ready()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut state = self.state.lock().unwrap();
        match pspec.name() {
            "padding" => state.padding = value.get::<i32>().unwrap_or(0),
            "target-width" => state.target_width = value.get::<i32>().unwrap_or(0),
            "target-height" => state.target_height = value.get::<i32>().unwrap_or(0),
            "resize-mode" => {
                state.resize_mode =
                    ResizeMode::from_property(&value.get::<String>().unwrap_or_default())
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let state = self.state.lock().unwrap();
        match pspec.name() {
            "padding" => state.padding.to_value(),
            "target-width" => state.target_width.to_value(),
            "target-height" => state.target_height.to_value(),
            "resize-mode" => state.resize_mode.as_str().to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for EdgeImpulseCrop {}

impl ElementImpl for EdgeImpulseCrop {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Dynamic Crop",
                "Filter/Video",
                "Crops detected object regions from video frames based on upstream inference metadata. \
                 Produces one output buffer per detection, each tagged with source coordinates.",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst_video::VideoCapsBuilder::new()
                .format(gst_video::VideoFormat::Rgb)
                .build();

            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                // Src accepts any video/x-raw RGB — dimensions vary per crop
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        });
        PAD_TEMPLATES.as_ref()
    }

    fn change_state(
        &self,
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        if transition == gst::StateChange::ReadyToNull {
            let mut state = self.state.lock().unwrap();
            state.video_info = None;
            state.src_caps_set = false;
            state.pending_segment = None;
        }
        self.parent_change_state(transition)
    }
}

// ─── Pad functions ───────────────────────────────────────────────────────────

impl EdgeImpulseCrop {
    fn sink_event(&self, pad: &gst::Pad, event: gst::Event) -> bool {
        match event.view() {
            gst::EventView::Caps(caps_event) => {
                let caps = caps_event.caps();
                gst::info!(CAT, obj = self.obj(), "Received caps: {:?}", caps);

                if let Ok(info) = gst_video::VideoInfo::from_caps(caps) {
                    let mut state = self.state.lock().unwrap();
                    state.video_info = Some(info);
                    state.src_caps_set = false;
                }

                // Don't forward caps downstream — we'll set our own when we
                // know the crop dimensions.
                true
            }
            gst::EventView::Segment(_) => {
                // Hold the segment event — we'll forward it after setting
                // src caps to avoid "segment before caps" misordering.
                let mut state = self.state.lock().unwrap();
                state.pending_segment = Some(event);
                true
            }
            _ => gst::Pad::event_default(pad, Some(&*self.obj()), event),
        }
    }

    fn sink_chain(
        &self,
        _pad: &gst::Pad,
        buffer: gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let state = self.state.lock().unwrap();

        let video_info = match &state.video_info {
            Some(info) => info.clone(),
            None => {
                gst::error!(
                    CAT,
                    obj = self.obj(),
                    "No video info available — caps not negotiated"
                );
                return Err(gst::FlowError::NotNegotiated);
            }
        };

        let padding = state.padding;
        let target_width = state.target_width;
        let target_height = state.target_height;
        let resize_mode = state.resize_mode;
        drop(state);

        let frame_width = video_info.width();
        let frame_height = video_info.height();

        // Extract detections from ROI metadata
        let detections = self.extract_detections(&buffer);

        // No detections → pass full frame through unchanged
        if detections.is_empty() {
            gst::log!(CAT, obj = self.obj(), "No detections — passing full frame");
            self.ensure_src_caps(frame_width, frame_height, target_width, target_height)?;
            if target_width > 0 && target_height > 0 {
                // Resize full frame to target dimensions
                let resized = self.resize_buffer(
                    &buffer,
                    &video_info,
                    frame_width,
                    frame_height,
                    target_width as u32,
                    target_height as u32,
                    resize_mode,
                )?;
                return self.src_pad.push(resized);
            }
            return self.src_pad.push(buffer);
        }

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Processing {} detections from frame {}x{}",
            detections.len(),
            frame_width,
            frame_height
        );

        // For each detection, extract the crop and push downstream
        for det in &detections {
            let result = self.push_crop(
                &buffer,
                &video_info,
                det,
                padding,
                target_width,
                target_height,
                resize_mode,
            );
            match result {
                Ok(gst::FlowSuccess::Ok) => {}
                Err(gst::FlowError::Flushing) => return Err(gst::FlowError::Flushing),
                Err(e) => {
                    gst::warning!(
                        CAT,
                        obj = self.obj(),
                        "Failed to push crop for detection '{}': {:?}",
                        det.label,
                        e
                    );
                }
                _ => {}
            }
        }

        Ok(gst::FlowSuccess::Ok)
    }

    /// Extract detection bounding boxes from ROI metadata on the buffer.
    fn extract_detections(&self, buffer: &gst::Buffer) -> Vec<Detection> {
        let mut detections = Vec::new();

        for roi in buffer.iter_meta::<gst_video::VideoRegionOfInterestMeta>() {
            let meta = unsafe { &*roi.as_ptr() };
            let x = meta.x;
            let y = meta.y;
            let width = meta.w;
            let height = meta.h;

            let mut label = String::new();
            let mut confidence = 0.0_f64;
            let mut object_id = 0_u64;

            for param in roi.params() {
                if param.name() == "detection" {
                    if let Ok(l) = param.get::<&str>("label") {
                        label = l.to_string();
                    }
                    if let Ok(c) = param.get::<f64>("confidence") {
                        confidence = c;
                    }
                    if let Ok(id) = param.get::<u64>("object_id") {
                        object_id = id;
                    }
                }
            }

            detections.push(Detection {
                x,
                y,
                width,
                height,
                label,
                confidence,
                object_id,
            });
        }

        detections
    }

    /// Create a cropped buffer for a single detection and push it downstream.
    #[allow(clippy::too_many_arguments)]
    fn push_crop(
        &self,
        buffer: &gst::Buffer,
        video_info: &gst_video::VideoInfo,
        det: &Detection,
        padding: i32,
        target_width: i32,
        target_height: i32,
        resize_mode: ResizeMode,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let frame_w = video_info.width() as i32;
        let frame_h = video_info.height() as i32;

        // Apply padding and clamp to frame bounds
        let crop_x = (det.x as i32 - padding).max(0) as u32;
        let crop_y = (det.y as i32 - padding).max(0) as u32;
        let crop_r = ((det.x + det.width) as i32 + padding).min(frame_w) as u32;
        let crop_b = ((det.y + det.height) as i32 + padding).min(frame_h) as u32;
        let crop_w = crop_r - crop_x;
        let crop_h = crop_b - crop_y;

        if crop_w == 0 || crop_h == 0 {
            gst::warning!(
                CAT,
                obj = self.obj(),
                "Zero-size crop for detection '{}' — skipping",
                det.label
            );
            return Ok(gst::FlowSuccess::Ok);
        }

        // Read the source frame
        let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buffer.as_ref(), video_info)
            .map_err(|_| {
                gst::error!(CAT, obj = self.obj(), "Failed to map video frame");
                gst::FlowError::Error
            })?;

        let stride = frame.plane_stride()[0] as usize;
        let src_data = frame.plane_data(0).map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to get plane data");
            gst::FlowError::Error
        })?;

        // Extract crop pixels (RGB, 3 bytes per pixel)
        let bpp = 3_usize; // RGB
        let crop_stride = crop_w as usize * bpp;
        let mut crop_data = vec![0u8; crop_h as usize * crop_stride];

        for row in 0..crop_h as usize {
            let src_offset = (crop_y as usize + row) * stride + crop_x as usize * bpp;
            let dst_offset = row * crop_stride;
            let src_end = src_offset + crop_stride;

            if src_end <= src_data.len() {
                crop_data[dst_offset..dst_offset + crop_stride]
                    .copy_from_slice(&src_data[src_offset..src_end]);
            }
        }

        // Drop the frame ref before creating new buffers
        drop(frame);

        // Determine output dimensions
        let (out_w, out_h, out_data) = if target_width > 0
            && target_height > 0
            && (crop_w != target_width as u32 || crop_h != target_height as u32)
        {
            let out = resize_rgb(
                &crop_data,
                crop_w,
                crop_h,
                target_width as u32,
                target_height as u32,
                resize_mode,
            );
            (target_width as u32, target_height as u32, out)
        } else {
            (crop_w, crop_h, crop_data)
        };

        // Set output caps if dimensions changed
        self.ensure_src_caps(
            video_info.width(),
            video_info.height(),
            target_width,
            target_height,
        )?;

        // Create output buffer
        let mut out_buf = gst::Buffer::from_mut_slice(out_data);
        {
            let out_buf_ref = out_buf.make_mut();
            // Copy timestamp from source
            out_buf_ref.set_pts(buffer.pts());
            out_buf_ref.set_dts(buffer.dts());
            out_buf_ref.set_duration(buffer.duration());

            // Attach crop origin metadata
            let mut crop_meta = CropOriginMeta::add(out_buf_ref);
            crop_meta.set_source_x(crop_x);
            crop_meta.set_source_y(crop_y);
            crop_meta.set_source_width(crop_w);
            crop_meta.set_source_height(crop_h);
            crop_meta.set_original_width(video_info.width());
            crop_meta.set_original_height(video_info.height());
            crop_meta.set_object_id(det.object_id);
            crop_meta.set_detection_label(det.label.clone());
            crop_meta.set_detection_confidence(det.confidence);
        }

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Pushing crop: '{}' ({:.0}%) region {}x{}+{}+{} → {}x{}",
            det.label,
            det.confidence * 100.0,
            crop_w,
            crop_h,
            crop_x,
            crop_y,
            out_w,
            out_h
        );

        self.src_pad.push(out_buf)
    }

    /// Resize a full buffer to target dimensions.
    #[allow(clippy::too_many_arguments)]
    fn resize_buffer(
        &self,
        buffer: &gst::Buffer,
        video_info: &gst_video::VideoInfo,
        frame_w: u32,
        frame_h: u32,
        target_w: u32,
        target_h: u32,
        resize_mode: ResizeMode,
    ) -> Result<gst::Buffer, gst::FlowError> {
        let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buffer.as_ref(), video_info)
            .map_err(|_| gst::FlowError::Error)?;

        let src_data = frame.plane_data(0).map_err(|_| gst::FlowError::Error)?;
        let stride = frame.plane_stride()[0] as usize;
        let bpp = 3_usize;

        // Copy frame data respecting stride
        let mut contiguous = vec![0u8; (frame_w * frame_h * 3) as usize];
        for row in 0..frame_h as usize {
            let src_start = row * stride;
            let dst_start = row * frame_w as usize * bpp;
            let len = frame_w as usize * bpp;
            contiguous[dst_start..dst_start + len]
                .copy_from_slice(&src_data[src_start..src_start + len]);
        }
        drop(frame);

        let resized = resize_rgb(
            &contiguous,
            frame_w,
            frame_h,
            target_w,
            target_h,
            resize_mode,
        );

        let mut out_buf = gst::Buffer::from_mut_slice(resized);
        {
            let out_ref = out_buf.make_mut();
            out_ref.set_pts(buffer.pts());
            out_ref.set_dts(buffer.dts());
            out_ref.set_duration(buffer.duration());
        }
        Ok(out_buf)
    }

    /// Ensure src pad caps are set for the given output dimensions.
    fn ensure_src_caps(
        &self,
        frame_w: u32,
        frame_h: u32,
        target_width: i32,
        target_height: i32,
    ) -> Result<(), gst::FlowError> {
        let out_w = if target_width > 0 {
            target_width as u32
        } else {
            frame_w
        };
        let out_h = if target_height > 0 {
            target_height as u32
        } else {
            frame_h
        };

        let mut state = self.state.lock().unwrap();
        if state.src_caps_set {
            return Ok(());
        }

        let caps = gst_video::VideoInfo::builder(gst_video::VideoFormat::Rgb, out_w, out_h)
            .build()
            .map_err(|_| gst::FlowError::Error)?
            .to_caps()
            .map_err(|_| gst::FlowError::Error)?;

        gst::info!(CAT, obj = self.obj(), "Setting src caps: {:?}", caps);
        state.src_caps_set = true;
        let pending_segment = state.pending_segment.take();
        drop(state);

        // Push caps first, then the deferred segment event — correct ordering
        self.src_pad.push_event(gst::event::Caps::new(&caps));
        if let Some(segment) = pending_segment {
            self.src_pad.push_event(segment);
        }
        Ok(())
    }
}

// ─── Resize policy ───────────────────────────────────────────────────────────

/// How a crop is fitted to the target dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ResizeMode {
    /// Stretch to the target, ignoring aspect ratio (legacy behavior).
    Squash,
    /// Scale preserving aspect ratio to fit within the target, then center on a
    /// zero-padded (black) canvas. Mirrors Edge Impulse's
    /// `EI_CLASSIFIER_RESIZE_FIT_LONGEST` preprocessing so recognizer/classifier
    /// input matches how the model was trained.
    FitLongest,
}

impl ResizeMode {
    /// Parse from a property string; unknown values fall back to `Squash`.
    fn from_property(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "fit-longest" | "fit" | "longest" => ResizeMode::FitLongest,
            _ => ResizeMode::Squash,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ResizeMode::Squash => "squash",
            ResizeMode::FitLongest => "fit-longest",
        }
    }
}

/// Resize `src` RGB (`src_w`×`src_h`, tightly packed) to `dst_w`×`dst_h` using
/// `mode`. Returns exactly `dst_w*dst_h*3` bytes. Invalid input yields a black
/// canvas of the target size.
fn resize_rgb(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    mode: ResizeMode,
) -> Vec<u8> {
    let black = || vec![0u8; (dst_w as usize) * (dst_h as usize) * 3];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return black();
    }
    let img: RgbImage = match ImageBuffer::from_raw(src_w, src_h, src.to_vec()) {
        Some(i) => i,
        None => return black(),
    };

    match mode {
        ResizeMode::Squash => {
            image::imageops::resize(&img, dst_w, dst_h, FilterType::Triangle).into_raw()
        }
        ResizeMode::FitLongest => {
            // Scale by the limiting axis, preserving aspect ratio, then center on
            // a zero-padded canvas (Edge Impulse's FIT_LONGEST, processing.cpp).
            let src_aspect = src_w as f32 / src_h as f32;
            let dst_aspect = dst_w as f32 / dst_h as f32;
            let (resize_w, resize_h) = if src_aspect > dst_aspect {
                (
                    dst_w,
                    ((dst_w as f32 / src_aspect) as u32).max(1).min(dst_h),
                )
            } else {
                (
                    ((dst_h as f32 * src_aspect) as u32).max(1).min(dst_w),
                    dst_h,
                )
            };

            let resized =
                image::imageops::resize(&img, resize_w, resize_h, FilterType::Triangle).into_raw();

            let start_x = (dst_w - resize_w) / 2;
            let start_y = (dst_h - resize_h) / 2;
            let dst_row = (dst_w as usize) * 3;
            let src_row = (resize_w as usize) * 3;
            let mut canvas = black();
            for row in 0..resize_h as usize {
                let s = row * src_row;
                let d = (start_y as usize + row) * dst_row + (start_x as usize) * 3;
                canvas[d..d + src_row].copy_from_slice(&resized[s..s + src_row]);
            }
            canvas
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: (u8, u8, u8)) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            v.push(rgb.0);
            v.push(rgb.1);
            v.push(rgb.2);
        }
        v
    }

    fn px(buf: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8) {
        let o = ((y * w + x) * 3) as usize;
        (buf[o], buf[o + 1], buf[o + 2])
    }

    #[test]
    fn fit_longest_pads_narrow_source_with_black_sides() {
        // 48x48 white (aspect 1) into 320x48 (aspect ~6.67): limited by height,
        // resized to 48x48 centered at x=136, black on both sides.
        let src = solid(48, 48, (255, 255, 255));
        let out = resize_rgb(&src, 48, 48, 320, 48, ResizeMode::FitLongest);
        assert_eq!(out.len(), 320 * 48 * 3);
        assert_eq!(
            px(&out, 320, 0, 24),
            (0, 0, 0),
            "left padding must be black"
        );
        assert_eq!(
            px(&out, 320, 160, 24),
            (255, 255, 255),
            "center must hold the image"
        );
        assert_eq!(
            px(&out, 320, 315, 24),
            (0, 0, 0),
            "right padding must be black"
        );
    }

    #[test]
    fn fit_longest_pads_wide_source_with_black_top_bottom() {
        // 640x48 white (aspect ~13.3) into 320x48: limited by width, resized to
        // 320x24 centered at y=12, black top and bottom.
        let src = solid(640, 48, (255, 255, 255));
        let out = resize_rgb(&src, 640, 48, 320, 48, ResizeMode::FitLongest);
        assert_eq!(
            px(&out, 320, 160, 0),
            (0, 0, 0),
            "top padding must be black"
        );
        assert_eq!(
            px(&out, 320, 160, 24),
            (255, 255, 255),
            "center must hold the image"
        );
        assert_eq!(
            px(&out, 320, 160, 47),
            (0, 0, 0),
            "bottom padding must be black"
        );
    }

    #[test]
    fn fit_longest_matching_aspect_fills_without_padding() {
        // 640x96 (aspect ~6.67) into 320x48 (same aspect): fills entirely.
        let src = solid(640, 96, (200, 100, 50));
        let out = resize_rgb(&src, 640, 96, 320, 48, ResizeMode::FitLongest);
        assert_eq!(
            px(&out, 320, 0, 24),
            (200, 100, 50),
            "left edge filled (no padding)"
        );
        assert_eq!(
            px(&out, 320, 319, 24),
            (200, 100, 50),
            "right edge filled (no padding)"
        );
    }

    #[test]
    fn squash_fills_entire_target_ignoring_aspect() {
        // Squash stretches a square to fill the wide target — no black padding.
        let src = solid(48, 48, (255, 255, 255));
        let out = resize_rgb(&src, 48, 48, 320, 48, ResizeMode::Squash);
        assert_eq!(px(&out, 320, 0, 24), (255, 255, 255));
        assert_eq!(px(&out, 320, 315, 24), (255, 255, 255));
    }

    #[test]
    fn resize_mode_parses_from_property_string() {
        assert_eq!(
            ResizeMode::from_property("fit-longest"),
            ResizeMode::FitLongest
        );
        assert_eq!(
            ResizeMode::from_property("Fit_Longest"),
            ResizeMode::FitLongest
        );
        assert_eq!(ResizeMode::from_property("squash"), ResizeMode::Squash);
        assert_eq!(ResizeMode::from_property("whatever"), ResizeMode::Squash);
        assert_eq!(ResizeMode::FitLongest.as_str(), "fit-longest");
    }
}
