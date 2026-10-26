//! # EdgeImpulseCrop — Dynamic crop element
//!
//! Reads bounding box metadata from upstream `edgeimpulsevideoinfer` and pushes
//! one cropped buffer per detection downstream.  Each crop carries a
//! [`CropOriginMeta`] so classification results can be mapped back to the
//! original frame coordinates.
//!
//! This is a 1-to-N element: one input buffer may produce N output buffers
//! (one per detected object).  If no detections are present the full frame
//! is passed through unchanged, or dropped when `drop-empty` is set.

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;

use super::meta::CropOriginMeta;
use crate::detection::{extract_detections, Detection};
use crate::resize::{resize_rgb, ResizeMode};

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
    /// Drop frames without detections instead of passing them through
    drop_empty: bool,
    /// Input video info (set during caps negotiation)
    video_info: Option<gst_video::VideoInfo>,
    /// Dimensions of the caps last pushed on the src pad, if any. Crops of
    /// their natural size change from buffer to buffer, and each change
    /// needs a new caps event before the buffer.
    src_dims: Option<(u32, u32)>,
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
            drop_empty: false,
            video_info: None,
            src_dims: None,
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
                         Takes effect only with target-height also set. A fixed size \
                         avoids renegotiating caps for every crop.",
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
                         Takes effect only with target-width also set. A fixed size \
                         avoids renegotiating caps for every crop.",
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
                         'fit-longest' preserves aspect ratio and zero-pads; \
                         'fit-shortest' preserves aspect ratio and center-crops \
                         (matching Edge Impulse model preprocessing).",
                    )
                    .default_value(Some("squash"))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("drop-empty")
                    .nick("Drop Empty")
                    .blurb(
                        "Drop frames without detections instead of passing the \
                         full frame through, so downstream elements only see crops",
                    )
                    .default_value(false)
                    .mutable_playing()
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
            "drop-empty" => state.drop_empty = value.get::<bool>().unwrap_or(false),
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
            "drop-empty" => state.drop_empty.to_value(),
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
            state.src_dims = None;
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
                    state.src_dims = None;
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
        let drop_empty = state.drop_empty;
        drop(state);

        let frame_width = video_info.width();
        let frame_height = video_info.height();

        // Extract detections from ROI metadata
        let detections = extract_detections(&buffer);

        // No detections → drop the frame, or pass the full frame through unchanged
        if detections.is_empty() {
            if drop_empty {
                gst::log!(CAT, obj = self.obj(), "No detections — dropping frame");
                return Ok(gst::FlowSuccess::Ok);
            }
            gst::log!(CAT, obj = self.obj(), "No detections — passing full frame");
            if let Some((out_w, out_h)) = fixed_size(target_width, target_height) {
                // Resize full frame to target dimensions
                let out_info = output_info(&video_info, out_w, out_h)?;
                let resized = self.resize_buffer(&buffer, &video_info, &out_info, resize_mode)?;
                self.ensure_src_caps(&out_info)?;
                return self.src_pad.push(resized);
            }
            self.ensure_src_caps(&output_info(&video_info, frame_width, frame_height)?)?;
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
        let (out_w, out_h, out_data) = match fixed_size(target_width, target_height) {
            Some((w, h)) if (crop_w, crop_h) != (w, h) => {
                let out = resize_rgb(&crop_data, crop_w, crop_h, w, h, resize_mode);
                (w, h, out)
            }
            _ => (crop_w, crop_h, crop_data),
        };

        // Announce this crop's size before pushing it, if it differs from the
        // last one's.
        let out_info = output_info(video_info, out_w, out_h)?;
        self.ensure_src_caps(&out_info)?;

        // Create output buffer
        let mut out_buf = rgb_buffer(out_data, &out_info);
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
            // The detection's own rect, unpadded and unclamped. `source_*` above
            // is this box grown by `padding` and clipped to the frame, which is
            // the right thing to *cut* but the wrong thing to match against.
            crop_meta.set_detection_x(det.x);
            crop_meta.set_detection_y(det.y);
            crop_meta.set_detection_width(det.width);
            crop_meta.set_detection_height(det.height);
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

    /// Resize a full buffer to the dimensions of `out_info`.
    fn resize_buffer(
        &self,
        buffer: &gst::Buffer,
        video_info: &gst_video::VideoInfo,
        out_info: &gst_video::VideoInfo,
        resize_mode: ResizeMode,
    ) -> Result<gst::Buffer, gst::FlowError> {
        let frame_w = video_info.width();
        let frame_h = video_info.height();
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
            out_info.width(),
            out_info.height(),
            resize_mode,
        );

        let mut out_buf = rgb_buffer(resized, out_info);
        {
            let out_ref = out_buf.make_mut();
            out_ref.set_pts(buffer.pts());
            out_ref.set_dts(buffer.dts());
            out_ref.set_duration(buffer.duration());
        }
        Ok(out_buf)
    }

    /// Push caps for `info` on the src pad when its size differs from the last
    /// caps pushed, then any segment held back until caps were known.
    fn ensure_src_caps(&self, info: &gst_video::VideoInfo) -> Result<(), gst::FlowError> {
        let dims = (info.width(), info.height());
        let mut state = self.state.lock().unwrap();
        let changed = state.src_dims != Some(dims);
        let pending_segment = state.pending_segment.take();
        if !changed && pending_segment.is_none() {
            return Ok(());
        }
        let caps = if changed {
            Some(info.to_caps().map_err(|_| gst::FlowError::Error)?)
        } else {
            None
        };
        state.src_dims = Some(dims);
        drop(state);

        // Caps first, then the deferred segment event — correct ordering
        if let Some(caps) = caps {
            gst::debug!(CAT, obj = self.obj(), "Setting src caps: {:?}", caps);
            if !self.src_pad.push_event(gst::event::Caps::new(&caps)) {
                gst::warning!(CAT, obj = self.obj(), "Downstream refused caps {:?}", caps);
            }
        }
        if let Some(segment) = pending_segment {
            self.src_pad.push_event(segment);
        }
        Ok(())
    }
}

/// The fixed output size, when both target dimensions are set.
fn fixed_size(target_width: i32, target_height: i32) -> Option<(u32, u32)> {
    (target_width > 0 && target_height > 0).then_some((target_width as u32, target_height as u32))
}

/// Video info for an RGB output of `width` x `height`, keeping the input's
/// framerate and pixel aspect ratio.
fn output_info(
    input: &gst_video::VideoInfo,
    width: u32,
    height: u32,
) -> Result<gst_video::VideoInfo, gst::FlowError> {
    gst_video::VideoInfo::builder(gst_video::VideoFormat::Rgb, width, height)
        .fps(input.fps())
        .par(input.par())
        .build()
        .map_err(|_| gst::FlowError::Error)
}

/// Lay tightly packed RGB rows out as `info` describes them. GStreamer pads
/// every RGB row to a multiple of 4 bytes, so a crop whose width times 3 is
/// not one needs the padding added, or downstream reads skewed rows.
fn rgb_buffer(tight: Vec<u8>, info: &gst_video::VideoInfo) -> gst::Buffer {
    let row = info.width() as usize * 3;
    let stride = info.stride()[0] as usize;
    if stride == row {
        return gst::Buffer::from_mut_slice(tight);
    }
    let mut laid_out = vec![0u8; info.size()];
    for (i, pixels) in tight.chunks_exact(row).enumerate() {
        laid_out[i * stride..i * stride + row].copy_from_slice(pixels);
    }
    gst::Buffer::from_mut_slice(laid_out)
}

#[cfg(test)]
mod tests {
    use gstreamer as gst;
    use gstreamer::prelude::*;
    use gstreamer_app as gst_app;
    use gstreamer_video as gst_video;

    const ELEMENT: &str = "eitestcrop";
    const FRAME_W: u32 = 8;
    const FRAME_H: u32 = 6;

    fn init() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            gst::init().unwrap();
            gst::Element::register(
                None,
                ELEMENT,
                gst::Rank::NONE,
                crate::crop::EdgeImpulseCrop::static_type(),
            )
            .unwrap();
        });
    }

    /// Byte of pixel (x, y), channel c, in the input frame.
    fn pixel(x: u32, y: u32, c: u32) -> u8 {
        ((y * FRAME_W + x) * 3 + c) as u8
    }

    struct Crop {
        width: u32,
        height: u32,
        /// Tightly packed RGB rows, read through the caps' stride.
        pixels: Vec<u8>,
    }

    /// Push one 8x6 frame carrying `boxes` (x, y, w, h) through the crop and
    /// return every buffer it pushed, read as its caps describe it.
    fn run(crop_props: &str, boxes: &[(u32, u32, u32, u32)]) -> Vec<Crop> {
        init();
        let pipeline = gst::parse::launch(&format!(
            "appsrc name=src format=time \
             caps=video/x-raw,format=RGB,width={FRAME_W},height={FRAME_H},framerate=10/1 \
             ! {ELEMENT} {crop_props} ! appsink name=out sync=false"
        ))
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let src = pipeline
            .by_name("src")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();
        let out = pipeline
            .by_name("out")
            .unwrap()
            .downcast::<gst_app::AppSink>()
            .unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();

        let data: Vec<u8> = (0..FRAME_H)
            .flat_map(|y| (0..FRAME_W).flat_map(move |x| (0..3).map(move |c| pixel(x, y, c))))
            .collect();
        let mut buffer = gst::Buffer::from_mut_slice(data);
        {
            let buffer = buffer.get_mut().unwrap();
            buffer.set_pts(gst::ClockTime::from_mseconds(100));
            for (i, rect) in boxes.iter().enumerate() {
                let mut roi = gst_video::VideoRegionOfInterestMeta::add(buffer, "obj", *rect);
                roi.add_param(
                    gst::Structure::builder("detection")
                        .field("label", "obj")
                        .field("confidence", 0.9f64)
                        .field("object_id", i as u64)
                        .build(),
                );
            }
        }
        src.push_buffer(buffer).unwrap();
        src.end_of_stream().unwrap();

        let mut crops = Vec::new();
        while let Ok(sample) = out.pull_sample() {
            let info = gst_video::VideoInfo::from_caps(sample.caps().unwrap()).unwrap();
            let buffer = sample.buffer().unwrap();
            assert_eq!(
                buffer.size(),
                info.size(),
                "buffer does not match its caps {}x{}",
                info.width(),
                info.height()
            );
            let map = buffer.map_readable().unwrap();
            let stride = info.stride()[0] as usize;
            let row = info.width() as usize * 3;
            let pixels = (0..info.height() as usize)
                .flat_map(|y| map[y * stride..y * stride + row].to_vec())
                .collect();
            crops.push(Crop {
                width: info.width(),
                height: info.height(),
                pixels,
            });
        }
        pipeline.set_state(gst::State::Null).unwrap();
        crops
    }

    fn expected_pixels(x0: u32, y0: u32, w: u32, h: u32) -> Vec<u8> {
        (y0..y0 + h)
            .flat_map(|y| (x0..x0 + w).flat_map(move |x| (0..3).map(move |c| pixel(x, y, c))))
            .collect()
    }

    #[test]
    fn natural_size_crops_carry_their_own_caps_and_pixels() {
        // 3 and 5 pixels wide: 9 and 15 bytes a row, both padded to 12 and 16.
        let crops = run("", &[(1, 1, 3, 2), (2, 2, 5, 3)]);

        assert_eq!(crops.len(), 2);
        assert_eq!((crops[0].width, crops[0].height), (3, 2));
        assert_eq!(crops[0].pixels, expected_pixels(1, 1, 3, 2));
        assert_eq!((crops[1].width, crops[1].height), (5, 3));
        assert_eq!(crops[1].pixels, expected_pixels(2, 2, 5, 3));
    }

    #[test]
    fn a_fixed_target_size_resizes_every_crop_to_it() {
        let crops = run(
            "target-width=5 target-height=4",
            &[(1, 1, 3, 2), (0, 0, 5, 4)],
        );

        assert_eq!(crops.len(), 2);
        assert!(crops.iter().all(|c| (c.width, c.height) == (5, 4)));
        // Already the target size: copied, not resampled.
        assert_eq!(crops[1].pixels, expected_pixels(0, 0, 5, 4));
    }

    #[test]
    fn a_frame_without_detections_passes_through_at_its_size() {
        let crops = run("", &[]);

        assert_eq!(crops.len(), 1);
        assert_eq!((crops[0].width, crops[0].height), (FRAME_W, FRAME_H));
        assert_eq!(crops[0].pixels, expected_pixels(0, 0, FRAME_W, FRAME_H));
    }

    #[test]
    fn a_frame_without_detections_is_resized_to_a_fixed_target() {
        let crops = run("target-width=3 target-height=3", &[]);

        assert_eq!(crops.len(), 1);
        assert_eq!((crops[0].width, crops[0].height), (3, 3));
    }

    #[test]
    fn a_single_target_dimension_keeps_the_natural_size() {
        let crops = run("target-width=4", &[(1, 1, 3, 2)]);

        assert_eq!(crops.len(), 1);
        assert_eq!((crops[0].width, crops[0].height), (3, 2));
    }
}
