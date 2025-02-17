use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base as gst_base;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::{VideoFormat, VideoFrameRef, VideoInfo};
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// A GStreamer element that draws bounding boxes on video frames based on ROI metadata.
///
/// The EdgeImpulseOverlay element is designed to visualize regions of interest (ROIs)
/// in video streams by drawing bounding boxes around them. It processes video frames
/// in RGB format and draws boxes based on VideoRegionOfInterestMeta metadata
/// attached to the video buffers.
///
/// # Features
/// - Draws bounding boxes around regions specified by VideoRegionOfInterestMeta
/// - Configurable box color and stroke width
/// - Supports RGB video format
/// - In-place buffer modification for efficient processing
/// - Safe bounds checking for all drawing operations
///
/// # Properties
/// - `bbox-color`: The color of the bounding box in RGB format (default: green 0x00FF00)
/// - `stroke-width`: Width of the bounding box lines in pixels (default: 2, minimum: 1)
///
/// # Formats
/// - Input: RGB
/// - Output: Same as input (in-place modification)
///
/// # Metadata
/// Reads VideoRegionOfInterestMeta from input buffers to determine where to draw boxes.
/// Each ROI meta contains:
/// - x, y: Top-left corner coordinates
/// - width, height: Dimensions of the region
/// - roi_type: Type identifier for the region
///
/// # Example Pipeline
/// ```text
/// gst-launch-1.0 \
///   videotestsrc ! \
///   video/x-raw,format=RGB,width=384,height=384 ! \
///   edgeimpulsevideoinfer ! \
///   edgeimpulseoverlay stroke-width=3 bbox-color=0xFF0000 ! \
///   autovideosink
/// ```
///
// Static category for logging
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulseoverlay",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Overlay Element"),
    )
});

// Settings for configuring the overlay appearance
#[derive(Debug, Clone)]
struct Settings {
    bbox_color: u32,
    stroke_width: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            bbox_color: 0x00FF00, // Default to green
            stroke_width: 2,      // Default to 2 pixels
        }
    }
}

// State struct to hold element properties and runtime data
#[derive(Default)]
pub struct EdgeImpulseOverlay {
    settings: Mutex<Settings>,
    video_info: Mutex<Option<VideoInfo>>,
}

// Implementation of GObject virtual methods
#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseOverlay {
    const NAME: &'static str = "EdgeImpulseOverlay";
    type Type = super::EdgeImpulseOverlay;
    type ParentType = gst_video::VideoFilter;
}

// Implementation of Object virtual methods
impl ObjectImpl for EdgeImpulseOverlay {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecUInt::builder("bbox-color")
                    .nick("Bounding Box Color")
                    .blurb("Color of the bounding box in RGB format (default: green 0x00FF00)")
                    .default_value(0x00FF00) // Green
                    .build(),
                glib::ParamSpecInt::builder("stroke-width")
                    .nick("Stroke Width")
                    .blurb("Width of the bounding box lines in pixels")
                    .default_value(2)
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "bbox-color" => {
                let mut settings = self.settings.lock().unwrap();
                settings.bbox_color = value.get().expect("type checked upstream");
            }
            "stroke-width" => {
                let mut settings = self.settings.lock().unwrap();
                settings.stroke_width = value.get().expect("type checked upstream");
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "bbox-color" => {
                let settings = self.settings.lock().unwrap();
                settings.bbox_color.to_value()
            }
            "stroke-width" => {
                let settings = self.settings.lock().unwrap();
                settings.stroke_width.to_value()
            }
            _ => unimplemented!(),
        }
    }

    fn constructed(&self) {
        self.parent_constructed();
    }
}

// Implementation of GstObject virtual methods
impl GstObjectImpl for EdgeImpulseOverlay {}

// Implementation of Element virtual methods
impl ElementImpl for EdgeImpulseOverlay {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Overlay",
                "Filter/Effect/Video",
                "Draws bounding boxes on video frames based on ROI metadata",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("video/x-raw")
                .field(
                    "format",
                    gst::List::new([
                        "RGB", "BGR", "RGBA", "BGRA", "UYVY", "YUY2", "YVYU", "NV12", "NV21",
                        "I420", "YV12",
                    ]),
                )
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
        PAD_TEMPLATES.as_slice()
    }
}

// Implementation of BaseTransform virtual methods
impl BaseTransformImpl for EdgeImpulseOverlay {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::debug!(CAT, obj = self.obj(), "transform_ip called");
        let res = self.parent_transform_ip(buf);
        gst::debug!(
            CAT,
            obj = self.obj(),
            "transform_ip completed with result: {:?}",
            res
        );
        res
    }

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        gst::debug!(CAT, obj = self.obj(), "Starting transform");
        self.parent_start()
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        gst::debug!(CAT, obj = self.obj(), "Stopping transform");
        self.parent_stop()
    }

    fn propose_allocation(
        &self,
        decide_query: Option<&gst::query::Allocation>,
        query: &mut gst::query::Allocation,
    ) -> Result<(), gst::LoggableError> {
        gst::debug!(CAT, obj = self.obj(), "Proposing allocation");
        self.parent_propose_allocation(decide_query, query)
    }
}

// Implementation of element specific methods
impl EdgeImpulseOverlay {
    fn draw_bbox(
        &self,
        frame: &mut VideoFrameRef<&mut gst::BufferRef>,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        roi_type: &str,
    ) -> Result<(), gst::LoggableError> {
        let format;
        {
            let video_info = self.video_info.lock().unwrap();
            let info = video_info
                .as_ref()
                .ok_or_else(|| gst::loggable_error!(CAT, "Video info not available"))?;
            format = info.format();
        }

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Drawing bbox for {} with format: {:?}",
            roi_type,
            format
        );

        let settings = self.settings.lock().unwrap();
        match format {
            VideoFormat::Rgb => {
                gst::debug!(CAT, obj = self.obj(), "Using RGB drawing for {}", roi_type);
                self.draw_bbox_rgb(frame, x, y, w, h, &settings)
            }
            VideoFormat::Nv12 | VideoFormat::Nv21 => {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Using NV12/21 drawing for {}",
                    roi_type
                );
                self.draw_bbox_nv12(frame, x, y, w, h, &settings)
            }
            _ => {
                gst::warning!(CAT, obj = self.obj(), "Unsupported format: {:?}", format);
                Err(gst::loggable_error!(CAT, "Unsupported format"))
            }
        }
    }

    fn draw_bbox_rgb(
        &self,
        frame: &mut VideoFrameRef<&mut gst::BufferRef>,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        settings: &Settings,
    ) -> Result<(), gst::LoggableError> {
        // Get the info we need first, then release the mutex
        let stride;
        let width;
        let height;
        {
            let info = self.video_info.lock().unwrap();
            let info = info
                .as_ref()
                .ok_or_else(|| gst::loggable_error!(CAT, "Video info not available"))?;
            stride = info.stride()[0] as i32;
            width = info.width() as i32;
            height = info.height() as i32;
        }

        // Ensure stroke width is at least 1
        let stroke_width = std::cmp::max(1, settings.stroke_width);

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Starting RGB bbox drawing: stride={}, width={}, height={}, data_len={}",
            stride,
            width,
            height,
            frame.plane_data(0).unwrap().len()
        );
        gst::debug!(
            CAT,
            obj = self.obj(),
            "Drawing bbox at ({}, {}) with size {}x{} and stroke_width={}",
            x,
            y,
            w,
            h,
            stroke_width
        );

        let data = frame.plane_data_mut(0).unwrap();

        let r = ((settings.bbox_color >> 16) & 0xFF) as u8;
        let g = ((settings.bbox_color >> 8) & 0xFF) as u8;
        let b = (settings.bbox_color & 0xFF) as u8;

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Using RGB color: ({}, {}, {})",
            r,
            g,
            b
        );

        let mut pixels_drawn = 0;

        // Draw horizontal lines with specified stroke width
        for i in x..x + w {
            if i < 0 || i >= width {
                continue;
            }

            // Top lines
            for s in 0..stroke_width {
                let y_pos = y + s as i32;
                if y_pos >= 0 && y_pos < height {
                    let idx = (y_pos * stride + i * 3) as usize;
                    if idx + 2 < data.len() {
                        data[idx] = r;
                        data[idx + 1] = g;
                        data[idx + 2] = b;
                        pixels_drawn += 1;
                    }
                }
            }

            // Bottom lines
            for s in 0..stroke_width {
                let y_pos = y + h - s as i32 - 1;
                if y_pos >= 0 && y_pos < height {
                    let idx = (y_pos * stride + i * 3) as usize;
                    if idx + 2 < data.len() {
                        data[idx] = r;
                        data[idx + 1] = g;
                        data[idx + 2] = b;
                        pixels_drawn += 1;
                    }
                }
            }
        }

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Completed horizontal lines ({} pixels), starting vertical",
            pixels_drawn
        );

        // Draw vertical lines with specified stroke width
        for j in y..y + h {
            if j < 0 || j >= height {
                continue;
            }

            // Left lines
            for s in 0..stroke_width {
                let x_pos = x + s as i32;
                if x_pos >= 0 && x_pos < width {
                    let idx = (j * stride + x_pos * 3) as usize;
                    if idx + 2 < data.len() {
                        data[idx] = r;
                        data[idx + 1] = g;
                        data[idx + 2] = b;
                        pixels_drawn += 1;
                    }
                }
            }

            // Right lines
            for s in 0..stroke_width {
                let x_pos = x + w - s as i32 - 1;
                if x_pos >= 0 && x_pos < width {
                    let idx = (j * stride + x_pos * 3) as usize;
                    if idx + 2 < data.len() {
                        data[idx] = r;
                        data[idx + 1] = g;
                        data[idx + 2] = b;
                        pixels_drawn += 1;
                    }
                }
            }
        }

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Completed RGB bbox drawing: {} total pixels drawn",
            pixels_drawn
        );

        if pixels_drawn == 0 {
            gst::warning!(
                CAT,
                obj = self.obj(),
                "No pixels were drawn - check coordinates and bounds"
            );
            return Err(gst::loggable_error!(CAT, "No pixels were drawn"));
        }

        Ok(())
    }

    fn draw_bbox_nv12(
        &self,
        frame: &mut VideoFrameRef<&mut gst::BufferRef>,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        settings: &Settings,
    ) -> Result<(), gst::LoggableError> {
        let info = self.video_info.lock().unwrap();
        let info = info.as_ref().unwrap();

        // Draw rectangle for bounding box
        // For now just draw on Y plane for NV12/NV21
        let y_stride = info.stride()[0] as i32;
        let y_data = frame.plane_data_mut(0).unwrap();

        // Convert RGB color to Y (luminance)
        let r = ((settings.bbox_color >> 16) & 0xFF) as f32;
        let g = ((settings.bbox_color >> 8) & 0xFF) as f32;
        let b = (settings.bbox_color & 0xFF) as f32;
        let y_value = (0.299 * r + 0.587 * g + 0.114 * b) as u8;

        // Draw horizontal lines with specified stroke width
        for i in x..x + w {
            // Top lines
            for s in 0..settings.stroke_width {
                if (y + s as i32) >= 0 && (y + s as i32) < info.height() as i32 {
                    let idx = ((y + s as i32) * y_stride + i) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Bottom lines
            for s in 0..settings.stroke_width {
                if (y + h - s as i32) >= 0 && (y + h - s as i32) < info.height() as i32 {
                    let idx = ((y + h - s as i32) * y_stride + i) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
        }

        // Draw vertical lines with specified stroke width
        for j in y..y + h {
            // Left lines
            for s in 0..settings.stroke_width {
                if j >= 0 && j < info.height() as i32 {
                    let idx = (j * y_stride + x + s as i32) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Right lines
            for s in 0..settings.stroke_width {
                if j >= 0 && j < info.height() as i32 {
                    let idx = (j * y_stride + x + w - s as i32) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
        }

        Ok(())
    }
}

// Implementation of VideoFilter virtual methods
impl VideoFilterImpl for EdgeImpulseOverlay {
    fn transform_frame_ip(
        &self,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::debug!(CAT, obj = self.obj(), "Processing frame");

        let regions: Vec<_> = frame
            .buffer()
            .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .map(|meta| (meta.rect(), meta.roi_type()))
            .collect();

        gst::debug!(CAT, obj = self.obj(), "Found {} regions", regions.len());

        for (rect, roi_type) in regions {
            let (x, y, w, h) = rect;
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Drawing bbox at ({}, {}) size {}x{} type {:?}",
                x,
                y,
                w,
                h,
                roi_type
            );

            if let Err(e) = self.draw_bbox(frame, x as i32, y as i32, w as i32, h as i32, roi_type)
            {
                gst::error!(CAT, obj = self.obj(), "Failed to draw bbox: {}", e);
                return Err(gst::FlowError::Error);
            }
            gst::debug!(CAT, obj = self.obj(), "Successfully drew bbox");
        }

        gst::debug!(CAT, obj = self.obj(), "Frame processing complete");
        Ok(gst::FlowSuccess::Ok)
    }

    fn set_info(
        &self,
        _incaps: &gst::Caps,
        in_info: &gst_video::VideoInfo,
        _outcaps: &gst::Caps,
        _out_info: &gst_video::VideoInfo,
    ) -> Result<(), gst::LoggableError> {
        gst::debug!(CAT, obj = self.obj(), "Setting info");
        let mut video_info = self.video_info.lock().unwrap();
        *video_info = Some(in_info.clone());
        Ok(())
    }
}
