use glib::ParamSpecBuilderExt;
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base as gst_base;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::{VideoFormat, VideoFrameRef, VideoInfo};
use once_cell::sync::Lazy;
use pangocairo::functions::*;
use std::sync::Mutex;

use crate::video::VideoClassificationMeta;

/// A GStreamer element that draws bounding boxes on video frames based on ROI metadata.
///
/// The EdgeImpulseOverlay element is designed to visualize regions of interest (ROIs)
/// in video streams by drawing bounding boxes around them. It processes video frames
/// in RGB format and draws boxes based on VideoRegionOfInterestMeta metadata
/// attached to the video buffers.
///
/// # Features
/// - Draws bounding boxes around regions specified by VideoRegionOfInterestMeta
/// - Automatic color assignment per object class
/// - Configurable stroke width and text properties
/// - Supports RGB and NV12/NV21 video formats
/// - In-place buffer modification for efficient processing
/// - Safe bounds checking for all drawing operations
///
/// # Properties
/// - `stroke-width`: Width of the bounding box lines in pixels (default: 2, minimum: 1)
/// - `text-color`: Color of the text in RGB format (default: white 0xFFFFFF)
/// - `text-font-size`: Size of the text font in pixels (default: 20)
/// - `text-font`: Font family to use for text rendering (default: Sans)
/// - `text-x`: X position for classification text (-1 for right-aligned)
/// - `text-y`: Y position for classification text (-1 for bottom-aligned)
/// - `show-labels`: Whether to draw labels on the video frames (default: true)
///
/// # Color Assignment
/// The element automatically assigns distinct colors to different object classes.
/// Each unique label (roi_type) gets assigned a color from a predefined palette,
/// ensuring consistent coloring across frames for the same object class.
///
/// # Formats
/// - Input: RGB, NV12, NV21
/// - Output: Same as input (in-place modification)
///
/// # Metadata
/// Reads VideoRegionOfInterestMeta from input buffers to determine where to draw boxes.
/// Each ROI meta contains:
/// - x, y: Top-left corner coordinates
/// - width, height: Dimensions of the region
/// - roi_type: Type identifier for the region
/// - confidence: Detection confidence value (0.0 - 1.0)
///
/// # Example Pipeline
/// ```text
/// gst-launch-1.0 \
///   videotestsrc ! \
///   video/x-raw,format=RGB,width=384,height=384 ! \
///   edgeimpulsevideoinfer ! \
///   edgeimpulseoverlay stroke-width=3 ! \
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

static COLORS: Lazy<Vec<(u8, u8, u8)>> = Lazy::new(|| {
    vec![
        (0xE6, 0x19, 0x4B), // '#E6194B'
        (0x3C, 0xB4, 0x4B), // '#3CB44B'
        (0xFF, 0xE1, 0x19), // '#FFE119'
        (0x43, 0x63, 0xD8), // '#4363D8'
        (0xF5, 0x82, 0x31), // '#F58231'
        (0x42, 0xD4, 0xF4), // '#42D4F4'
        (0xF0, 0x32, 0xE6), // '#F032E6'
        (0xFA, 0xBE, 0xD4), // '#FABED4'
        (0x46, 0x99, 0x90), // '#469990'
        (0xDC, 0xBE, 0xFF), // '#DCBEFF'
        (0x9A, 0x63, 0x24), // '#9A6324'
        (0xFF, 0xFA, 0xC8), // '#FFFAC8'
        (0x80, 0x00, 0x00), // '#800000'
        (0xAA, 0xFF, 0xC3), // '#AAFFC3'
    ]
});

// Settings for configuring the overlay appearance
#[derive(Debug, Clone)]
struct Settings {
    stroke_width: i32,
    text_color: u32,
    text_font_size: u32,
    text_font: String,
    text_x: i32, // X position for classification text
    text_y: i32, // Y position for classification text
    show_labels: bool,
}

#[derive(Debug)]
struct BBoxParams {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    roi_type: String,
    color: (u8, u8, u8),
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            stroke_width: 2,               // Default to 2 pixels
            text_color: 0xFFFFFF,          // Default to white
            text_font_size: 14,            // Default font size
            text_font: "Sans".to_string(), // Default font
            text_x: -1,                    // Default to right-aligned (-1 means right)
            text_y: -1,                    // Default to bottom-aligned (-1 means bottom)
            show_labels: true,
        }
    }
}

// State struct to hold element properties and runtime data
#[derive(Default)]
pub struct EdgeImpulseOverlay {
    settings: Mutex<Settings>,
    video_info: Mutex<Option<VideoInfo>>,
    label_colors: Mutex<std::collections::HashMap<String, (u8, u8, u8)>>,
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
                glib::ParamSpecInt::builder("stroke-width")
                    .nick("Stroke Width")
                    .blurb("Width of the bounding box lines in pixels")
                    .minimum(1)
                    .maximum(100)
                    .default_value(2)
                    .build(),
                glib::ParamSpecUInt::builder("text-color")
                    .nick("Text Color")
                    .blurb("Color of the text in RGB format (default: white 0xFFFFFF)")
                    .default_value(0xFFFFFF) // White
                    .build(),
                glib::ParamSpecUInt::builder("text-font-size")
                    .nick("Text Font Size")
                    .blurb("Size of the text font in pixels")
                    .default_value(20)
                    .build(),
                glib::ParamSpecString::builder("text-font")
                    .nick("Text Font")
                    .blurb("Font family to use for text rendering (default: Sans)")
                    .default_value(Some("Sans"))
                    .build(),
                glib::ParamSpecInt::builder("text-x")
                    .nick("Text X Position")
                    .blurb("X position for classification text (-1 for right-aligned)")
                    .minimum(-1)
                    .maximum(10000)
                    .default_value(-1)
                    .build(),
                glib::ParamSpecInt::builder("text-y")
                    .nick("Text Y Position")
                    .blurb("Y position for classification text (-1 for bottom-aligned)")
                    .minimum(-1)
                    .maximum(10000)
                    .default_value(-1)
                    .build(),
                glib::ParamSpecBoolean::builder("show-labels")
                    .nick("Show Labels")
                    .blurb("Whether to draw labels on the video frames")
                    .default_value(true)
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "stroke-width" => {
                let mut settings = self.settings.lock().unwrap();
                settings.stroke_width = value.get().expect("type checked upstream");
            }
            "text-color" => {
                let mut settings = self.settings.lock().unwrap();
                settings.text_color = value.get().expect("type checked upstream");
            }
            "text-font-size" => {
                let mut settings = self.settings.lock().unwrap();
                settings.text_font_size = value.get().expect("type checked upstream");
            }
            "text-font" => {
                let mut settings = self.settings.lock().unwrap();
                settings.text_font = value.get().expect("type checked upstream");
            }
            "text-x" => {
                let mut settings = self.settings.lock().unwrap();
                settings.text_x = value.get().expect("type checked upstream");
            }
            "text-y" => {
                let mut settings = self.settings.lock().unwrap();
                settings.text_y = value.get().expect("type checked upstream");
            }
            "show-labels" => {
                let mut settings = self.settings.lock().unwrap();
                settings.show_labels = value.get().expect("type checked upstream");
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "stroke-width" => {
                let settings = self.settings.lock().unwrap();
                settings.stroke_width.to_value()
            }
            "text-color" => {
                let settings = self.settings.lock().unwrap();
                settings.text_color.to_value()
            }
            "text-font-size" => {
                let settings = self.settings.lock().unwrap();
                settings.text_font_size.to_value()
            }
            "text-font" => {
                let settings = self.settings.lock().unwrap();
                settings.text_font.to_value()
            }
            "text-x" => {
                let settings = self.settings.lock().unwrap();
                settings.text_x.to_value()
            }
            "text-y" => {
                let settings = self.settings.lock().unwrap();
                settings.text_y.to_value()
            }
            "show-labels" => {
                let settings = self.settings.lock().unwrap();
                settings.show_labels.to_value()
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
        params: &BBoxParams,
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
            params.roi_type,
            format
        );

        let settings = self.settings.lock().unwrap();
        match format {
            VideoFormat::Rgb => {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Using RGB drawing for {}",
                    params.roi_type
                );
                self.draw_bbox_rgb(frame, params, &settings)
            }
            VideoFormat::Nv12 | VideoFormat::Nv21 => {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Using NV12/21 drawing for {}",
                    params.roi_type
                );
                self.draw_bbox_nv12(frame, params, &settings)
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
        params: &BBoxParams,
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
            stride = info.stride()[0];
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
            params.x,
            params.y,
            params.width,
            params.height,
            stroke_width
        );

        let data = frame.plane_data_mut(0).unwrap();

        let (r, g, b) = params.color;

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
        for i in params.x..params.x + params.width {
            if i < 0 || i >= width {
                continue;
            }

            // Top lines
            for s in 0..stroke_width {
                let y_pos = params.y + s;
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
                let y_pos = params.y + params.height - s - 1;
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
        for j in params.y..params.y + params.height {
            if j < 0 || j >= height {
                continue;
            }

            // Left lines
            for s in 0..stroke_width {
                let x_pos = params.x + s;
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
                let x_pos = params.x + params.width - s - 1;
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
        params: &BBoxParams,
        settings: &Settings,
    ) -> Result<(), gst::LoggableError> {
        let info = self.video_info.lock().unwrap();
        let info = info.as_ref().unwrap();

        // Draw rectangle for bounding box
        // For now just draw on Y plane for NV12/NV21
        let y_stride = info.stride()[0];
        let y_data = frame.plane_data_mut(0).unwrap();

        // Convert RGB color to Y (luminance)
        let (r, g, b) = params.color;
        let r = r as f32;
        let g = g as f32;
        let b = b as f32;
        let y_value = (0.299 * r + 0.587 * g + 0.114 * b) as u8;

        // Draw horizontal lines with specified stroke width
        for i in params.x..params.x + params.width {
            // Top lines
            for s in 0..settings.stroke_width {
                if (params.y + s) >= 0 && (params.y + s) < info.height() as i32 {
                    let idx = ((params.y + s) * y_stride + i) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Bottom lines
            for s in 0..settings.stroke_width {
                if (params.y + params.height - s) >= 0
                    && (params.y + params.height - s) < info.height() as i32
                {
                    let idx = ((params.y + params.height - s) * y_stride + i) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
        }

        // Draw vertical lines with specified stroke width
        for j in params.y..params.y + params.height {
            // Left lines
            for s in 0..settings.stroke_width {
                if j >= 0 && j < info.height() as i32 {
                    let idx = (j * y_stride + params.x + s) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Right lines
            for s in 0..settings.stroke_width {
                if j >= 0 && j < info.height() as i32 {
                    let idx = (j * y_stride + params.x + params.width - s) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
        }

        Ok(())
    }

    fn draw_text(
        &self,
        frame: &mut VideoFrameRef<&mut gst::BufferRef>,
        text: &str,
        x: i32,
        y: i32,
        settings: &Settings,
        _color: (u8, u8, u8),
    ) -> Result<(), gst::LoggableError> {
        let (width, height, stride) = {
            let video_info = self.video_info.lock().unwrap();
            let info = video_info
                .as_ref()
                .ok_or_else(|| gst::loggable_error!(CAT, "Video info not available"))?;
            (info.width() as i32, info.height() as i32, info.stride()[0])
        };

        // Create a temporary surface with proper stride alignment
        let mut surface = cairo::ImageSurface::create(cairo::Format::Rgb24, width, height)
            .map_err(|e| gst::loggable_error!(CAT, "Failed to create Cairo surface: {}", e))?;

        let surface_stride = surface.stride();

        // Copy frame data to the temporary surface
        {
            let frame_data = frame.plane_data(0).unwrap();
            let mut surface_data = surface.data().unwrap();

            for y in 0..height {
                for x in 0..width {
                    let frame_idx = (y * stride + x * 3) as usize;
                    let surface_idx = (y * surface_stride + x * 4) as usize;

                    if frame_idx + 2 < frame_data.len() && surface_idx + 3 < surface_data.len() {
                        surface_data[surface_idx] = frame_data[frame_idx + 2]; // B
                        surface_data[surface_idx + 1] = frame_data[frame_idx + 1]; // G
                        surface_data[surface_idx + 2] = frame_data[frame_idx]; // R
                        surface_data[surface_idx + 3] = 255; // A
                    }
                }
            }
        }

        // Get text dimensions first
        let (text_width, text_height) = {
            let cr = cairo::Context::new(&surface)
                .map_err(|e| gst::loggable_error!(CAT, "Failed to create Cairo context: {}", e))?;
            let layout = create_layout(&cr);
            let mut font_desc = pango::FontDescription::new();
            font_desc.set_family(&settings.text_font);
            font_desc.set_absolute_size(settings.text_font_size as f64 * pango::SCALE as f64);
            layout.set_font_description(Some(&font_desc));
            layout.set_text(text);
            layout.pixel_size()
        };

        // Create Cairo context and draw text
        {
            let cr = cairo::Context::new(&surface)
                .map_err(|e| gst::loggable_error!(CAT, "Failed to create Cairo context: {}", e))?;

            // Create Pango context and layout
            let pango_context = pango::Context::new();
            let layout = pango::Layout::new(&pango_context);
            layout.set_text(text);
            layout.set_font_description(Some(&pango::FontDescription::from_string(
                &settings.text_font,
            )));
            let font_size = settings.text_font_size as i32 * pango::SCALE;
            let attr_list = pango::AttrList::new();
            let size_attr = pango::AttrSize::new(font_size);
            attr_list.insert(size_attr);
            layout.set_attributes(Some(&attr_list));

            let (text_width, text_height) = layout.pixel_size();
            let text_width = text_width as i32;
            let text_height = text_height as i32;

            // Draw text background
            cr.save()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo save failed: {}", e))?;
            cr.set_source_rgba(0.0, 0.0, 0.0, 0.5);
            cr.rectangle(
                x as f64 - 5.0,
                y as f64 - text_height as f64 - 5.0,
                text_width as f64 + 10.0,
                text_height as f64 + 10.0,
            );
            cr.fill()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo fill failed: {}", e))?;
            cr.restore()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo restore failed: {}", e))?;

            // Draw text
            cr.save()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo save failed: {}", e))?;
            cr.set_source_rgba(
                ((settings.text_color >> 16) & 0xFF) as f64 / 255.0,
                ((settings.text_color >> 8) & 0xFF) as f64 / 255.0,
                (settings.text_color & 0xFF) as f64 / 255.0,
                1.0,
            );
            cr.move_to(x as f64, y as f64 - text_height as f64);
            show_layout(&cr, &layout);
            cr.restore()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo restore failed: {}", e))?;
        }

        // Copy the rendered text to the video frame
        let stride = surface.stride() as usize;
        let surface_data = match surface.data() {
            Ok(data) => data,
            Err(e) => {
                gst::error!(CAT, obj = self.obj(), "Failed to get surface data: {}", e);
                return Err(gst::loggable_error!(
                    CAT,
                    "Failed to get surface data: {}",
                    e
                ));
            }
        };

        let comp_stride = frame.comp_stride(0) as usize;
        let frame_data = match frame.plane_data_mut(0) {
            Ok(data) => data,
            Err(e) => {
                gst::error!(CAT, obj = self.obj(), "Failed to get frame data: {}", e);
                return Err(gst::loggable_error!(CAT, "Failed to get frame data: {}", e));
            }
        };

        // Copy only the region where we drew the text
        let y_start = (y - text_height - 5).max(0) as usize;
        let y_end = (y + 5).min(height) as usize;
        let x_start = (x - 5).max(0) as usize;
        let x_end = (x + text_width + 5).min(width) as usize;

        for y in y_start..y_end {
            let src_offset = y * stride + x_start * 4;
            let dst_offset = y * comp_stride + x_start * 4;
            frame_data[dst_offset..dst_offset + (x_end - x_start) * 4]
                .copy_from_slice(&surface_data[src_offset..src_offset + (x_end - x_start) * 4]);
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

        // Get video dimensions
        let (width, height) = {
            let video_info = self.video_info.lock().unwrap();
            let info = video_info.as_ref().ok_or_else(|| {
                gst::error!(CAT, obj = self.obj(), "Video info not available");
                gst::FlowError::Error
            })?;
            (info.width() as i32, info.height() as i32)
        };

        // Get settings
        let settings = self.settings.lock().unwrap();

        // Check for classification metadata
        if let Some(meta) = frame.buffer().meta::<VideoClassificationMeta>() {
            // Find the highest confidence classification
            if let Some((label, confidence)) = meta
                .params()
                .iter()
                .filter_map(|param| {
                    Some((
                        param.get::<String>("label").ok()?,
                        param.get::<f64>("confidence").ok()?,
                    ))
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                // Draw the classification text
                let text = format!("{} ({:.1}%)", label, confidence * 100.0);
                let text_x = if settings.text_x >= 0 {
                    settings.text_x
                } else {
                    width - 10 // Right-aligned
                };
                let text_y = if settings.text_y >= 0 {
                    settings.text_y
                } else {
                    height - 10 // Bottom-aligned
                };

                // Create a temporary surface for text rendering
                let mut surface =
                    cairo::ImageSurface::create(cairo::Format::Rgb24, width as i32, height as i32)
                        .map_err(|e| {
                            gst::error!(
                                CAT,
                                obj = self.obj(),
                                "Failed to create Cairo surface: {}",
                                e
                            );
                            gst::FlowError::Error
                        })?;

                let context = cairo::Context::new(&surface).map_err(|e| {
                    gst::error!(
                        CAT,
                        obj = self.obj(),
                        "Failed to create Cairo context: {}",
                        e
                    );
                    gst::FlowError::Error
                })?;

                // Create Pango context and layout
                let pango_context = pango::Context::new();
                let layout = pango::Layout::new(&pango_context);
                layout.set_text(&text);
                layout.set_font_description(Some(&pango::FontDescription::from_string(
                    &settings.text_font,
                )));
                let font_size = settings.text_font_size as i32 * pango::SCALE;
                let attr_list = pango::AttrList::new();
                let size_attr = pango::AttrSize::new(font_size);
                attr_list.insert(size_attr);
                layout.set_attributes(Some(&attr_list));

                let (text_width, text_height) = layout.pixel_size();
                let text_width = text_width as i32;
                let text_height = text_height as i32;

                // Draw text background
                context.save().map_err(|e| {
                    gst::error!(CAT, "Cairo save failed: {}", e);
                    gst::FlowError::Error
                })?;
                context.set_source_rgba(0.0, 0.0, 0.0, 0.5);
                context.rectangle(
                    text_x as f64 - 5.0,
                    text_y as f64 - text_height as f64 - 5.0,
                    text_width as f64 + 10.0,
                    text_height as f64 + 10.0,
                );
                context.fill().map_err(|e| {
                    gst::error!(CAT, "Cairo fill failed: {}", e);
                    gst::FlowError::Error
                })?;
                context.restore().map_err(|e| {
                    gst::error!(CAT, "Cairo restore failed: {}", e);
                    gst::FlowError::Error
                })?;

                // Draw text
                context.save().map_err(|e| {
                    gst::error!(CAT, "Cairo save failed: {}", e);
                    gst::FlowError::Error
                })?;
                context.set_source_rgba(
                    ((settings.text_color >> 16) & 0xFF) as f64 / 255.0,
                    ((settings.text_color >> 8) & 0xFF) as f64 / 255.0,
                    (settings.text_color & 0xFF) as f64 / 255.0,
                    1.0,
                );
                context.move_to(text_x as f64, text_y as f64 - text_height as f64);
                show_layout(&context, &layout);
                context.restore().map_err(|e| {
                    gst::error!(CAT, "Cairo restore failed: {}", e);
                    gst::FlowError::Error
                })?;

                // Copy the rendered text to the video frame
                let stride = surface.stride() as usize;
                let surface_data = match surface.data() {
                    Ok(data) => data,
                    Err(e) => {
                        gst::error!(CAT, obj = self.obj(), "Failed to get surface data: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                };

                let comp_stride = frame.comp_stride(0) as usize;
                let frame_data = match frame.plane_data_mut(0) {
                    Ok(data) => data,
                    Err(e) => {
                        gst::error!(CAT, obj = self.obj(), "Failed to get frame data: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                };

                // Copy only the region where we drew the text
                let y_start = (text_y - text_height - 5).max(0) as usize;
                let y_end = (text_y + 5).min(height as i32) as usize;
                let x_start = (text_x - 5).max(0) as usize;
                let x_end = (text_x + text_width + 5).min(width as i32) as usize;

                for y in y_start..y_end {
                    let src_offset = y * stride + x_start * 4;
                    let dst_offset = y * comp_stride + x_start * 4;
                    frame_data[dst_offset..dst_offset + (x_end - x_start) * 4].copy_from_slice(
                        &surface_data[src_offset..src_offset + (x_end - x_start) * 4],
                    );
                }
            }
        }

        // Process object detection ROIs
        let roi_data: Vec<_> = frame
            .buffer()
            .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .map(|meta| {
                let rect = meta.rect();
                let roi_type = meta.roi_type().to_string();
                let confidence = meta
                    .param("ObjectDetection")
                    .and_then(|v| v.get::<f64>("confidence").ok())
                    .unwrap_or(0.0) as f32;
                (rect, roi_type, confidence)
            })
            .collect();

        gst::debug!(CAT, obj = self.obj(), "Found {} regions", roi_data.len());

        // Process object detection ROIs
        for ((x, y, w, h), roi_type, confidence) in roi_data {
            // Get or assign color for this label
            let color = {
                let mut label_colors = self.label_colors.lock().unwrap();
                if !label_colors.contains_key(&roi_type) {
                    let next_color = COLORS.get(label_colors.len() % COLORS.len()).unwrap();
                    label_colors.insert(roi_type.clone(), *next_color);
                }
                *label_colors.get(&roi_type).unwrap()
            };

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Drawing bbox at ({}, {}) size {}x{} type {:?} with color {:?}",
                x,
                y,
                w,
                h,
                roi_type,
                color
            );

            let params = BBoxParams {
                x: x as i32,
                y: y as i32,
                width: w as i32,
                height: h as i32,
                roi_type: roi_type.clone(),
                color,
            };

            if let Err(e) = self.draw_bbox(frame, &params) {
                gst::error!(CAT, obj = self.obj(), "Failed to draw bbox: {}", e);
                return Err(gst::FlowError::Error);
            }

            // Format text with label and confidence
            let text = format!("{} {:.1}%", roi_type, confidence * 100.0);

            // Draw text if enabled
            if settings.show_labels {
                match self.draw_text(frame, &text, x as i32, y as i32 + 4, &settings, color) {
                    Ok(_) => {}
                    Err(e) => {
                        gst::error!(CAT, obj = self.obj(), "Failed to draw text: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                }
            }
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
