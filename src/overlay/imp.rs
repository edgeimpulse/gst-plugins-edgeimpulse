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
use pangocairo::functions::*;
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
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            stroke_width: 2,               // Default to 2 pixels
            text_color: 0xFFFFFF,          // Default to white
            text_font_size: 14,            // Default font size
            text_font: "Sans".to_string(), // Default font
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
        color: (u8, u8, u8),
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
                self.draw_bbox_rgb(frame, x, y, w, h, &settings, color)
            }
            VideoFormat::Nv12 | VideoFormat::Nv21 => {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Using NV12/21 drawing for {}",
                    roi_type
                );
                self.draw_bbox_nv12(frame, x, y, w, h, &settings, color)
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
        color: (u8, u8, u8),
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
            x,
            y,
            w,
            h,
            stroke_width
        );

        let data = frame.plane_data_mut(0).unwrap();

        let (r, g, b) = color;

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
        color: (u8, u8, u8),
    ) -> Result<(), gst::LoggableError> {
        let info = self.video_info.lock().unwrap();
        let info = info.as_ref().unwrap();

        // Draw rectangle for bounding box
        // For now just draw on Y plane for NV12/NV21
        let y_stride = info.stride()[0];
        let y_data = frame.plane_data_mut(0).unwrap();

        // Convert RGB color to Y (luminance)
        let (r, g, b) = color;
        let r = r as f32;
        let g = g as f32;
        let b = b as f32;
        let y_value = (0.299 * r + 0.587 * g + 0.114 * b) as u8;

        // Draw horizontal lines with specified stroke width
        for i in x..x + w {
            // Top lines
            for s in 0..settings.stroke_width {
                if (y + s) >= 0 && (y + s) < info.height() as i32 {
                    let idx = ((y + s) * y_stride + i) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Bottom lines
            for s in 0..settings.stroke_width {
                if (y + h - s) >= 0 && (y + h - s) < info.height() as i32 {
                    let idx = ((y + h - s) * y_stride + i) as usize;
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
                    let idx = (j * y_stride + x + s) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Right lines
            for s in 0..settings.stroke_width {
                if j >= 0 && j < info.height() as i32 {
                    let idx = (j * y_stride + x + w - s) as usize;
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
        color: (u8, u8, u8),
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

            // Set up Pango layout
            let layout = create_layout(&cr);
            let mut font_desc = pango::FontDescription::new();
            font_desc.set_family(&settings.text_font);
            font_desc.set_absolute_size(settings.text_font_size as f64 * pango::SCALE as f64);
            layout.set_font_description(Some(&font_desc));
            layout.set_text(text);

            // Draw text background for better visibility
            cr.save()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo save failed: {}", e))?;
            let (r, g, b) = color;
            cr.set_source_rgba(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0, 0.6);
            cr.rectangle(x as f64, y as f64, text_width as f64, text_height as f64);
            cr.fill()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo fill failed: {}", e))?;
            cr.restore()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo restore failed: {}", e))?;

            // Set text color
            let r = ((settings.text_color >> 16) & 0xFF) as f64 / 255.0;
            let g = ((settings.text_color >> 8) & 0xFF) as f64 / 255.0;
            let b = (settings.text_color & 0xFF) as f64 / 255.0;
            cr.set_source_rgb(r, g, b);

            // Position and draw text
            cr.move_to(x as f64, y as f64);
            show_layout(&cr, &layout);
        }

        // Copy the modified region back to the frame
        {
            let surface_data = surface.data().unwrap();
            let frame_data = frame.plane_data_mut(0).unwrap();

            let text_y_start = std::cmp::max(0, y);
            let text_y_end = std::cmp::min(height, y + text_height);
            let text_x_start = std::cmp::max(0, x);
            let text_x_end = std::cmp::min(width, x + text_width);

            for y in text_y_start..text_y_end {
                for x in text_x_start..text_x_end {
                    let frame_idx = (y * stride + x * 3) as usize;
                    let surface_idx = (y * surface_stride + x * 4) as usize;

                    if frame_idx + 2 < frame_data.len() && surface_idx + 3 < surface_data.len() {
                        frame_data[frame_idx] = surface_data[surface_idx + 2]; // R
                        frame_data[frame_idx + 1] = surface_data[surface_idx + 1]; // G
                        frame_data[frame_idx + 2] = surface_data[surface_idx]; // B
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

        // Collect all ROI data first
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

        // Get settings once and clone what we need
        let settings_clone = {
            let settings = self.settings.lock().unwrap();
            settings.clone()
        };

        // Now process all collected data
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

            if let Err(e) = self.draw_bbox(
                frame, x as i32, y as i32, w as i32, h as i32, &roi_type, color,
            ) {
                gst::error!(CAT, obj = self.obj(), "Failed to draw bbox: {}", e);
                return Err(gst::FlowError::Error);
            }

            // Format text with label and confidence
            let text = format!("{} {:.1}%", roi_type, confidence * 100.0);

            // Draw text inside the bounding box at the top
            if let Err(e) =
                self.draw_text(frame, &text, x as i32, y as i32 + 4, &settings_clone, color)
            {
                gst::error!(CAT, obj = self.obj(), "Failed to draw text: {}", e);
                return Err(gst::FlowError::Error);
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
