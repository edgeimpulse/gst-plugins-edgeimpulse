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

#[derive(Debug, Clone)]
struct Settings {
    stroke_width: i32,
    text_color: u32,
    text_font_size: u32,
    text_font: String,
    text_x: i32,
    text_y: i32,
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
            stroke_width: 2,
            text_color: 0xFFFFFF,
            text_font_size: 14,
            text_font: "Sans".to_string(),
            text_x: -1,
            text_y: -1,
            show_labels: true,
        }
    }
}

#[derive(Default)]
pub struct EdgeImpulseOverlay {
    settings: Mutex<Settings>,
    video_info: Mutex<Option<VideoInfo>>,
    label_colors: Mutex<std::collections::HashMap<String, (u8, u8, u8)>>,
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseOverlay {
    const NAME: &'static str = "EdgeImpulseOverlay";
    type Type = super::EdgeImpulseOverlay;
    type ParentType = gst_video::VideoFilter;
}

// Implementation of GObject virtual methods
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
                    .default_value(0xFFFFFF)
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
}

impl GstObjectImpl for EdgeImpulseOverlay {}

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

impl BaseTransformImpl for EdgeImpulseOverlay {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        self.parent_transform_ip(buf)
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

impl VideoFilterImpl for EdgeImpulseOverlay {
    fn transform_frame_ip(
        &self,
        frame: &mut VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Get all settings upfront and clone what we need
        let settings = {
            let settings = self.settings.lock().unwrap();
            settings.clone()
        };

        // Debug log the frame info
        gst::debug!(
            CAT,
            obj = self.obj(),
            "Processing frame: {}x{} format={:?}",
            frame.width(),
            frame.height(),
            frame.format()
        );

        // First check for classification metadata
        let classification_meta = frame.buffer().meta::<VideoClassificationMeta>();
        if let Some(meta) = classification_meta {
            // Get all parameters and find the one with highest confidence
            let mut best_result: Option<(String, f64)> = None;
            for param in meta.params() {
                if param.name() != "Classification" {
                    continue;
                }

                if let (Ok(label), Ok(confidence)) =
                    (param.get::<String>("label"), param.get::<f64>("confidence"))
                {
                    match best_result {
                        None => best_result = Some((label, confidence)),
                        Some((_, prev_conf)) if confidence > prev_conf => {
                            best_result = Some((label, confidence))
                        }
                        _ => {}
                    }
                }
            }

            // If we found a classification result, render it
            if let Some((label, confidence)) = best_result {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Rendering classification: {} ({:.1}%)",
                    label,
                    confidence * 100.0
                );

                // Draw classification text
                let text = format!("{} {:.1}%", label, confidence * 100.0);
                let text_x = if settings.text_x < 0 {
                    10
                } else {
                    settings.text_x
                };
                let text_y = if settings.text_y < 0 {
                    frame.height() as i32 - settings.text_font_size as i32 - 10
                } else {
                    settings.text_y
                };

                // Get or assign color for this label
                let color = {
                    let mut label_colors = self.label_colors.lock().unwrap();
                    if !label_colors.contains_key(&label as &str) {
                        let next_color = COLORS.get(label_colors.len() % COLORS.len()).unwrap();
                        label_colors.insert(label.clone(), *next_color);
                    }
                    *label_colors.get(&label as &str).unwrap()
                };

                if let Err(e) = self.draw_text(frame, &text, text_x, text_y, &settings, color) {
                    gst::error!(CAT, obj = self.obj(), "Failed to draw text: {}", e);
                    return Err(gst::FlowError::Error);
                }
            }
        }

        // Then check for ROI metadata (object detection)
        let rois: Vec<_> = frame
            .buffer()
            .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .filter_map(|roi| {
                let meta = unsafe { &*roi.as_ptr() };
                let x = meta.x as i32;
                let y = meta.y as i32;
                let width = meta.w as i32;
                let height = meta.h as i32;

                // Get label and confidence from parameters
                let params_result = roi.params().find_map(|param| {
                    if param.name() != "detection" {
                        return None;
                    }

                    let label = param.get::<String>("label").ok()?;
                    let confidence = param.get::<f64>("confidence").ok()?;
                    Some((label, confidence))
                });

                params_result.map(|(label, confidence)| (x, y, width, height, label, confidence))
            })
            .collect();

        if !rois.is_empty() {
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Rendering {} regions of interest",
                rois.len()
            );

            // Process each ROI
            for (x, y, width, height, label, confidence) in rois {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Rendering ROI: {} ({:.1}%) at ({}, {}) {}x{}",
                    label,
                    confidence * 100.0,
                    x,
                    y,
                    width,
                    height
                );

                // Get or assign color for this label
                let color = {
                    let mut label_colors = self.label_colors.lock().unwrap();
                    if !label_colors.contains_key(&label) {
                        let next_color = COLORS.get(label_colors.len() % COLORS.len()).unwrap();
                        label_colors.insert(label.clone(), *next_color);
                    }
                    *label_colors.get(&label).unwrap()
                };

                // Draw bounding box if stroke width > 0
                if settings.stroke_width > 0 {
                    let bbox_params = BBoxParams {
                        x,
                        y,
                        width,
                        height,
                        roi_type: label.clone(),
                        color,
                    };

                    if let Err(e) = self.draw_bbox(frame, &bbox_params) {
                        gst::error!(CAT, obj = self.obj(), "Failed to draw box: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                }

                // Draw label if enabled
                if settings.show_labels {
                    let text = format!("{} {:.1}%", label, confidence * 100.0);
                    let text_x = x + 2;
                    let text_y = y + settings.text_font_size as i32 + 2;

                    if let Err(e) = self.draw_text(frame, &text, text_x, text_y, &settings, color) {
                        gst::error!(CAT, obj = self.obj(), "Failed to draw text: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                }
            }
        }

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

// Implementation of element specific methods
impl EdgeImpulseOverlay {
    fn draw_bbox(
        &self,
        frame: &mut VideoFrameRef<&mut gst::BufferRef>,
        params: &BBoxParams,
    ) -> Result<(), gst::LoggableError> {
        // Get all video info upfront
        let (format, stride, width, height) = {
            let video_info = self.video_info.lock().unwrap();
            let info = video_info
                .as_ref()
                .ok_or_else(|| gst::loggable_error!(CAT, "Video info not available"))?;
            (
                info.format(),
                info.stride()[0],
                info.width() as i32,
                info.height() as i32,
            )
        };

        // Get settings upfront
        let stroke_width = {
            let settings = self.settings.lock().unwrap();
            std::cmp::max(1, settings.stroke_width)
        };

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Drawing bbox for {} with format: {:?}",
            params.roi_type,
            format
        );

        match format {
            VideoFormat::Rgb => {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Using RGB drawing for {}",
                    params.roi_type
                );
                self.draw_bbox_rgb(frame, params, stroke_width, stride, width, height)
            }
            VideoFormat::Nv12 | VideoFormat::Nv21 => {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Using NV12/21 drawing for {}",
                    params.roi_type
                );
                self.draw_bbox_nv12(frame, params, stroke_width, stride, height)
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
        stroke_width: i32,
        stride: i32,
        width: i32,
        height: i32,
    ) -> Result<(), gst::LoggableError> {
        gst::debug!(
            CAT,
            obj = self.obj(),
            "Starting RGB bbox drawing: stride={}, width={}, height={}, data_len={}",
            stride,
            width,
            height,
            frame.plane_data(0).unwrap().len()
        );

        let data = frame.plane_data_mut(0).unwrap();
        let (r, g, b) = params.color;

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
        stroke_width: i32,
        stride: i32,
        height: i32,
    ) -> Result<(), gst::LoggableError> {
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
            for s in 0..stroke_width {
                if (params.y + s) >= 0 && (params.y + s) < height {
                    let idx = ((params.y + s) * stride + i) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Bottom lines
            for s in 0..stroke_width {
                if (params.y + params.height - s) >= 0 && (params.y + params.height - s) < height {
                    let idx = ((params.y + params.height - s) * stride + i) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
        }

        // Draw vertical lines with specified stroke width
        for j in params.y..params.y + params.height {
            // Left lines
            for s in 0..stroke_width {
                if j >= 0 && j < height {
                    let idx = (j * stride + params.x + s) as usize;
                    if idx < y_data.len() {
                        y_data[idx] = y_value;
                    }
                }
            }
            // Right lines
            for s in 0..stroke_width {
                if j >= 0 && j < height {
                    let idx = (j * stride + params.x + params.width - s) as usize;
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
        // Get all video info upfront
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

        // Create Cairo context and get text dimensions
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

        // Draw text and background
        {
            let cr = cairo::Context::new(&surface)
                .map_err(|e| gst::loggable_error!(CAT, "Failed to create Cairo context: {}", e))?;

            // Draw text background using the bounding box color with some transparency
            cr.save()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo save failed: {}", e))?;
            cr.set_source_rgba(
                color.0 as f64 / 255.0,
                color.1 as f64 / 255.0,
                color.2 as f64 / 255.0,
                0.7,
            );
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

            // Draw text in white or black depending on background color brightness
            cr.save()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo save failed: {}", e))?;

            // Calculate perceived brightness of background color
            let brightness =
                (0.299 * color.0 as f64 + 0.587 * color.1 as f64 + 0.114 * color.2 as f64) / 255.0;

            // Use white text for dark backgrounds, black text for light backgrounds
            if brightness < 0.5 {
                cr.set_source_rgb(1.0, 1.0, 1.0); // White text
            } else {
                cr.set_source_rgb(0.0, 0.0, 0.0); // Black text
            }

            cr.move_to(x as f64, y as f64 - text_height as f64);

            let layout = create_layout(&cr);
            let mut font_desc = pango::FontDescription::new();
            font_desc.set_family(&settings.text_font);
            font_desc.set_absolute_size(settings.text_font_size as f64 * pango::SCALE as f64);
            layout.set_font_description(Some(&font_desc));
            layout.set_text(text);

            show_layout(&cr, &layout);
            cr.restore()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo restore failed: {}", e))?;
        }

        // Ensure surface is finished before accessing data
        surface.flush();

        // Calculate bounds for the region we need to copy
        let y_start = (y - text_height - 5).max(0) as usize;
        let y_end = (y + 5).min(height) as usize;
        let x_start = (x - 5).max(0) as usize;
        let x_end = (x + text_width + 5).min(width) as usize;

        let surface_stride = surface.stride() as usize;

        // Get surface data after all drawing is complete
        let surface_data = surface
            .data()
            .map_err(|e| gst::loggable_error!(CAT, "Failed to get surface data: {}", e))?;

        // Get frame data
        let frame_data = frame
            .plane_data_mut(0)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to get frame data"))?;

        // Copy the rendered text to the video frame with bounds checking
        for y in y_start..y_end {
            let src_offset = y * surface_stride + x_start * 4;
            let dst_offset = y * stride as usize + x_start * 3;

            for x in 0..(x_end - x_start) {
                let src_idx = src_offset + x * 4;
                let dst_idx = dst_offset + x * 3;

                if dst_idx + 2 < frame_data.len() && src_idx + 2 < surface_data.len() {
                    frame_data[dst_idx] = surface_data[src_idx + 2]; // R
                    frame_data[dst_idx + 1] = surface_data[src_idx + 1]; // G
                    frame_data[dst_idx + 2] = surface_data[src_idx]; // B
                }
            }
        }

        Ok(())
    }
}
