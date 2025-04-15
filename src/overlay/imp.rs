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

use crate::video::{VideoAnomalyMeta, VideoClassificationMeta};

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
pub struct Settings {
    pub stroke_width: i32,
    pub text_color: u32,
    pub font_size: i32,
    pub font_type: String,
    pub text_position: String,
    pub show_labels: bool,
    pub model_input_width: i32,
    pub model_input_height: i32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            stroke_width: 2,
            text_color: 0xFFFFFF,
            font_size: 12,
            font_type: "Sans".to_string(),
            text_position: "top-left".to_string(),
            show_labels: true,
            model_input_width: 160,
            model_input_height: 160,
        }
    }
}

#[derive(Debug)]
struct BBoxParams {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    #[allow(dead_code)]
    roi_type: String,
    color: (u8, u8, u8),
}

struct TextParams {
    text: String,
    x: i32,
    y: i32,
    settings: Settings,
    color: (u8, u8, u8),
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
                    .blurb("Width of the bounding box stroke")
                    .minimum(0)
                    .default_value(2)
                    .build(),
                glib::ParamSpecUInt::builder("text-color")
                    .nick("Text Color")
                    .blurb("Color of the text in RGB format (0xRRGGBB)")
                    .default_value(0xFFFFFF)
                    .build(),
                glib::ParamSpecInt::builder("font-size")
                    .nick("Font Size")
                    .blurb("Size of the font")
                    .minimum(1)
                    .default_value(12)
                    .build(),
                glib::ParamSpecString::builder("font-type")
                    .nick("Font Type")
                    .blurb("Type of font to use")
                    .default_value(Some("Sans"))
                    .build(),
                glib::ParamSpecString::builder("text-position")
                    .nick("Text Position")
                    .blurb("Position of the text")
                    .default_value(Some("top-left"))
                    .build(),
                glib::ParamSpecBoolean::builder("show-labels")
                    .nick("Show Labels")
                    .blurb("Whether to show labels")
                    .default_value(true)
                    .build(),
                glib::ParamSpecInt::builder("model-input-width")
                    .nick("Model Input Width")
                    .blurb("Width of the model input")
                    .minimum(1)
                    .default_value(160)
                    .build(),
                glib::ParamSpecInt::builder("model-input-height")
                    .nick("Model Input Height")
                    .blurb("Height of the model input")
                    .minimum(1)
                    .default_value(160)
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings.lock().unwrap();
        match pspec.name() {
            "stroke-width" => {
                settings.stroke_width = value.get().unwrap();
            }
            "text-color" => {
                settings.text_color = value.get().unwrap();
            }
            "font-size" => {
                settings.font_size = value.get().unwrap();
            }
            "font-type" => {
                settings.font_type = value.get().unwrap();
            }
            "text-position" => {
                settings.text_position = value.get().unwrap();
            }
            "show-labels" => {
                settings.show_labels = value.get().unwrap();
            }
            "model-input-width" => {
                settings.model_input_width = value.get().unwrap();
            }
            "model-input-height" => {
                settings.model_input_height = value.get().unwrap();
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings.lock().unwrap();
        match pspec.name() {
            "stroke-width" => settings.stroke_width.to_value(),
            "text-color" => settings.text_color.to_value(),
            "font-size" => settings.font_size.to_value(),
            "font-type" => settings.font_type.to_value(),
            "text-position" => settings.text_position.to_value(),
            "show-labels" => settings.show_labels.to_value(),
            "model-input-width" => settings.model_input_width.to_value(),
            "model-input-height" => settings.model_input_height.to_value(),
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

        // Get all label colors upfront
        let label_colors = {
            let label_colors = self.label_colors.lock().unwrap();
            label_colors.clone()
        };

        // Get video info upfront
        let video_info = {
            let video_info = self.video_info.lock().unwrap();
            video_info.clone()
        };

        // Debug log the frame info and metadata
        gst::debug!(
            CAT,
            obj = self.obj(),
            "Processing frame: {}x{} format={:?}",
            frame.width(),
            frame.height(),
            frame.format()
        );

        // Log all metadata types present on the buffer
        let mut meta_types = Vec::new();
        frame.buffer().iter_meta::<gst::Meta>().for_each(|meta| {
            meta_types.push(meta.api().name());
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Found metadata type: {}",
                meta.api().name()
            );
        });
        gst::debug!(
            CAT,
            obj = self.obj(),
            "Metadata types present: {:?}",
            meta_types
        );

        // Collect all metadata and data upfront to avoid borrow checker issues
        let classification_data = frame.buffer().meta::<VideoClassificationMeta>().and_then(|meta| {
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Found VideoClassificationMeta with {} params",
                meta.params().len()
            );
            let mut best_result: Option<(String, f64)> = None;
            for param in meta.params() {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Processing classification param: name={}, has_field_label={}, has_field_confidence={}",
                    param.name(),
                    param.has_field("label"),
                    param.has_field("confidence")
                );
                if param.name() != "classification" {
                    continue;
                }

                if let (Ok(label), Ok(confidence)) =
                    (param.get::<String>("label"), param.get::<f64>("confidence"))
                {
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Found classification result: {} ({:.1}%)",
                        label,
                        confidence * 100.0
                    );
                    match best_result {
                        None => best_result = Some((label, confidence)),
                        Some((_, prev_conf)) if confidence > prev_conf => {
                            best_result = Some((label, confidence))
                        }
                        _ => {}
                    }
                }
            }
            best_result
        });

        let anomaly_data = frame.buffer().meta::<VideoAnomalyMeta>().map(|meta| {
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Found VideoAnomalyMeta: anomaly={:.1}%, grid_size={}, frame_size={}x{}",
                meta.anomaly() * 100.0,
                meta.visual_anomaly_grid().len(),
                frame.width(),
                frame.height()
            );
            let anomaly = meta.anomaly();

            let grid: Vec<(i32, i32, i32, i32, f64)> = meta
                .visual_anomaly_grid()
                .iter()
                .map(|roi| {
                    let score = roi.label.parse::<f64>().unwrap_or(0.0);
                    // Scale coordinates based on input and output dimensions
                    let scale_x = frame.width() as f64 / settings.model_input_width as f64;
                    let scale_y = frame.height() as f64 / settings.model_input_height as f64;
                    let scaled_x = (roi.x as f64 * scale_x) as i32;
                    let scaled_y = (roi.y as f64 * scale_y) as i32;
                    let scaled_w = (roi.width as f64 * scale_x) as i32;
                    let scaled_h = (roi.height as f64 * scale_y) as i32;
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Processing grid cell: original=({}, {}) {}x{} -> scaled=({}, {}) {}x{} score={:.1}% (scale_x={:.2}, scale_y={:.2}, model_size={}x{})",
                        roi.x, roi.y, roi.width, roi.height,
                        scaled_x, scaled_y, scaled_w, scaled_h,
                        score * 100.0,
                        scale_x, scale_y,
                        settings.model_input_width, settings.model_input_height
                    );
                    (scaled_x, scaled_y, scaled_w, scaled_h, score)
                })
                .collect();
            (anomaly, grid)
        });

        if anomaly_data.is_none() {
            gst::debug!(CAT, obj = self.obj(), "No VideoAnomalyMeta found in frame");
        }

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

        // First handle classification data
        if let Some((label, confidence)) = classification_data {
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Rendering classification: {} ({:.1}%)",
                label,
                confidence * 100.0
            );

            // Draw classification text
            let text = format!("{} {:.1}%", label, confidence * 100.0);
            let text_x = if settings.text_position == "top-left" || settings.text_position == "bottom-left" {
                10
            } else if settings.text_position == "top-right" || settings.text_position == "bottom-right" {
                frame.width() as i32 - settings.font_size - 10
            } else {
                -1
            };
            let text_y = if settings.text_position == "top-left" || settings.text_position == "top-right" {
                10
            } else if settings.text_position == "bottom-left" || settings.text_position == "bottom-right" {
                frame.height() as i32 - settings.font_size - 10
            } else {
                -1
            };

            // Get or assign color for this label
            let color = if let Some(color) = label_colors.get(&label) {
                *color
            } else {
                let next_color = COLORS.get(label_colors.len() % COLORS.len()).unwrap();
                let mut label_colors = self.label_colors.lock().unwrap();
                label_colors.insert(label.clone(), *next_color);
                *next_color
            };

            if let Err(e) =
                self.draw_text(frame, TextParams {
                    text,
                    x: text_x,
                    y: text_y,
                    settings: settings.clone(),
                    color,
                }, &video_info)
            {
                gst::error!(CAT, obj = self.obj(), "Failed to draw text: {}", e);
                return Err(gst::FlowError::Error);
            }
        }

        // Then handle anomaly detection data
        if let Some((anomaly, grid)) = anomaly_data {
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Processing anomaly grid: {} cells, frame size: {}x{}, anomaly: {:.1}%",
                grid.len(),
                frame.width(),
                frame.height(),
                anomaly * 100.0
            );

            // Draw anomaly score
            let text = format!("Anomaly: {:.1}%", anomaly * 100.0);
            let text_x = if settings.text_position == "top-left" || settings.text_position == "bottom-left" {
                10
            } else if settings.text_position == "top-right" || settings.text_position == "bottom-right" {
                frame.width() as i32 - settings.font_size - 10
            } else {
                -1
            };
            let text_y = if settings.text_position == "top-left" || settings.text_position == "top-right" {
                10
            } else if settings.text_position == "bottom-left" || settings.text_position == "bottom-right" {
                frame.height() as i32 - settings.font_size - 10
            } else {
                -1
            };

            // Use red color for anomaly score
            let color = (255, 0, 0);

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Drawing anomaly score text at ({}, {}): {}",
                text_x,
                text_y,
                text
            );

            if let Err(e) =
                self.draw_text(frame, TextParams {
                    text,
                    x: text_x,
                    y: text_y,
                    settings: settings.clone(),
                    color,
                }, &video_info)
            {
                gst::error!(CAT, obj = self.obj(), "Failed to draw text: {}", e);
                return Err(gst::FlowError::Error);
            }

            // Draw visual anomaly grid
            for (x, y, width, height, score) in grid {
                // Note: coordinates are already scaled from the grid creation
                let normalized_score = (score / 30.0).min(1.0);
                let color = self.get_color_for_score(normalized_score);

                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Processing grid cell: ({}, {}) {}x{} score={:.1}% color={:?} stroke_width={}",
                    x,
                    y,
                    width,
                    height,
                    normalized_score * 100.0,
                    color,
                    settings.stroke_width
                );

                // Draw bounding box if stroke width > 0
                if settings.stroke_width > 0 {
                    let bbox_params = BBoxParams {
                        x,
                        y,
                        width,
                        height,
                        roi_type: format!("anomaly {:.1}%", normalized_score * 100.0),
                        color,
                    };

                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Drawing bounding box: {:?}",
                        bbox_params
                    );

                    if let Err(e) = self.draw_bbox(frame, &bbox_params, &settings, &video_info) {
                        gst::error!(CAT, obj = self.obj(), "Failed to draw box: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                } else {
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Skipping bounding box drawing because stroke_width is 0"
                    );
                }

                // Draw label if enabled
                if settings.show_labels {
                    let text = format!("{:.1}%", normalized_score * 100.0);
                    let text_x = x + 2;
                    let text_y = y + 2;

                    if let Err(e) =
                        self.draw_text(frame, TextParams {
                            text,
                            x: text_x,
                            y: text_y,
                            settings: settings.clone(),
                            color,
                        }, &video_info)
                    {
                        gst::error!(CAT, obj = self.obj(), "Failed to draw text: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                }
            }
        }

        // Finally handle ROI data (object detection)
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
                let color = if let Some(color) = label_colors.get(&label) {
                    *color
                } else {
                    let next_color = COLORS.get(label_colors.len() % COLORS.len()).unwrap();
                    let mut label_colors = self.label_colors.lock().unwrap();
                    label_colors.insert(label.clone(), *next_color);
                    *next_color
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

                    if let Err(e) = self.draw_bbox(frame, &bbox_params, &settings, &video_info) {
                        gst::error!(CAT, obj = self.obj(), "Failed to draw box: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                }

                // Draw label if enabled
                if settings.show_labels {
                    let text = format!("{} {:.1}%", label, confidence * 100.0);
                    let text_x = x + 2;
                    // Position the text just slightly below the top of the bounding box
                    let text_y = y + 2;

                    if let Err(e) =
                        self.draw_text(frame, TextParams {
                            text,
                            x: text_x,
                            y: text_y,
                            settings: settings.clone(),
                            color,
                        }, &video_info)
                    {
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
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
        params: &BBoxParams,
        settings: &Settings,
        video_info: &Option<VideoInfo>,
    ) -> Result<(), gst::LoggableError> {
        // The coordinates are already scaled, no need to scale them again
        let x = params.x;
        let y = params.y;
        let width = params.width;
        let height = params.height;

        gst::debug!(
            CAT,
            "Drawing bbox at ({}, {}) with size {}x{}",
            x,
            y,
            width,
            height,
        );

        // Draw horizontal lines
        for i in 0..settings.stroke_width {
            for j in 0..width {
                let x_pos = x + j;
                let y_pos_top = y + i;
                let y_pos_bottom = y + height - settings.stroke_width + i;

                if x_pos >= 0
                    && x_pos < frame.width() as i32
                    && y_pos_top >= 0
                    && y_pos_top < frame.height() as i32
                {
                    self.set_pixel(frame, x_pos, y_pos_top, params.color, video_info);
                }

                if x_pos >= 0
                    && x_pos < frame.width() as i32
                    && y_pos_bottom >= 0
                    && y_pos_bottom < frame.height() as i32
                {
                    self.set_pixel(frame, x_pos, y_pos_bottom, params.color, video_info);
                }
            }
        }

        // Draw vertical lines
        for i in 0..settings.stroke_width {
            for j in 0..height {
                let x_pos_left = x + i;
                let x_pos_right = x + width - settings.stroke_width + i;
                let y_pos = y + j;

                if x_pos_left >= 0
                    && x_pos_left < frame.width() as i32
                    && y_pos >= 0
                    && y_pos < frame.height() as i32
                {
                    self.set_pixel(frame, x_pos_left, y_pos, params.color, video_info);
                }

                if x_pos_right >= 0
                    && x_pos_right < frame.width() as i32
                    && y_pos >= 0
                    && y_pos < frame.height() as i32
                {
                    self.set_pixel(frame, x_pos_right, y_pos, params.color, video_info);
                }
            }
        }

        Ok(())
    }

    fn draw_text(
        &self,
        frame: &mut VideoFrameRef<&mut gst::BufferRef>,
        params: TextParams,
        video_info: &Option<VideoInfo>,
    ) -> Result<(), gst::LoggableError> {
        // Get all video info upfront
        let (width, height, stride) = match video_info {
            Some(info) => (info.width() as i32, info.height() as i32, info.stride()[0]),
            None => return Err(gst::loggable_error!(CAT, "Video info not available")),
        };

        // Create a temporary surface at 2x resolution for better text quality
        let scale = 2.0; // Use f64 for Cairo scaling
        let surface_width = (width as f64 * scale) as i32;
        let surface_height = (height as f64 * scale) as i32;

        let mut surface =
            cairo::ImageSurface::create(cairo::Format::Rgb24, surface_width, surface_height)
                .map_err(|e| gst::loggable_error!(CAT, "Failed to create Cairo surface: {}", e))?;

        // Variables for copy region and dimensions
        let (copy_y_start, copy_y_end, copy_x_start, copy_x_end);
        let (text_width, text_height, total_width, total_height);

        // Draw text and background
        {
            let cr = cairo::Context::new(&surface)
                .map_err(|e| gst::loggable_error!(CAT, "Failed to create Cairo context: {}", e))?;

            // Scale the context for high resolution rendering
            cr.scale(scale, scale);

            // Create and configure text layout
            let layout = create_layout(&cr);
            let mut font_desc = pango::FontDescription::new();
            font_desc.set_family(&params.settings.font_type);
            // Scale up the font size for high resolution
            font_desc.set_absolute_size(params.settings.font_size as f64 * pango::SCALE as f64);
            if height < 200 {
                font_desc.set_weight(pango::Weight::Bold);
            }
            layout.set_font_description(Some(&font_desc));
            layout.set_text(&params.text);

            // Get the exact text dimensions including any descent
            let (w, h) = layout.pixel_size();
            text_width = w as f64;
            text_height = h as f64;

            // Add padding around text for better visibility
            let bg_padding = 4.0;
            total_width = text_width + (bg_padding * 2.0);
            total_height = text_height + (bg_padding * 2.0);

            // Draw background rectangle at high resolution
            cr.rectangle(params.x as f64, params.y as f64, total_width, total_height);
            cr.set_source_rgba(
                params.color.0 as f64 / 255.0,
                params.color.1 as f64 / 255.0,
                params.color.2 as f64 / 255.0,
                0.7,
            );
            cr.fill()
                .map_err(|e| gst::loggable_error!(CAT, "Cairo fill failed: {}", e))?;

            // Calculate perceived brightness of background color
            let brightness =
                (0.299 * params.color.0 as f64 + 0.587 * params.color.1 as f64 + 0.114 * params.color.2 as f64) / 255.0;

            // Use white text for dark backgrounds, black text for light backgrounds
            if brightness < 0.5 {
                cr.set_source_rgb(1.0, 1.0, 1.0); // White text
            } else {
                cr.set_source_rgb(0.0, 0.0, 0.0); // Black text
            }

            // Position text inside the background box with padding
            cr.move_to(params.x as f64 + bg_padding, params.y as f64 + bg_padding);
            show_layout(&cr, &layout);

            // Calculate the region to copy, ensuring we stay within bounds
            copy_y_start = params.y.max(0) as usize;
            copy_y_end = ((params.y as f64 + total_height) as i32).min(height) as usize;
            copy_x_start = params.x.max(0) as usize;
            copy_x_end = ((params.x as f64 + total_width) as i32).min(width) as usize;
        }

        // Get stride before dropping context
        let surface_stride = surface.stride() as usize;

        // Ensure surface is finished before accessing data
        surface.flush();

        // Get surface data after all drawing is complete
        let surface_data = surface
            .data()
            .map_err(|e| gst::loggable_error!(CAT, "Failed to get surface data: {}", e))?;

        // Get frame data
        let frame_data = frame
            .plane_data_mut(0)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to get frame data"))?;

        // Copy only the affected region from surface to frame, with downscaling
        for y in copy_y_start..copy_y_end {
            let src_y = (y as f64 * scale) as usize;
            let src_offset = src_y * surface_stride + (copy_x_start as f64 * scale) as usize * 4;
            let dst_offset = y * stride as usize + copy_x_start * 3;

            for x in 0..(copy_x_end - copy_x_start) {
                let src_idx = src_offset + (x as f64 * scale) as usize * 4;
                let dst_idx = dst_offset + x * 3;

                // Ensure we don't exceed buffer boundaries
                if dst_idx + 2 < frame_data.len() && src_idx + 2 < surface_data.len() {
                    frame_data[dst_idx] = surface_data[src_idx + 2]; // R
                    frame_data[dst_idx + 1] = surface_data[src_idx + 1]; // G
                    frame_data[dst_idx + 2] = surface_data[src_idx]; // B
                }
            }
        }

        Ok(())
    }

    fn get_color_for_score(&self, score: f64) -> (u8, u8, u8) {
        // Convert score to color gradient from green (0.0) to red (1.0)
        let r = (score * 255.0) as u8;
        let g = ((1.0 - score) * 255.0) as u8;
        let b = 0;
        (r, g, b)
    }

    fn set_pixel(
        &self,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
        x: i32,
        y: i32,
        color: (u8, u8, u8),
        video_info: &Option<VideoInfo>,
    ) {
        let info = match video_info {
            Some(info) => info,
            None => {
                gst::warning!(CAT, "Video info not available");
                return;
            }
        };

        let format = info.format();
        let stride = info.stride()[0];
        let (r, g, b) = color;

        // Get the plane data directly from the frame
        if let Ok(data) = frame.plane_data_mut(0) {
            match format {
                VideoFormat::Rgb => {
                    let idx = (y * stride + x * 3) as usize;
                    if idx + 2 < data.len() {
                        data[idx] = r;
                        data[idx + 1] = g;
                        data[idx + 2] = b;
                    }
                }
                VideoFormat::Nv12 | VideoFormat::Nv21 => {
                    let idx = (y * stride + x) as usize;
                    if idx < data.len() {
                        // Convert RGB to Y (luminance)
                        let y_value =
                            (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                        data[idx] = y_value;
                    }
                }
                _ => {
                    gst::warning!(CAT, "Unsupported format: {:?}", format);
                }
            }
        }
    }
}
