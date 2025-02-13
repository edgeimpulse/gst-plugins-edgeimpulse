//! Video inference element implementation for Edge Impulse models
//!
//! This module implements a GStreamer BaseTransform element that performs
//! machine learning inference on video frames using Edge Impulse models.
//! The element operates in two modes:
//!
//! 1. Pass-through mode: When no model is loaded, frames are passed through unchanged
//! 2. Inference mode: When a model is loaded, each frame is processed for inference
//!
//! The element maintains the original video stream while performing inference,
//! making it suitable for real-time video processing applications.
//!
//! # Pipeline Example
//! ```bash
//! gst-launch-1.0 v4l2src ! videoconvert ! \
//!     edgeimpulsevideoinfer model-path=/path/to/model.eim ! \
//!     videoconvert ! autovideosink
//! ```

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use gstreamer_video::{VideoFormat, VideoFrameRef, VideoInfo};
use std::sync::Mutex;
use once_cell::sync::Lazy;
use serde_json;

use crate::common::State;

/// Debug category for the video inference element
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulsevideoinfer",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Video Inference Element"),
    )
});

/// Video inference element structure
///
/// This element processes video frames through an Edge Impulse model.
/// It operates as a BaseTransform element, which means it processes
/// input buffers one at a time and produces corresponding output buffers.
#[derive(Default)]
pub struct EdgeImpulseVideoInfer {
    /// Shared state protected by a mutex for thread-safe access
    state: Mutex<State>,
}

// This macro implements ObjectSubclassType and other required traits
#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseVideoInfer {
    const NAME: &'static str = "EdgeImpulseVideoInfer";
    type Type = super::EdgeImpulseVideoInfer;
    type ParentType = gstreamer_base::BaseTransform;
}

impl ObjectImpl for EdgeImpulseVideoInfer {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            crate::common::create_common_properties()
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "model-path" => {
                let mut state = self.state.lock().unwrap();
                let model_path: Option<String> = value.get().expect("type checked upstream");

                // Initialize the model when the path is set
                if let Some(model_path) = model_path {
                    match edge_impulse_runner::EimModel::new(&model_path) {
                        Ok(model) => {
                            gst::debug!(CAT, obj = self.obj(), "Successfully loaded model from {}", model_path);
                            state.model = Some(model);
                        }
                        Err(err) => {
                            gst::error!(CAT, obj = self.obj(), "Failed to load model: {}", err);
                        }
                    }
                }
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "model-path" => {
                let state = self.state.lock().unwrap();
                if let Some(ref model) = state.model {
                    model.path().to_value()
                } else {
                    None::<String>.to_value()
                }
            }
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for EdgeImpulseVideoInfer {}

impl ElementImpl for EdgeImpulseVideoInfer {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Video Inference",
                "Filter/Video/AI",
                "Runs video inference on Edge Impulse models (EIM)",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::new(1, i32::MAX))
                .field("height", gst::IntRange::new(1, i32::MAX))
                .build();

            vec![
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                ).unwrap(),
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                ).unwrap(),
            ]
        });
        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for EdgeImpulseVideoInfer {
    /// Configure transform to never operate in-place
    const MODE: BaseTransformMode = BaseTransformMode::NeverInPlace;
    /// Allow pass-through when caps are unchanged
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    /// Don't transform in-place even in passthrough mode
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    /// Process a single video frame
    ///
    /// This function is called for each frame in the video stream. It either:
    /// 1. Passes the frame through unchanged if no model is loaded
    /// 2. Performs inference on the frame if a model is loaded
    ///
    /// The original frame is always copied to the output to maintain the video
    /// stream, while inference results can be emitted as messages or signals.
    fn transform(
        &self,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::debug!(CAT, imp = self, "Transform called!");

        // Map the input buffer for reading
        let in_map = inbuf.map_readable().map_err(|_| {
            gst::error!(CAT, imp = self, "Failed to map input buffer readable");
            gst::FlowError::Error
        })?;

        // Map the output buffer for writing
        let mut out_map = outbuf.map_writable().map_err(|_| {
            gst::error!(CAT, imp = self, "Failed to map output buffer writable");
            gst::FlowError::Error
        })?;

        // Copy the input buffer to the output buffer
        out_map.copy_from_slice(&in_map);

        // Drop the mappings before doing inference
        drop(out_map);
        drop(in_map);

        // Run inference on the input buffer
        let mut state = self.state.lock().unwrap();
        if let Some(ref mut model) = state.model {
            // Get all parameters upfront
            let params = match model.parameters() {
                Ok(p) => p,
                Err(e) => {
                    gst::error!(CAT, obj = self.obj(), "Failed to get model parameters: {}", e);
                    return Err(gst::FlowError::Error);
                }
            };

            let width = params.image_input_width;
            let height = params.image_input_height;
            let channels = params.image_channel_count;
            let is_object_detection = params.model_type == "constrained_object_detection";

            gst::debug!(CAT, obj = self.obj(),
                "Processing frame for {} model with dimensions {}x{} and {} channels",
                params.model_type, width, height, channels);

            // Extract frame data for inference
            let in_frame = match VideoFrameRef::from_buffer_ref_readable(
                inbuf.as_ref(),
                &VideoInfo::builder(VideoFormat::Rgb, width as u32, height as u32)
                    .build()
                    .unwrap(),
            ) {
                Ok(frame) => {
                    gst::debug!(CAT, obj = self.obj(), "Successfully mapped input buffer");
                    frame
                },
                Err(err) => {
                    gst::error!(CAT, obj = self.obj(), "Failed to map input buffer: {:?}", err);
                    return Err(gst::FlowError::Error);
                }
            };

            // Get the raw frame data
            let frame_data = match in_frame.plane_data(0) {
                Ok(data) => {
                    gst::debug!(CAT, obj = self.obj(), "Successfully got frame data of size {}", data.len());
                    data
                },
                Err(err) => {
                    gst::error!(CAT, obj = self.obj(), "Failed to get frame data: {:?}", err);
                    return Err(gst::FlowError::Error);
                }
            };

            // Convert frame data to features based on channel count
            let features = if channels == 3 {
                // RGB: Pack RGB values into single numbers
                let mut features = Vec::with_capacity((width * height) as usize);
                for chunk in frame_data.chunks_exact(3) {
                    if let [r, g, b] = chunk {
                        // Pack RGB values into a single number: (r << 16) + (g << 8) + b
                        let packed = (*r as u32) << 16 | (*g as u32) << 8 | (*b as u32);
                        features.push(packed as f32);
                    }
                }
                features
            } else {
                // Grayscale: Convert RGB to grayscale and pack
                let mut features = Vec::with_capacity((width * height) as usize);
                for chunk in frame_data.chunks_exact(3) {
                    if let [r, g, b] = chunk {
                        // Convert RGB to grayscale using standard weights
                        let gray = (0.299 * (*r as f32) + 0.587 * (*g as f32) + 0.114 * (*b as f32)) as u8;
                        // Pack grayscale value into RGB format
                        let packed = (gray as u32) << 16 | (gray as u32) << 8 | (gray as u32);
                        features.push(packed as f32);
                    }
                }
                features
            };

            gst::debug!(CAT, obj = self.obj(), "Running inference with {} features", features.len());

            // Run inference
            match model.classify(features, None) {
                Ok(result) => {
                    // Convert result to JSON string
                    let result_json = serde_json::to_string(&result.result).unwrap_or_else(|e| {
                        gst::warning!(CAT, obj = self.obj(), "Failed to serialize result: {}", e);
                        String::from("{}")
                    });

                    if is_object_detection {
                        gst::info!(CAT, obj = self.obj(), "Object detection result: {}", result_json);

                        // Create message structure for object detection results
                        let s = gst::Structure::builder("edge-impulse-inference-result")
                            .field("timestamp", inbuf.pts().unwrap_or(gst::ClockTime::ZERO))
                            .field("type", "object-detection")
                            .field("result", result_json)
                            .build();

                        // Post the message
                        let _ = self.obj().post_message(gst::message::Element::new(s));
                    } else {
                        gst::info!(CAT, obj = self.obj(), "Classification result: {}", result_json);

                        // Create message structure for classification results
                        let s = gst::Structure::builder("edge-impulse-inference-result")
                            .field("timestamp", inbuf.pts().unwrap_or(gst::ClockTime::ZERO))
                            .field("type", "classification")
                            .field("result", result_json)
                            .build();

                        // Post the message
                        let _ = self.obj().post_message(gst::message::Element::new(s));
                    }
                }
                Err(e) => {
                    gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);

                    // Post error message
                    let s = gst::Structure::builder("edge-impulse-inference-result")
                        .field("timestamp", inbuf.pts().unwrap_or(gst::ClockTime::ZERO))
                        .field("type", "error")
                        .field("error", e.to_string())
                        .build();

                    let _ = self.obj().post_message(gst::message::Element::new(s));
                }
            }
        } else {
            gst::debug!(CAT, obj = self.obj(), "No model loaded, skipping inference");
        }

        gst::debug!(CAT, imp = self, "Transform completed successfully");
        Ok(gst::FlowSuccess::Ok)
    }

    /// Handle caps (format) negotiation
    ///
    /// Stores the video dimensions from the negotiated caps for use during
    /// frame processing.
    fn set_caps(
        &self,
        incaps: &gst::Caps,
        outcaps: &gst::Caps,
    ) -> Result<(), gst::LoggableError> {
        gst::debug!(CAT, obj = self.obj(), "Set caps called with incaps: {:?}, outcaps: {:?}", incaps, outcaps);

        let mut state = self.state.lock().unwrap();

        // Parse input caps
        let in_info = VideoInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse input caps"))?;

        gst::info!(CAT, obj = self.obj(), "Setting caps: width={}, height={}",
            in_info.width(), in_info.height());

        // Store dimensions
        state.width = Some(in_info.width() as u32);
        state.height = Some(in_info.height() as u32);

        Ok(())
    }

    /// Transform caps between pads
    ///
    /// Handles format negotiation between input and output pads.
    /// Currently maintains the same caps, optionally filtered.
    fn transform_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        _filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        gst::debug!(CAT, obj = self.obj(), "Transform caps called with direction {:?}", direction);
        Some(caps.clone())
    }
}