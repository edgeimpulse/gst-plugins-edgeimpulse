//! Video inference element implementation for Edge Impulse models
//!
//! This module implements a GStreamer BaseTransform element that performs
//! machine learning inference on video frames using Edge Impulse models.
//! The element always passes the original frames through unchanged while
//! optionally performing inference when a model is loaded.
//!
//! # Video Processing Flow
//! 1. The element receives RGB video frames
//! 2. Input frames are copied directly to output (always)
//! 3. If a model is loaded, frames are also:
//!    - Mapped to raw RGB data
//!    - Converted to features based on model requirements (RGB or grayscale)
//!    - Processed through the Edge Impulse model
//!    - Results are emitted as GStreamer messages
//!
//! # Result Output Mechanisms
//! For object detection models, the element provides two mechanisms to consume results:
//!
//! 1. Bus Messages:
//!    - Sends element messages on the GStreamer bus using create_inference_message
//!    - Messages contain the raw JSON results and timing information
//!    - Useful for custom applications that want to process detection results
//!
//! 2. Video Frame Metadata (QC IM SDK Compatible):
//!    - Attaches VideoRegionOfInterestMeta to each video frame
//!    - Compatible with Qualcomm IM SDK `qtioverlay` element for automatic visualization
//!    - Each ROI includes the bounding box coordinates, label and confidence
//!    - QC IM SDK's `qtioverlay` element will automatically render these as boxes on the video
//!
//! This dual mechanism allows both custom applications and QC IM SDK
//! visualization tools to work with the detection results.
//!
//! # Properties
//! - `model-path`: Path to the Edge Impulse model file (.eim)
//!   - When set, loads the model and begins inference
//!   - When unset or invalid, only passes frames through
//!
//! # Messages
//! The element emits "edge-impulse-inference-result" messages with:
//! - timestamp: Frame presentation timestamp
//! - type: "classification" or "object-detection" based on model type
//! - result: JSON string containing model output
//!
//! For classification models, result contains:
//! ```json
//! {
//!   "classification": [
//!     {"label": "class1", "value": 0.95},
//!     {"label": "class2", "value": 0.05}
//!   ]
//! }
//! ```
//!
//! For object detection models, result contains:
//! ```json
//! {
//!   "bounding_boxes": [
//!     {
//!       "label": "object1",
//!       "value": 0.95,
//!       "x": 100,
//!       "y": 100,
//!       "width": 50,
//!       "height": 50
//!     }
//!   ]
//! }
//! ```
//!
//! # Pipeline Examples
//! ```bash
//! # Basic webcam pipeline
//! gst-launch-1.0 \
//!   avfvideosrc ! \
//!   queue max-size-buffers=2 leaky=downstream ! \
//!   videoconvert n-threads=4 ! \
//!   videoscale method=nearest-neighbour ! \
//!   video/x-raw,format=RGB,width=384,height=384 ! \
//!   queue max-size-buffers=2 leaky=downstream ! \
//!   edgeimpulsevideoinfer model-path=<model path> ! \
//!   queue max-size-buffers=2 leaky=downstream ! \
//!   videoscale method=nearest-neighbour ! \
//!   video/x-raw,width=480,height=480 ! \
//!   videoconvert n-threads=4 ! \
//!   autovideosink sync=false
//! ```
//!
//! # Implementation Details
//!
//! ## Frame Processing
//! - Input frames must be in RGB format
//! - Each frame is copied directly to output buffer
//! - For inference, RGB data is converted to features based on model type:
//!
//!   RGB models (3 channels):
//!   ```text
//!   for each pixel:
//!     packed = (r << 16) | (g << 8) | b
//!     feature = packed as f32
//!   ```
//!
//!   Grayscale models (1 channel):
//!   ```text
//!   for each pixel:
//!     gray = 0.299*r + 0.587*g + 0.114*b
//!     packed = (gray << 16) | (gray << 8) | gray
//!     feature = packed as f32
//!   ```
//!
//! ## Model Types
//! The element supports two types of Edge Impulse models:
//!
//! 1. Classification models:
//!    - Process entire frame
//!    - Output class probabilities
//!    - Message type: "classification"
//!
//! 2. Object Detection models:
//!    - Detect and locate objects in frame
//!    - Output bounding boxes and class probabilities
//!    - Message type: "object-detection"
//!
//! ## Video Format Requirements
//! - Format must be RGB (no other color formats supported)
//! - Width and height must match model input dimensions
//! - Frame rate is unrestricted
//! - Stride must be width * 3 (no padding supported)
//!
//! ## Error Handling
//! The element handles errors gracefully:
//! - Model loading failures:
//!   - Logged as errors
//!   - Element continues in pass-through mode
//! - Buffer mapping errors:
//!   - Return GST_FLOW_ERROR
//!   - May cause pipeline to stop
//! - Inference errors:
//!   - Logged as errors
//!   - Emitted as error messages
//!   - Pipeline continues running
//!
//! ## Memory Management
//! - Input frames are copied once to output buffer
//! - Feature conversion uses temporary vectors
//! - Buffer mappings are dropped immediately after use
//! - No frame data is retained between transforms
//!
//! ## Performance Considerations
//! - Single frame copy operation
//! - Feature conversion optimized for both RGB and grayscale
//! - No additional scaling or format conversion
//! - Pipeline should handle any necessary pre-processing
//! - Inference runs in transform thread
//!
//! ## Pad Templates
//! ```text
//! SRC template: 'src'
//! Availability: Always
//! Capabilities:
//!   video/x-raw
//!     format: RGB
//!     width: [ 1, 2147483647 ]
//!     height: [ 1, 2147483647 ]
//!
//! SINK template: 'sink'
//! Availability: Always
//! Capabilities:
//!   video/x-raw
//!     format: RGB
//!     width: [ 1, 2147483647 ]
//!     height: [ 1, 2147483647 ]
//! ```
//!
//! ## Element Information
//! - Name: "edgeimpulsevideoinfer"
//! - Classification: Filter/Video/AI
//! - Description: "Runs video inference on Edge Impulse models (EIM)"
//!
//! ## Debug Categories
//! The element uses the "edgeimpulsevideoinfer" debug category for logging.
//! Enable with:
//! ```bash
//! GST_DEBUG=edgeimpulsevideoinfer:4
//! ```
//!

use edge_impulse_runner::EimModel;
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use gstreamer_video as gst_video;
use gstreamer_video::{VideoInfo, VideoFrameRef, VideoFormat};
use gstreamer_video::prelude::VideoFrameExt;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use serde_json;
use gstreamer_sys;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulsevideoinfer",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Video Inference"),
    )
});

#[derive(Default)]
pub struct VideoState {
    /// The loaded Edge Impulse model instance
    pub model: Option<EimModel>,

    /// Width of the input frames (for video models)
    pub width: Option<u32>,

    /// Height of the input frames (for video models)
    pub height: Option<u32>,
}

/// EdgeImpulseVideoInfer element
///
/// This element performs ML inference on video frames using Edge Impulse models.
/// For object detection models, it provides two mechanisms to consume the results:
///
/// 1. Bus Messages:
///    - Sends element messages on the GStreamer bus using create_inference_message
///    - Messages contain the raw JSON results and timing information
///    - Useful for custom applications that want to process detection results
///
/// 2. Video Frame Metadata (QC IM SDK Compatible):
///    - Attaches VideoRegionOfInterestMeta to each video frame
///    - Compatible with Qualcomm IM SDK  element for automatic visualization
///    - Each ROI includes the bounding box coordinates, label and confidence
///    - QC IM SDK's qtioverlay element will automatically render these as boxes on the video
///
/// This dual mechanism allows both custom applications and QC IM SDK
/// visualization tools to work with the detection results.
#[derive(Default)]
pub struct EdgeImpulseVideoInfer {
    /// Shared state protected by a mutex for thread-safe access
    state: Mutex<VideoState>,
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
        gst::debug!(
            CAT,
            "Creating Edge Impulse Video Inference Element properties - Version {}",
            env!("CARGO_PKG_VERSION")
        );

        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> =
            Lazy::new(crate::common::create_common_properties);
        PROPERTIES.as_ref()
    }

    fn constructed(&self) {
        gst::debug!(
            CAT,
            imp = self,
            "Edge Impulse Video Inference Element constructed - Version {}",
            env!("CARGO_PKG_VERSION")
        );
        self.parent_constructed();
    }

    fn set_property(&self, id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        gst::info!(
            // Changed to info level for more visibility
            CAT,
            imp = self,
            "Setting property '{}' - Element Version {}",
            pspec.name(),
            env!("CARGO_PKG_VERSION")
        );

        crate::common::set_common_property::<VideoState>(
            &self.state,
            id,
            value,
            pspec,
            &*self.obj(),
            &CAT,
        );
    }

    fn property(&self, id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        gst::debug!(CAT, imp = self, "Getting property '{}'", pspec.name());
        crate::common::get_common_property::<VideoState>(&self.state, id, pspec)
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
            // Create base caps structure
            let base_caps = gst::Structure::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::new(1, i32::MAX))
                .field("height", gst::IntRange::new(1, i32::MAX))
                .build();

            // Create caps for both regular and GBM memory
            let caps = gst::Caps::builder_full()
                // Add regular memory caps
                .structure(base_caps.clone())
                // Add GBM memory caps
                .structure_with_features(
                    base_caps,
                    gst::CapsFeatures::new([String::from("memory:GBM")]),
                )
                .build();

            vec![
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
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

    /// Get the size of one unit for the given caps
    fn unit_size(&self, caps: &gst::Caps) -> Option<usize> {
        // Parse the caps into video info
        let info = match gst_video::VideoInfo::from_caps(caps) {
            Ok(info) => info,
            Err(_) => return None,
        };

        // Return the size of one video frame
        Some(info.size())
    }

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
                    gst::error!(
                        CAT,
                        obj = self.obj(),
                        "Failed to get model parameters: {}",
                        e
                    );
                    return Ok(gst::FlowSuccess::Ok);
                }
            };

            let width = params.image_input_width;
            let height = params.image_input_height;
            let channels = params.image_channel_count;
            let is_object_detection = params.model_type == "constrained_object_detection"
                || params.model_type == "object_detection";

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Processing frame for {} model with dimensions {}x{} and {} channels",
                params.model_type,
                width,
                height,
                channels
            );

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
                }
                Err(err) => {
                    gst::error!(
                        CAT,
                        obj = self.obj(),
                        "Failed to map input buffer: {:?}",
                        err
                    );
                    return Ok(gst::FlowSuccess::Ok);
                }
            };

            // Get the raw frame data
            let frame_data = match in_frame.plane_data(0) {
                Ok(data) => {
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Successfully got frame data of size {}",
                        data.len()
                    );
                    data
                }
                Err(err) => {
                    gst::error!(CAT, obj = self.obj(), "Failed to get frame data: {:?}", err);
                    return Ok(gst::FlowSuccess::Ok);
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
                        let gray =
                            (0.299 * (*r as f32) + 0.587 * (*g as f32) + 0.114 * (*b as f32)) as u8;
                        // Pack grayscale value into RGB format
                        let packed = (gray as u32) << 16 | (gray as u32) << 8 | (gray as u32);
                        features.push(packed as f32);
                    }
                }
                features
            };

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Running inference with {} features",
                features.len()
            );

            // Run inference
            match model.classify(features, None) {
                Ok(result) => {
                    let result_json = serde_json::to_string(&result.result).unwrap_or_else(|e| {
                        gst::warning!(CAT, obj = self.obj(), "Failed to serialize result: {}", e);
                        String::from("{}")
                    });

                    let now = std::time::Instant::now();
                    if is_object_detection {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result_json) {
                            gst::debug!(CAT, obj = self.obj(), "Parsed JSON result: {:?}", json);

                            if let Some(boxes) =
                                json.get("bounding_boxes").and_then(|b| b.as_array())
                            {
                                gst::debug!(
                                    CAT,
                                    obj = self.obj(),
                                    "Processing {} detections",
                                    boxes.len()
                                );

                                // Create detection metadata
                                for bbox in boxes {
                                    gst::debug!(
                                        CAT,
                                        obj = self.obj(),
                                        "Processing bbox: {:?}",
                                        bbox
                                    );

                                    if let (
                                        Some(label),
                                        Some(value),
                                        Some(x),
                                        Some(y),
                                        Some(w),
                                        Some(h),
                                    ) = (
                                        bbox.get("label").and_then(|v| v.as_str()),
                                        bbox.get("value").and_then(|v| v.as_f64()),
                                        bbox.get("x").and_then(|v| v.as_i64()),
                                        bbox.get("y").and_then(|v| v.as_i64()),
                                        bbox.get("width").and_then(|v| v.as_i64()),
                                        bbox.get("height").and_then(|v| v.as_i64()),
                                    ) {
                                        gst::debug!(
                                            CAT,
                                            obj = self.obj(),
                                            "Creating ROI metadata for {} at ({}, {}, {}, {}) with confidence {}",
                                            label, x, y, w, h, value
                                        );

                                        // Create ROI metadata
                                        let mut roi_meta =
                                            gst_video::VideoRegionOfInterestMeta::add(
                                                outbuf,
                                                label,
                                                (x as u32, y as u32, w as u32, h as u32),
                                            );

                                        // Add detection parameters
                                        let s = gst::Structure::builder("ObjectDetection")
                                            .field("confidence", value)
                                            .field("color", 0xFF0000FFu32)
                                            .build();
                                        roi_meta.add_param(s);

                                        gst::debug!(
                                            CAT,
                                            obj = self.obj(),
                                            "Successfully added ROI metadata"
                                        );
                                    } else {
                                        gst::warning!(
                                            CAT,
                                            obj = self.obj(),
                                            "Invalid bbox format: {:?}",
                                            bbox
                                        );
                                    }
                                }
                            } else {
                                gst::warning!(
                                    CAT,
                                    obj = self.obj(),
                                    "No bounding_boxes array in result"
                                );
                            }

                            let elapsed = now.elapsed();

                            // Create and post inference message to the bus as well.
                            let s = crate::common::create_inference_message(
                                "video",
                                inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                                "object-detection",
                                result_json,
                                elapsed.as_millis() as u32,
                            );

                            // Post the message
                            let _ = self.obj().post_message(gst::message::Element::new(s));
                        } else {
                            gst::warning!(CAT, obj = self.obj(), "Failed to parse JSON result");
                        }
                    } else {
                        let elapsed = now.elapsed();
                        let s = crate::common::create_inference_message(
                            "video",
                            inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                            "classification",
                            result_json,
                            elapsed.as_millis() as u32,
                        );
                        let _ = self.obj().post_message(gst::message::Element::new(s));
                    }
                }
                Err(e) => {
                    gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);
                    let s = crate::common::create_error_message(
                        "video",
                        inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                        e.to_string(),
                    );
                    let _ = self.obj().post_message(gst::message::Element::new(s));
                }
            }
        } else {
            gst::debug!(CAT, obj = self.obj(), "No model loaded, skipping inference");
        }

        Ok(gst::FlowSuccess::Ok)
    }

    /// Handle caps (format) negotiation
    ///
    /// Stores the video dimensions from the negotiated caps for use during
    /// frame processing.
    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        gst::info!(
            CAT,
            obj = self.obj(),
            "Set caps called with incaps: {:?}, outcaps: {:?}",
            incaps,
            outcaps
        );

        let mut state = self.state.lock().unwrap();

        // Parse input caps
        let in_info = VideoInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse input caps"))?;

        // Verify format is RGB
        if in_info.format() != gst_video::VideoFormat::Rgb {
            return Err(gst::loggable_error!(
                CAT,
                "Unsupported format: {:?}, only RGB is supported",
                in_info.format()
            ));
        }

        gst::info!(
            CAT,
            obj = self.obj(),
            "Setting caps: width={}, height={}, format={:?}",
            in_info.width(),
            in_info.height(),
            in_info.format()
        );

        // Store dimensions
        state.width = Some(in_info.width());
        state.height = Some(in_info.height());

        // Parse output caps and verify they match
        let out_info = VideoInfo::from_caps(outcaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse output caps"))?;

        if in_info.width() != out_info.width() || in_info.height() != out_info.height() {
            return Err(gst::loggable_error!(
                CAT,
                "Input and output dimensions must match: input={}x{}, output={}x{}",
                in_info.width(),
                in_info.height(),
                out_info.width(),
                out_info.height()
            ));
        }

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
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        gst::debug!(
            CAT,
            obj = self.obj(),
            "Transforming caps {} in direction {:?}",
            caps,
            direction
        );

        // Handle empty caps
        if caps.is_empty() {
            // Return a new caps with our supported formats
            let mut transformed_caps = gst::Caps::new_empty();

            // Create base caps structure for RGB format
            let base_structure = gst::Structure::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::new(1, i32::MAX))
                .field("height", gst::IntRange::new(1, i32::MAX))
                .build();

            // Add regular system memory caps
            transformed_caps.get_mut().unwrap().append_structure(base_structure.clone());

            // Add GBM memory caps
            let gbm_features = gst::CapsFeatures::new(&[String::from("memory:GBM")]);
            transformed_caps
                .get_mut()
                .unwrap()
                .append_structure_full(base_structure, Some(gbm_features));

            // If we have a filter, intersect with it
            if let Some(filter) = filter {
                transformed_caps = transformed_caps.intersect(filter);
            }

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Transformed empty caps to {} in direction {:?}",
                transformed_caps,
                direction
            );

            return Some(transformed_caps);
        }

        // Get the structure from the caps
        let structure = match caps.structure(0) {
            Some(s) => s,
            None => {
                gst::error!(
                    CAT,
                    obj = self.obj(),
                    "Failed to get structure from caps {}",
                    caps
                );
                return None;
            }
        };

        // Create base caps structure for RGB format
        let mut base_structure = gst::Structure::builder("video/x-raw")
            .field("format", "RGB");

        // If we have fixed dimensions in the input caps, maintain them
        if let Ok(width) = structure.get::<i32>("width") {
            base_structure = base_structure.field("width", width);
        } else {
            base_structure = base_structure.field("width", gst::IntRange::new(1, i32::MAX));
        }

        if let Ok(height) = structure.get::<i32>("height") {
            base_structure = base_structure.field("height", height);
        } else {
            base_structure = base_structure.field("height", gst::IntRange::new(1, i32::MAX));
        }

        // Build the structure
        let base_structure = base_structure.build();

        // Create transformed caps that will hold both memory types
        let mut transformed_caps = gst::Caps::new_empty();

        // Add regular system memory caps
        transformed_caps.get_mut().unwrap().append_structure(base_structure.clone());

        // Add GBM memory caps
        let gbm_features = gst::CapsFeatures::new(&[String::from("memory:GBM")]);
        transformed_caps
            .get_mut()
            .unwrap()
            .append_structure_full(base_structure, Some(gbm_features));

        // If we have a filter, intersect with it
        if let Some(filter) = filter {
            transformed_caps = transformed_caps.intersect(filter);
        }

        gst::debug!(
            CAT,
            obj = self.obj(),
            "Transformed caps from {} to {} in direction {:?}",
            caps,
            transformed_caps,
            direction
        );

        Some(transformed_caps)
    }
}

impl EdgeImpulseVideoInfer {
    fn handle_regular_memory(
        &self,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
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
                    gst::error!(
                        CAT,
                        obj = self.obj(),
                        "Failed to get model parameters: {}",
                        e
                    );
                    return Ok(gst::FlowSuccess::Ok);
                }
            };

            let width = params.image_input_width;
            let height = params.image_input_height;
            let channels = params.image_channel_count;
            let is_object_detection = params.model_type == "constrained_object_detection"
                || params.model_type == "object_detection";

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Processing frame for {} model with dimensions {}x{} and {} channels",
                params.model_type,
                width,
                height,
                channels
            );

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
                }
                Err(err) => {
                    gst::error!(
                        CAT,
                        obj = self.obj(),
                        "Failed to map input buffer: {:?}",
                        err
                    );
                    return Ok(gst::FlowSuccess::Ok);
                }
            };

            // Calculate scaling and padding to maintain aspect ratio
            let input_width = in_frame.width() as f32;
            let input_height = in_frame.height() as f32;
            let target_width = width as f32;
            let target_height = height as f32;

            let input_aspect = input_width / input_height;
            let target_aspect = target_width / target_height;

            // Calculate scale to fit within target dimensions while preserving aspect ratio
            let scale = if input_aspect > target_aspect {
                // Width limited
                target_width / input_width
            } else {
                // Height limited
                target_height / input_height
            };

            // Calculate scaled dimensions
            let scaled_width = (input_width * scale) as u32;
            let scaled_height = (input_height * scale) as u32;

            // Calculate padding to center the image
            let pad_left = ((width as u32 - scaled_width) / 2) as u32;
            let pad_top = ((height as u32 - scaled_height) / 2) as u32;

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Scaling frame: input={}x{}, scale={}, scaled={}x{}, padding left={}, top={}",
                input_width,
                input_height,
                scale,
                scaled_width,
                scaled_height,
                pad_left,
                pad_top
            );

            // Get the raw frame data
            let frame_data = match in_frame.plane_data(0) {
                Ok(data) => {
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Successfully got frame data of size {}",
                        data.len()
                    );
                    data
                }
                Err(err) => {
                    gst::error!(CAT, obj = self.obj(), "Failed to get frame data: {:?}", err);
                    return Ok(gst::FlowSuccess::Ok);
                }
            };

            // Create a buffer for the scaled and padded image
            let mut features = vec![0.0f32; (width * height) as usize];

            // Fill the features buffer with scaled and padded image data
            for y in 0..height as u32 {
                for x in 0..width as u32 {
                    let dst_idx = (y * width as u32 + x) as usize;

                    let (r, g, b) = if x >= pad_left
                        && x < pad_left + scaled_width
                        && y >= pad_top
                        && y < pad_top + scaled_height
                    {
                        // Map target coordinates back to source image
                        let src_x = ((x - pad_left) as f32 / scale) as u32;
                        let src_y = ((y - pad_top) as f32 / scale) as u32;

                        // Ensure we don't exceed source dimensions
                        let src_x = src_x.min(input_width as u32 - 1);
                        let src_y = src_y.min(input_height as u32 - 1);

                        let idx = ((src_y * in_frame.width() + src_x) * 3) as usize;
                        (
                            frame_data[idx],
                            frame_data[idx + 1],
                            frame_data[idx + 2],
                        )
                    } else {
                        // Fill padding with black
                        (0, 0, 0)
                    };

                    if channels == 3 {
                        // Pack RGB values into a single number
                        let packed = (r as u32) << 16 | (g as u32) << 8 | (b as u32);
                        features[dst_idx] = packed as f32;
                    } else {
                        // Convert to grayscale
                        let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                        let packed = (gray as u32) << 16 | (gray as u32) << 8 | (gray as u32);
                        features[dst_idx] = packed as f32;
                    }
                }
            }

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Running inference with {} features",
                features.len()
            );

            // Run inference
            match model.classify(features, None) {
                Ok(result) => {
                    let result_json = serde_json::to_string(&result.result).unwrap_or_else(|e| {
                        gst::warning!(CAT, obj = self.obj(), "Failed to serialize result: {}", e);
                        String::from("{}")
                    });

                    let now = std::time::Instant::now();
                    if is_object_detection {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&result_json) {
                            gst::debug!(CAT, obj = self.obj(), "Parsed JSON result: {:?}", json);

                            if let Some(boxes) =
                                json.get("bounding_boxes").and_then(|b| b.as_array())
                            {
                                gst::debug!(
                                    CAT,
                                    obj = self.obj(),
                                    "Processing {} detections",
                                    boxes.len()
                                );

                                // Create detection metadata
                                for bbox in boxes {
                                    gst::debug!(
                                        CAT,
                                        obj = self.obj(),
                                        "Processing bbox: {:?}",
                                        bbox
                                    );

                                    if let (
                                        Some(label),
                                        Some(value),
                                        Some(x),
                                        Some(y),
                                        Some(w),
                                        Some(h),
                                    ) = (
                                        bbox.get("label").and_then(|v| v.as_str()),
                                        bbox.get("value").and_then(|v| v.as_f64()),
                                        bbox.get("x").and_then(|v| v.as_i64()),
                                        bbox.get("y").and_then(|v| v.as_i64()),
                                        bbox.get("width").and_then(|v| v.as_i64()),
                                        bbox.get("height").and_then(|v| v.as_i64()),
                                    ) {
                                        gst::debug!(
                                            CAT,
                                            obj = self.obj(),
                                            "Creating ROI metadata for {} at ({}, {}, {}, {}) with confidence {}",
                                            label, x, y, w, h, value
                                        );

                                        // Create ROI metadata
                                        let mut roi_meta =
                                            gst_video::VideoRegionOfInterestMeta::add(
                                                outbuf,
                                                label,
                                                (x as u32, y as u32, w as u32, h as u32),
                                            );

                                        // Add detection parameters
                                        let s = gst::Structure::builder("ObjectDetection")
                                            .field("confidence", value)
                                            .field("color", 0xFF0000FFu32)
                                            .build();
                                        roi_meta.add_param(s);

                                        gst::debug!(
                                            CAT,
                                            obj = self.obj(),
                                            "Successfully added ROI metadata"
                                        );
                                    } else {
                                        gst::warning!(
                                            CAT,
                                            obj = self.obj(),
                                            "Invalid bbox format: {:?}",
                                            bbox
                                        );
                                    }
                                }
                            } else {
                                gst::warning!(
                                    CAT,
                                    obj = self.obj(),
                                    "No bounding_boxes array in result"
                                );
                            }

                            let elapsed = now.elapsed();

                            // Create and post inference message to the bus as well.
                            let s = crate::common::create_inference_message(
                                "video",
                                inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                                "object-detection",
                                result_json,
                                elapsed.as_millis() as u32,
                            );

                            // Post the message
                            let _ = self.obj().post_message(gst::message::Element::new(s));
                        } else {
                            gst::warning!(CAT, obj = self.obj(), "Failed to parse JSON result");
                        }
                    } else {
                        let elapsed = now.elapsed();
                        let s = crate::common::create_inference_message(
                            "video",
                            inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                            "classification",
                            result_json,
                            elapsed.as_millis() as u32,
                        );
                        let _ = self.obj().post_message(gst::message::Element::new(s));
                    }
                    Ok(gst::FlowSuccess::Ok)
                }
                Err(e) => {
                    gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);
                    let s = crate::common::create_error_message(
                        "video",
                        inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                        e.to_string(),
                    );
                    let _ = self.obj().post_message(gst::message::Element::new(s));
                    Ok(gst::FlowSuccess::Ok)
                }
            }
        } else {
            gst::debug!(CAT, obj = self.obj(), "No model loaded, skipping inference");
            Ok(gst::FlowSuccess::Ok)
        }
    }
}

impl AsRef<Option<EimModel>> for VideoState {
    fn as_ref(&self) -> &Option<EimModel> {
        &self.model
    }
}

impl AsMut<Option<EimModel>> for VideoState {
    fn as_mut(&mut self) -> &mut Option<EimModel> {
        &mut self.model
    }
}
