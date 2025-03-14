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
use gstreamer_video::{VideoInfo, VideoFrameRef};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use serde_json;

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
            // Create base caps structure for system memory
            let system_caps = gst::Structure::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::<i32>::new(1, i32::MAX))
                .field("height", gst::IntRange::<i32>::new(1, i32::MAX))
                .build();

            // Create base caps structure for GBM memory
            let gbm_caps = gst::Structure::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::<i32>::new(1, i32::MAX))
                .field("height", gst::IntRange::<i32>::new(1, i32::MAX))
                .build();

            // Create caps for both regular and GBM memory
            let caps = gst::Caps::builder_full()
                // Add regular memory caps
                .structure(system_caps)
                // Add GBM memory caps
                .structure_with_features(
                    gbm_caps,
                    gst::CapsFeatures::new(&[String::from("memory:GBM")]),
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
    /// Don't allow pass-through when caps are unchanged
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

        // Get input caps and info
        let element = self.obj();
        let sinkpad = element.static_pad("sink").unwrap();
        let caps = sinkpad.current_caps().unwrap();
        let info = gst_video::VideoInfo::from_caps(&caps).map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to get video info from caps");
            gst::FlowError::Error
        })?;

        // Check if we have GBM memory
        let has_gbm = caps.features(0).map(|f| f.contains("memory:GBM")).unwrap_or(false);
        gst::debug!(CAT, imp = self, "Using GBM memory: {}", has_gbm);

        // Create VideoFrame from input buffer - no need for flags here
        let in_frame = VideoFrameRef::from_buffer_ref_readable(inbuf.as_ref(), &info)
            .map_err(|_| {
                gst::error!(CAT, obj = self.obj(), "Failed to map input buffer");
                gst::FlowError::Error
            })?;

        // Create VideoFrame from output buffer - no need for flags here
        let mut out_frame = VideoFrameRef::from_buffer_ref_writable(outbuf, &info)
            .map_err(|_| {
                gst::error!(CAT, obj = self.obj(), "Failed to map output buffer");
                gst::FlowError::Error
            })?;

        // Copy frame data using VideoFrameRef's copy method
        in_frame.copy(&mut out_frame).map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to copy frame data");
            gst::FlowError::Error
        })?;

        // Check if we have a model loaded and get model parameters
        let (_model_type, _channels) = {
            let state = self.state.lock().unwrap();
            if state.model.is_none() {
                // No model loaded, just pass through the buffer
                return Ok(gst::FlowSuccess::Ok);
            }

            // Get model parameters while holding the lock
            let model = state.model.as_ref().unwrap();
            let params = model.parameters().map_err(|e| {
                gst::error!(
                    CAT,
                    obj = self.obj(),
                    "Failed to get model parameters: {}",
                    e
                );
                gst::FlowError::Error
            })?;

            (params.model_type.clone(), params.image_channel_count)
        };

        // Get the plane data for inference
        let plane_data = in_frame.plane_data(0).map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to get plane data");
            gst::FlowError::Error
        })?;

        // Make a copy of the data for inference processing
        let data_copy = plane_data.to_vec();

        // Drop the frame references before processing
        drop(in_frame);
        drop(out_frame);

        // Process the frame data and run inference
        self.process_frame_data(data_copy, inbuf, outbuf)
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

        // Check if we have GBM buffers
        let has_gbm = incaps
            .features(0)
            .map(|f| f.contains("memory:GBM"))
            .unwrap_or(false);

        // For regular buffers, verify format is RGB
        if !has_gbm && in_info.format() != gst_video::VideoFormat::Rgb {
            return Err(gst::loggable_error!(
                CAT,
                "Unsupported format: {:?}, only RGB is supported for regular memory",
                in_info.format()
            ));
        }

        // For GBM buffers, verify format is RGB
        if has_gbm && in_info.format() != gst_video::VideoFormat::Rgb {
            return Err(gst::loggable_error!(
                CAT,
                "Unsupported format: {:?}, only RGB is supported for GBM memory",
                in_info.format()
            ));
        }

        gst::info!(
            CAT,
            obj = self.obj(),
            "Setting caps: width={}, height={}, format={:?}, is_gbm={}",
            in_info.width(),
            in_info.height(),
            in_info.format(),
            has_gbm
        );

        // Store dimensions
        state.width = Some(in_info.width());
        state.height = Some(in_info.height());

        // Parse output caps and verify they match
        let out_info = VideoInfo::from_caps(outcaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse output caps"))?;

        // Verify dimensions match
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

        // Verify memory types match
        let out_has_gbm = outcaps
            .features(0)
            .map(|f| f.contains("memory:GBM"))
            .unwrap_or(false);

        if has_gbm != out_has_gbm {
            return Err(gst::loggable_error!(
                CAT,
                "Input and output memory types must match: input_gbm={}, output_gbm={}",
                has_gbm,
                out_has_gbm
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
            imp = self,
            "Transform caps called with direction {:?} and caps {:?}",
            direction,
            caps
        );

        let mut builder = gst::Caps::builder_full();
        let mut has_structures = false;

        // Iterate through each structure in the input caps
        for structure in caps.iter() {
            has_structures = true;

            // Get width and height, handling both fixed values and ranges
            let width = match structure.get::<gst::IntRange<i32>>("width") {
                Ok(range) => range,
                Err(_) => {
                    // If not a range, try to get as a fixed value
                    match structure.get::<i32>("width") {
                        Ok(w) => gst::IntRange::new(w, w + 1), // Create a range that includes the fixed value
                        Err(_) => gst::IntRange::new(1, 2), // Default to a valid range
                    }
                }
            };

            let height = match structure.get::<gst::IntRange<i32>>("height") {
                Ok(range) => range,
                Err(_) => {
                    // If not a range, try to get as a fixed value
                    match structure.get::<i32>("height") {
                        Ok(h) => gst::IntRange::new(h, h + 1), // Create a range that includes the fixed value
                        Err(_) => gst::IntRange::new(1, 2), // Default to a valid range
                    }
                }
            };

            // Create a new structure with RGB format
            let new_structure = gst::Structure::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", width)
                .field("height", height)
                .build();

            // Check if this structure has GBM memory feature
            if let Ok(features) = structure.get::<gst::CapsFeatures>("features") {
                if features.contains("memory:GBM") {
                    // If it has GBM, add it with GBM feature
                    builder = builder.structure_with_features(
                        new_structure,
                        gst::CapsFeatures::new(["memory:GBM"]),
                    );
                } else {
                    // If it doesn't have GBM, add it without features
                    builder = builder.structure(new_structure);
                }
            } else {
                // If no features, add without features
                builder = builder.structure(new_structure);
            }
        }

        // If no structures were added, create a default one with GBM memory
        if !has_structures {
            let default_structure = gst::Structure::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", gst::IntRange::new(1, 2))
                .field("height", gst::IntRange::new(1, 2))
                .build();
            builder = builder.structure_with_features(
                default_structure,
                gst::CapsFeatures::new(["memory:GBM"]),
            );
        }

        let mut new_caps = builder.build();

        // Apply filter if provided
        if let Some(filter) = filter {
            new_caps = new_caps.intersect(filter);
        }

        gst::debug!(
            CAT,
            imp = self,
            "Transformed caps from {:?} to {:?}",
            caps,
            new_caps
        );

        Some(new_caps)
    }
}

impl EdgeImpulseVideoInfer {
    // Helper function to process frame data and run inference
    fn process_frame_data(&self, data: Vec<u8>, inbuf: &gst::Buffer, outbuf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        let mut state = self.state.lock().unwrap();
        if let Some(ref mut model) = state.model {
            // Get all parameters upfront
            let model_type = match model.parameters() {
                Ok(p) => p.model_type.clone(),
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

            let channels = match model.parameters() {
                Ok(p) => p.image_channel_count,
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

            // Convert frame data to features based on channel count
            let features = if channels == 3 {
                // RGB: Pack RGB values into single numbers
                let mut features = Vec::with_capacity(data.len() / 3);
                for chunk in data.chunks_exact(3) {
                    if let [r, g, b] = chunk {
                        // Pack RGB values into a single number: (r << 16) + (g << 8) + b
                        let packed = (*r as u32) << 16 | (*g as u32) << 8 | (*b as u32);
                        features.push(packed as f32);
                    }
                }
                features
            } else {
                // Grayscale: Convert RGB to grayscale and pack
                let mut features = Vec::with_capacity(data.len() / 3);
                for chunk in data.chunks_exact(3) {
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
            let now = std::time::Instant::now();
            let result = match model.classify(features, None) {
                Ok(result) => result,
                Err(e) => {
                    gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);

                    let s = crate::common::create_error_message(
                        "video",
                        inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                        e.to_string(),
                    );

                    // Post the message
                    let _ = self.obj().post_message(gst::message::Element::new(s));
                    return Ok(gst::FlowSuccess::Ok);
                }
            };

            let result_json = serde_json::to_string(&result.result).unwrap();
            let elapsed = now.elapsed();

            if model_type == "constrained_object_detection" || model_type == "object_detection" {
                // For object detection models, also add ROI metadata
                if let Ok(result) = serde_json::from_str::<serde_json::Value>(&result_json) {
                    if let Some(boxes) = result.get("bounding_boxes") {
                        if let Some(boxes) = boxes.as_array() {
                            for bbox in boxes {
                                if let (Some(x), Some(y), Some(w), Some(h), Some(label), Some(value)) = (
                                    bbox.get("x").and_then(|v| v.as_f64()),
                                    bbox.get("y").and_then(|v| v.as_f64()),
                                    bbox.get("width").and_then(|v| v.as_f64()),
                                    bbox.get("height").and_then(|v| v.as_f64()),
                                    bbox.get("label").and_then(|v| v.as_str()),
                                    bbox.get("value").and_then(|v| v.as_f64()),
                                ) {
                                    let mut meta = gst_video::VideoRegionOfInterestMeta::add(
                                        outbuf,
                                        label,
                                        (x as u32, y as u32, w as u32, h as u32),
                                    );

                                    meta.add_param(gst::Structure::builder("ObjectDetection")
                                        .field("confidence", value)
                                        .build());
                                }
                            }
                        }
                    }
                }

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
                let s = crate::common::create_inference_message(
                    "video",
                    inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                    "classification",
                    result_json,
                    elapsed.as_millis() as u32,
                );

                // Post the message
                let _ = self.obj().post_message(gst::message::Element::new(s));
            }
        }

        Ok(gst::FlowSuccess::Ok)
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
