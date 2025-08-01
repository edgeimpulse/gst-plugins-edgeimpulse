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
//! - `model-path`: Path to the Edge Impulse model file (.eim) - EIM mode only (legacy)
//!   - When set, loads the model and begins inference
//!   - When unset, uses FFI mode (default)
//! - `debug`: Enable debug mode for FFI inference (FFI mode only)
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
//! # Basic webcam pipeline (FFI mode - default)
//! gst-launch-1.0 \
//!   avfvideosrc ! \
//!   queue max-size-buffers=2 leaky=downstream ! \
//!   videoconvert n-threads=4 ! \
//!   videoscale method=nearest-neighbour ! \
//!   video/x-raw,format=RGB,width=384,height=384 ! \
//!   queue max-size-buffers=2 leaky=downstream ! \
//!   edgeimpulsevideoinfer ! \
//!   queue max-size-buffers=2 leaky=downstream ! \
//!   videoscale method=nearest-neighbour ! \
//!   video/x-raw,width=480,height=480 ! \
//!   videoconvert n-threads=4 ! \
//!   autovideosink sync=false
//!
//! # EIM mode (legacy)
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
//! - Description: "Runs video inference on Edge Impulse models (FFI default, EIM legacy)"
//!
//! ## Debug Categories
//! The element uses the "edgeimpulsevideoinfer" debug category for logging.
//! Enable with:
//! ```bash
//! GST_DEBUG=edgeimpulsevideoinfer:4
//! ```
//!

use edge_impulse_runner::EdgeImpulseModel;
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use gstreamer_video as gst_video;
use gstreamer_video::{VideoFormat, VideoInfo};
use once_cell::sync::Lazy;
use std::sync::Mutex;

use super::meta::VideoRegionOfInterestMeta;
use super::VideoAnomalyMeta;
use super::VideoClassificationMeta;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulsevideoinfer",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Video Inference"),
    )
});

#[derive(Debug)]
pub struct VideoState {
    /// The loaded Edge Impulse model instance
    pub model: Option<EdgeImpulseModel>,

    /// Width of the input frames (for video models)
    pub width: Option<u32>,

    /// Height of the input frames (for video models)
    pub height: Option<u32>,

    /// Format of the input frames
    pub format: Option<VideoFormat>,

    /// Debug mode flag for FFI mode (lazy initialization)
    #[cfg(feature = "ffi")]
    pub debug_enabled: bool,
}

impl Default for VideoState {
    fn default() -> Self {
        Self {
            model: None,
            width: None,
            height: None,
            format: None,
            #[cfg(feature = "ffi")]
            debug_enabled: false,
        }
    }
}

impl VideoState {}

impl crate::common::DebugState for VideoState {
    fn set_debug(&mut self, enabled: bool) {
        #[cfg(feature = "ffi")]
        {
            self.debug_enabled = enabled;
        }
        #[cfg(not(feature = "ffi"))]
        {
            // No-op in non-FFI mode
            let _ = enabled;
        }
    }

    fn get_debug(&self) -> bool {
        #[cfg(feature = "ffi")]
        {
            self.debug_enabled
        }
        #[cfg(not(feature = "ffi"))]
        {
            false
        }
    }
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
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            let mut props = crate::common::create_common_properties();
            if !props.iter().any(|p| p.name() == "model-path-with-debug") {
                props.push(
                    glib::ParamSpecString::builder("model-path-with-debug")
                        .nick("Model Path With Debug")
                        .blurb("Path to Edge Impulse model file (debug mode enabled)")
                        .build(),
                );
            }
            props
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
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
                "Runs video inference on Edge Impulse models (FFI default, EIM legacy)",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", gst::List::new(["RGB", "GRAY8"]))
                .field("width", gst::IntRange::new(1, i32::MAX))
                .field("height", gst::IntRange::new(1, i32::MAX))
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
        // Get all state values upfront to minimize mutex lock time
        let (width, height, format, model) = {
            let mut state = self.state.lock().unwrap();
            let width = state.width.unwrap_or(0);
            let height = state.height.unwrap_or(0);
            let format = state.format;
            let model_exists = state.model.is_some();

            #[cfg(feature = "ffi")]
            {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Transform called with current state: width={}, height={}, format={:?}, model_exists={}, debug_enabled={}",
                    width,
                    height,
                    format,
                    model_exists,
                    state.debug_enabled
                );
            }
            #[cfg(not(feature = "ffi"))]
            {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Transform called with current state: width={}, height={}, format={:?}, model_exists={}, debug_enabled=false",
                    width,
                    height,
                    format,
                    model_exists
                );
            }

            // Try to get existing model first
            let model = if let Some(model) = state.model.take() {
                gst::debug!(CAT, obj = self.obj(), "Using existing model from state");
                Some(model)
            } else {
                // No model exists, try lazy initialization (FFI mode only)
                #[cfg(feature = "ffi")]
                {
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "No model in state, attempting lazy FFI initialization (debug={})",
                        state.debug_enabled
                    );

                    let model_result = if state.debug_enabled {
                        gst::debug!(
                            CAT,
                            obj = self.obj(),
                            "Creating FFI model with debug enabled"
                        );
                        EdgeImpulseModel::new_with_debug(true)
                    } else {
                        gst::debug!(CAT, obj = self.obj(), "Creating FFI model without debug");
                        EdgeImpulseModel::new()
                    };

                    match model_result {
                        Ok(model) => {
                            gst::debug!(
                                CAT,
                                obj = self.obj(),
                                "Successfully created FFI model lazily (debug={})",
                                state.debug_enabled
                            );
                            Some(model)
                        }
                        Err(err) => {
                            gst::error!(
                                CAT,
                                obj = self.obj(),
                                "Failed to create FFI model lazily (debug={}): {}",
                                state.debug_enabled,
                                err
                            );
                            None
                        }
                    }
                }
                #[cfg(not(feature = "ffi"))]
                {
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "FFI feature not enabled, cannot create model lazily"
                    );
                    None
                }
            };

            (width, height, format, model)
        };

        // Map the input buffer for reading (keep it mapped for inference)
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

        // Drop output mapping but keep input mapping for inference
        drop(out_map);

        if let Some(mut model) = model {
            gst::debug!(CAT, obj = self.obj(), "Model loaded, running inference");

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
                    // Put the model back in the state
                    let mut state = self.state.lock().unwrap();
                    state.model = Some(model);
                    return Err(gst::FlowError::Error);
                }
            };

            let _model_width = params.image_input_width;
            let _model_height = params.image_input_height;
            let channels = params.image_channel_count;
            let is_object_detection = params.model_type == "constrained_object_detection"
                || params.model_type == "object_detection"
                || params.model_type == "object-detection";
            let is_anomaly_detection = params.model_type == "anomaly_detection"
                || matches!(
                    params.has_anomaly,
                    edge_impulse_runner::types::RunnerHelloHasAnomaly::VisualGMM
                );

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Model type detection: model_type='{}', is_object_detection={}, is_anomaly_detection={}",
                params.model_type,
                is_object_detection,
                is_anomaly_detection
            );

            // Add debug output for the condition branches
            if is_anomaly_detection {
                gst::debug!(CAT, obj = self.obj(), "Taking anomaly detection branch");
            } else if is_object_detection {
                gst::debug!(CAT, obj = self.obj(), "Taking object detection branch");
            } else {
                gst::debug!(CAT, obj = self.obj(), "Taking classification branch");
            }

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Model parameters: width={}, height={}, channels={}, type={}, has_anomaly={:?}",
                _model_width,
                _model_height,
                channels,
                params.model_type,
                params.has_anomaly
            );

            // Use the already mapped input buffer data directly for feature conversion
            let frame_data = &in_map;

            // Pre-allocate features vector with exact capacity
            let pixel_count = (width * height) as usize;
            let mut features = Vec::with_capacity(pixel_count);

            // Optimized feature conversion - avoid redundant VideoFrameRef creation
            if format == Some(VideoFormat::Gray8) {
                // For grayscale images, create RGB values by repeating the grayscale value
                features.extend(frame_data.iter().map(|&pixel| {
                    // Create 24-bit RGB value: 0xRRGGBB where R=G=B=pixel
                    let feature = ((pixel as u32) << 16) | ((pixel as u32) << 8) | (pixel as u32);
                    feature as f32
                }));
            } else {
                // For RGB images, combine channels into 24-bit RGB values
                // Use chunks_exact for better performance and avoid bounds checking
                features.extend(frame_data.chunks_exact(3).map(|chunk| {
                    // Create 24-bit RGB value: 0xRRGGBB
                    let feature =
                        ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
                    feature as f32
                }));
            }

            // Run inference
            let start = std::time::Instant::now();
            let result = match model.infer(features, None) {
                Ok(result) => result,
                Err(e) => {
                    gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);
                    let s = crate::common::create_error_message(
                        "video",
                        inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                        e.to_string(),
                    );
                    let _ = self.obj().post_message(gst::message::Element::new(s));
                    // Put the model back in the state
                    let mut state = self.state.lock().unwrap();
                    state.model = Some(model);
                    return Err(gst::FlowError::Error);
                }
            };

            let elapsed = start.elapsed();

            // Convert result.result to serde_json::Value for normalization
            let mut result_value = serde_json::to_value(&result.result).unwrap();

            // Standardize classification output: always as object {label: value, ...}
            if let Some(classification) = result_value.get_mut("classification") {
                if classification.is_array() {
                    let mut map = serde_json::Map::new();
                    for entry in classification.as_array().unwrap() {
                        if let (Some(label), Some(value)) = (entry.get("label"), entry.get("value"))
                        {
                            if let (Some(label), Some(value)) = (label.as_str(), value.as_f64()) {
                                map.insert(label.to_string(), serde_json::Value::from(value));
                            }
                        }
                    }
                    *classification = serde_json::Value::Object(map);
                }
            }

            // Standardize object detection: always array of objects for bounding_boxes
            if let Some(bboxes) = result_value.get_mut("bounding_boxes") {
                if bboxes.is_object() {
                    // Convert object to array if needed (legacy)
                    let mut arr = Vec::new();
                    for (_k, v) in bboxes.as_object().unwrap() {
                        arr.push(v.clone());
                    }
                    *bboxes = serde_json::Value::Array(arr);
                }
            }

            // Standardize visual anomaly detection grid: always array of objects
            if let Some(grid) = result_value.get_mut("visual_anomaly_grid") {
                if grid.is_object() {
                    let mut arr = Vec::new();
                    for (_k, v) in grid.as_object().unwrap() {
                        arr.push(v.clone());
                    }
                    *grid = serde_json::Value::Array(arr);
                }
            }

            // Convert to JSON string once and reuse
            let result_json = serde_json::to_string(&result_value).unwrap();
            gst::debug!(CAT, obj = self.obj(), "Inference result: {}", result_json);

            // --- Handle visual anomaly detection metadata ---
            if let Some(grid) = result_value
                .get("visual_anomaly_grid")
                .and_then(|g| g.as_array())
            {
                if !grid.is_empty() {
                    // Create VideoAnomalyMeta for visual anomaly detection
                    let mut anomaly_meta = VideoAnomalyMeta::add(outbuf);

                    // Set anomaly values from the result
                    if let Some(anomaly) = result_value.get("anomaly").and_then(|a| a.as_f64()) {
                        anomaly_meta.set_anomaly(anomaly as f32);
                    }
                    if let Some(max) = result_value
                        .get("visual_anomaly_max")
                        .and_then(|m| m.as_f64())
                    {
                        anomaly_meta.set_visual_anomaly_max(max as f32);
                    }
                    if let Some(mean) = result_value
                        .get("visual_anomaly_mean")
                        .and_then(|m| m.as_f64())
                    {
                        anomaly_meta.set_visual_anomaly_mean(mean as f32);
                    }

                    // Convert grid cells to VideoRegionOfInterestMeta format
                    let mut grid_rois = Vec::new();
                    for cell in grid {
                        if let (Some(x), Some(y), Some(width), Some(height), Some(score)) = (
                            cell["x"].as_u64(),
                            cell["y"].as_u64(),
                            cell["width"].as_u64(),
                            cell["height"].as_u64(),
                            cell["value"].as_f64(),
                        ) {
                            grid_rois.push(VideoRegionOfInterestMeta {
                                x: x as u32,
                                y: y as u32,
                                width: width as u32,
                                height: height as u32,
                                label: format!("{:.1}", score * 100.0), // Store score as label
                            });
                        }
                    }
                    anomaly_meta.set_visual_anomaly_grid(grid_rois);

                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Added visual anomaly metadata: anomaly={:.1}%, max={:.1}%, mean={:.1}%, grid_cells={}",
                        anomaly_meta.anomaly() * 100.0,
                        anomaly_meta.visual_anomaly_max() * 100.0,
                        anomaly_meta.visual_anomaly_mean() * 100.0,
                        anomaly_meta.visual_anomaly_grid().len()
                    );

                    // Post inference message for visual anomaly detection
                    let s = crate::common::create_inference_message(
                        "video",
                        inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                        "visual-anomaly",
                        result_json.clone(),
                        elapsed.as_millis() as u32,
                    );
                    let _ = self.obj().post_message(gst::message::Element::new(s));
                }
            }

            // --- Unified result handling: prefer object detection if bounding_boxes is present and non-empty ---
            // Use the already parsed result_value instead of parsing JSON again
            if let Some(boxes) = result_value
                .get("bounding_boxes")
                .and_then(|b| b.as_array())
            {
                if !boxes.is_empty() {
                    // Object detection branch
                    for bbox in boxes {
                        if let (
                            Some(label),
                            Some(value),
                            Some(x),
                            Some(y),
                            Some(width),
                            Some(height),
                        ) = (
                            bbox["label"].as_str(),
                            bbox["value"].as_f64(),
                            bbox["x"].as_u64(),
                            bbox["y"].as_u64(),
                            bbox["width"].as_u64(),
                            bbox["height"].as_u64(),
                        ) {
                            let mut roi_meta = gst_video::VideoRegionOfInterestMeta::add(
                                outbuf,
                                label,
                                (x as u32, y as u32, width as u32, height as u32),
                            );
                            let s = gst::Structure::builder("detection")
                                .field("label", label)
                                .field("confidence", value)
                                .build();
                            roi_meta.add_param(s);
                        }
                    }
                    let s = crate::common::create_inference_message(
                        "video",
                        inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                        "object-detection",
                        result_json,
                        elapsed.as_millis() as u32,
                    );
                    let _ = self.obj().post_message(gst::message::Element::new(s));
                } else {
                    // Classification fallback if bounding_boxes is empty
                    if let Some(classification) = result_value
                        .get("classification")
                        .and_then(|c| c.as_object())
                    {
                        let mut best_label = None;
                        let mut best_confidence = 0.0;
                        for (label, confidence) in classification {
                            if let Some(conf) = confidence.as_f64() {
                                if conf > best_confidence {
                                    best_confidence = conf;
                                    best_label = Some(label);
                                }
                            }
                        }
                        if let Some(label) = best_label {
                            let mut meta = VideoClassificationMeta::add(outbuf);
                            let s = gst::Structure::builder("classification")
                                .field("label", label)
                                .field("confidence", best_confidence)
                                .build();
                            meta.add_param(s);
                            gst::debug!(
                                CAT,
                                obj = self.obj(),
                                "Added classification metadata: label={}, confidence={}",
                                label,
                                best_confidence
                            );
                        }
                    }
                    let s = crate::common::create_inference_message(
                        "video",
                        inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                        "classification",
                        result_json,
                        elapsed.as_millis() as u32,
                    );
                    let _ = self.obj().post_message(gst::message::Element::new(s));
                }
            } else {
                // No bounding_boxes field, treat as classification
                if let Some(classification) = result_value
                    .get("classification")
                    .and_then(|c| c.as_object())
                {
                    let mut best_label = None;
                    let mut best_confidence = 0.0;
                    for (label, confidence) in classification {
                        if let Some(conf) = confidence.as_f64() {
                            if conf > best_confidence {
                                best_confidence = conf;
                                best_label = Some(label);
                            }
                        }
                    }
                    if let Some(label) = best_label {
                        let mut meta = VideoClassificationMeta::add(outbuf);
                        let s = gst::Structure::builder("classification")
                            .field("label", label)
                            .field("confidence", best_confidence)
                            .build();
                        meta.add_param(s);
                        gst::debug!(
                            CAT,
                            obj = self.obj(),
                            "Added classification metadata: label={}, confidence={}",
                            label,
                            best_confidence
                        );
                    }
                }
                let s = crate::common::create_inference_message(
                    "video",
                    inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                    "classification",
                    result_json,
                    elapsed.as_millis() as u32,
                );
                let _ = self.obj().post_message(gst::message::Element::new(s));
            }

            // Put the model back in the state
            let mut state = self.state.lock().unwrap();
            state.model = Some(model);
        } else {
            #[cfg(feature = "ffi")]
            {
                let debug_enabled = {
                    let state = self.state.lock().unwrap();
                    state.debug_enabled
                };
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "No model loaded, skipping inference. FFI mode enabled with debug={}, lazy initialization may have failed",
                    debug_enabled
                );
            }
            #[cfg(not(feature = "ffi"))]
            {
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "No model loaded, skipping inference. FFI mode not enabled, EIM mode requires model-path property"
                );
            }
        }

        // Drop the input mapping at the end
        drop(in_map);

        Ok(gst::FlowSuccess::Ok)
    }

    /// Handle caps (format) negotiation
    ///
    /// Stores the video dimensions from the negotiated caps for use during
    /// frame processing.
    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        gst::debug!(
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

        gst::info!(
            CAT,
            obj = self.obj(),
            "Setting caps: width={}, height={}, format={:?}",
            in_info.width(),
            in_info.height(),
            in_info.format()
        );

        // Store dimensions and format
        state.width = Some(in_info.width());
        state.height = Some(in_info.height());
        state.format = Some(in_info.format());

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
        gst::debug!(
            CAT,
            obj = self.obj(),
            "Transform caps called with direction {:?}",
            direction
        );
        Some(caps.clone())
    }
}

impl AsRef<Option<EdgeImpulseModel>> for VideoState {
    fn as_ref(&self) -> &Option<EdgeImpulseModel> {
        &self.model
    }
}

impl AsMut<Option<EdgeImpulseModel>> for VideoState {
    fn as_mut(&mut self) -> &mut Option<EdgeImpulseModel> {
        &mut self.model
    }
}
