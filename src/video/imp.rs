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
use gstreamer_video::{VideoFormat, VideoFrameRef, VideoInfo};
use once_cell::sync::Lazy;
use std::sync::Mutex;

#[cfg(feature = "dma-buf")]
#[repr(C)]
struct dma_buf_sync {
    flags: u64,
}

#[cfg(feature = "dma-buf")]
const DMA_BUF_SYNC_START: u64 = 0;
#[cfg(feature = "dma-buf")]
const DMA_BUF_SYNC_END: u64 = 1;
#[cfg(feature = "dma-buf")]
const DMA_BUF_SYNC_RW: u64 = 1 << 1;
#[cfg(feature = "dma-buf")]
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x40087b0d;

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
        let state = self.state.lock().unwrap();

        // Check if input buffer uses GBM memory
        let is_gbm = inbuf.n_memory() > 0 && inbuf.peek_memory(0).flags().contains(gst::MemoryFlags::NO_SHARE);

        let width = state.width.unwrap();
        let height = state.height.unwrap();

        // Create video info for frame mapping
        let video_info = VideoInfo::builder(
                gst_video::VideoFormat::Rgb,
                width,
                height
            )
            .build()
            .map_err(|_| {
                gst::error!(CAT, imp = self, "Failed to create video info");
                gst::FlowError::Error
            })?;

        if is_gbm {
            // Handle GBM memory buffer
            #[cfg(feature = "dma-buf")]
            {
                // Get FD from GBM memory
                let fd = if let Ok(mem) = inbuf.peek_memory(0).map(|m| m.downcast_memory_ref::<gst::FdMemory>()) {
                    mem.and_then(|fd_mem| Some(fd_mem.fd()))
                } else {
                    None
                };

                if let Some(fd) = fd {
                    // Sync start
                    unsafe {
                        let mut sync = dma_buf_sync {
                            flags: DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW,
                        };
                        if libc::ioctl(fd, DMA_BUF_IOCTL_SYNC, &mut sync) != 0 {
                            gst::warning!(CAT, "DMA buffer sync start failed");
                        }
                    }

                    // Map input frame
                    let in_frame = gst_video::VideoFrameRef::from_buffer_ref_readable(inbuf.as_ref(), &video_info)
                        .map_err(|_| {
                            gst::error!(CAT, imp = self, "Failed to map input buffer");
                            gst::FlowError::Error
                        })?;

                    // Map output frame
                    let mut out_frame = gst_video::VideoFrameRef::from_buffer_ref_writable(outbuf, &video_info)
                        .map_err(|_| {
                            gst::error!(CAT, imp = self, "Failed to map output buffer");
                            gst::FlowError::Error
                        })?;

                    // Copy frame data
                    let in_data = in_frame.plane_data(0).map_err(|_| {
                        gst::error!(CAT, imp = self, "Failed to get input plane data");
                        gst::FlowError::Error
                    })?;

                    let out_data = out_frame.plane_data_mut(0).map_err(|_| {
                        gst::error!(CAT, imp = self, "Failed to get output plane data");
                        gst::FlowError::Error
                    })?;

                    out_data.copy_from_slice(in_data);

                    // Drop frame mappings
                    drop(out_frame);
                    drop(in_frame);

                    // Process the buffer for inference
                    if let Some(ref model) = state.model {
                        let input_data: Vec<f32> = in_data.iter().map(|&x| x as f32).collect();
                        model.classify(input_data, None).map_err(|_| {
                            gst::error!(CAT, imp = self, "Failed to run inference");
                            gst::FlowError::Error
                        })?;
                    }

                    // Sync end
                    unsafe {
                        let mut sync = dma_buf_sync {
                            flags: DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW,
                        };
                        if libc::ioctl(fd, DMA_BUF_IOCTL_SYNC, &mut sync) != 0 {
                            gst::warning!(CAT, "DMA buffer sync end failed");
                        }
                    }

                    Ok(gst::FlowSuccess::Ok)
                } else {
                    gst::error!(CAT, "Failed to get FD from GBM memory");
                    Err(gst::FlowError::Error)
                }
            }
            #[cfg(not(feature = "dma-buf"))]
            {
                gst::error!(CAT, "GBM memory support not enabled");
                Err(gst::FlowError::NotSupported)
            }
        } else {
            // Handle regular system memory buffer
            // Map input frame
            let in_frame = gst_video::VideoFrameRef::from_buffer_ref_readable(inbuf.as_ref(), &video_info)
                .map_err(|_| {
                    gst::error!(CAT, imp = self, "Failed to map input buffer");
                    gst::FlowError::Error
                })?;

            // Map output frame
            let mut out_frame = gst_video::VideoFrameRef::from_buffer_ref_writable(outbuf, &video_info)
                .map_err(|_| {
                    gst::error!(CAT, imp = self, "Failed to map output buffer");
                    gst::FlowError::Error
                })?;

            // Copy frame data
            let in_data = in_frame.plane_data(0).map_err(|_| {
                gst::error!(CAT, imp = self, "Failed to get input plane data");
                gst::FlowError::Error
            })?;

            let out_data = out_frame.plane_data_mut(0).map_err(|_| {
                gst::error!(CAT, imp = self, "Failed to get output plane data");
                gst::FlowError::Error
            })?;

            out_data.copy_from_slice(in_data);

            // Drop frame mappings
            drop(out_frame);
            drop(in_frame);

            // Process the buffer for inference
            if let Some(ref model) = state.model {
                let input_data: Vec<f32> = in_data.iter().map(|&x| x as f32).collect();
                model.classify(input_data, None).map_err(|_| {
                    gst::error!(CAT, imp = self, "Failed to run inference");
                    gst::FlowError::Error
                })?;
            }

            Ok(gst::FlowSuccess::Ok)
        }
    }

    /// Handle caps (format) negotiation
    ///
    /// Stores the video dimensions from the negotiated caps for use during
    /// frame processing.
    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        gst::info!(
            // Changed to info level for more visibility
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
            "Setting caps: width={}, height={}",
            in_info.width(),
            in_info.height()
        );

        // Store dimensions
        state.width = Some(in_info.width());
        state.height = Some(in_info.height());

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
        // Create base caps structure for RGB format
        let base_structure = gst::Structure::builder("video/x-raw")
            .field("format", "RGB")
            .field("width", gst::IntRange::new(1, i32::MAX))
            .field("height", gst::IntRange::new(1, i32::MAX))
            .build();

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
