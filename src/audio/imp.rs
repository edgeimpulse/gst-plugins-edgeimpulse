//! Audio inference element implementation for Edge Impulse models
//!
//! This module implements a GStreamer BaseTransform element that performs
//! machine learning inference on audio streams using Edge Impulse models.
//! The element operates in two modes:
//!
//! 1. Pass-through mode: When no model is loaded, audio is passed through unchanged
//! 2. Inference mode: When a model is loaded, audio is processed for inference
//!
//! # Audio Processing Flow
//! 1. The element receives S16LE mono audio samples at 16kHz
//! 2. Input samples are copied directly to output (pass-through)
//! 3. In parallel, samples are:
//!    - Converted from S16LE to normalized f32 [-1, 1] range
//!    - Accumulated in a buffer until enough samples for inference
//!    - Processed through the Edge Impulse model when buffer is full
//!    - Results are emitted as GStreamer messages
//!
//! # Properties
//! - `model-path`: Path to the Edge Impulse model file (.eim)
//!
//! # Messages
//! The element emits "edge-impulse-inference-result" messages with:
//! - timestamp: Buffer presentation timestamp
//! - type: "classification"
//! - result: JSON string containing model output
//! - timing_ms: Time taken for inference
//!
//! # Pipeline Example
//! ```bash
//! # Basic pipeline
//! gst-launch-1.0 \
//!     autoaudiosrc ! \
//!     capsfilter caps="audio/x-raw,format=F32LE" ! \
//!     audioconvert ! \
//!     audioresample ! \
//!     capsfilter caps="audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved" ! \
//!     edgeimpulseaudioinfer model-path=<path-to-model> ! \
//!     audioconvert ! \
//!     audioresample ! \
//!     capsfilter caps="audio/x-raw,format=F32LE,channels=2,rate=44100" ! \
//!     autoaudiosink
//! ```
//!
//! # Implementation Details
//!
//! ## Sample Buffer Management
//! - The element maintains a VecDeque<f32> to accumulate audio samples
//! - New samples are added to the buffer after normalization
//! - When buffer reaches model's slice_size, samples are drained for inference
//! - This creates a sliding window effect for continuous audio processing
//!
//! ## Audio Format Requirements
//! - Input must be S16LE (16-bit signed integer, little-endian)
//! - Single channel (mono)
//! - 16kHz sample rate
//! - Interleaved layout
//!
//! ## Sample Normalization
//! - Input samples are converted from S16LE [-32768, 32767] to f32 [-1, 1]
//! - Conversion: float_sample = int_sample / 32768.0
//!
//! ## Threading and Synchronization
//! - State is protected by Mutex for thread-safe access
//! - Buffer operations are atomic through Mutex locking
//! - GStreamer buffer processing is synchronized with pipeline clock
//!
//! ## Error Handling
//! - Model loading failures are logged but don't stop the pipeline
//! - Buffer mapping errors result in flow errors
//! - Inference errors are logged and reported via messages
//!
//! ## Memory Management
//! - Input buffers are copied to output immediately
//! - Sample conversions happen on temporary Vec<f32>
//! - Buffer mappings are dropped as soon as possible
//! - Sample buffer is cleared after each inference

use crate::common::CAT;
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::AudioInfo;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use once_cell::sync::Lazy;
use serde_json;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Audio-specific state structure
pub struct AudioState {
    /// The loaded Edge Impulse model
    pub model: Option<edge_impulse_runner::EimModel>,
    /// Audio sample rate
    pub sample_rate: Option<u32>,
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            model: None,
            sample_rate: None,
        }
    }
}

impl AsRef<Option<edge_impulse_runner::EimModel>> for AudioState {
    fn as_ref(&self) -> &Option<edge_impulse_runner::EimModel> {
        &self.model
    }
}

impl AsMut<Option<edge_impulse_runner::EimModel>> for AudioState {
    fn as_mut(&mut self) -> &mut Option<edge_impulse_runner::EimModel> {
        &mut self.model
    }
}

/// Audio inference element structure
#[derive(Default)]
pub struct EdgeImpulseAudioInfer {
    /// Shared state protected by a mutex for thread-safe access
    state: Mutex<AudioState>,
    /// Buffer for collecting audio samples
    sample_buffer: Mutex<VecDeque<f32>>,
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseAudioInfer {
    const NAME: &'static str = "EdgeImpulseAudioInfer";
    type Type = super::EdgeImpulseAudioInfer;
    type ParentType = gstreamer_base::BaseTransform;
}

impl ObjectImpl for EdgeImpulseAudioInfer {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> =
            Lazy::new(|| crate::common::create_common_properties());
        PROPERTIES.as_ref()
    }

    fn set_property(&self, id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        crate::common::set_common_property::<AudioState>(
            &self.state,
            id,
            value,
            pspec,
            &*self.obj(),
        );
    }

    fn property(&self, id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        crate::common::get_common_property::<AudioState>(&self.state, id, pspec)
    }
}

impl GstObjectImpl for EdgeImpulseAudioInfer {}

impl ElementImpl for EdgeImpulseAudioInfer {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Audio Inference",
                "Filter/Audio/AI",
                "Runs audio inference on Edge Impulse models (EIM)",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("audio/x-raw")
                .field("format", "S16LE")
                .field("rate", gst::IntRange::new(8000, 48000))
                .field("channels", 1i32)
                .field("layout", "interleaved")
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

impl BaseTransformImpl for EdgeImpulseAudioInfer {
    const MODE: BaseTransformMode = BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn unit_size(&self, caps: &gst::Caps) -> Option<usize> {
        gst::debug!(CAT, imp = self, "Getting unit size for caps: {:?}", caps);

        // Parse the caps to get audio info
        AudioInfo::from_caps(caps)
            .map(|info| info.bpf() as usize)
            .ok()
    }

    fn transform(
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

        // Copy the input buffer to the output buffer (passthrough)
        out_map.copy_from_slice(&in_map);

        // Convert input samples from S16LE to f32 normalized [-1, 1] for inference
        let samples: Vec<f32> = in_map
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0 // Normalize to [-1, 1]
            })
            .collect();

        // Drop the mappings after we're done with them
        drop(out_map);
        drop(in_map);

        // Run inference if we have enough samples and a model is loaded
        let mut state = self.state.lock().unwrap();
        if let Some(ref mut model) = state.model {
            let params = match model.parameters() {
                Ok(p) => p,
                Err(e) => {
                    gst::error!(
                        CAT,
                        obj = self.obj(),
                        "Failed to get model parameters: {}",
                        e
                    );
                    return Err(gst::FlowError::Error);
                }
            };

            let slice_size = params.slice_size;

            // Add new samples to buffer
            let mut sample_buffer = self.sample_buffer.lock().unwrap();
            sample_buffer.extend(samples);

            // Process when we have enough samples
            if sample_buffer.len() >= slice_size {
                let now = std::time::Instant::now();

                // Extract features for inference
                let features: Vec<f32> = sample_buffer.drain(..slice_size).collect();

                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Running inference with {} samples",
                    features.len()
                );

                // Run inference
                match model.classify(features, None) {
                    Ok(result) => {
                        let elapsed = now.elapsed();

                        let result_json =
                            serde_json::to_string(&result.result).unwrap_or_else(|e| {
                                gst::warning!(
                                    CAT,
                                    obj = self.obj(),
                                    "Failed to serialize result: {}",
                                    e
                                );
                                String::from("{}")
                            });

                        // Create message structure for classification results
                        let s = gst::Structure::builder("edge-impulse-inference-result")
                            .field("timestamp", inbuf.pts().unwrap_or(gst::ClockTime::ZERO))
                            .field("type", "classification")
                            .field("result", result_json)
                            .field("timing_ms", elapsed.as_millis() as u32)
                            .build();

                        // Post the message
                        let _ = self.obj().post_message(gst::message::Element::new(s));
                    }
                    Err(e) => {
                        gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);
                    }
                }
            }
        }

        Ok(gst::FlowSuccess::Ok)
    }

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
        let in_info = AudioInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse input caps"))?;

        gst::info!(
            CAT,
            obj = self.obj(),
            "Setting caps: rate={}, channels={}",
            in_info.rate(),
            in_info.channels()
        );

        // Store audio parameters
        state.sample_rate = Some(in_info.rate() as u32);

        Ok(())
    }

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
