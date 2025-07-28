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
//! - `model-path`: Path to the Edge Impulse model file (.eim) - EIM mode only (legacy)
//! - `debug`: Enable debug mode for FFI inference (FFI mode only)
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
//! # Basic pipeline (FFI mode - default)
//! gst-launch-1.0 \
//!     autoaudiosrc ! \
//!     capsfilter caps="audio/x-raw,format=F32LE" ! \
//!     audioconvert ! \
//!     audioresample ! \
//!     capsfilter caps="audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved" ! \
//!     edgeimpulseaudioinfer ! \
//!     audioconvert ! \
//!     audioresample ! \
//!     capsfilter caps="audio/x-raw,format=F32LE,channels=2,rate=44100" ! \
//!     autoaudiosink
//!
//! # EIM mode (legacy)
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
//! - The element maintains a `VecDeque<f32>` to accumulate audio samples
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
//! - Sample conversions happen on temporary `Vec<f32>`
//! - Buffer mappings are dropped as soon as possible
//! - Sample buffer is cleared after each inference

use edge_impulse_runner::EdgeImpulseModel;
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::AudioInfo;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::sync::Mutex;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulseaudioinfer",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Audio Inference"),
    )
});

/// Audio-specific state structure
#[derive(Default)]
pub struct AudioState {
    /// The loaded Edge Impulse model
    pub model: Option<EdgeImpulseModel>,
    /// Audio sample rate
    pub sample_rate: Option<u32>,
    /// Debug mode flag for FFI mode (lazy initialization)
    #[cfg(feature = "ffi")]
    pub debug_enabled: bool,
}

impl AsRef<Option<EdgeImpulseModel>> for AudioState {
    fn as_ref(&self) -> &Option<EdgeImpulseModel> {
        &self.model
    }
}

impl AsMut<Option<EdgeImpulseModel>> for AudioState {
    fn as_mut(&mut self) -> &mut Option<EdgeImpulseModel> {
        &mut self.model
    }
}

impl crate::common::DebugState for AudioState {
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

impl AudioState {}

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
            Lazy::new(crate::common::create_common_properties);
        PROPERTIES.as_ref()
    }

    fn set_property(&self, id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        crate::common::set_common_property::<AudioState>(
            &self.state,
            id,
            value,
            pspec,
            &*self.obj(),
            &CAT,
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
                "Runs audio inference on Edge Impulse models (FFI default, EIM legacy)",
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

        // Convert input samples from S16LE to f32 for inference
        let samples: Vec<f32> = in_map
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32
            })
            .collect();

        // Drop the mappings after we're done with them
        drop(out_map);
        drop(in_map);

        // Run inference if we have enough samples and a model is loaded
        let mut state = self.state.lock().unwrap();
        let model_exists = state.model.is_some();
        let sample_rate = state.sample_rate;

        #[cfg(feature = "ffi")]
        {
            gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Audio transform called with current state: sample_rate={:?}, model_exists={}, debug_enabled={}",
                    sample_rate,
                    model_exists,
                    state.debug_enabled
                );
        }
        #[cfg(not(feature = "ffi"))]
        {
            gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Audio transform called with current state: sample_rate={:?}, model_exists={}, debug_enabled=false",
                    sample_rate,
                    model_exists
                );
        }

        // Try to get existing model or create lazily
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

        if let Some(mut model) = model {
            gst::debug!(CAT, obj = self.obj(), "Got model, getting parameters...");

            let _params = match model.parameters() {
                Ok(p) => {
                    gst::debug!(CAT, obj = self.obj(), "Successfully got model parameters");
                    p
                }
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

            // Get the required number of raw audio samples from the model parameters
            // For FFI mode, we should use the raw sample count from metadata
            #[cfg(feature = "ffi")]
            let required_samples =
                { edge_impulse_runner::ffi::ModelMetadata::get().raw_sample_count };
            #[cfg(not(feature = "ffi"))]
            let required_samples = { _params.slice_size as usize };
            gst::debug!(
                CAT,
                obj = self.obj(),
                "Model expects {} raw audio samples for inference (ModelMetadata::get().raw_sample_count)",
                required_samples
            );

            // Add new samples to buffer (convert S16LE to f32, no normalization)
            let mut sample_buffer = self.sample_buffer.lock().unwrap();
            let samples_len = samples.len();
            sample_buffer.extend(samples);

            gst::debug!(
                CAT,
                obj = self.obj(),
                "Buffer status: {} samples in buffer, need {} samples, received {} new samples (total: {})",
                sample_buffer.len(),
                required_samples,
                samples_len,
                sample_buffer.len() + samples_len
            );

            // Run inference when we have enough samples
            while sample_buffer.len() >= required_samples {
                let now = std::time::Instant::now();
                // Take exactly the number of raw samples we need
                let features: Vec<f32> = sample_buffer.drain(..required_samples).collect();
                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Running inference with {} samples",
                    features.len()
                );
                // Run inference
                match model.infer(features, None) {
                    Ok(result) => {
                        let elapsed = now.elapsed();

                        // Convert result.result to serde_json::Value for normalization
                        let mut result_value = serde_json::to_value(&result.result).unwrap();
                        // Standardize classification output: always as object {label: value, ...}
                        if let Some(classification) = result_value.get_mut("classification") {
                            if classification.is_array() {
                                let mut map = serde_json::Map::new();
                                for entry in classification.as_array().unwrap() {
                                    if let (Some(label), Some(value)) =
                                        (entry.get("label"), entry.get("value"))
                                    {
                                        if let (Some(label), Some(value)) =
                                            (label.as_str(), value.as_f64())
                                        {
                                            map.insert(
                                                label.to_string(),
                                                serde_json::Value::from(value),
                                            );
                                        }
                                    }
                                }
                                *classification = serde_json::Value::Object(map);
                            }
                        }
                        let result_json =
                            serde_json::to_string(&result_value).unwrap_or_else(|e| {
                                gst::warning!(
                                    CAT,
                                    obj = self.obj(),
                                    "Failed to serialize result: {}",
                                    e
                                );
                                String::from("{}")
                            });

                        let s = crate::common::create_inference_message(
                            "audio",
                            inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                            "classification",
                            result_json,
                            elapsed.as_millis() as u32,
                        );

                        // Post the message
                        let _ = self.obj().post_message(gst::message::Element::new(s));
                    }
                    Err(e) => {
                        gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);
                    }
                }
            }

            // Put the model back in the state
            state.model = Some(model);
        }

        // Handle end-of-stream inference if we have remaining samples
        let is_eos = inbuf.size() == 0;
        if is_eos {
            let mut sample_buffer = self.sample_buffer.lock().unwrap();
            if !sample_buffer.is_empty() {
                // For EIM mode, we need to get the required samples from the model parameters
                // Since we don't have a model in EIM mode without a valid model path, we'll use a default
                let required_samples = 16000; // Default for most audio models

                gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "End of stream reached with {} samples in buffer, running final inference",
                    sample_buffer.len()
                );

                // Take as many samples as we have, pad with zeros if needed
                let mut final_samples: Vec<f32> = sample_buffer.drain(..).collect();
                if final_samples.len() < required_samples {
                    final_samples.extend(vec![0.0; required_samples - final_samples.len()]);
                    gst::debug!(
                    CAT,
                    obj = self.obj(),
                    "Using {} real samples + {} zero padding = {} total samples for classification",
                    final_samples.len() - (required_samples - final_samples.len()),
                    required_samples - final_samples.len(),
                    final_samples.len()
                );
                } else {
                    // Take exactly the required number of samples
                    final_samples = final_samples[..required_samples].to_vec();
                    gst::debug!(
                        CAT,
                        obj = self.obj(),
                        "Using exactly {} raw audio samples for classification (no padding)",
                        final_samples.len()
                    );
                }

                if let Some(mut model) = state.model.take() {
                    let now = std::time::Instant::now();
                    match model.infer(final_samples, None) {
                        Ok(result) => {
                            let elapsed = now.elapsed();
                            let mut result_value = serde_json::to_value(&result.result).unwrap();
                            if let Some(classification) = result_value.get_mut("classification") {
                                if classification.is_array() {
                                    let mut map = serde_json::Map::new();
                                    for entry in classification.as_array().unwrap() {
                                        if let (Some(label), Some(value)) =
                                            (entry.get("label"), entry.get("value"))
                                        {
                                            if let (Some(label), Some(value)) =
                                                (label.as_str(), value.as_f64())
                                            {
                                                map.insert(
                                                    label.to_string(),
                                                    serde_json::Value::from(value),
                                                );
                                            }
                                        }
                                    }
                                    *classification = serde_json::Value::Object(map);
                                }
                            }
                            let result_json =
                                serde_json::to_string(&result_value).unwrap_or_else(|e| {
                                    gst::warning!(
                                        CAT,
                                        obj = self.obj(),
                                        "Failed to serialize result: {}",
                                        e
                                    );
                                    String::from("{}")
                                });

                            let s = crate::common::create_inference_message(
                                "audio",
                                inbuf.pts().unwrap_or(gst::ClockTime::ZERO),
                                "classification",
                                result_json,
                                elapsed.as_millis() as u32,
                            );

                            let _ = self.obj().post_message(gst::message::Element::new(s));
                        }
                        Err(e) => {
                            gst::error!(CAT, obj = self.obj(), "Final inference failed: {}", e);
                        }
                    }
                    state.model = Some(model);
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
        state.sample_rate = Some(in_info.rate());

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
