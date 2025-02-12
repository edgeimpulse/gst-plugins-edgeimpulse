//! # GStreamer Edge Impulse Plugin
//!
//! This plugin integrates Edge Impulse machine learning models into GStreamer pipelines,
//! allowing real-time inference on audio streams.
//!
//! ## Architecture
//!
//! The plugin is implemented as a GStreamer BaseTransform element. BaseTransform is designed
//! for elements that process data in a 1:1 fashion - each input buffer produces exactly one
//! output buffer of the same size. This is ideal for our use case where we want to:
//! 1. Process incoming audio data
//! 2. Run inference on that data
//! 3. Pass the original audio through unchanged
//!
//! ## Data Flow
//!
//! 1. Audio data arrives as raw PCM samples (S16LE format)
//! 2. The transform() function is called for each input buffer
//! 3. Input data is copied to output to maintain audio flow
//! 4. Samples are converted to f32 format required by Edge Impulse
//! 5. Inference is run in a separate thread to avoid blocking the pipeline
//! 6. Results are logged (future: could be sent as events/messages)
//!
//! ## Edge Impulse Integration
//!
//! The Edge Impulse Runner provides a C API wrapped in Rust bindings:
//! - Models are loaded from .eim files
//! - classify_continuous() handles sliding window inference
//! - Results include detected classes and probabilities
//!
//! ## Threading Model
//!
//! - GStreamer pipeline runs on main thread
//! - Inference runs on background thread using Arc<Mutex<>> for thread safety
//! - Audio processing is never blocked by ML inference

use gstreamer as gst;
use gstreamer_base::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use gstreamer_audio::AudioInfo;
use gstreamer_audio::AudioFormat;
use std::sync::Mutex;
use edge_impulse_runner::{EimModel, SensorType};
use edge_impulse_runner::InferenceResult::Classification;
use gst::glib;
use once_cell::sync::Lazy;
use glib::subclass::Signal;
use glib::value::ToValue;
use std::collections::VecDeque;

/// Debug category for logging
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulseinfer",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Inference Element"),
    )
});

/// Plugin settings that can be configured at runtime
#[derive(Debug, Default)]
struct Settings {
    /// Path to the .eim model file
    model_path: Option<String>,
}

/// Main plugin state
#[derive(Default)]
pub struct EdgeImpulseInfer {
    /// Runtime settings protected by mutex
    settings: Mutex<Settings>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    model: Option<EimModel>,
    sample_rate: Option<i32>,
    samples_per_inference: Option<usize>,
    sample_buffer: VecDeque<i16>,
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseInfer {
    const NAME: &'static str = "EdgeImpulseInfer";
    type Type = super::EdgeImpulseInfer;
    type ParentType = gstreamer_base::BaseTransform;
}

impl EdgeImpulseInfer {
    /// Scale factor to normalize 16-bit audio samples to -1.0 to +1.0 range
    const I16_TO_F32_SCALE: f32 = 32768.0;
}

impl BaseTransformImpl for EdgeImpulseInfer {
    /// Configure transform behavior
    const MODE: BaseTransformMode = BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    /// Calculate size of one audio frame
    fn unit_size(&self, caps: &gst::Caps) -> Option<usize> {
        AudioInfo::from_caps(caps)
            .map(|info| info.bpf() as usize)
            .ok()
    }

    /// Process one buffer of audio data
    fn transform(
        &self,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let mut state = self.state.lock().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to lock state");
            gst::FlowError::Error
        })?;

        // Verify we have a model loaded
        if state.model.is_none() {
            gst::error!(CAT, obj = self.obj(), "No model loaded");
            return Err(gst::FlowError::Error);
        }

        let samples_needed = state.samples_per_inference.ok_or_else(|| {
            gst::error!(CAT, obj = self.obj(), "Samples per inference not set");
            gst::FlowError::Error
        })?;

        // Get input samples from buffer
        let data = inbuf.map_readable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Can't map buffer readable");
            gst::FlowError::Error
        })?;

        // Copy input to output
        let mut out_data = outbuf.map_writable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Can't map buffer writable");
            gst::FlowError::Error
        })?;
        out_data.copy_from_slice(&data);

        // Convert bytes to i16 samples
        let samples = data.chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]));

        // Add new samples to our buffer
        state.sample_buffer.extend(samples);

        // Keep only the most recent samples
        while state.sample_buffer.len() > samples_needed {
            state.sample_buffer.pop_front();
        }

        // Run inference if we have enough samples
        if state.sample_buffer.len() >= samples_needed {
            let inference_samples: Vec<f32> = state.sample_buffer.iter()
                .map(|&s| s as f32 / Self::I16_TO_F32_SCALE)
                .collect();

            if let Some(model) = &mut state.model {
                match model.classify(inference_samples, None) {
                    Ok(result) => {
                        // Create a structure with the classification results
                        let mut s = gst::Structure::builder("edge-impulse-inference-result")
                            .field("timestamp", inbuf.pts().unwrap_or(gst::ClockTime::ZERO))
                            .build();

                        // Add classification results as a nested structure
                        if let Classification { ref classification } = result.result {
                            let mut class_struct = gst::Structure::new_empty("classification");
                            for (label, confidence) in classification {
                                class_struct.set(label, confidence);
                            }
                            s.set("classification", class_struct);
                        }

                        // Post the message on the bus
                        let msg = gst::message::Element::new(s);
                        let _ = self.obj().post_message(msg);

                        gst::debug!(CAT, obj = self.obj(), "Inference result: {:?}", result);
                    }
                    Err(e) => {
                        gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);
                    }
                }
            }

            state.sample_buffer.clear();
        }

        Ok(gst::FlowSuccess::Ok)
    }

    fn set_caps(
        &self,
        incaps: &gst::Caps,
        _outcaps: &gst::Caps,
    ) -> Result<(), gst::LoggableError> {
        let s = incaps.structure(0).unwrap();
        let rate = s.get::<i32>("rate").map_err(|_| {
            gst::loggable_error!(CAT, "Failed to get rate from caps")
        })?;

        let state = self.state.lock().unwrap();

        // Verify rate matches model's expected rate
        if let Some(model_rate) = state.sample_rate {
            if rate != model_rate {
                return Err(gst::loggable_error!(CAT,
                    "Input sample rate {} Hz doesn't match model's expected rate {} Hz",
                    rate, model_rate
                ));
            }
        }

        Ok(())
    }

    fn transform_caps(
        &self,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        _filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let state = self.state.lock().unwrap();
        let mut output_caps = caps.clone();

        // Ensure we get the right format
        if direction == gst::PadDirection::Sink {
            if let Some(s) = output_caps.make_mut().structure_mut(0) {
                s.set("format", "S16LE");
                if let Some(rate) = state.sample_rate {
                    s.set("rate", rate);
                }
                s.set("channels", 1i32);
            }
        }

        Some(output_caps)
    }
}

impl ObjectImpl for EdgeImpulseInfer {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("model-path")
                    .nick("Model Path")
                    .blurb("Path to Edge Impulse model file")
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    fn constructed(&self) {
        self.parent_constructed();

        // Initialize state
        let mut state = self.state.lock().unwrap();

        // Load model and get its properties
        if let Some(model_path) = &self.settings.lock().unwrap().model_path {
            // Check debug level for edgeimpulseinfer category
            let debug_enabled = gst::DebugCategory::get("edgeimpulseinfer")
                .map(|cat| cat.threshold() > gst::DebugLevel::Debug)
                .unwrap_or(false);

            match EimModel::new_with_debug(model_path, debug_enabled) {
                Ok(model) => {
                    // Get model parameters
                    match model.parameters() {
                        Ok(params) => {
                            // Verify this is an audio model
                            if params.sensor != SensorType::Microphone as u32 {
                                gst::error!(CAT, obj = self.obj(),
                                    "Model is not an audio model (sensor type: {})",
                                    params.sensor
                                );
                                return;
                            }

                            // Store the parameters we need
                            let frequency = params.frequency;
                            let slice_size = params.slice_size;
                            let labels = params.labels.clone();

                            // Calculate sample rate from frequency
                            state.sample_rate = Some(frequency as i32);

                            // Calculate samples per inference from slice_size
                            state.samples_per_inference = Some(slice_size);

                            // Store model
                            state.model = Some(model);

                            gst::info!(CAT, obj = self.obj(),
                                "Audio model loaded: sample_rate={}Hz, window_size={} samples, labels={:?}",
                                frequency,
                                slice_size,
                                labels
                            );
                        }
                        Err(err) => {
                            gst::error!(CAT, obj = self.obj(),
                                "Failed to get model parameters: {}",
                                err
                            );
                        }
                    }
                }
                Err(err) => {
                    gst::error!(CAT, obj = self.obj(), "Failed to load model: {}", err);
                }
            }
        }
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "model-path" => {
                let mut settings = self.settings.lock().unwrap();
                let mut state = self.state.lock().unwrap();

                settings.model_path = value.get().expect("type checked upstream");

                if let Some(path) = &settings.model_path {
                    gst::info!(CAT, obj = self.obj(), "Loading model from {}", path);

                    // Check debug level for edgeimpulseinfer category
                    let debug_enabled = gst::DebugCategory::get("edgeimpulseinfer")
                        .map(|cat| cat.threshold() > gst::DebugLevel::Debug)
                        .unwrap_or(false);

                    match EimModel::new_with_debug(path, debug_enabled) {
                        Ok(model) => {
                            // Get parameters and clone the values we need
                            let (frequency, slice_size, sensor) = match model.parameters() {
                                Ok(p) => (p.frequency, p.slice_size, p.sensor),
                                Err(e) => {
                                    gst::error!(CAT, obj = self.obj(),
                                        "Failed to get model parameters: {}", e);
                                    return;
                                }
                            };

                            // Validate sensor type
                            if sensor != SensorType::Microphone as u32 {
                                gst::error!(CAT, obj = self.obj(),
                                    "Invalid sensor type {}", sensor);
                                return;
                            }

                            // Update state
                            state.sample_rate = Some(frequency as i32);
                            state.samples_per_inference = Some(slice_size);
                            state.model = Some(model);

                            gst::info!(CAT, obj = self.obj(),
                                "Model loaded: {}Hz, {} samples",
                                frequency, slice_size);
                        }
                        Err(e) => {
                            gst::error!(CAT, obj = self.obj(),
                                "Failed to load model: {}", e);
                        }
                    }
                }
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        gst::info!(CAT, obj = self.obj(), "property called for {}", pspec.name());
        match pspec.name() {
            "model-path" => {
                let settings = self.settings.lock().unwrap();
                settings.model_path.to_value()
            }
            _ => unimplemented!(),
        }
    }

    fn signals() -> &'static [Signal] {
        static SIGNALS: Lazy<Vec<Signal>> = Lazy::new(|| {
            vec![Signal::builder("edge-impulse-inference-result")
                .param_types([String::static_type()])
                .build()]
        });
        SIGNALS.as_ref()
    }
}

impl ElementImpl for EdgeImpulseInfer {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Inference",
                "Filter/Audio/Video/Motion",
                "Runs Edge Impulse ML inference on input data",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            // Default caps that support all sensor types
            let audio_caps = gst::Caps::builder("audio/x-raw")
                .field("format", AudioFormat::S16le.to_str())
                .field("rate", gst::IntRange::new(8000, 48000))
                .field("channels", 1i32)
                .field("layout", "interleaved")
                .build();

            // TODO: Add caps for other sensor types (video, etc)
            let caps = audio_caps;

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

impl GstObjectImpl for EdgeImpulseInfer {}