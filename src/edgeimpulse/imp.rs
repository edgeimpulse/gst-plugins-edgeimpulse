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
use std::sync::{Arc, Mutex};
use edge_impulse_runner::{EimModel, SensorType};
use gst::glib;
use once_cell::sync::Lazy;
use std::error::Error;

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
    /// Model parameters obtained after loading
    input_size: Option<usize>,
    sensor_type: Option<SensorType>,
}

/// Main plugin state
#[derive(Default)]
pub struct EdgeImpulseInfer {
    /// Runtime settings protected by mutex
    settings: Mutex<Settings>,
    /// The loaded ML model wrapped in Arc for thread-safety
    model: Arc<Mutex<Option<EimModel>>>,
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

    fn transform_buffer(
        &self,
        buffer: &gst::Buffer,
    ) -> Result<gst::Buffer, gst::FlowError> {
        let element = self.obj();
        let settings = self.settings.lock().unwrap();

        // Get buffer data
        let data = buffer.map_readable().map_err(|_| {
            gst::error!(CAT, obj = element, "Failed to map buffer readable");
            gst::FlowError::Error
        })?;

        // Convert input data based on sensor type
        let features: Vec<f32> = match settings.sensor_type {
            Some(SensorType::Microphone) => {
                // Convert audio samples (S16LE) to f32
                data.chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / Self::I16_TO_F32_SCALE
                    })
                    .collect()
            },
            Some(SensorType::Camera) => {
                // TODO: Handle image data conversion
                gst::error!(CAT, obj = element, "Camera input not yet implemented");
                return Err(gst::FlowError::Error);
            },
            Some(SensorType::Accelerometer) => {
                // TODO: Handle accelerometer data conversion
                gst::error!(CAT, obj = element, "Accelerometer input not yet implemented");
                return Err(gst::FlowError::Error);
            },
            Some(SensorType::Positional) => {
                // TODO: Handle positional data conversion
                gst::error!(CAT, obj = element, "Positional input not yet implemented");
                return Err(gst::FlowError::Error);
            },
            _ => {
                gst::error!(CAT, obj = element, "Unknown or unset sensor type");
                return Err(gst::FlowError::Error);
            }
        };

        // Verify input size matches model requirements
        if let Some(expected_size) = settings.input_size {
            if features.len() != expected_size {
                gst::error!(CAT, obj = element,
                    "Input size mismatch. Got {} features, expected {}",
                    features.len(), expected_size);
                return Err(gst::FlowError::Error);
            }
        }

        // Run inference
        if let Some(ref mut model) = *self.model.lock().unwrap() {
            match model.classify(features, None) {
                Ok(result) => {
                    gst::debug!(CAT, obj = element, "Classification result: {:?}", result);
                    Ok(buffer.clone())
                }
                Err(e) => {
                    gst::error!(CAT, obj = element, "Failed to run inference: {}", e);
                    Err(gst::FlowError::Error)
                }
            }
        } else {
            gst::error!(CAT, obj = element, "Model not loaded");
            Err(gst::FlowError::Error)
        }
    }

    fn load_model(&self, path: &str) -> Result<(), Box<dyn Error>> {
        gst::debug!(CAT, obj = self.obj(), "Loading model from path: {}", path);

        let model = EimModel::new(path)?;

        // Get model parameters
        let input_size = model.input_size()?;
        let sensor_type = model.sensor_type()?;
        let params = model.parameters()?;

        gst::info!(CAT, obj = self.obj(),
            "Model loaded: input_size={}, sensor_type={:?}, params={:?}",
            input_size, sensor_type, params);

        // Update settings
        let mut settings = self.settings.lock().unwrap();
        settings.input_size = Some(input_size);
        settings.sensor_type = Some(sensor_type);

        // Store model
        *self.model.lock().unwrap() = Some(model);

        Ok(())
    }
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
        gst::debug!(CAT, obj = self.obj(), "Processing buffer of size {}", inbuf.size());

        // Map input buffer for reading
        let data = inbuf.map_readable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to map input buffer readable");
            gst::FlowError::Error
        })?;

        // Copy input to output to maintain audio flow
        let mut out_map = outbuf.map_writable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to map output buffer writable");
            gst::FlowError::Error
        })?;
        out_map.copy_from_slice(&data);

        // Convert samples to f32 format required by Edge Impulse
        let samples: Vec<f32> = data
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / Self::I16_TO_F32_SCALE
            })
            .collect();

        // Clone Arc for the inference thread
        let model = Arc::clone(&self.model);

        // Run inference in background thread
        std::thread::spawn(move || {
            let mut model = model.lock().unwrap();
            if let Some(ref mut model) = *model {
                match model.classify(samples, None) {
                    Ok(result) => {
                        gst::info!(CAT, "Inference result: {:?}", result);
                    }
                    Err(e) => {
                        gst::error!(CAT, "Failed to run inference: {}", e);
                    }
                }
            }
        });

        Ok(gst::FlowSuccess::Ok)
    }

    fn set_caps(&self, incaps: &gst::Caps, _outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        gst::info!(CAT, obj = self.obj(), "set_caps called with incaps: {}", incaps);

        let audio_info = AudioInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse input caps"))?;

        gst::info!(CAT, obj = self.obj(), "Audio info: rate={}, channels={}, format={:?}",
            audio_info.rate(), audio_info.channels(), audio_info.format());

        let mut settings = self.settings.lock().unwrap();
        settings.input_size = Some(audio_info.bpf() as usize);
        settings.sensor_type = Some(SensorType::Microphone);

        gst::info!(CAT, obj = self.obj(), "State initialized with input_size: {}", settings.input_size.unwrap());
        Ok(())
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
        gst::info!(CAT, obj = self.obj(), "EdgeImpulseInfer constructed");
        self.parent_constructed();
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        gst::info!(CAT, obj = self.obj(), "set_property called for {}", pspec.name());
        match pspec.name() {
            "model-path" => {
                let mut settings = self.settings.lock().unwrap();
                settings.model_path = value.get().expect("type checked upstream");
                gst::info!(CAT, obj = self.obj(), "Model path set to: {:?}", settings.model_path);

                // Try to load the model
                if let Some(path) = &settings.model_path {
                    gst::info!(CAT, obj = self.obj(), "Attempting to load model from {}", path);
                    match EimModel::new(path) {
                        Ok(model) => {
                            gst::info!(CAT, obj = self.obj(), "Model loaded successfully");
                            let mut model_guard = self.model.lock().unwrap();
                            *model_guard = Some(model);
                        }
                        Err(e) => {
                            gst::error!(CAT, obj = self.obj(), "Failed to load model: {}", e);
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
}

impl ElementImpl for EdgeImpulseInfer {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Inference",
                "Filter/Audio",
                "Runs Edge Impulse ML inference on audio data",
                "Your Name <your.email@example.com>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("audio/x-raw")
                .field("format", AudioFormat::S16le.to_str())
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

impl GstObjectImpl for EdgeImpulseInfer {}