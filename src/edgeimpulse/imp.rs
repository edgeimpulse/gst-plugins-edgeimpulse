use gst::prelude::*;
use gst::subclass::prelude::*;
use gstreamer as gst;
use gstreamer_audio as gst_audio;
use gstreamer_base::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use edge_impulse_runner::{EimModel, InferenceResult};
use gst::glib;
use once_cell::sync::Lazy;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "edgeimpulseinfer",
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Inference Element"),
    )
});

#[derive(Debug, Default)]
struct Settings {
    model_path: Option<String>,
}

struct State {
    buffer: VecDeque<i16>,
    window_size: usize,
}

#[derive(Default)]
pub struct EdgeImpulseInfer {
    settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
    model: Arc<Mutex<Option<EimModel>>>,
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseInfer {
    const NAME: &'static str = "EdgeImpulseInfer";
    type Type = super::EdgeImpulseInfer;
    type ParentType = gstreamer_base::BaseTransform;
}

impl EdgeImpulseInfer {
    fn transform_ip_impl(
        &self,
        element: &super::EdgeImpulseInfer,
        buf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::trace!(CAT, obj = element, "transform_ip_impl called");

        let mut state = self.state.lock().unwrap();
        let state = state.as_mut().unwrap();

        let samples = buf.map_readable().map_err(|_| {
            gst::error!(CAT, obj = element, "Can't map buffer readable");
            gst::FlowError::Error
        })?;

        gst::debug!(CAT, obj = element, "Processing buffer of size {} bytes", samples.len());

        // Convert to i16 and add to buffer
        let samples_i16: Vec<i16> = samples
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();

        gst::trace!(CAT, obj = element, "Converted {} bytes to {} i16 samples", samples.len(), samples_i16.len());

        state.buffer.extend(samples_i16);
        gst::debug!(CAT, obj = element, "Buffer size after adding samples: {} samples", state.buffer.len());

        // Check if we have enough samples
        if state.buffer.len() >= state.window_size {
            gst::debug!(CAT, obj = element, "Have enough samples ({} >= {}), running inference",
                state.buffer.len(), state.window_size);

            // Convert samples to f32
            let samples_f32: Vec<f32> = state.buffer
                .drain(0..state.window_size)
                .map(|s| s as f32 / 32768.0)
                .collect();

            gst::trace!(CAT, obj = element, "Converted {} samples to f32 for inference", samples_f32.len());

            // Run inference
            let mut model = self.model.lock().unwrap();
            gst::trace!(CAT, obj = element, "Got model lock, model is_some: {}", model.is_some());

            if let Some(ref mut model) = *model {
                gst::debug!(CAT, obj = element, "Running classification on {} samples", samples_f32.len());
                match model.classify(samples_f32, None) {
                    Ok(response) => {
                        gst::debug!(CAT, obj = element, "Classification completed, success: {}", response.success);
                        if !response.success {
                            gst::error!(CAT, obj = element, "Inference failed: success = false");
                            return Err(gst::FlowError::Error);
                        }
                        match response.result {
                            InferenceResult::Classification { classification } => {
                                gst::info!(CAT, obj = element, "Classification results:");
                                for (label, probability) in classification.iter() {
                                    gst::info!(CAT, obj = element, "  {}: {:.2}%", label, probability * 100.0);
                                }
                            }
                            InferenceResult::ObjectDetection { .. } => {
                                gst::error!(CAT, obj = element, "Unexpected object detection result");
                                return Err(gst::FlowError::Error);
                            }
                        }
                    }
                    Err(e) => {
                        gst::error!(CAT, obj = element, "Inference error: {}", e);
                        return Err(gst::FlowError::Error);
                    }
                }
            } else {
                gst::error!(CAT, obj = element, "Model not loaded");
                return Err(gst::FlowError::Error);
            }
        }

        Ok(gst::FlowSuccess::Ok)
    }
}

impl BaseTransformImpl for EdgeImpulseInfer {
    const MODE: BaseTransformMode = BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = true;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        self.transform_ip_impl(self.obj().upcast_ref(), buf)
    }

    fn set_caps(&self, incaps: &gst::Caps, _outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let _audio_info = gst_audio::AudioInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse input caps"))?;

        let mut state = self.state.lock().unwrap();
        *state = Some(State {
            buffer: VecDeque::new(),
            window_size: 16000, // 1 second of audio at 16kHz
        });

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
        gst::debug!(CAT, obj = self.obj(), "EdgeImpulseInfer constructed");
        self.parent_constructed();
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        gst::debug!(CAT, obj = self.obj(), "set_property called for {}", pspec.name());
        match pspec.name() {
            "model-path" => {
                let mut settings = self.settings.lock().unwrap();
                settings.model_path = value.get().expect("type checked upstream");
                gst::debug!(CAT, obj = self.obj(), "Model path set to: {:?}", settings.model_path);

                // Try to load the model
                if let Some(path) = &settings.model_path {
                    gst::debug!(CAT, obj = self.obj(), "Attempting to load model from {}", path);
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
        gst::trace!(CAT, obj = self.obj(), "property called for {}", pspec.name());
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
                .field("format", gst_audio::AudioFormat::S16le.to_str())
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