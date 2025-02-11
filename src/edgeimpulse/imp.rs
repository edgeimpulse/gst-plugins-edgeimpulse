use gstreamer as gst;
use gstreamer_base::prelude::*;
use gstreamer_base::subclass::prelude::*;
use gstreamer_base::subclass::BaseTransformMode;
use gstreamer_audio::AudioInfo;
use gstreamer_audio::AudioFormat;
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
    window_size: usize,
    window_increment: usize,
}

#[derive(Default)]
pub struct EdgeImpulseInfer {
    settings: Mutex<Settings>,
    model: Arc<Mutex<Option<EimModel>>>,
}

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseInfer {
    const NAME: &'static str = "EdgeImpulseInfer";
    type Type = super::EdgeImpulseInfer;
    type ParentType = gstreamer_base::BaseTransform;
}

impl EdgeImpulseInfer {
}

impl BaseTransformImpl for EdgeImpulseInfer {
    const MODE: BaseTransformMode = BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn unit_size(&self, caps: &gst::Caps) -> Option<usize> {
        AudioInfo::from_caps(caps)
            .map(|info| info.bpf() as usize)
            .ok()
    }

    fn transform(
        &self,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::debug!(CAT, obj = self.obj(), "Processing buffer of size {}", inbuf.size());

        let data = inbuf.map_readable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to map input buffer readable");
            gst::FlowError::Error
        })?;

        // Convert samples to f32 in range [-1.0, 1.0]
        let samples: Vec<f32> = data
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                (sample as f32) / 32768.0
            })
            .collect();

        // Process samples with the model
        let mut model = self.model.lock().unwrap();
        if let Some(model) = model.as_mut() {
            match model.classify(samples, None) {
                Ok(response) => {
                    if response.success {
                        gst::debug!(CAT, obj = self.obj(), "Raw result: {:?}", response.result);

                        match response.result {
                            InferenceResult::Classification { classification } => {
                                for (label, score) in classification {
                                    gst::info!(CAT, obj = self.obj(), "{}: {:.2}%", label, score * 100.0);
                                }
                            }
                            InferenceResult::ObjectDetection { .. } => {
                                gst::warning!(CAT, obj = self.obj(), "Received object detection results for audio model");
                            }
                        }
                    }
                }
                Err(e) => {
                    gst::error!(CAT, obj = self.obj(), "Inference error: {}", e);
                    return Err(gst::FlowError::Error);
                }
            }
        }

        // Copy input to output
        let mut out_map = outbuf.map_writable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to map output buffer writable");
            gst::FlowError::Error
        })?;
        out_map.copy_from_slice(&data);

        Ok(gst::FlowSuccess::Ok)
    }

    fn set_caps(&self, incaps: &gst::Caps, _outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        gst::info!(CAT, obj = self.obj(), "set_caps called with incaps: {}", incaps);

        let audio_info = AudioInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "Failed to parse input caps"))?;

        gst::info!(CAT, obj = self.obj(), "Audio info: rate={}, channels={}, format={:?}",
            audio_info.rate(), audio_info.channels(), audio_info.format());

        let mut settings = self.settings.lock().unwrap();
        settings.window_size = 16000; // 1 second of audio at 16kHz
        settings.window_increment = settings.window_size / 2;

        gst::info!(CAT, obj = self.obj(), "State initialized with window_size: {}", settings.window_size);
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