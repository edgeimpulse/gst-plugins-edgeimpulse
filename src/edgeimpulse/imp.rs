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
use edge_impulse_runner::{EimModel, SensorType, InferenceResult};
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
    sensor_type: Option<SensorType>,
    // Audio specific
    sample_rate: Option<i32>,
    samples_per_inference: Option<usize>,
    sample_buffer: VecDeque<i16>,
    // Image/Video specific
    width: Option<u32>,
    height: Option<u32>,
    channels: Option<u32>,
    frames_needed: Option<u32>,
    frame_buffer: VecDeque<Vec<f32>>,
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

    fn process_audio(
        &self,
        state: &mut State,
        data: &[u8],
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let samples_needed = state.samples_per_inference.ok_or_else(|| {
            gst::error!(CAT, obj = self.obj(), "Samples per inference not set");
            gst::FlowError::Error
        })?;

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
                self.run_inference(model, inference_samples)?;
            }
            state.sample_buffer.clear();
        }

        Ok(gst::FlowSuccess::Ok)
    }

    fn process_video(
        &self,
        state: &mut State,
        data: &[u8],
        _inbuf: &gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let width = state.width.ok_or_else(|| {
            gst::error!(CAT, obj = self.obj(), "Width not set");
            gst::FlowError::Error
        })?;
        let height = state.height.ok_or_else(|| {
            gst::error!(CAT, obj = self.obj(), "Height not set");
            gst::FlowError::Error
        })?;
        let channels = state.channels.ok_or_else(|| {
            gst::error!(CAT, obj = self.obj(), "Channels not set");
            gst::FlowError::Error
        })?;
        let frames_needed = state.frames_needed.unwrap_or(1);

        // Process frame data into features
        let features = self.process_frame(data, width, height, channels)?;

        // Add frame to buffer
        state.frame_buffer.push_back(features);

        // Keep only needed frames
        while state.frame_buffer.len() > frames_needed as usize {
            state.frame_buffer.pop_front();
        }

        // Run inference if we have enough frames
        if state.frame_buffer.len() >= frames_needed as usize {
            // Concatenate all frames
            let inference_features: Vec<f32> = state.frame_buffer
                .iter()
                .flat_map(|frame| frame.iter().copied())
                .collect();

            if let Some(model) = &mut state.model {
                self.run_inference(model, inference_features)?;
            }

            if frames_needed == 1 {
                state.frame_buffer.clear();
            }
        }

        Ok(gst::FlowSuccess::Ok)
    }

    fn process_frame(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Vec<f32>, gst::FlowError> {
        let mut features = Vec::with_capacity((width * height) as usize);

        // Process RGB or grayscale
        if channels == 3 {
            for chunk in data.chunks(3) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let feature = ((r as u32) << 16) + ((g as u32) << 8) + (b as u32);
                features.push(feature as f32);
            }
        } else {
            for &pixel in data {
                let feature = ((pixel as u32) << 16) + ((pixel as u32) << 8) + (pixel as u32);
                features.push(feature as f32);
            }
        }

        Ok(features.into_iter().map(|x| x / 255.0).collect())
    }

    fn run_inference(
        &self,
        model: &mut EimModel,
        features: Vec<f32>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        match model.classify(features, None) {
            Ok(result) => {
                // Create message structure based on result type
                let mut s = gst::Structure::builder("edge-impulse-inference-result").build();

                match result.result {
                    InferenceResult::Classification { classification } => {
                        let mut class_struct = gst::Structure::new_empty("classification");
                        for (label, confidence) in classification {
                            class_struct.set(label, confidence);
                        }
                        s.set("classification", class_struct);
                    }
                    InferenceResult::ObjectDetection { bounding_boxes, classification } => {
                        // Convert bounding boxes to array value
                        let boxes: Vec<gst::Structure> = bounding_boxes.into_iter()
                            .map(|bb| {
                                gst::Structure::builder("bbox")
                                    .field("label", bb.label)
                                    .field("confidence", bb.value)
                                    .field("x", bb.x)
                                    .field("y", bb.y)
                                    .field("width", bb.width)
                                    .field("height", bb.height)
                                    .build()
                            })
                            .collect();

                        // Convert Vec<Structure> to array value
                        let boxes_array = gst::Array::from_iter(
                            boxes.into_iter()
                                .map(|s| s.to_send_value())
                        );
                        s.set("bounding_boxes", boxes_array);

                        // Add classification if present
                        if !classification.is_empty() {
                            let mut class_struct = gst::Structure::new_empty("classification");
                            for (label, confidence) in classification {
                                class_struct.set(label, confidence);
                            }
                            s.set("classification", class_struct);
                        }
                    }
                }

                let msg = gst::message::Element::new(s);
                let _ = self.obj().post_message(msg);
                Ok(gst::FlowSuccess::Ok)
            }
            Err(e) => {
                gst::error!(CAT, obj = self.obj(), "Inference failed: {}", e);
                Err(gst::FlowError::Error)
            }
        }
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
        let mut state = self.state.lock().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Failed to lock state");
            gst::FlowError::Error
        })?;

        // Get sensor type first to avoid borrow issues
        let sensor_type = state.sensor_type;

        // Map buffers
        let data = inbuf.map_readable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Can't map buffer readable");
            gst::FlowError::Error
        })?;
        let mut out_data = outbuf.map_writable().map_err(|_| {
            gst::error!(CAT, obj = self.obj(), "Can't map buffer writable");
            gst::FlowError::Error
        })?;
        out_data.copy_from_slice(&data);

        // Process based on sensor type
        match sensor_type {
            Some(SensorType::Microphone) => self.process_audio(&mut state, &data),
            Some(SensorType::Camera) => self.process_video(&mut state, &data, inbuf),
            _ => {
                gst::error!(CAT, obj = self.obj(), "Unsupported sensor type");
                Err(gst::FlowError::Error)
            }
        }
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
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "model-path" => {
                let mut settings = self.settings.lock().unwrap();
                let mut state = self.state.lock().unwrap();

                // Only create new model if path changed
                if settings.model_path != value.get().expect("type checked upstream") {
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
                                match model.parameters() {
                                    Ok(params) => {
                                        // Store the parameters we need
                                        let frequency = params.frequency;
                                        let slice_size = params.slice_size;
                                        let sensor_type = params.sensor;

                                        // Set sensor type based on model parameters
                                        state.sensor_type = match SensorType::try_from(sensor_type) {
                                            Ok(sensor_type) => Some(sensor_type),
                                            Err(e) => {
                                                gst::error!(CAT, obj = self.obj(),
                                                    "Invalid sensor type {}: {}", sensor_type, e);
                                                None
                                            }
                                        };

                                        // Set audio-specific parameters
                                        if sensor_type == 1 {  // Microphone
                                            state.sample_rate = Some(frequency as i32);
                                            state.samples_per_inference = Some(slice_size);
                                        }
                                        // Set video-specific parameters
                                        else if sensor_type == 2 {  // Camera
                                            state.width = Some(params.image_input_width);
                                            state.height = Some(params.image_input_height);
                                            state.channels = Some(params.image_channel_count);
                                            state.frames_needed = Some(params.image_input_frames);
                                        }

                                        state.model = Some(model);

                                        gst::info!(CAT, obj = self.obj(),
                                            "Model loaded: sensor_type={:?}, frequency={}Hz, slice_size={} samples",
                                            state.sensor_type,
                                            frequency,
                                            slice_size
                                        );
                                    }
                                    Err(e) => {
                                        gst::error!(CAT, obj = self.obj(),
                                            "Failed to get model parameters: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                gst::error!(CAT, obj = self.obj(),
                                    "Failed to load model: {}", e);
                            }
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
            // Audio caps
            let audio_caps = gst::Caps::builder("audio/x-raw")
                .field("format", "S16LE")
                .field("rate", gst::IntRange::new(8000, 48000))
                .field("channels", 1i32)
                .field("layout", "interleaved")
                .build();

            // Video caps
            let video_caps = gst::Caps::builder("video/x-raw")
                .field("format", gst::List::new(["RGB", "RGBA", "BGR", "BGRA"]))
                .field("width", gst::IntRange::new(1, i32::MAX))
                .field("height", gst::IntRange::new(1, i32::MAX))
                .field("framerate", gst::FractionRange::new(
                    gst::Fraction::new(0, 1),
                    gst::Fraction::new(i32::MAX, 1)
                ))
                .build();

            // Combine caps
            let caps = gst::Caps::builder_full()
                .structure(audio_caps.structure(0).unwrap().to_owned())
                .structure(video_caps.structure(0).unwrap().to_owned())
                .build();

            // Create pad templates
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