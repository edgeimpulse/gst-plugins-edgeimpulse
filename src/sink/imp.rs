use edge_impulse_runner::ingestion::{Category, Ingestion, UploadOptions};
use glib::ParamSpec;
use glib::ParamSpecString;
use glib::Value;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_base as gst_base;
use gstreamer_base::subclass::prelude::*;
use hound;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageBuffer, ImageEncoder, Rgb, Rgba};
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tempfile::NamedTempFile;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    let variant = env!("PLUGIN_VARIANT");
    let name = if variant.is_empty() {
        "edgeimpulsesink".to_string()
    } else {
        format!("edgeimpulsesink_{}", variant)
    };
    gst::DebugCategory::new(
        &name,
        gst::DebugColorFlags::empty(),
        Some("Edge Impulse Sink"),
    )
});

#[derive(Default)]
pub struct EdgeImpulseSink {
    pub api_key: Mutex<Option<String>>,
    pub hmac_key: Mutex<Option<String>>,
    pub label: Mutex<Option<String>>,
    pub category: Mutex<Option<String>>,
    pub ingestion: Mutex<Option<Arc<Ingestion>>>,
    pub upload_interval_ms: Mutex<u32>,
    // Audio accumulation
    pub audio_buffer: Mutex<Vec<u8>>,
    pub audio_last_upload: Mutex<Option<Instant>>,
    pub audio_sample_rate: Mutex<Option<u32>>,
    // Video accumulation
    pub video_last_upload: Mutex<Option<Instant>>,
}

// Helper to create the ingestion message structure
fn create_ingestion_message_structure_with_error(
    filename: &str,
    media_type: &str,
    label: &Option<String>,
    category: &str,
    error: Option<&str>,
    length: Option<u64>,
) -> gst::Structure {
    let mut builder = gst::Structure::builder(if error.is_some() {
        "edge-impulse-ingestion-error"
    } else {
        "edge-impulse-ingestion-result"
    })
    .field("filename", filename)
    .field("media_type", media_type)
    .field("label", label)
    .field("category", category);
    if let Some(e) = error {
        builder = builder.field("error", e);
    }
    if let Some(l) = length {
        builder = builder.field("length", l);
    }
    builder.build()
}

fn post_ingestion_success(
    obj: &gst::Element,
    filename: &str,
    media_type: &str,
    length: u64,
    label: &Option<String>,
    category: &str,
) {
    let s = create_ingestion_message_structure_with_error(
        filename,
        media_type,
        label,
        category,
        None,
        Some(length),
    );
    let _ = obj.post_message(gst::message::Element::new(s));
}

fn post_ingestion_error(
    obj: &gst::Element,
    filename: &str,
    media_type: &str,
    label: &Option<String>,
    category: &str,
    error: &str,
) {
    let s = create_ingestion_message_structure_with_error(
        filename,
        media_type,
        label,
        category,
        Some(error),
        None,
    );
    let _ = obj.post_message(gst::message::Element::new(s));
}

// Include generated type names for variant-specific builds
include!(concat!(env!("OUT_DIR"), "/type_names.rs"));

#[glib::object_subclass]
impl ObjectSubclass for EdgeImpulseSink {
    const NAME: &'static str = SINK_TYPE_NAME;
    type Type = super::EdgeImpulseSink;
    type ParentType = gst_base::BaseSink;
}

impl ObjectImpl for EdgeImpulseSink {
    fn properties() -> &'static [ParamSpec] {
        static PROPERTIES: Lazy<Vec<ParamSpec>> = Lazy::new(|| {
            vec![
                ParamSpecString::builder("api-key")
                    .nick("API Key")
                    .blurb("Edge Impulse API key")
                    .build(),
                ParamSpecString::builder("hmac-key")
                    .nick("HMAC Key")
                    .blurb("Optional HMAC key for signing requests")
                    .build(),
                ParamSpecString::builder("label")
                    .nick("Label")
                    .blurb("Optional label for the file or sample")
                    .build(),
                ParamSpecString::builder("category")
                    .nick("Category")
                    .blurb("Category: training, testing, or anomaly")
                    .default_value(Some("training"))
                    .build(),
                glib::ParamSpecUInt::builder("upload-interval-ms")
                    .nick("Upload Interval (ms)")
                    .blurb("Minimum interval in milliseconds between uploads (0 = every buffer)")
                    .default_value(0)
                    .minimum(0)
                    .maximum(u32::MAX)
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &Value, pspec: &ParamSpec) {
        match pspec.name() {
            "api-key" => {
                let mut v = self.api_key.lock().unwrap();
                *v = value.get().ok();
            }
            "hmac-key" => {
                let mut v = self.hmac_key.lock().unwrap();
                *v = value.get().ok();
            }
            "label" => {
                let mut v = self.label.lock().unwrap();
                *v = value.get().ok();
            }
            "category" => {
                let mut v = self.category.lock().unwrap();
                *v = value.get().ok();
            }
            "upload-interval-ms" => {
                let mut v = self.upload_interval_ms.lock().unwrap();
                *v = value.get().unwrap_or(0);
            }
            _ => {}
        }
    }

    fn property(&self, _id: usize, pspec: &ParamSpec) -> Value {
        match pspec.name() {
            "api-key" => self.api_key.lock().unwrap().to_value(),
            "hmac-key" => self.hmac_key.lock().unwrap().to_value(),
            "label" => self.label.lock().unwrap().to_value(),
            "category" => self.category.lock().unwrap().to_value(),
            "upload-interval-ms" => self.upload_interval_ms.lock().unwrap().to_value(),
            _ => Value::from_type(pspec.value_type()),
        }
    }
}

impl GstObjectImpl for EdgeImpulseSink {}
impl ElementImpl for EdgeImpulseSink {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Edge Impulse Sink",
                "Sink/AI",
                "Uploads audio or video frames to Edge Impulse via ingestion API",
                "Fernando Jiménez Moreno <fernando@edgeimpulse.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let audio_caps = gst::Caps::builder("audio/x-raw")
                .field("format", "S16LE")
                .field("channels", gst::IntRange::<i32>::new(1, 2))
                .field("rate", gst::IntRange::<i32>::new(1, 192000))
                .build();
            let video_caps = gst::Caps::builder("video/x-raw")
                .field("format", gst::List::new(["RGB", "RGBA"]))
                .field("width", gst::IntRange::<i32>::new(1, 4096))
                .field("height", gst::IntRange::<i32>::new(1, 4096))
                .build();
            let mut caps = gst::Caps::new_empty();
            caps.merge(audio_caps);
            caps.merge(video_caps);
            vec![gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap()]
        });
        PAD_TEMPLATES.as_ref()
    }
}

impl BaseSinkImpl for EdgeImpulseSink {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let api_key = self.api_key.lock().unwrap().clone();
        let hmac_key = self.hmac_key.lock().unwrap().clone();
        if let Some(api_key) = api_key {
            let mut ingestion = Ingestion::new(api_key);
            if let Some(hmac) = hmac_key {
                ingestion = ingestion.with_hmac(hmac);
            }
            gst::info!(CAT, "Ingestion client initialized");
            *self.ingestion.lock().unwrap() = Some(Arc::new(ingestion));
        } else {
            gst::warning!(CAT, "No API key set, ingestion client not initialized");
            *self.ingestion.lock().unwrap() = None;
        }
        *self.audio_buffer.lock().unwrap() = Vec::new();
        *self.audio_last_upload.lock().unwrap() = None;
        *self.audio_sample_rate.lock().unwrap() = None;
        *self.video_last_upload.lock().unwrap() = None;

        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        gst::info!(CAT, "Ingestion client cleared");
        *self.ingestion.lock().unwrap() = None;
        *self.audio_buffer.lock().unwrap() = Vec::new();
        *self.audio_last_upload.lock().unwrap() = None;
        *self.audio_sample_rate.lock().unwrap() = None;
        *self.video_last_upload.lock().unwrap() = None;
        Ok(())
    }

    fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
        let ingestion = self.ingestion.lock().unwrap().as_ref().cloned();
        let label = self.label.lock().unwrap().clone();
        let category = self
            .category
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| "training".to_string());
        let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
        let data = map.as_slice();
        let pad = self.obj().static_pad("sink").ok_or(gst::FlowError::Error)?;
        let caps = pad.current_caps().ok_or(gst::FlowError::Error)?;
        let structure = caps.structure(0).ok_or(gst::FlowError::Error)?;
        let media_type = structure.name();
        let path;
        let tmpfile;
        if media_type.starts_with("audio/x-raw") {
            let sample_rate = structure
                .get::<i32>("rate")
                .map_err(|_| gst::FlowError::Error)? as u32;
            *self.audio_sample_rate.lock().unwrap() = Some(sample_rate);
            let mut audio_buffer = self.audio_buffer.lock().unwrap();
            audio_buffer.extend_from_slice(data);
            let interval_ms = *self.upload_interval_ms.lock().unwrap() as u64;
            let now = Instant::now();
            let mut last_upload = self.audio_last_upload.lock().unwrap();
            let enough = if interval_ms == 0 {
                true
            } else if let Some(last) = *last_upload {
                now.duration_since(last).as_millis() as u64 >= interval_ms
            } else {
                true
            };
            if !enough {
                return Ok(gst::FlowSuccess::Ok);
            }
            *last_upload = Some(now);
            tmpfile = NamedTempFile::with_suffix(".wav").map_err(|_| gst::FlowError::Error)?;
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer =
                hound::WavWriter::new(&tmpfile, spec).map_err(|_| gst::FlowError::Error)?;
            for chunk in audio_buffer.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                writer
                    .write_sample(sample)
                    .map_err(|_| gst::FlowError::Error)?;
            }
            writer.finalize().map_err(|_| gst::FlowError::Error)?;
            let audio_length = (audio_buffer.len() as u64) / 2 * 1000 / sample_rate as u64; // ms
            audio_buffer.clear();
            path = tmpfile.path().to_path_buf();
            let label_clone = label.clone();
            let category_clone = category.clone();
            let media_type_str = media_type.to_string();
            let filename = path.to_string_lossy().to_string();
            let obj = self.obj().clone().upcast::<gst::Element>();
            std::thread::spawn(move || {
                let _tmpfile = tmpfile;
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                    if let Some(ingestion) = ingestion {
                        let cat = match category_clone.to_lowercase().as_str() {
                            "training" => Category::Training,
                            "testing" => Category::Testing,
                            "anomaly" => Category::Anomaly,
                            _ => Category::Training,
                        };
                        let options = UploadOptions::default();
                        match ingestion
                            .upload_file(&path, cat, label_clone.clone(), Some(options))
                            .await
                        {
                            Ok(_resp) => {
                                post_ingestion_success(
                                    &obj,
                                    &filename,
                                    &media_type_str,
                                    audio_length,
                                    &label_clone,
                                    &category_clone,
                                );
                            }
                            Err(e) => {
                                gst::error!(CAT, "upload error: {e:?}");
                                post_ingestion_error(
                                    &obj,
                                    &filename,
                                    &media_type_str,
                                    &label_clone,
                                    &category_clone,
                                    &format!("{e:?}"),
                                );
                            }
                        }
                    } else {
                        gst::warning!(CAT, "ingestion client not initialized");
                    }
                });
            });
            return Ok(gst::FlowSuccess::Ok);
        } else if media_type.starts_with("video/x-raw") {
            let interval_ms = *self.upload_interval_ms.lock().unwrap() as u64;
            let now = Instant::now();
            let mut last_upload = self.video_last_upload.lock().unwrap();
            let enough = if interval_ms == 0 {
                true
            } else if let Some(last) = *last_upload {
                now.duration_since(last).as_millis() as u64 >= interval_ms
            } else {
                true
            };
            if !enough {
                return Ok(gst::FlowSuccess::Ok);
            }
            *last_upload = Some(now);
            let width = structure
                .get::<i32>("width")
                .map_err(|_| gst::FlowError::Error)? as u32;
            let height = structure
                .get::<i32>("height")
                .map_err(|_| gst::FlowError::Error)? as u32;
            let format = structure
                .get::<&str>("format")
                .map_err(|_| gst::FlowError::Error)?;
            tmpfile = NamedTempFile::with_suffix(".png").map_err(|_| gst::FlowError::Error)?;
            let mut file = tmpfile.reopen().map_err(|_| gst::FlowError::Error)?;
            match format {
                "RGB" => {
                    let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data)
                        .ok_or(gst::FlowError::Error)?;
                    let encoder = PngEncoder::new(&mut file);
                    encoder
                        .write_image(img.as_raw(), width, height, ColorType::Rgb8)
                        .map_err(|_| gst::FlowError::Error)?;
                }
                "RGBA" => {
                    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data)
                        .ok_or(gst::FlowError::Error)?;
                    let encoder = PngEncoder::new(&mut file);
                    encoder
                        .write_image(img.as_raw(), width, height, ColorType::Rgba8)
                        .map_err(|_| gst::FlowError::Error)?;
                }
                _ => return Err(gst::FlowError::Error),
            }
            path = tmpfile.path().to_path_buf();
            let label_clone = label.clone();
            let category_clone = category.clone();
            let media_type_str = media_type.to_string();
            let filename = path.to_string_lossy().to_string();
            let obj = self.obj().clone().upcast::<gst::Element>();
            std::thread::spawn(move || {
                let _tmpfile = tmpfile;
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                    if let Some(ingestion) = ingestion {
                        let cat = match category_clone.to_lowercase().as_str() {
                            "training" => Category::Training,
                            "testing" => Category::Testing,
                            "anomaly" => Category::Anomaly,
                            _ => Category::Training,
                        };
                        let options = UploadOptions::default();
                        match ingestion
                            .upload_file(&path, cat, label_clone.clone(), Some(options))
                            .await
                        {
                            Ok(_resp) => {
                                post_ingestion_success(
                                    &obj,
                                    &filename,
                                    &media_type_str,
                                    1,
                                    &label_clone,
                                    &category_clone,
                                );
                            }
                            Err(e) => {
                                gst::error!(CAT, "upload error: {e:?}");
                                post_ingestion_error(
                                    &obj,
                                    &filename,
                                    &media_type_str,
                                    &label_clone,
                                    &category_clone,
                                    &format!("{e:?}"),
                                );
                            }
                        }
                    } else {
                        gst::warning!(CAT, "ingestion client not initialized");
                    }
                });
            });
            return Ok(gst::FlowSuccess::Ok);
        }
        Err(gst::FlowError::Error)
    }
}
