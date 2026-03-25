//! # InferenceResultMeta — Generic inference result metadata for any buffer
//!
//! A media-agnostic metadata type attached to both audio and video buffers by
//! [`edgeimpulsevideoinfer`](crate::video) and [`edgeimpulseaudioinfer`](crate::audio).
//! Provides a unified interface for downstream elements like
//! [`edgeimpulsecontinueif`](crate::filter) to read inference results without
//! knowing the media type.
//!
//! This is an **additional convenience layer** for flow-control elements and
//! does **not** replace the video-specific metadata types
//! (`VideoRegionOfInterestMeta`, `VideoClassificationMeta`, `VideoAnomalyMeta`).
//! Those remain the **primary API** for downstream consumers — including
//! `edgeimpulseoverlay`, `edgeimpulsecrop`, and external elements such as
//! Qualcomm IM SDK's `qtioverlay`.
//!
//! ## Fields
//!
//! | Field | Type | Description |
//! |-------|------|-------------|
//! | `inference_type` | `String` | `"object-detection"`, `"classification"`, `"anomaly-detection"`, etc. |
//! | `result_json` | `String` | Raw JSON result string from the model |
//! | `detection_count` | `u32` | Number of detected bounding boxes |
//! | `max_confidence` | `f64` | Highest confidence across all detections/classifications |
//! | `top_class` | `String` | Label of the highest-confidence class |
//! | `top_confidence` | `f64` | Confidence of `top_class` |
//! | `anomaly_score` | `f64` | Overall anomaly score (`0.0` if not anomaly) |
//! | `visual_anomaly_max` | `f64` | Peak visual anomaly grid score (`0.0` if not visual anomaly) |
//!
//! ## How it is populated
//!
//! The [`populate_from_result`] helper parses the raw JSON result string and
//! fills in the summary fields. Both `edgeimpulsevideoinfer` and
//! `edgeimpulseaudioinfer` call this after each inference, before posting
//! the bus message.
//!
//! ## Reading metadata downstream
//!
//! ```rust,ignore
//! use gstreamer::prelude::*;
//! use crate::meta::InferenceResultMeta;
//!
//! // Inside a transform or probe callback:
//! if let Some(meta) = buffer.meta::<InferenceResultMeta>() {
//!     println!("type={}, detections={}, top={}({:.0}%)",
//!         meta.inference_type(),
//!         meta.detection_count(),
//!         meta.top_class(),
//!         meta.top_confidence() * 100.0,
//!     );
//! }
//! ```

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use std::fmt;
use std::ptr;

/// Public Rust wrapper for the inference result metadata.
#[repr(C)]
pub struct InferenceResultMeta(imp::InferenceResultMeta);

unsafe impl Send for InferenceResultMeta {}
unsafe impl Sync for InferenceResultMeta {}

impl InferenceResultMeta {
    pub fn add(buffer: &mut gst::BufferRef) -> gst::MetaRefMut<'_, Self, gst::meta::Standalone> {
        unsafe {
            let meta = gst::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::inference_result_meta_get_info(),
                ptr::null_mut(),
            ) as *mut imp::InferenceResultMeta;

            {
                let meta = &mut *meta;
                meta.inference_type = String::new();
                meta.result_json = String::new();
                meta.detection_count = 0;
                meta.max_confidence = 0.0;
                meta.top_class = String::new();
                meta.top_confidence = 0.0;
                meta.anomaly_score = 0.0;
                meta.visual_anomaly_max = 0.0;
            }

            Self::from_mut_ptr(buffer, meta)
        }
    }

    /// Inference type: "object-detection", "classification", "anomaly-detection", etc.
    pub fn inference_type(&self) -> &str {
        &self.0.inference_type
    }
    pub fn set_inference_type(&mut self, v: String) {
        self.0.inference_type = v;
    }

    /// Raw JSON result string from the model
    pub fn result_json(&self) -> &str {
        &self.0.result_json
    }
    pub fn set_result_json(&mut self, v: String) {
        self.0.result_json = v;
    }

    /// Number of detected objects (bounding boxes)
    pub fn detection_count(&self) -> u32 {
        self.0.detection_count
    }
    pub fn set_detection_count(&mut self, v: u32) {
        self.0.detection_count = v;
    }

    /// Highest confidence score across all detections/classifications
    pub fn max_confidence(&self) -> f64 {
        self.0.max_confidence
    }
    pub fn set_max_confidence(&mut self, v: f64) {
        self.0.max_confidence = v;
    }

    /// Top class label (highest confidence)
    pub fn top_class(&self) -> &str {
        &self.0.top_class
    }
    pub fn set_top_class(&mut self, v: String) {
        self.0.top_class = v;
    }

    /// Confidence of the top class
    pub fn top_confidence(&self) -> f64 {
        self.0.top_confidence
    }
    pub fn set_top_confidence(&mut self, v: f64) {
        self.0.top_confidence = v;
    }

    /// Anomaly score (0.0 if not anomaly detection)
    pub fn anomaly_score(&self) -> f64 {
        self.0.anomaly_score
    }
    pub fn set_anomaly_score(&mut self, v: f64) {
        self.0.anomaly_score = v;
    }

    /// Visual anomaly max score (0.0 if not visual anomaly)
    pub fn visual_anomaly_max(&self) -> f64 {
        self.0.visual_anomaly_max
    }
    pub fn set_visual_anomaly_max(&mut self, v: f64) {
        self.0.visual_anomaly_max = v;
    }
}

unsafe impl gst::prelude::MetaAPI for InferenceResultMeta {
    type GstType = imp::InferenceResultMeta;

    fn meta_api() -> glib::Type {
        imp::inference_result_meta_api_get_type()
    }
}

impl fmt::Debug for InferenceResultMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("InferenceResultMeta")
            .field("type", &self.inference_type())
            .field("detection_count", &self.detection_count())
            .field("top_class", &self.top_class())
            .field("top_confidence", &self.top_confidence())
            .field("max_confidence", &self.max_confidence())
            .field("anomaly_score", &self.anomaly_score())
            .finish()
    }
}

/// Helper to populate InferenceResultMeta from the raw JSON result.
/// Call this after inference, before posting the bus message.
pub fn populate_from_result(
    meta: &mut gst::MetaRefMut<'_, InferenceResultMeta, gst::meta::Standalone>,
    inference_type: &str,
    result_json: &str,
) {
    meta.set_inference_type(inference_type.to_string());
    meta.set_result_json(result_json.to_string());

    let result: serde_json::Value = match serde_json::from_str(result_json) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Object detection
    if let Some(boxes) = result.get("bounding_boxes").and_then(|b| b.as_array()) {
        meta.set_detection_count(boxes.len() as u32);
        let mut max_conf = 0.0_f64;
        let mut top_label = String::new();
        for bbox in boxes {
            if let Some(conf) = bbox.get("value").and_then(|v| v.as_f64()) {
                if conf > max_conf {
                    max_conf = conf;
                    if let Some(label) = bbox.get("label").and_then(|l| l.as_str()) {
                        top_label = label.to_string();
                    }
                }
            }
        }
        meta.set_max_confidence(max_conf);
        meta.set_top_class(top_label);
        meta.set_top_confidence(max_conf);
    }

    // Classification
    if let Some(classification) = result.get("classification").and_then(|c| c.as_object()) {
        let mut best_label = String::new();
        let mut best_conf = 0.0_f64;
        for (label, score) in classification {
            if let Some(conf) = score.as_f64() {
                if conf > best_conf {
                    best_conf = conf;
                    best_label = label.clone();
                }
            }
        }
        if best_conf > meta.max_confidence() {
            meta.set_max_confidence(best_conf);
        }
        meta.set_top_class(best_label);
        meta.set_top_confidence(best_conf);
    }

    // Anomaly
    if let Some(anomaly) = result.get("anomaly").and_then(|a| a.as_f64()) {
        meta.set_anomaly_score(anomaly);
    }
    if let Some(vam) = result.get("visual_anomaly_max").and_then(|v| v.as_f64()) {
        meta.set_visual_anomaly_max(vam);
    }
}

// ─── Unsafe meta registration ────────────────────────────────────────────────

mod imp {
    use super::*;
    use glib::translate::*;
    use once_cell::sync::Lazy;
    use std::ffi::CString;
    use std::mem;

    include!(concat!(env!("OUT_DIR"), "/type_names.rs"));

    #[repr(C)]
    pub struct InferenceResultMeta {
        pub(super) meta: gst::ffi::GstMeta,
        pub(super) inference_type: String,
        pub(super) result_json: String,
        pub(super) detection_count: u32,
        pub(super) max_confidence: f64,
        pub(super) top_class: String,
        pub(super) top_confidence: f64,
        pub(super) anomaly_score: f64,
        pub(super) visual_anomaly_max: f64,
    }

    pub(super) fn inference_result_meta_api_get_type() -> glib::Type {
        static TYPE: Lazy<glib::Type> = Lazy::new(|| unsafe {
            let name =
                CString::new(INFERENCE_RESULT_META_API_NAME).expect("Failed to create CString");
            let t = from_glib(gst::ffi::gst_meta_api_type_register(
                name.as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));
            assert_ne!(t, glib::Type::INVALID);
            t
        });
        *TYPE
    }

    unsafe extern "C" fn inference_result_meta_init(
        meta: *mut gst::ffi::GstMeta,
        _params: glib::ffi::gpointer,
        _buffer: *mut gst::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        let meta = &mut *(meta as *mut InferenceResultMeta);
        ptr::write(&mut meta.inference_type, String::new());
        ptr::write(&mut meta.result_json, String::new());
        ptr::write(&mut meta.detection_count, 0);
        ptr::write(&mut meta.max_confidence, 0.0);
        ptr::write(&mut meta.top_class, String::new());
        ptr::write(&mut meta.top_confidence, 0.0);
        ptr::write(&mut meta.anomaly_score, 0.0);
        ptr::write(&mut meta.visual_anomaly_max, 0.0);
        true.into_glib()
    }

    unsafe extern "C" fn inference_result_meta_free(
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut InferenceResultMeta);
        ptr::drop_in_place(&mut meta.inference_type);
        ptr::drop_in_place(&mut meta.result_json);
        ptr::drop_in_place(&mut meta.top_class);
    }

    unsafe extern "C" fn inference_result_meta_transform(
        dest: *mut gst::ffi::GstBuffer,
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &*(meta as *const InferenceResultMeta);
        let mut new = super::InferenceResultMeta::add(gst::BufferRef::from_mut_ptr(dest));
        new.0.inference_type = meta.inference_type.clone();
        new.0.result_json = meta.result_json.clone();
        new.0.detection_count = meta.detection_count;
        new.0.max_confidence = meta.max_confidence;
        new.0.top_class = meta.top_class.clone();
        new.0.top_confidence = meta.top_confidence;
        new.0.anomaly_score = meta.anomaly_score;
        new.0.visual_anomaly_max = meta.visual_anomaly_max;
        true.into_glib()
    }

    pub(super) fn inference_result_meta_get_info() -> *const gst::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gst::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: Lazy<MetaInfo> = Lazy::new(|| unsafe {
            let name = CString::new(INFERENCE_RESULT_META_NAME).expect("Failed to create CString");
            MetaInfo(
                ptr::NonNull::new(gst::ffi::gst_meta_register(
                    inference_result_meta_api_get_type().into_glib(),
                    name.as_ptr() as *const _,
                    mem::size_of::<InferenceResultMeta>(),
                    Some(inference_result_meta_init),
                    Some(inference_result_meta_free),
                    Some(inference_result_meta_transform),
                ) as *mut gst::ffi::GstMetaInfo)
                .expect("Failed to register InferenceResultMeta"),
            )
        });

        META_INFO.0.as_ptr()
    }
}
