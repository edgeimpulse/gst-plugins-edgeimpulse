//! Custom GStreamer buffer metadata for tracking crop origin.
//!
//! Each cropped buffer carries a `CropOriginMeta` that records where the crop
//! came from in the original frame, so downstream classification results can
//! be mapped back to the full-frame coordinate space.

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use std::fmt;
use std::ptr;

/// Public Rust wrapper for the crop origin metadata.
#[repr(C)]
pub struct CropOriginMeta(imp::CropOriginMeta);

unsafe impl Send for CropOriginMeta {}
unsafe impl Sync for CropOriginMeta {}

impl CropOriginMeta {
    pub fn add(buffer: &mut gst::BufferRef) -> gst::MetaRefMut<'_, Self, gst::meta::Standalone> {
        unsafe {
            let meta = gst::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::crop_origin_meta_get_info(),
                ptr::null_mut(),
            ) as *mut imp::CropOriginMeta;

            {
                let meta = &mut *meta;
                meta.source_x = 0;
                meta.source_y = 0;
                meta.source_width = 0;
                meta.source_height = 0;
                meta.original_width = 0;
                meta.original_height = 0;
                meta.object_id = 0;
                meta.detection_label = String::new();
                meta.detection_confidence = 0.0;
            }

            Self::from_mut_ptr(buffer, meta)
        }
    }

    pub fn source_x(&self) -> u32 {
        self.0.source_x
    }
    pub fn set_source_x(&mut self, v: u32) {
        self.0.source_x = v;
    }

    pub fn source_y(&self) -> u32 {
        self.0.source_y
    }
    pub fn set_source_y(&mut self, v: u32) {
        self.0.source_y = v;
    }

    pub fn source_width(&self) -> u32 {
        self.0.source_width
    }
    pub fn set_source_width(&mut self, v: u32) {
        self.0.source_width = v;
    }

    pub fn source_height(&self) -> u32 {
        self.0.source_height
    }
    pub fn set_source_height(&mut self, v: u32) {
        self.0.source_height = v;
    }

    pub fn original_width(&self) -> u32 {
        self.0.original_width
    }
    pub fn set_original_width(&mut self, v: u32) {
        self.0.original_width = v;
    }

    pub fn original_height(&self) -> u32 {
        self.0.original_height
    }
    pub fn set_original_height(&mut self, v: u32) {
        self.0.original_height = v;
    }

    pub fn object_id(&self) -> u64 {
        self.0.object_id
    }
    pub fn set_object_id(&mut self, v: u64) {
        self.0.object_id = v;
    }

    pub fn detection_label(&self) -> &str {
        &self.0.detection_label
    }
    pub fn set_detection_label(&mut self, v: String) {
        self.0.detection_label = v;
    }

    pub fn detection_confidence(&self) -> f64 {
        self.0.detection_confidence
    }
    pub fn set_detection_confidence(&mut self, v: f64) {
        self.0.detection_confidence = v;
    }
}

unsafe impl gst::prelude::MetaAPI for CropOriginMeta {
    type GstType = imp::CropOriginMeta;

    fn meta_api() -> glib::Type {
        imp::crop_origin_meta_api_get_type()
    }
}

impl fmt::Debug for CropOriginMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("CropOriginMeta")
            .field(
                "source",
                &format!(
                    "{}x{}+{}+{}",
                    self.source_width(),
                    self.source_height(),
                    self.source_x(),
                    self.source_y()
                ),
            )
            .field(
                "original",
                &format!("{}x{}", self.original_width(), self.original_height()),
            )
            .field("object_id", &self.object_id())
            .field("label", &self.detection_label())
            .field("confidence", &self.detection_confidence())
            .finish()
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
    pub struct CropOriginMeta {
        pub(super) meta: gst::ffi::GstMeta,
        /// X offset of the crop in the original frame
        pub(super) source_x: u32,
        /// Y offset of the crop in the original frame
        pub(super) source_y: u32,
        /// Width of the crop region (before any resize)
        pub(super) source_width: u32,
        /// Height of the crop region (before any resize)
        pub(super) source_height: u32,
        /// Width of the original frame
        pub(super) original_width: u32,
        /// Height of the original frame
        pub(super) original_height: u32,
        /// Object tracking ID from upstream detection
        pub(super) object_id: u64,
        /// Detection label from upstream
        pub(super) detection_label: String,
        /// Detection confidence from upstream
        pub(super) detection_confidence: f64,
    }

    pub(super) fn crop_origin_meta_api_get_type() -> glib::Type {
        static TYPE: Lazy<glib::Type> = Lazy::new(|| unsafe {
            let name = CString::new(CROP_ORIGIN_META_API_NAME)
                .expect("Failed to create CString for CropOriginMetaAPI");
            let t = from_glib(gst::ffi::gst_meta_api_type_register(
                name.as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));
            assert_ne!(t, glib::Type::INVALID);
            t
        });
        *TYPE
    }

    unsafe extern "C" fn crop_origin_meta_init(
        meta: *mut gst::ffi::GstMeta,
        _params: glib::ffi::gpointer,
        _buffer: *mut gst::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        let meta = &mut *(meta as *mut CropOriginMeta);
        ptr::write(&mut meta.source_x, 0);
        ptr::write(&mut meta.source_y, 0);
        ptr::write(&mut meta.source_width, 0);
        ptr::write(&mut meta.source_height, 0);
        ptr::write(&mut meta.original_width, 0);
        ptr::write(&mut meta.original_height, 0);
        ptr::write(&mut meta.object_id, 0);
        ptr::write(&mut meta.detection_label, String::new());
        ptr::write(&mut meta.detection_confidence, 0.0);
        true.into_glib()
    }

    unsafe extern "C" fn crop_origin_meta_free(
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut CropOriginMeta);
        ptr::drop_in_place(&mut meta.detection_label);
    }

    unsafe extern "C" fn crop_origin_meta_transform(
        dest: *mut gst::ffi::GstBuffer,
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &*(meta as *const CropOriginMeta);
        let mut new_meta = super::CropOriginMeta::add(gst::BufferRef::from_mut_ptr(dest));
        new_meta.0.source_x = meta.source_x;
        new_meta.0.source_y = meta.source_y;
        new_meta.0.source_width = meta.source_width;
        new_meta.0.source_height = meta.source_height;
        new_meta.0.original_width = meta.original_width;
        new_meta.0.original_height = meta.original_height;
        new_meta.0.object_id = meta.object_id;
        new_meta.0.detection_label = meta.detection_label.clone();
        new_meta.0.detection_confidence = meta.detection_confidence;
        true.into_glib()
    }

    pub(super) fn crop_origin_meta_get_info() -> *const gst::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gst::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: Lazy<MetaInfo> = Lazy::new(|| unsafe {
            let name = CString::new(CROP_ORIGIN_META_NAME)
                .expect("Failed to create CString for CropOriginMeta");
            MetaInfo(
                ptr::NonNull::new(gst::ffi::gst_meta_register(
                    crop_origin_meta_api_get_type().into_glib(),
                    name.as_ptr() as *const _,
                    mem::size_of::<CropOriginMeta>(),
                    Some(crop_origin_meta_init),
                    Some(crop_origin_meta_free),
                    Some(crop_origin_meta_transform),
                ) as *mut gst::ffi::GstMetaInfo)
                .expect("Failed to register CropOriginMeta"),
            )
        });

        META_INFO.0.as_ptr()
    }
}
