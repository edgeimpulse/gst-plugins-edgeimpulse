use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer::Meta;
use gstreamer_base as gst_base;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use once_cell::sync::Lazy;
use std::fmt;
use std::ptr;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct VideoRegionOfInterestMeta {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub label: String,
}

// Public Rust type for the custom meta
#[repr(C)]
pub struct VideoClassificationMeta(imp::VideoClassificationMeta);

// Public Rust type for the custom meta
#[repr(C)]
pub struct VideoAnomalyMeta(imp::VideoAnomalyMeta);

// Metas must be Send+Sync
unsafe impl Send for VideoClassificationMeta {}
unsafe impl Sync for VideoClassificationMeta {}
unsafe impl Send for VideoAnomalyMeta {}
unsafe impl Sync for VideoAnomalyMeta {}

impl VideoClassificationMeta {
    pub fn add(buffer: &mut gst::BufferRef) -> gst::MetaRefMut<Self, gst::meta::Standalone> {
        unsafe {
            // First add it with empty params
            let meta = gst::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::video_classification_meta_get_info(),
                ptr::null_mut(),
            ) as *mut imp::VideoClassificationMeta;

            // Initialize params
            {
                let meta = &mut *meta;
                meta.params = Vec::new();
            }

            Self::from_mut_ptr(buffer, meta)
        }
    }

    pub fn add_param(&mut self, param: gst::Structure) {
        self.0.params.push(param);
    }

    pub fn params(&self) -> &[gst::Structure] {
        &self.0.params
    }
}

impl VideoAnomalyMeta {
    pub fn add(buffer: &mut gst::BufferRef) -> gst::MetaRefMut<Self, gst::meta::Standalone> {
        unsafe {
            // First add it with empty params
            let meta = gst::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::video_anomaly_meta_get_info(),
                ptr::null_mut(),
            ) as *mut imp::VideoAnomalyMeta;

            // Initialize fields
            {
                let meta = &mut *meta;
                meta.anomaly = 0.0;
                meta.visual_anomaly_max = 0.0;
                meta.visual_anomaly_mean = 0.0;
                meta.visual_anomaly_grid = Vec::new();
            }

            Self::from_mut_ptr(buffer, meta)
        }
    }

    pub fn anomaly(&self) -> f32 {
        self.0.anomaly
    }

    pub fn set_anomaly(&mut self, value: f32) {
        self.0.anomaly = value;
    }

    pub fn visual_anomaly_max(&self) -> f32 {
        self.0.visual_anomaly_max
    }

    pub fn set_visual_anomaly_max(&mut self, value: f32) {
        self.0.visual_anomaly_max = value;
    }

    pub fn visual_anomaly_mean(&self) -> f32 {
        self.0.visual_anomaly_mean
    }

    pub fn set_visual_anomaly_mean(&mut self, value: f32) {
        self.0.visual_anomaly_mean = value;
    }

    pub fn visual_anomaly_grid(&self) -> &[VideoRegionOfInterestMeta] {
        &self.0.visual_anomaly_grid
    }

    pub fn set_visual_anomaly_grid(&mut self, value: Vec<VideoRegionOfInterestMeta>) {
        self.0.visual_anomaly_grid = value;
    }
}

// Trait to allow using the gst::Buffer API with this meta
unsafe impl MetaAPI for VideoClassificationMeta {
    type GstType = imp::VideoClassificationMeta;

    fn meta_api() -> glib::Type {
        imp::video_classification_meta_api_get_type()
    }
}

unsafe impl MetaAPI for VideoAnomalyMeta {
    type GstType = imp::VideoAnomalyMeta;

    fn meta_api() -> glib::Type {
        imp::video_anomaly_meta_api_get_type()
    }
}

impl fmt::Debug for VideoClassificationMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("VideoClassificationMeta")
            .field("params", &self.params())
            .finish()
    }
}

impl fmt::Debug for VideoAnomalyMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("VideoAnomalyMeta")
            .field("anomaly", &self.anomaly())
            .field("visual_anomaly_max", &self.visual_anomaly_max())
            .field("visual_anomaly_mean", &self.visual_anomaly_mean())
            .field("visual_anomaly_grid", &self.visual_anomaly_grid())
            .finish()
    }
}

// Actual unsafe implementation of the meta
mod imp {
    use super::*;
    use glib::translate::*;
    use gstreamer as gst;
    use gstreamer_video as gst_video;
    use once_cell::sync::Lazy;
    use std::mem;
    use std::ptr;

    // This is the C type that is actually stored as meta inside the buffers
    #[repr(C)]
    pub struct VideoClassificationMeta {
        pub(super) meta: gst::ffi::GstMeta,
        pub(super) params: Vec<gst::Structure>,
    }

    // This is the C type that is actually stored as meta inside the buffers
    #[repr(C)]
    pub struct VideoAnomalyMeta {
        pub(super) meta: gst::ffi::GstMeta,
        pub(super) anomaly: f32,
        pub(super) visual_anomaly_max: f32,
        pub(super) visual_anomaly_mean: f32,
        pub(super) visual_anomaly_grid: Vec<VideoRegionOfInterestMeta>,
    }

    // Function to register the meta API and get a type back
    pub(super) fn video_classification_meta_api_get_type() -> glib::Type {
        static TYPE: Lazy<glib::Type> = Lazy::new(|| unsafe {
            let t = from_glib(gst::ffi::gst_meta_api_type_register(
                c"VideoClassificationMetaAPI".as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));

            assert_ne!(t, glib::Type::INVALID);

            t
        });

        *TYPE
    }

    pub(super) fn video_anomaly_meta_api_get_type() -> glib::Type {
        static TYPE: Lazy<glib::Type> = Lazy::new(|| unsafe {
            let t = from_glib(gst::ffi::gst_meta_api_type_register(
                c"VideoAnomalyMetaAPI".as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));

            assert_ne!(t, glib::Type::INVALID);

            t
        });

        *TYPE
    }

    // Initialization function for our meta
    unsafe extern "C" fn video_classification_meta_init(
        meta: *mut gst::ffi::GstMeta,
        _params: glib::ffi::gpointer,
        _buffer: *mut gst::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        let meta = &mut *(meta as *mut VideoClassificationMeta);

        // Initialize params
        ptr::write(&mut meta.params, Vec::new());

        true.into_glib()
    }

    unsafe extern "C" fn video_anomaly_meta_init(
        meta: *mut gst::ffi::GstMeta,
        _params: glib::ffi::gpointer,
        _buffer: *mut gst::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        let meta = &mut *(meta as *mut VideoAnomalyMeta);

        // Initialize fields
        ptr::write(&mut meta.anomaly, 0.0);
        ptr::write(&mut meta.visual_anomaly_max, 0.0);
        ptr::write(&mut meta.visual_anomaly_mean, 0.0);
        ptr::write(&mut meta.visual_anomaly_grid, Vec::new());

        true.into_glib()
    }

    // Free function for our meta
    unsafe extern "C" fn video_classification_meta_free(
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut VideoClassificationMeta);

        // Drop params
        ptr::drop_in_place(&mut meta.params);
    }

    unsafe extern "C" fn video_anomaly_meta_free(
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut VideoAnomalyMeta);

        // Drop fields
        ptr::drop_in_place(&mut meta.visual_anomaly_grid);
    }

    // Transform function for our meta
    unsafe extern "C" fn video_classification_meta_transform(
        dest: *mut gst::ffi::GstBuffer,
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &mut *(meta as *mut VideoClassificationMeta);

        // Copy over our meta
        let mut new_meta = super::VideoClassificationMeta::add(gst::BufferRef::from_mut_ptr(dest));
        new_meta.0.params = meta.params.clone();

        true.into_glib()
    }

    unsafe extern "C" fn video_anomaly_meta_transform(
        dest: *mut gst::ffi::GstBuffer,
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &mut *(meta as *mut VideoAnomalyMeta);

        // Copy over our meta
        let mut new_meta = super::VideoAnomalyMeta::add(gst::BufferRef::from_mut_ptr(dest));
        new_meta.0.anomaly = meta.anomaly;
        new_meta.0.visual_anomaly_max = meta.visual_anomaly_max;
        new_meta.0.visual_anomaly_mean = meta.visual_anomaly_mean;
        new_meta.0.visual_anomaly_grid = meta.visual_anomaly_grid.clone();

        true.into_glib()
    }

    // Register the meta itself with its functions
    pub(super) fn video_classification_meta_get_info() -> *const gst::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gst::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: Lazy<MetaInfo> = Lazy::new(|| unsafe {
            MetaInfo(
                ptr::NonNull::new(gst::ffi::gst_meta_register(
                    video_classification_meta_api_get_type().into_glib(),
                    c"VideoClassificationMeta".as_ptr() as *const _,
                    mem::size_of::<VideoClassificationMeta>(),
                    Some(video_classification_meta_init),
                    Some(video_classification_meta_free),
                    Some(video_classification_meta_transform),
                ) as *mut gst::ffi::GstMetaInfo)
                .expect("Failed to register meta API"),
            )
        });

        META_INFO.0.as_ptr()
    }

    pub(super) fn video_anomaly_meta_get_info() -> *const gst::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gst::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: Lazy<MetaInfo> = Lazy::new(|| unsafe {
            MetaInfo(
                ptr::NonNull::new(gst::ffi::gst_meta_register(
                    video_anomaly_meta_api_get_type().into_glib(),
                    c"VideoAnomalyMeta".as_ptr() as *const _,
                    mem::size_of::<VideoAnomalyMeta>(),
                    Some(video_anomaly_meta_init),
                    Some(video_anomaly_meta_free),
                    Some(video_anomaly_meta_transform),
                ) as *mut gst::ffi::GstMetaInfo)
                .expect("Failed to register meta API"),
            )
        });

        META_INFO.0.as_ptr()
    }
}
