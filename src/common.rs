use gstreamer::glib;
use gstreamer::glib::ParamSpecBuilderExt;
use edge_impulse_runner::EimModel;

/// State structure for Edge Impulse elements
///
/// This structure maintains the runtime state of the Edge Impulse elements,
/// including the loaded model and media format information. It's protected
/// by a Mutex to ensure thread-safe access in streaming contexts.
pub struct State {
    /// The loaded Edge Impulse model instance
    pub model: Option<EimModel>,

    /// Width of the input frames (for video models)
    pub width: Option<u32>,

    /// Height of the input frames (for video models)
    pub height: Option<u32>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            model: None,
            width: None,
            height: None,
        }
    }
}

/// Creates common GStreamer properties shared between Edge Impulse elements
///
/// This function returns a vector of parameter specifications that define
/// the properties available on Edge Impulse GStreamer elements. These
/// properties can be set through standard GStreamer property mechanisms
/// (e.g., gst-launch-1.0 element property=value).
///
/// # Returns
///
/// Returns a `Vec<glib::ParamSpec>` containing the property specifications:
/// - model-path: String property for specifying the path to the .eim model file
///
/// # Example
///
/// ```bash
/// # Setting properties in gst-launch-1.0:
/// gst-launch-1.0 edgeimpulsevideoinfer model-path=/path/to/model.eim ! ...
/// ```
pub fn create_common_properties() -> Vec<glib::ParamSpec> {
    vec![
        glib::ParamSpecString::builder("model-path")
            .nick("Model Path")
            .blurb("Path to Edge Impulse model file")
            .build()
    ]
}