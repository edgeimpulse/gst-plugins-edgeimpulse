use edge_impulse_runner::EimModel;
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::glib::ParamSpecBuilderExt;
use gstreamer::prelude::*;
use std::sync::Mutex;

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
    vec![glib::ParamSpecString::builder("model-path")
        .nick("Model Path")
        .blurb("Path to Edge Impulse model file")
        .build()]
}

pub fn set_common_property<T>(
    state: &Mutex<T>,
    _id: usize,
    value: &glib::Value,
    pspec: &glib::ParamSpec,
    obj: &impl GstObjectExt,
    cat: &gst::DebugCategory,
) where
    T: AsMut<Option<EimModel>>,
{
    match pspec.name() {
        "model-path" => {
            let mut state = state.lock().unwrap();
            let model_path: Option<String> = value.get().expect("type checked upstream");

            // Initialize the model when the path is set
            if let Some(model_path) = model_path {
                match edge_impulse_runner::EimModel::new(&model_path) {
                    Ok(model) => {
                        gst::debug!(
                            cat,
                            obj = obj,
                            "Successfully loaded model from {}",
                            model_path
                        );
                        *state.as_mut() = Some(model);
                    }
                    Err(err) => {
                        gst::error!(cat, obj = obj, "Failed to load model: {}", err);
                    }
                }
            }
        }
        _ => unimplemented!(),
    }
}

pub fn get_common_property<T>(state: &Mutex<T>, _id: usize, pspec: &glib::ParamSpec) -> glib::Value
where
    T: AsRef<Option<EimModel>>,
{
    match pspec.name() {
        "model-path" => {
            let state = state.lock().unwrap();
            if let Some(ref model) = *state.as_ref() {
                model.path().to_value()
            } else {
                None::<String>.to_value()
            }
        }
        _ => unimplemented!(),
    }
}

/// Creates a standard inference result message structure
pub fn create_inference_message(
    element_type: &str,
    timestamp: gst::ClockTime,
    result_type: &str,
    result_json: String,
    timing_ms: u32,
) -> gst::Structure {
    gst::Structure::builder(&format!("edge-impulse-{}-inference-result", element_type))
        .field("timestamp", timestamp)
        .field("type", result_type)
        .field("result", result_json)
        .field("timing_ms", timing_ms)
        .build()
}

/// Creates an error message structure
pub fn create_error_message(
    element_type: &str,
    timestamp: gst::ClockTime,
    error: String,
) -> gst::Structure {
    gst::Structure::builder(&format!("edge-impulse-{}-inference-result", element_type))
        .field("timestamp", timestamp)
        .field("type", "error")
        .field("error", error)
        .build()
}
