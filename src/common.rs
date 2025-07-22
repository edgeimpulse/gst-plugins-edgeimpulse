use edge_impulse_runner::EdgeImpulseModel;
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::glib::ParamSpecBuilderExt;
use gstreamer::prelude::*;
use std::sync::Mutex;

/// Trait for state types that support debug mode
#[allow(dead_code)]
pub trait DebugState {
    fn set_debug(&mut self, enabled: bool);
    fn get_debug(&self) -> bool;
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
/// - model-path: String property for specifying the path to the .eim model file (EIM mode only)
/// - model-path-with-debug: String property for specifying the path to the .eim model file with debug enabled (EIM mode only)
/// - debug: Boolean property for enabling debug mode (FFI mode only)
///
/// # Example
///
/// ```bash
/// # Setting properties in gst-launch-1.0:
/// # FFI mode (default):
/// gst-launch-1.0 edgeimpulsevideoinfer debug=true ! ...
/// # EIM mode (legacy):
/// gst-launch-1.0 edgeimpulsevideoinfer model-path=/path/to/model.eim ! ...
/// ```
pub fn create_common_properties() -> Vec<glib::ParamSpec> {
    let base_properties = vec![
        glib::ParamSpecString::builder("model-path")
            .nick("Model Path")
            .blurb("Path to Edge Impulse model file (.eim for EIM mode only, legacy)")
            .build(),
        glib::ParamSpecString::builder("model-path-with-debug")
            .nick("Model Path With Debug")
            .blurb("Path to Edge Impulse model file (debug mode enabled, EIM mode only, legacy)")
            .build(),
        glib::ParamSpecString::builder("threshold")
            .nick("Model Block Threshold")
            .blurb("Threshold value for model blocks in format 'blockId.type=value'. Examples: '5.min_score=0.6' for object detection blocks, '4.min_anomaly_score=0.35' for anomaly detection blocks. Multiple thresholds can be set by calling the property multiple times.")
            .build(),
    ];

    #[cfg(feature = "ffi")]
    {
        let mut properties = base_properties;
        properties.push(
            glib::ParamSpecBoolean::builder("debug")
                .nick("Debug Mode")
                .blurb("Enable debug mode for FFI inference (FFI mode only)")
                .default_value(false)
                .build(),
        );
        properties
    }
    #[cfg(not(feature = "ffi"))]
    {
        base_properties
    }
}

pub fn set_common_property<T>(
    state: &Mutex<T>,
    _id: usize,
    value: &glib::Value,
    pspec: &glib::ParamSpec,
    obj: &impl GstObjectExt,
    cat: &gst::DebugCategory,
) where
    T: AsMut<Option<EdgeImpulseModel>> + DebugState,
{
    match pspec.name() {
        "model-path" => {
            let model_path: Option<String> = value.get().expect("type checked upstream");

            gst::debug!(
                cat,
                obj = obj,
                "Setting model-path property: {:?}",
                model_path
            );

            // Runtime mode selection: if model path is provided, use EIM mode (legacy)
            // model_path is only used in EIM feature blocks, but we need to handle the case when EIM is not enabled
            if let Some(_model_path) = model_path {
                #[cfg(feature = "eim")]
                {
                    let model_result = EdgeImpulseModel::new_eim(&_model_path);
                    match model_result {
                        Ok(model) => {
                            gst::debug!(
                                cat,
                                obj = obj,
                                "Successfully loaded EIM model from {} (debug=false, legacy mode)",
                                _model_path
                            );
                            let mut state_guard = state.lock().unwrap();
                            *state_guard.as_mut() = Some(model);
                        }
                        Err(err) => {
                            gst::error!(cat, obj = obj, "Failed to load EIM model: {}", err);
                        }
                    }
                }
                #[cfg(not(feature = "eim"))]
                {
                    gst::error!(
                        cat,
                        obj = obj,
                        "EIM mode not enabled. Enable the 'eim' feature to use model files (legacy mode)."
                    );
                }
            } else {
                // No model path provided, use FFI mode (default)
                #[cfg(feature = "ffi")]
                {
                    gst::debug!(
                        cat,
                        obj = obj,
                        "No model path provided, using FFI mode (default): model will be created lazily on first inference"
                    );
                }
                #[cfg(not(feature = "ffi"))]
                {
                    gst::error!(cat, obj = obj, "FFI mode not enabled. Enable the 'ffi' feature or provide a model path for EIM mode (legacy).");
                }
            }
        }
        "model-path-with-debug" => {
            let mut state = state.lock().unwrap();
            let model_path: Option<String> = value.get().expect("type checked upstream");

            gst::debug!(
                cat,
                obj = obj,
                "Setting model-path-with-debug property: {:?}",
                model_path
            );

            // Runtime mode selection: if model path is provided, use EIM mode (legacy)
            // model_path is only used in EIM feature blocks, but we need to handle the case when EIM is not enabled
            if let Some(_model_path) = model_path {
                #[cfg(feature = "eim")]
                {
                    let model_result = EdgeImpulseModel::new_eim_with_debug(&_model_path, true);
                    match model_result {
                        Ok(model) => {
                            gst::debug!(
                                cat,
                                obj = obj,
                                "Successfully loaded EIM model from {} (debug=true, legacy mode)",
                                _model_path
                            );
                            *state.as_mut() = Some(model);
                        }
                        Err(err) => {
                            gst::error!(cat, obj = obj, "Failed to load EIM model: {}", err);
                        }
                    }
                }
                #[cfg(not(feature = "eim"))]
                {
                    gst::error!(
                        cat,
                        obj = obj,
                        "EIM mode not enabled. Enable the 'eim' feature to use model files (legacy mode)."
                    );
                }
            } else {
                // No model path provided, use FFI mode with debug (default)
                #[cfg(feature = "ffi")]
                {
                    // Set debug mode for lazy initialization
                    state.set_debug(true);
                    gst::debug!(
                        cat,
                        obj = obj,
                        "No model path provided, using FFI mode with debug (default): model will be created lazily on first inference"
                    );
                }
                #[cfg(not(feature = "ffi"))]
                {
                    gst::error!(cat, obj = obj, "FFI mode not enabled. Enable the 'ffi' feature or provide a model path for EIM mode (legacy).");
                }
            }
        }
        "debug" => {
            #[cfg(feature = "ffi")]
            {
                let debug_enabled: bool = value.get().expect("type checked upstream");
                let mut state = state.lock().unwrap();

                gst::debug!(
                    cat,
                    obj = obj,
                    "Setting debug property: {} (model will be created lazily on first inference)",
                    debug_enabled
                );

                // Update the debug flag in the state using the trait
                state.set_debug(debug_enabled);
            }
            #[cfg(not(feature = "ffi"))]
            {
                gst::warning!(
                    cat,
                    obj = obj,
                    "debug property is only available in FFI mode"
                );
            }
        }
        "threshold" => {
            let mut state = state.lock().unwrap();
            let threshold_str: Option<String> = value.get().expect("type checked upstream");

            gst::debug!(
                cat,
                obj = obj,
                "Setting threshold property: {:?}",
                threshold_str
            );

            if let Some(threshold_str) = threshold_str {
                if let Some(_model) = state.as_mut() {
                    // Parse threshold string in format "blockId.type=value"
                    let re = regex::Regex::new(r"^(\d+)\.([a-zA-Z0-9_-]+)=([\d\.]+)$").unwrap();

                    if let Some(captures) = re.captures(&threshold_str) {
                        let id: u32 = match captures[1].parse() {
                            Ok(id) => id,
                            Err(_) => {
                                gst::error!(
                                    cat,
                                    obj = obj,
                                    "Invalid block ID '{}', must be a number",
                                    &captures[1]
                                );
                                return;
                            }
                        };

                        let key = captures[2].to_string();
                        let value: f32 = match captures[3].parse() {
                            Ok(val) => val,
                            Err(_) => {
                                gst::error!(
                                    cat,
                                    obj = obj,
                                    "Invalid threshold value '{}', must be a number",
                                    &captures[3]
                                );
                                return;
                            }
                        };

                        // Create appropriate threshold config based on key
                        let threshold_config = match key.as_str() {
                            "min_score" => edge_impulse_runner::types::ModelThreshold::ObjectDetection {
                                id,
                                min_score: value,
                            },
                            "min_anomaly_score" => edge_impulse_runner::types::ModelThreshold::AnomalyGMM {
                                id,
                                min_anomaly_score: value,
                            },
                            _ => {
                                gst::error!(
                                    cat,
                                    obj = obj,
                                    "Invalid threshold type '{}', must be 'min_score' or 'min_anomaly_score'",
                                    key
                                );
                                return;
                            }
                        };

                        // Set the threshold using the new API
                        match _model.set_threshold(threshold_config) {
                            Ok(()) => {
                                gst::debug!(
                                    cat,
                                    obj = obj,
                                    "Successfully set threshold: {}={} for block ID {}",
                                    key, value, id
                                );
                            }
                            Err(err) => {
                                gst::error!(
                                    cat,
                                    obj = obj,
                                    "Failed to set threshold: {}",
                                    err
                                );
                            }
                        }
                    } else {
                        gst::error!(
                            cat,
                            obj = obj,
                            "Invalid threshold format. Expected 'blockId.type=value', got '{}'",
                            threshold_str
                        );
                    }
                }
            }
        }
        _ => unimplemented!(),
    }
}

pub fn get_common_property<T>(state: &Mutex<T>, _id: usize, pspec: &glib::ParamSpec) -> glib::Value
where
    T: AsRef<Option<EdgeImpulseModel>> + DebugState,
{
    match pspec.name() {
        "model-path" => {
            let state = state.lock().unwrap();
            if let Some(ref _model) = *state.as_ref() {
                // EdgeImpulseModel now has a path() method, so we return the actual path if available
                #[cfg(feature = "eim")]
                {
                    if let Some(path) = _model.path() {
                        return path.display().to_string().to_value();
                    }
                }
                // Return empty string if no path available (FFI mode or no model)
                "".to_value()
            } else {
                "".to_value()
            }
        }
        "model-path-with-debug" => None::<String>.to_value(),
        "debug" => {
            #[cfg(feature = "ffi")]
            {
                let state = state.lock().unwrap();
                state.get_debug().to_value()
            }
            #[cfg(not(feature = "ffi"))]
            {
                false.to_value()
            }
        }
        "threshold" => {
            let state = state.lock().unwrap();
            if let Some(ref _model) = *state.as_ref() {
                // Try to get the current thresholds from model parameters
                if let Ok(params) = _model.parameters() {
                    // Return a string representation of all thresholds
                    let thresholds: Vec<String> = params
                        .thresholds
                        .iter()
                        .map(|t| match t {
                            edge_impulse_runner::types::ModelThreshold::ObjectDetection {
                                id,
                                min_score,
                            } => {
                                format!("{id}.min_score={min_score}")
                            }
                            edge_impulse_runner::types::ModelThreshold::AnomalyGMM {
                                id,
                                min_anomaly_score,
                            } => {
                                format!("{id}.min_anomaly_score={min_anomaly_score}")
                            }
                            edge_impulse_runner::types::ModelThreshold::ObjectTracking {
                                id,
                                threshold,
                                ..
                            } => {
                                format!("{id}.threshold={threshold}")
                            }
                            edge_impulse_runner::types::ModelThreshold::Unknown { id, unknown } => {
                                format!("{id}.unknown={unknown}")
                            }
                        })
                        .collect();

                    if !thresholds.is_empty() {
                        return thresholds.join(",").to_value();
                    }
                }
                // Return empty string if no thresholds found
                "".to_value()
            } else {
                "".to_value()
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
    gst::Structure::builder(format!("edge-impulse-{element_type}-inference-result"))
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
    gst::Structure::builder(format!("edge-impulse-{element_type}-inference-result"))
        .field("timestamp", timestamp)
        .field("type", "error")
        .field("error", error)
        .build()
}
