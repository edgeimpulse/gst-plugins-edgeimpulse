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
    vec![
        glib::ParamSpecString::builder("model-path")
            .nick("Model Path")
            .blurb("Path to Edge Impulse model file")
            .build(),
        glib::ParamSpecString::builder("model-path-with-debug")
            .nick("Model Path With Debug")
            .blurb("Path to Edge Impulse model file (debug mode enabled)")
            .build(),
        glib::ParamSpecString::builder("threshold")
            .nick("Model Block Threshold")
            .blurb("Threshold value for model blocks in format 'blockId.type=value'. Examples: '5.min_score=0.6' for object detection blocks, '4.min_anomaly_score=0.35' for anomaly detection blocks. Multiple thresholds can be set by calling the property multiple times.")
            .build(),
    ]
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
            if let Some(model_path) = model_path {
                let model_result = edge_impulse_runner::EimModel::new(&model_path);
                match model_result {
                    Ok(model) => {
                        gst::debug!(
                            cat,
                            obj = obj,
                            "Successfully loaded model from {} (debug=false)",
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
        "model-path-with-debug" => {
            let mut state = state.lock().unwrap();
            let model_path: Option<String> = value.get().expect("type checked upstream");
            if let Some(model_path) = model_path {
                let model_result = edge_impulse_runner::EimModel::new_with_debug(&model_path, true);
                match model_result {
                    Ok(model) => {
                        gst::debug!(
                            cat,
                            obj = obj,
                            "Successfully loaded model from {} (debug=true)",
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
        "threshold" => {
            let mut state = state.lock().unwrap();
            let threshold_str: Option<String> = value.get().expect("type checked upstream");

            if let Some(threshold_str) = threshold_str {
                if let Some(model) = state.as_mut() {
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
                            "min_score" => edge_impulse_runner::inference::messages::ThresholdConfig::ObjectDetection {
                                id,
                                min_score: value,
                            },
                            "min_anomaly_score" => edge_impulse_runner::inference::messages::ThresholdConfig::AnomalyGMM {
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

                        // Set the threshold asynchronously
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        match rt.block_on(model.set_learn_block_threshold(threshold_config)) {
                            Ok(_) => {
                                gst::debug!(
                                    cat,
                                    obj = obj,
                                    "Successfully set threshold for block {} to {}",
                                    id,
                                    value
                                );
                            }
                            Err(err) => {
                                gst::error!(cat, obj = obj, "Failed to set threshold: {}", err);
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
        "model-path-with-debug" => None::<String>.to_value(),
        "threshold" => {
            let state = state.lock().unwrap();
            if let Some(ref model) = *state.as_ref() {
                // Try to get the current thresholds from model parameters
                if let Ok(params) = model.parameters() {
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
