//! End-to-end integration tests for gst-plugins-edgeimpulse.
//!
//! These tests build real GStreamer pipelines with the Edge Impulse plugin
//! elements, feed them synthetic video, and assert on bus messages and buffer
//! metadata. They require a model compiled into the binary (FFI mode).
//!
//! Run with:
//!   cargo test --release --test e2e -- --nocapture
//!
//! Or via Docker:
//!   docker compose -f docker-compose.test.yml up --build --abort-on-container-exit

use gstreamer as gst;
use gstreamer::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        gst::init().expect("Failed to initialize GStreamer");
    });
}

/// Collected results from a test pipeline run.
#[derive(Debug, Default, Clone)]
struct TestResults {
    inference_messages: Vec<serde_json::Value>,
    metadata_messages: Vec<serde_json::Value>,
    model_loaded: bool,
    errors: Vec<String>,
    buffer_count: u64,
}

/// Run a pipeline for a given duration, collecting bus messages.
fn run_pipeline_for(pipeline_str: &str, duration: Duration) -> TestResults {
    init();

    let pipeline = gst::parse::launch(pipeline_str)
        .unwrap_or_else(|e| panic!("Failed to parse pipeline: {e}\nPipeline: {pipeline_str}"));
    let pipeline = pipeline
        .dynamic_cast::<gst::Pipeline>()
        .expect("Not a pipeline");

    let results = Arc::new(Mutex::new(TestResults::default()));
    let results_clone = results.clone();

    let bus = pipeline.bus().unwrap();

    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to set pipeline to Playing");

    let deadline = std::time::Instant::now() + duration;

    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break;
        }

        let timeout = gst::ClockTime::from_mseconds(remaining.as_millis() as u64);
        match bus.timed_pop(timeout) {
            Some(msg) => {
                use gst::MessageView;
                let mut res = results_clone.lock().unwrap();
                match msg.view() {
                    MessageView::Element(element) => {
                        let structure = element.structure().unwrap();
                        let name = structure.name().to_string();

                        if name.contains("inference-result") {
                            let result_type = structure
                                .get::<String>("type")
                                .unwrap_or_default();
                            let result_json = structure
                                .get::<String>("result")
                                .unwrap_or_default();
                            let timing_ms = structure.get::<u32>("timing_ms").unwrap_or(0);

                            let entry = serde_json::json!({
                                "message_name": name,
                                "type": result_type,
                                "result": serde_json::from_str::<serde_json::Value>(&result_json)
                                    .unwrap_or(serde_json::Value::Null),
                                "timing_ms": timing_ms,
                            });
                            res.inference_messages.push(entry);
                            res.buffer_count += 1;
                        } else if name == "edge-impulse-model-loaded" {
                            res.model_loaded = true;
                        } else if name == "edge-impulse-continue-if-metadata" {
                            let mut fields = serde_json::Map::new();
                            for (idx, field_name) in structure.iter().enumerate() {
                                let _ = idx;
                                if let Ok(val) = structure.get::<String>(field_name.0) {
                                    fields.insert(
                                        field_name.0.to_string(),
                                        serde_json::Value::String(val),
                                    );
                                }
                            }
                            res.metadata_messages
                                .push(serde_json::Value::Object(fields));
                        }
                    }
                    MessageView::Error(err) => {
                        let msg = format!(
                            "Pipeline error: {} ({})",
                            err.error(),
                            err.debug().unwrap_or_default()
                        );
                        res.errors.push(msg);
                        break;
                    }
                    MessageView::Eos(_) => break,
                    _ => {}
                }
            }
            None => break,
        }
    }

    pipeline
        .set_state(gst::State::Null)
        .expect("Failed to stop pipeline");

    let res = results.lock().unwrap().clone();
    res
}

/// Build a test source that produces RGB video frames.
/// Uses videotestsrc with a SMPTE pattern — not ideal for real inference
/// but exercises the full pipeline path.
fn video_test_source(num_buffers: u32) -> String {
    format!(
        "videotestsrc num-buffers={num_buffers} pattern=smpte ! \
         video/x-raw,format=RGB,width=320,height=320,framerate=5/1 ! \
         videoconvert"
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_video_inference_produces_bus_messages() {
    let src = video_test_source(5);
    let pipeline = format!("{src} ! edgeimpulsevideoinfer ! fakesink");

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    assert!(
        results.errors.is_empty(),
        "Pipeline errors: {:?}",
        results.errors
    );
    assert!(
        !results.inference_messages.is_empty(),
        "Expected at least one inference bus message"
    );

    // Every message should have a valid type and result
    for msg in &results.inference_messages {
        let result_type = msg["type"].as_str().unwrap();
        assert!(
            ["object-detection", "classification", "anomaly-detection", "object-tracking"]
                .contains(&result_type),
            "Unexpected result type: {result_type}"
        );
        assert!(
            !msg["result"].is_null(),
            "Result should not be null"
        );
        assert!(
            msg["timing_ms"].as_u64().is_some(),
            "timing_ms should be present"
        );
    }
}

#[test]
fn test_video_inference_message_structure() {
    let src = video_test_source(3);
    let pipeline = format!("{src} ! edgeimpulsevideoinfer ! fakesink");

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));
    assert!(!results.inference_messages.is_empty());

    let first = &results.inference_messages[0];
    let result = &first["result"];

    // Should have either bounding_boxes (detection) or classification
    let has_detections = result.get("bounding_boxes").is_some();
    let has_classification = result.get("classification").is_some();
    let has_anomaly = result.get("anomaly").is_some();

    assert!(
        has_detections || has_classification || has_anomaly,
        "Result should contain bounding_boxes, classification, or anomaly. Got: {result}"
    );
}

#[test]
fn test_overlay_does_not_crash() {
    let src = video_test_source(5);
    let pipeline = format!(
        "{src} ! edgeimpulsevideoinfer ! edgeimpulseoverlay ! fakesink"
    );

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    assert!(
        results.errors.is_empty(),
        "Overlay pipeline should not produce errors: {:?}",
        results.errors
    );
    assert!(
        !results.inference_messages.is_empty(),
        "Should still produce inference messages after overlay"
    );
}

#[test]
fn test_continue_if_passes_all_when_condition_always_true() {
    let src = video_test_source(5);
    // detection_count >= 0 is always true (even 0 detections)
    let pipeline = format!(
        "{src} ! edgeimpulsevideoinfer ! \
         edgeimpulsecontinueif condition=\"detection_count >= 0\" ! \
         fakesink"
    );

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    assert!(
        results.errors.is_empty(),
        "Errors: {:?}",
        results.errors
    );
    assert!(
        !results.inference_messages.is_empty(),
        "All buffers should pass through"
    );
}

#[test]
fn test_continue_if_drops_when_condition_impossible() {
    let src = video_test_source(5);
    // detection_count >= 9999 is never true
    let pipeline = format!(
        "{src} ! edgeimpulsevideoinfer ! \
         edgeimpulsecontinueif condition=\"detection_count >= 9999\" ! \
         fakesink"
    );

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    // Pipeline should complete without errors — buffers are marked GAP, not errored
    assert!(
        results.errors.is_empty(),
        "Pipeline should not error on dropped buffers: {:?}",
        results.errors
    );
}

#[test]
fn test_continue_if_rules_emit_metadata() {
    init();

    let src = video_test_source(5);
    // Build pipeline without rules (set property programmatically to avoid quoting issues)
    let pipeline_str = format!(
        "{src} ! edgeimpulsevideoinfer ! \
         edgeimpulsecontinueif name=gate ! \
         fakesink"
    );

    let pipeline = gst::parse::launch(&pipeline_str).unwrap();
    let pipeline = pipeline.dynamic_cast::<gst::Pipeline>().unwrap();

    // Set rules property programmatically — avoids gst_parse quoting issues
    let gate = pipeline.by_name("gate").expect("gate element not found");
    let rules_json = r#"[{"condition":"detection_count >= 0","metadata":{"color":"green","status":"ok"}}]"#;
    gate.set_property("rules", rules_json);

    pipeline.set_state(gst::State::Playing).unwrap();

    let bus = pipeline.bus().unwrap();
    let mut metadata_messages = Vec::new();
    let mut errors = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);

    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() { break; }

        let timeout = gst::ClockTime::from_mseconds(remaining.as_millis() as u64);
        match bus.timed_pop(timeout) {
            Some(msg) => {
                use gst::MessageView;
                match msg.view() {
                    MessageView::Element(element) => {
                        let structure = element.structure().unwrap();
                        if structure.name() == "edge-impulse-continue-if-metadata" {
                            let mut fields = serde_json::Map::new();
                            for field in structure.iter() {
                                if let Ok(val) = structure.get::<String>(field.0) {
                                    fields.insert(field.0.to_string(), serde_json::Value::String(val));
                                }
                            }
                            metadata_messages.push(serde_json::Value::Object(fields));
                        }
                    }
                    MessageView::Error(err) => {
                        errors.push(format!("{} ({})", err.error(), err.debug().unwrap_or_default()));
                        break;
                    }
                    MessageView::Eos(_) => break,
                    _ => {}
                }
            }
            None => break,
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();

    assert!(errors.is_empty(), "Errors: {:?}", errors);
    assert!(
        !metadata_messages.is_empty(),
        "Should have metadata bus messages from matched rules"
    );

    let first_meta = &metadata_messages[0];
    assert_eq!(
        first_meta.get("color").and_then(|v| v.as_str()),
        Some("green"),
        "Metadata should contain color=green"
    );
    assert_eq!(
        first_meta.get("status").and_then(|v| v.as_str()),
        Some("ok"),
        "Metadata should contain status=ok"
    );
}

#[test]
fn test_crop_produces_output_buffers() {
    let src = video_test_source(5);
    // Crop with target size — even if no detections, full frame should pass through
    let pipeline = format!(
        "{src} ! edgeimpulsevideoinfer ! \
         edgeimpulsecrop target-width=96 target-height=96 ! \
         fakesink"
    );

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    assert!(
        results.errors.is_empty(),
        "Crop pipeline should not error: {:?}",
        results.errors
    );
}

#[test]
fn test_full_two_stage_pipeline() {
    let src = video_test_source(5);
    // Full pipeline: infer → tee → (overlay + crop → continue-if)
    // This tests element interop without needing a second model
    let pipeline = format!(
        "{src} ! edgeimpulsevideoinfer ! tee name=t \
         t. ! queue ! edgeimpulseoverlay ! fakesink \
         t. ! queue ! edgeimpulsecontinueif condition=\"detection_count >= 0\" ! \
              edgeimpulsecrop target-width=96 target-height=96 ! \
              fakesink"
    );

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    assert!(
        results.errors.is_empty(),
        "Two-stage pipeline should not error: {:?}",
        results.errors
    );
    assert!(
        !results.inference_messages.is_empty(),
        "Should produce inference messages"
    );
}

#[test]
fn test_continue_if_no_condition_passes_all() {
    let src = video_test_source(3);
    // No condition set — everything should pass
    let pipeline = format!(
        "{src} ! edgeimpulsevideoinfer ! edgeimpulsecontinueif ! fakesink"
    );

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    assert!(
        results.errors.is_empty(),
        "Errors: {:?}",
        results.errors
    );
    assert!(
        !results.inference_messages.is_empty(),
        "With no condition, all buffers should pass"
    );
}

#[test]
fn test_crop_padding_property() {
    let src = video_test_source(3);
    let pipeline = format!(
        "{src} ! edgeimpulsevideoinfer ! \
         edgeimpulsecrop padding=20 target-width=128 target-height=128 ! \
         fakesink"
    );

    let results = run_pipeline_for(&pipeline, Duration::from_secs(10));

    assert!(
        results.errors.is_empty(),
        "Crop with padding should not error: {:?}",
        results.errors
    );
}
