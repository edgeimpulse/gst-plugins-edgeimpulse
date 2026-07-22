//! Text recognizer core: turns an RGB crop into `(text, confidence)` by
//! producing sequence logits (via a `LogitSource`) and greedy-CTC-decoding them.
//!
//! The FFI model call is isolated behind `LogitSource` so the decode path is
//! unit-testable without a deployed Edge Impulse model.

use super::ctc::{ctc_greedy_decode, parse_charset, parse_dictionary, passes_dictionary};

/// Produces flat `[T, C]` logits for one RGB crop. Implemented by the FFI model
/// wrapper in production and by a fake in tests.
pub trait LogitSource {
    /// Returns `(logits, num_classes)` or an error string. `rgb` is tightly
    /// packed 24-bit RGB, row-major, `width * height * 3` bytes.
    fn logits(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, usize), String>;
}

/// Recognizer configuration + decode pipeline. Owns the charset/dictionary and
/// delegates inference to a `LogitSource`.
pub struct Recognizer<S: LogitSource> {
    source: S,
    charset: Vec<char>,
    dictionary: Vec<String>,
}

impl<S: LogitSource> Recognizer<S> {
    pub fn new(source: S, charset: &str, dictionary: &str) -> Self {
        Self {
            source,
            charset: parse_charset(charset),
            dictionary: parse_dictionary(dictionary),
        }
    }

    /// Recognize one crop. Returns `Ok(None)` when the read is empty or fails
    /// the dictionary gate, `Ok(Some((text, confidence)))` otherwise.
    pub fn recognize_text(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Option<(String, f32)>, String> {
        let (logits, num_classes) = self.source.logits(rgb, width, height)?;
        let decoded = ctc_greedy_decode(&logits, num_classes, &self.charset);
        if decoded.text.is_empty() || !passes_dictionary(&decoded.text, &self.dictionary) {
            return Ok(None);
        }
        Ok(Some((decoded.text, decoded.confidence)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeSource {
        logits: Vec<f32>,
        num_classes: usize,
    }
    impl LogitSource for FakeSource {
        fn logits(&mut self, _rgb: &[u8], _w: u32, _h: u32) -> Result<(Vec<f32>, usize), String> {
            Ok((self.logits.clone(), self.num_classes))
        }
    }

    #[test]
    fn decodes_text_through_source() {
        // charset "_AB": timesteps A, A(repeat), B => "AB"
        let src = FakeSource {
            logits: vec![0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            num_classes: 3,
        };
        let mut rec = Recognizer::new(src, "_AB", "");
        let out = rec.recognize_text(&[], 4, 4).unwrap();
        assert_eq!(out.unwrap().0, "AB");
    }

    #[test]
    fn dictionary_rejects_unlisted_text() {
        let src = FakeSource {
            logits: vec![0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            num_classes: 3,
        }; // decodes "AB"
        let mut rec = Recognizer::new(src, "_AB", "XY,ZZ");
        assert_eq!(rec.recognize_text(&[], 4, 4).unwrap(), None);
    }

    #[test]
    fn propagates_source_error() {
        struct ErrSource;
        impl LogitSource for ErrSource {
            fn logits(&mut self, _: &[u8], _: u32, _: u32) -> Result<(Vec<f32>, usize), String> {
                Err("boom".into())
            }
        }
        let mut rec = Recognizer::new(ErrSource, "_AB", "");
        assert_eq!(rec.recognize_text(&[], 1, 1).unwrap_err(), "boom");
    }
}

#[cfg(feature = "ffi")]
pub mod ffi {
    use super::LogitSource;
    use edge_impulse_runner::EdgeImpulseModel;

    /// FFI-backed logit source. Loads the model baked into this plugin variant
    /// `.so` (same mechanism as `edgeimpulsevideoinfer`) and returns the raw
    /// classification tensor flattened as `[T, C]`.
    pub struct FfiLogitSource {
        model: EdgeImpulseModel,
        /// One-shot latch for the on-device shape diagnostic (see `logits`).
        diagnosed: bool,
    }

    impl FfiLogitSource {
        pub fn new(debug: bool) -> Result<Self, String> {
            let model = if debug {
                EdgeImpulseModel::new_with_debug(true)
            } else {
                EdgeImpulseModel::new()
            }
            .map_err(|e| format!("failed to load recognizer model: {e:?}"))?;
            Ok(Self {
                model,
                diagnosed: false,
            })
        }
    }

    impl LogitSource for FfiLogitSource {
        fn logits(
            &mut self,
            rgb: &[u8],
            width: u32,
            height: u32,
        ) -> Result<(Vec<f32>, usize), String> {
            // Pack RGB -> 0xRRGGBB f32 features (mirror video/imp.rs).
            let expected = (width as usize) * (height as usize) * 3;
            if rgb.len() < expected {
                return Err(format!("crop too small: {} < {}", rgb.len(), expected));
            }
            let mut features = Vec::with_capacity(expected / 3);
            for px in rgb[..expected].chunks_exact(3) {
                let packed =
                    ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32);
                features.push(packed as f32);
            }

            let response = self
                .model
                .infer(features, None)
                .map_err(|e| format!("recognizer inference failed: {e:?}"))?;

            // On the very first inference, surface a one-shot diagnostic of the
            // real model metadata + response shape so `extract_logits` can be
            // implemented against the deployed CRNN with certainty (Step 3). It
            // rides the `Err` channel because the recognizer path already logs
            // that as a warning (imp.rs), so no extra logging wiring is needed;
            // this is a probe, not a genuine failure mode.
            if !self.diagnosed {
                self.diagnosed = true;
                return Err(diagnostic(&self.model, &response));
            }

            extract_logits(&response)
        }
    }

    /// One-shot on-device probe: dump the recognizer model's metadata
    /// (`model_type`, `label_count`, `labels`, input dimensions) and the raw
    /// `InferenceResponse` shape for a single crop. This is what tells us how the
    /// CRNN output actually surfaces through the runner, which is the one
    /// genuinely uncertain input needed to implement `extract_logits`.
    ///
    /// Read the logged line as follows: if `label_count` equals `T * C` (with
    /// `C` = charset length) and `labels` are unique, the flat `[T, C]` tensor
    /// can be reconstructed from the classification map via the ordered `labels`
    /// index; if `label_count` equals just `C`, the sequence has been collapsed
    /// and the raw tensor must be reached another way.
    fn diagnostic(
        model: &EdgeImpulseModel,
        response: &edge_impulse_runner::InferenceResponse,
    ) -> String {
        use std::fmt::Write as _;
        let mut s = String::from("extract_logits DIAGNOSTIC (one-shot; pending validation)\n");
        match model.parameters() {
            Ok(p) => {
                let _ = writeln!(
                    s,
                    "  model_type={:?} label_count={} input_features_count={} \
                     image={}x{}x{} frames={} engine={} has_anomaly={:?}",
                    p.model_type,
                    p.label_count,
                    p.input_features_count,
                    p.image_input_width,
                    p.image_input_height,
                    p.image_channel_count,
                    p.image_input_frames,
                    p.inferencing_engine,
                    p.has_anomaly,
                );
                let n = p.labels.len();
                let head: Vec<&String> = p.labels.iter().take(32).collect();
                let _ = writeln!(s, "  labels(n={n}) head={head:?}");
                if n > 36 {
                    let tail: Vec<&String> = p.labels.iter().skip(n - 4).collect();
                    let _ = writeln!(s, "  labels tail={tail:?}");
                }
            }
            Err(e) => {
                let _ = writeln!(s, "  parameters() unavailable: {e:?}");
            }
        }
        match model.input_size() {
            Ok(sz) => {
                let _ = writeln!(s, "  input_size={sz}");
            }
            Err(e) => {
                let _ = writeln!(s, "  input_size() unavailable: {e:?}");
            }
        }
        match &response.result {
            edge_impulse_runner::InferenceResult::Classification { classification } => {
                let mut keys: Vec<&String> = classification.keys().collect();
                keys.sort();
                let sample: Vec<String> = keys
                    .iter()
                    .take(8)
                    .map(|k| format!("{k}={:.4}", classification[*k]))
                    .collect();
                let _ = writeln!(
                    s,
                    "  result=Classification entries={} sample={sample:?}",
                    classification.len()
                );
            }
            edge_impulse_runner::InferenceResult::ObjectDetection {
                bounding_boxes,
                classification,
                ..
            } => {
                let _ = writeln!(
                    s,
                    "  result=ObjectDetection boxes={} classification_entries={}",
                    bounding_boxes.len(),
                    classification.len()
                );
            }
            edge_impulse_runner::InferenceResult::VisualAnomaly { anomaly, .. } => {
                let _ = writeln!(s, "  result=VisualAnomaly anomaly={anomaly}");
            }
        }
        s
    }

    /// DEVICE-VALIDATED SEAM: pull flattened `[T, C]` sequence logits out of the
    /// runner response. A CRNN deployed from EI Studio surfaces its raw output
    /// tensor here; the exact accessor MUST be confirmed against a deployed
    /// model (Step 3, deferred). The high-level `Classification { HashMap }` enum
    /// collapses the temporal dimension, so the raw output tensor is required.
    fn extract_logits(
        response: &edge_impulse_runner::InferenceResponse,
    ) -> Result<(Vec<f32>, usize), String> {
        // NOTE: pending on-device confirmation of the CRNN output shape. Until
        // validated, fail explicitly rather than guess a wrong tensor shape.
        let _ = response;
        Err("extract_logits: pending on-device validation of CRNN output shape".into())
    }
}
