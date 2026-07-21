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
    }

    impl FfiLogitSource {
        pub fn new(debug: bool) -> Result<Self, String> {
            let model = if debug {
                EdgeImpulseModel::new_with_debug(true)
            } else {
                EdgeImpulseModel::new()
            }
            .map_err(|e| format!("failed to load recognizer model: {e:?}"))?;
            Ok(Self { model })
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
            for px in rgb.chunks_exact(3) {
                let packed =
                    ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32);
                features.push(packed as f32);
            }

            let response = self
                .model
                .infer(features, None)
                .map_err(|e| format!("recognizer inference failed: {e:?}"))?;

            extract_logits(&response)
        }
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
