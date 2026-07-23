//! Text recognizer core: turns an RGB crop into `(text, confidence)` by
//! producing sequence logits (via a `LogitSource`) and greedy-CTC-decoding them.
//!
//! The FFI model call is isolated behind `LogitSource` so the decode path is
//! unit-testable without a deployed Edge Impulse model.

use super::ctc::{ctc_greedy_decode, parse_charset, parse_dictionary, passes_dictionary};
use super::normalize::{normalize_text, Normalize};

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
    normalize: Normalize,
}

impl<S: LogitSource> Recognizer<S> {
    pub fn new(source: S, charset: &str, dictionary: &str, normalize: Normalize) -> Self {
        Self {
            source,
            charset: parse_charset(charset),
            dictionary: parse_dictionary(dictionary),
            normalize,
        }
    }

    /// Recognize one crop. Returns `Ok(None)` when the read is empty or fails
    /// the dictionary gate, `Ok(Some((text, confidence)))` otherwise.
    ///
    /// Normalization is applied to the decoded text *before* the empty/dictionary
    /// gates so a dictionary is matched against (and truncation counts) the
    /// emitted form.
    pub fn recognize_text(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Option<(String, f32)>, String> {
        let (logits, num_classes) = self.source.logits(rgb, width, height)?;
        let decoded = ctc_greedy_decode(&logits, num_classes, &self.charset);
        let text = normalize_text(&decoded.text, self.normalize);
        if text.is_empty() || !passes_dictionary(&text, &self.dictionary) {
            return Ok(None);
        }
        Ok(Some((text, decoded.confidence)))
    }
}

/// Turn the runner's freeform output tensors into flat `[T, C]` sequence logits
/// for CTC decoding. A CRNN recognizer exposes a single output tensor of shape
/// `[T, num_classes]` (row-major), so the first tensor is taken as-is and
/// validated to contain a whole number of timesteps. `num_classes` is the
/// charset size (blank + symbols) the model was trained against.
#[cfg(any(feature = "ffi", test))]
pub(crate) fn logits_from_freeform(
    outputs: Vec<Vec<f32>>,
    num_classes: usize,
) -> Result<(Vec<f32>, usize), String> {
    if num_classes == 0 {
        return Err("recognizer num_classes must be > 0".to_string());
    }
    let logits = outputs
        .into_iter()
        .next()
        .ok_or_else(|| "recognizer produced no freeform output tensors".to_string())?;
    if logits.is_empty() {
        return Err("recognizer freeform output tensor is empty".to_string());
    }
    if logits.len() % num_classes != 0 {
        return Err(format!(
            "recognizer output length {} is not a multiple of num_classes {num_classes}",
            logits.len(),
        ));
    }
    Ok((logits, num_classes))
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
        let mut rec = Recognizer::new(src, "_AB", "", Normalize::None);
        let out = rec.recognize_text(&[], 4, 4).unwrap();
        assert_eq!(out.unwrap().0, "AB");
    }

    #[test]
    fn normalize_is_applied_before_dictionary_gate() {
        // charset "_ab": timesteps a, a(repeat), b => "ab"; upper-alnum => "AB".
        let src = FakeSource {
            logits: vec![0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            num_classes: 3,
        };
        // Dictionary lists the *normalized* form; the raw decode ("ab") only
        // passes because normalization runs first.
        let mut rec = Recognizer::new(src, "_ab", "AB", Normalize::UpperAlnum);
        assert_eq!(rec.recognize_text(&[], 4, 4).unwrap().unwrap().0, "AB");
    }

    #[test]
    fn dictionary_rejects_unlisted_text() {
        let src = FakeSource {
            logits: vec![0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            num_classes: 3,
        }; // decodes "AB"
        let mut rec = Recognizer::new(src, "_AB", "XY,ZZ", Normalize::None);
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
        let mut rec = Recognizer::new(ErrSource, "_AB", "", Normalize::None);
        assert_eq!(rec.recognize_text(&[], 1, 1).unwrap_err(), "boom");
    }

    #[test]
    fn freeform_takes_first_tensor_as_flat_logits() {
        let out = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let (logits, c) = logits_from_freeform(out, 3).unwrap();
        assert_eq!(c, 3);
        assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn freeform_rejects_length_not_multiple_of_classes() {
        let out = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]]; // 5 not divisible by 3
        assert!(logits_from_freeform(out, 3).is_err());
    }

    #[test]
    fn freeform_rejects_empty_or_zero_classes() {
        assert!(logits_from_freeform(Vec::new(), 3).is_err());
        assert!(logits_from_freeform(vec![vec![]], 3).is_err());
        assert!(logits_from_freeform(vec![vec![1.0, 2.0]], 0).is_err());
    }

    #[test]
    fn freeform_output_decodes_through_ctc() {
        // 3 timesteps, C=3 charset "_AB": argmax A, A(repeat), B => "AB".
        let out = vec![vec![0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]];
        let (logits, c) = logits_from_freeform(out, 3).unwrap();
        let decoded = ctc_greedy_decode(&logits, c, &parse_charset("_AB"));
        assert_eq!(decoded.text, "AB");
    }
}

#[cfg(feature = "ffi")]
pub mod ffi {
    use super::{logits_from_freeform, LogitSource};
    use edge_impulse_runner::EdgeImpulseModel;

    /// FFI-backed logit source. Loads the model baked into this plugin variant
    /// `.so` (same mechanism as `edgeimpulsevideoinfer`) and returns the
    /// recognizer's raw sequence logits as a flat `[T, C]` tensor via the
    /// runner's freeform-output API. CRNN/OCR models expose their raw output
    /// tensor this way rather than through the collapsed classification map.
    pub struct FfiLogitSource {
        model: EdgeImpulseModel,
        /// Charset size (blank + symbols) the model was trained against, i.e.
        /// the number of classes `C` per timestep in the `[T, C]` output.
        num_classes: usize,
    }

    impl FfiLogitSource {
        pub fn new(debug: bool, num_classes: usize) -> Result<Self, String> {
            let model = if debug {
                EdgeImpulseModel::new_with_debug(true)
            } else {
                EdgeImpulseModel::new()
            }
            .map_err(|e| format!("failed to load recognizer model: {e:?}"))?;
            Ok(Self { model, num_classes })
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
                let packed = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32);
                features.push(packed as f32);
            }

            let outputs = self
                .model
                .infer_freeform(features, None)
                .map_err(|e| format!("recognizer inference failed: {e:?}"))?;

            logits_from_freeform(outputs, self.num_classes)
        }
    }
}
