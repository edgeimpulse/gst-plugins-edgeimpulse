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
