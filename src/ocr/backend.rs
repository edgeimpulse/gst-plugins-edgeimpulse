//! Backend abstraction for the OCR element.

/// One recognized line of text with its axis-aligned box in full-frame pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Recognizes text in a tightly-packed RGB frame (stride == width * 3).
pub trait OcrBackend: Send {
    fn recognize(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<OcrLine>, String>;
}

/// Which recognition engine the element uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Ocrs,
    EdgeImpulse,
    EdgeImpulseRecognizer,
}

impl Backend {
    pub fn parse(s: &str) -> Self {
        match s {
            "ocrs" => Backend::Ocrs,
            "edge-impulse-characters" => Backend::EdgeImpulse,
            "edge-impulse-recognizer" => Backend::EdgeImpulseRecognizer,
            _ => Backend::Ocrs,
        }
    }
}

/// A backend that recognizes nothing. Used as a safe default before a real
/// backend is configured, and for tests.
pub struct NoopBackend;
impl OcrBackend for NoopBackend {
    fn recognize(&mut self, _rgb: &[u8], _w: u32, _h: u32) -> Result<Vec<OcrLine>, String> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_maps_known_backends() {
        assert!(matches!(Backend::parse("ocrs"), Backend::Ocrs));
        assert!(matches!(
            Backend::parse("edge-impulse-characters"),
            Backend::EdgeImpulse
        ));
        // legacy name no longer recognized (never released) -> falls back to default
        assert!(matches!(Backend::parse("edge-impulse"), Backend::Ocrs));
    }

    #[test]
    fn noop_backend_returns_no_lines() {
        assert!(NoopBackend.recognize(&[0u8; 12], 2, 2).unwrap().is_empty());
    }

    #[test]
    fn parses_recognizer_backend() {
        assert!(matches!(
            Backend::parse("edge-impulse-recognizer"),
            Backend::EdgeImpulseRecognizer
        ));
    }
}
