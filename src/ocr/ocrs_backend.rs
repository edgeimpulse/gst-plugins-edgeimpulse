//! Pure-Rust OCR backend built on the `ocrs` engine (`rten` runtime).
use crate::ocr::backend::{OcrBackend, OcrLine};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten_imageproc::BoundingRect;
use rten_tensor::{Layout, NdTensor};

/// Vendored ocrs models, committed under `<crate-root>/models` for offline,
/// reproducible builds. Populated by `examples/download-ocr-models.sh`.
mod embedded {
    pub static DETECTION: &[u8] = include_bytes!("../../models/text-detection.rten");
    pub static RECOGNITION: &[u8] = include_bytes!("../../models/text-recognition.rten");
}

pub struct OcrsBackend {
    engine: OcrEngine,
}

fn build_engine(detection: rten::Model, recognition: rten::Model) -> Result<OcrEngine, String> {
    OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection),
        recognition_model: Some(recognition),
        ..Default::default()
    })
    .map_err(|e| e.to_string())
}

/// Load a model from `path`, or fall back to the `embedded` bytes when `path`
/// is empty (zero-copy over the `&'static` slice compiled into the binary).
fn load_model(path: &str, embedded: &'static [u8]) -> Result<rten::Model, String> {
    if path.is_empty() {
        rten::Model::load_static_slice(embedded).map_err(|e| e.to_string())
    } else {
        rten::Model::load_file(path).map_err(|e| e.to_string())
    }
}

/// Mean of the detection probability `mask` over the half-open integer box
/// `[x0, x1) x [y0, y1)`, clamped to the mask bounds. Returns `0.0` for an
/// empty or fully out-of-bounds box.
fn mean_mask_prob(mask: &NdTensor<f32, 2>, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
    let [h, w] = mask.shape();
    let (x0, x1) = (x0.min(w), x1.min(w));
    let (y0, y1) = (y0.min(h), y1.min(h));
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += mask[[y, x]];
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

impl OcrsBackend {
    /// Build the engine, loading each model from its explicit path when set and
    /// otherwise from the model embedded in the binary. Two empty paths yield
    /// the zero-config all-embedded default; a user may also override just one
    /// model (e.g. a custom detector) and keep the embedded other.
    pub fn new(detection_model_path: &str, recognition_model_path: &str) -> Result<Self, String> {
        let detection = load_model(detection_model_path, embedded::DETECTION)?;
        let recognition = load_model(recognition_model_path, embedded::RECOGNITION)?;
        Ok(Self {
            engine: build_engine(detection, recognition)?,
        })
    }
}

impl OcrBackend for OcrsBackend {
    fn recognize(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<OcrLine>, String> {
        let source = ImageSource::from_bytes(rgb, (width, height)).map_err(|e| e.to_string())?;
        let input = self
            .engine
            .prepare_input(source)
            .map_err(|e| e.to_string())?;
        let words = self
            .engine
            .detect_words(&input)
            .map_err(|e| e.to_string())?;
        let line_rects = self.engine.find_text_lines(&input, &words);
        let recognized = self
            .engine
            .recognize_text(&input, &line_rects)
            .map_err(|e| e.to_string())?;
        let text_mask = self
            .engine
            .detect_text_pixels(&input)
            .map_err(|e| e.to_string())?;

        let mut out = Vec::new();
        for (line, word_rects) in recognized.iter().zip(line_rects.iter()) {
            let Some(line) = line else { continue };
            let text = line.to_string();
            if text.trim().is_empty() || word_rects.is_empty() {
                continue;
            }
            let (mut min_x, mut min_y, mut max_x, mut max_y) =
                (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for wr in word_rects {
                let r = wr.bounding_rect();
                min_x = min_x.min(r.left());
                min_y = min_y.min(r.top());
                max_x = max_x.max(r.right());
                max_y = max_y.max(r.bottom());
            }
            // The detector expands each rect by a few pixels, so clamp both
            // corners to the frame before deriving the size — otherwise a word
            // touching an edge would over-count width/height.
            let x0 = min_x.max(0.0);
            let y0 = min_y.max(0.0);
            let x1 = max_x.min(width as f32);
            let y1 = max_y.min(height as f32);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            // Confidence is the mean detection-mask probability over this line's
            // box: a text-density score, not a recognition-accuracy score. It is
            // near zero on blank frames, which is what lets `min-confidence`
            // drop false positives.
            let confidence = mean_mask_prob(
                &text_mask,
                x0 as usize,
                y0 as usize,
                x1 as usize,
                y1 as usize,
            );
            out.push(OcrLine {
                text,
                confidence,
                x: x0 as u32,
                y: y0 as u32,
                w: (x1 - x0) as u32,
                h: (y1 - y0) as u32,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::mean_mask_prob;
    use rten_tensor::NdTensor;

    // Row-major [H=2, W=3]:
    //   row 0: 0.0 1.0 0.5
    //   row 1: 0.0 0.0 1.0
    fn mask_2x3() -> NdTensor<f32, 2> {
        NdTensor::from_data([2, 3], vec![0.0, 1.0, 0.5, 0.0, 0.0, 1.0])
    }

    #[test]
    fn mean_over_full_mask() {
        let mask = mask_2x3();
        let mean = mean_mask_prob(&mask, 0, 0, 3, 2);
        assert!((mean - (2.5 / 6.0)).abs() < 1e-6, "got {mean}");
    }

    #[test]
    fn mean_over_sub_box() {
        // row 0, cols 1..3 -> (1.0 + 0.5) / 2 = 0.75
        let mask = mask_2x3();
        let mean = mean_mask_prob(&mask, 1, 0, 3, 1);
        assert!((mean - 0.75).abs() < 1e-6, "got {mean}");
    }

    #[test]
    fn clamps_out_of_bounds_upper_corner() {
        // x1/y1 past the edges clamp to [3, 2]; cols 1..3, rows 0..2
        // -> (1.0 + 0.5 + 0.0 + 1.0) / 4 = 0.625
        let mask = mask_2x3();
        let mean = mean_mask_prob(&mask, 1, 0, 10, 10);
        assert!((mean - 0.625).abs() < 1e-6, "got {mean}");
    }

    #[test]
    fn empty_box_is_zero() {
        let mask = mask_2x3();
        assert_eq!(mean_mask_prob(&mask, 2, 0, 2, 2), 0.0);
    }

    #[test]
    fn fully_out_of_bounds_is_zero() {
        let mask = mask_2x3();
        assert_eq!(mean_mask_prob(&mask, 5, 5, 6, 6), 0.0);
    }
}
