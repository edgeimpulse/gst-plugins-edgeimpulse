//! Glue between recognized OCR lines and the generic [`crate::tracker`].
//!
//! Maps `OcrLine`s into tracker detections, advances the tracker for one
//! recognition, and maps the confirmed tracks back into consolidated lines. The
//! OCR worker calls [`stabilize`] once per recognition when text stabilization
//! is enabled. Pure and host-testable.

use crate::ocr::backend::OcrLine;
use crate::tracker::{BBox, Detection, Tracker};

fn line_to_detection(line: &OcrLine) -> Detection {
    Detection {
        label: line.text.clone(),
        confidence: line.confidence,
        bbox: BBox {
            x: line.x as f32,
            y: line.y as f32,
            w: line.w as f32,
            h: line.h as f32,
        },
    }
}

/// Feed one recognition's `lines` through `tracker` (advancing it by `dt`
/// seconds since the previous recognition) and return the stabilized lines
/// (most-frequent text + mean confidence per tracked object, box = latest
/// read). Empty-text lines are dropped before tracking so they never spawn
/// tracks. Returned lines are ordered by track age (oldest first), not by the
/// input reading order.
pub fn stabilize(tracker: &mut Tracker, lines: Vec<OcrLine>, dt: f32) -> Vec<OcrLine> {
    let detections: Vec<Detection> = lines
        .iter()
        .filter(|l| !l.text.is_empty())
        .map(line_to_detection)
        .collect();
    tracker.update(&detections, dt);
    tracker
        .confirmed()
        .into_iter()
        .map(|c| OcrLine {
            text: c.label,
            confidence: c.confidence,
            x: c.bbox.x as u32,
            y: c.bbox.y as u32,
            w: c.bbox.w as u32,
            h: c.bbox.h as u32,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::TrackerConfig;

    fn line(text: &str, conf: f32, x: u32, y: u32, w: u32, h: u32) -> OcrLine {
        OcrLine {
            text: text.into(),
            confidence: conf,
            x,
            y,
            w,
            h,
        }
    }

    fn tracker(window: usize, min_hits: u32, max_misses: u32) -> Tracker {
        Tracker::new(TrackerConfig {
            iou_threshold: 0.3,
            window,
            min_hits,
            max_misses,
            kalman: None,
            consolidate_labels: true,
        })
    }

    #[test]
    fn single_stable_read_is_reported_verbatim() {
        let mut t = tracker(10, 1, 5);
        let out = stabilize(&mut t, vec![line("Hello", 0.5, 0, 0, 20, 10)], 1.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "Hello");
        assert_eq!((out[0].x, out[0].y, out[0].w, out[0].h), (0, 0, 20, 10));
    }

    #[test]
    fn empty_text_lines_are_dropped() {
        let mut t = tracker(10, 1, 5);
        let out = stabilize(&mut t, vec![line("", 0.9, 0, 0, 20, 10)], 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn recurring_text_wins_over_high_confidence_fragment() {
        let mut t = tracker(10, 1, 5);
        let bx = (0u32, 0u32, 40u32, 10u32);
        stabilize(
            &mut t,
            vec![line("Qualcomm robotics", 0.30, bx.0, bx.1, bx.2, bx.3)],
            1.0,
        );
        stabilize(
            &mut t,
            vec![line("robotics", 0.90, bx.0, bx.1, bx.2, bx.3)],
            1.0,
        );
        let out = stabilize(
            &mut t,
            vec![line("Qualcomm robotics", 0.35, bx.0, bx.1, bx.2, bx.3)],
            1.0,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "Qualcomm robotics");
    }
}
