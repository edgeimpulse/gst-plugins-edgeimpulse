//! `edge-impulse` OCR backend: decode the per-character detections an upstream
//! Edge Impulse object-detection model (run by `edgeimpulsevideoinfer`) attached
//! to the buffer into lines of text, reusing the shared `shaping` output path.
//!
//! Unlike the `ocrs` backend this evaluates no model: the character boxes are
//! already on the buffer as `VideoRegionOfInterestMeta` (in full-frame pixels,
//! with `label` + `confidence` in a `detection` param). We group them into rows,
//! concatenate each row into a line, and hand the lines to `shaping`.

use crate::ocr::backend::OcrLine;
use gstreamer as gst;
use gstreamer_video as gst_video;

/// One upstream character detection, in full-frame pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Overlap length of the 1-D intervals `[a0, a1)` and `[b0, b1)` (0 if disjoint).
fn overlap_1d(a0: u32, a1: u32, b0: u32, b1: u32) -> u32 {
    a1.min(b1).saturating_sub(a0.max(b0))
}

/// Group character detections into text lines: rows by vertical overlap,
/// left-to-right within a row, one `OcrLine` per row, ordered top-to-bottom.
///
/// Detections with an empty label are dropped. A box joins the current row when
/// its vertical span overlaps the row's span by at least half the shorter of the
/// two heights; otherwise it starts a new row. Line text is the row's labels
/// concatenated in x order (no spaces inserted); line confidence is the minimum
/// member confidence; line bbox is the union of member boxes.
pub fn assemble_lines(dets: &[Detection]) -> Vec<OcrLine> {
    let mut items: Vec<&Detection> = dets.iter().filter(|d| !d.label.is_empty()).collect();
    if items.is_empty() {
        return Vec::new();
    }
    // Sort top-to-bottom (tie-break left) so a single greedy pass can grow rows:
    // each box either extends the current (last) row or starts a new one.
    items.sort_by(|a, b| (a.y, a.x).cmp(&(b.y, b.x)));

    let mut rows: Vec<Vec<&Detection>> = Vec::new();
    for d in items {
        let joins_last = rows.last().is_some_and(|row| {
            let top = row.iter().map(|r| r.y).min().unwrap();
            let bottom = row.iter().map(|r| r.y + r.h).max().unwrap();
            let shorter = (bottom - top).min(d.h);
            shorter > 0 && overlap_1d(d.y, d.y + d.h, top, bottom) * 2 >= shorter
        });
        if joins_last {
            rows.last_mut().unwrap().push(d);
        } else {
            rows.push(vec![d]);
        }
    }

    let mut lines: Vec<OcrLine> = rows
        .into_iter()
        .map(|mut row| {
            row.sort_by(|a, b| (a.x, a.y).cmp(&(b.x, b.y)));
            let text: String = row.iter().map(|d| d.label.as_str()).collect();
            let confidence = row
                .iter()
                .map(|d| d.confidence)
                .fold(f32::INFINITY, f32::min);
            let x = row.iter().map(|d| d.x).min().unwrap();
            let y = row.iter().map(|d| d.y).min().unwrap();
            let right = row.iter().map(|d| d.x + d.w).max().unwrap();
            let bottom = row.iter().map(|d| d.y + d.h).max().unwrap();
            OcrLine {
                text,
                confidence,
                x,
                y,
                w: right - x,
                h: bottom - y,
            }
        })
        .collect();
    lines.sort_by(|a, b| (a.y, a.x).cmp(&(b.y, b.x)));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(label: &str, conf: f32, x: u32, y: u32, w: u32, h: u32) -> Detection {
        Detection {
            label: label.into(),
            confidence: conf,
            x,
            y,
            w,
            h,
        }
    }

    #[test]
    fn empty_input_yields_no_lines() {
        assert!(assemble_lines(&[]).is_empty());
    }

    #[test]
    fn single_detection_is_one_line() {
        let lines = assemble_lines(&[det("A", 0.9, 5, 10, 8, 20)]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "A");
        assert_eq!(
            (lines[0].x, lines[0].y, lines[0].w, lines[0].h),
            (5, 10, 8, 20)
        );
    }

    #[test]
    fn same_row_orders_left_to_right() {
        // Fed out of order; must come back "AB".
        let lines = assemble_lines(&[det("B", 0.8, 20, 10, 10, 20), det("A", 0.9, 0, 10, 10, 20)]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "AB");
    }

    #[test]
    fn union_bbox_covers_all_members() {
        let lines = assemble_lines(&[det("A", 0.9, 0, 10, 10, 20), det("B", 0.8, 20, 10, 10, 24)]);
        // x: 0..30, y: 10, bottom: max(30, 34) = 34 -> h = 24.
        assert_eq!(
            (lines[0].x, lines[0].y, lines[0].w, lines[0].h),
            (0, 10, 30, 24)
        );
    }

    #[test]
    fn line_confidence_is_minimum_member() {
        let lines = assemble_lines(&[det("A", 0.9, 0, 10, 10, 20), det("B", 0.3, 20, 10, 10, 20)]);
        assert!((lines[0].confidence - 0.3).abs() < 1e-6);
    }

    #[test]
    fn two_rows_are_separate_and_top_first() {
        let lines = assemble_lines(&[
            det("X", 0.9, 0, 100, 10, 20),
            det("Y", 0.9, 20, 100, 10, 20),
            det("A", 0.9, 0, 0, 10, 20),
            det("B", 0.9, 20, 0, 10, 20),
        ]);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "AB");
        assert_eq!(lines[1].text, "XY");
    }

    #[test]
    fn empty_labels_are_dropped() {
        let lines = assemble_lines(&[det("A", 0.9, 0, 10, 10, 20), det("", 0.9, 20, 10, 10, 20)]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "A");
    }

    #[test]
    fn slight_vertical_offset_still_one_row() {
        // 18/20 overlap of the shorter height -> same row.
        let lines = assemble_lines(&[det("A", 0.9, 0, 10, 10, 20), det("B", 0.9, 20, 12, 10, 20)]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "AB");
    }
}
