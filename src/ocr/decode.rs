//! `edge-impulse` OCR backend: decode the per-character detections an upstream
//! Edge Impulse object-detection model (run by `edgeimpulsevideoinfer`) attached
//! to the buffer into lines of text, reusing the shared `shaping` output path.
//!
//! Unlike the `ocrs` backend this evaluates no model: the character boxes are
//! already on the buffer as `VideoRegionOfInterestMeta` (in full-frame pixels,
//! with `label` + `confidence` in a `detection` param). We group them into rows,
//! concatenate each row into a line, and hand the lines to `shaping`.

use crate::ocr::backend::OcrLine;
use gst::buffer::BufferMetaForeachAction;
use gstreamer as gst;
use gstreamer_video as gst_video;
use std::ops::ControlFlow;

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
/// its vertical span overlaps the row's *current* span by at least half of the
/// shorter of (that span's height, the box's height); otherwise it starts a new
/// row. Because the row's span grows as boxes join, a monotonic half-height
/// vertical ramp (a slightly tilted single line) is intentionally chained into
/// one line — this tilt tolerance is by design. Line text is the row's labels
/// concatenated in x order (no spaces inserted); line confidence is the minimum
/// finite member confidence (NaN ignored, all-NaN → 0.0); line bbox is the union
/// of member boxes.
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
            // rows are non-empty by construction, so min/max never see an empty iter
            let top = row.iter().map(|r| r.y).min().unwrap();
            let bottom = row.iter().map(|r| r.y.saturating_add(r.h)).max().unwrap();
            let shorter = (bottom - top).min(d.h);
            shorter > 0 && overlap_1d(d.y, d.y.saturating_add(d.h), top, bottom) * 2 >= shorter
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
            // Ignore NaN confidences; an all-NaN (or empty) result yields a finite
            // 0.0 so a degenerate line can't slip past `min-confidence`.
            let confidence = row
                .iter()
                .map(|d| d.confidence)
                .filter(|c| !c.is_nan())
                .reduce(f32::min)
                .unwrap_or(0.0);
            let x = row.iter().map(|d| d.x).min().unwrap();
            let y = row.iter().map(|d| d.y).min().unwrap();
            let right = row.iter().map(|d| d.x.saturating_add(d.w)).max().unwrap();
            let bottom = row.iter().map(|d| d.y.saturating_add(d.h)).max().unwrap();
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

/// Read every character detection from the buffer's `VideoRegionOfInterestMeta`
/// entries **and remove those metas**. We consume the per-character boxes so
/// downstream renders only the assembled line, not the raw glyphs. Only ROIs
/// carrying a `detection` param (the Edge Impulse convention) are taken; any
/// other ROI meta is left in place.
pub fn take_detections(buf: &mut gst::BufferRef) -> Vec<Detection> {
    let mut out = Vec::new();
    buf.foreach_meta_mut(|mut meta| {
        let action = match meta.downcast_ref::<gst_video::VideoRegionOfInterestMeta>() {
            Some(roi) => match roi.params().find(|p| p.name() == "detection") {
                Some(p) => {
                    let (x, y, w, h) = roi.rect();
                    let label = p.get::<String>("label").unwrap_or_default();
                    let confidence = p.get::<f64>("confidence").unwrap_or(0.0) as f32;
                    out.push(Detection {
                        label,
                        confidence,
                        x,
                        y,
                        w,
                        h,
                    });
                    BufferMetaForeachAction::Remove
                }
                None => BufferMetaForeachAction::Keep,
            },
            None => BufferMetaForeachAction::Keep,
        };
        ControlFlow::Continue(action)
    });
    out
}

/// Full edge-impulse transform for one buffer: consume the upstream character
/// detections, assemble them into lines, filter by confidence / length, attach
/// one ROI meta per line, and return the lines so the caller can post `ocr`
/// bus messages.
///
/// Not idempotent: the attached line ROIs carry the same `detection` param shape
/// that `take_detections` consumes, so exactly one `edge-impulse` OCR element
/// should appear in a pipeline branch — a second pass would re-consume the
/// assembled lines as if they were characters.
pub fn process_buffer(
    buf: &mut gst::BufferRef,
    min_confidence: f32,
    max_len: usize,
) -> Vec<OcrLine> {
    let dets = take_detections(buf);
    let lines = assemble_lines(&dets);
    let lines = crate::ocr::shaping::filter_and_truncate(lines, min_confidence, max_len);
    crate::ocr::shaping::attach_results(buf, &lines);
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
    fn nan_confidence_does_not_corrupt_min() {
        // A stray NaN among finite confidences must not swallow the real minimum.
        let lines = assemble_lines(&[
            det("A", 0.9, 0, 10, 10, 20),
            det("B", f32::NAN, 20, 10, 10, 20),
            det("C", 0.3, 40, 10, 10, 20),
        ]);
        assert_eq!(lines.len(), 1);
        assert!((lines[0].confidence - 0.3).abs() < 1e-6);
    }

    #[test]
    fn all_nan_confidence_is_finite_not_infinity() {
        // A line whose characters all carry NaN confidence must not leak past a
        // confidence filter as +inf; it collapses to a finite 0.0.
        let lines = assemble_lines(&[
            det("A", f32::NAN, 0, 10, 10, 20),
            det("B", f32::NAN, 20, 10, 10, 20),
        ]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "AB");
        assert!(lines[0].confidence.is_finite());
        assert_eq!(lines[0].confidence, 0.0);
    }

    #[test]
    fn staircase_half_height_ramp_chains_into_one_line() {
        // A monotonic half-height vertical ramp (e.g. a slightly tilted single
        // line) is INTENTIONALLY grouped into one line by the growing-span join
        // rule. Pinning this so the tilt-tolerant behavior stays deliberate.
        let lines = assemble_lines(&[
            det("A", 0.9, 0, 0, 10, 10),
            det("B", 0.9, 10, 5, 10, 10),
            det("C", 0.9, 20, 10, 10, 10),
            det("D", 0.9, 30, 15, 10, 10),
        ]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "ABCD");
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

    #[test]
    fn nonzero_overlap_below_threshold_splits_rows() {
        // Vertical overlap is 4px; the shorter height is 20px, so 4*2 = 8 < 20
        // -> the two boxes must land on SEPARATE rows (top box first).
        let lines = assemble_lines(&[det("A", 0.9, 0, 0, 10, 20), det("B", 0.9, 0, 16, 10, 20)]);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "A");
        assert_eq!(lines[1].text, "B");
    }

    fn init() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| gst::init().expect("gst init"));
    }

    fn char_roi(buf: &mut gst::BufferRef, label: &str, x: u32, conf: f64) {
        let mut roi = gst_video::VideoRegionOfInterestMeta::add(buf, label, (x, 10, 10, 20));
        roi.add_param(
            gst::Structure::builder("detection")
                .field("label", label)
                .field("confidence", conf)
                .build(),
        );
    }

    #[test]
    fn take_detections_reads_and_consumes_detection_rois() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        char_roi(b, "A", 0, 0.9);
        char_roi(b, "B", 20, 0.8);
        let dets = take_detections(b);
        assert_eq!(dets.len(), 2);
        assert!(dets
            .iter()
            .any(|d| d.label == "A" && (d.confidence - 0.9).abs() < 1e-6));
        assert!(dets.iter().any(|d| d.label == "B"));
        // The character ROIs are consumed.
        assert_eq!(
            b.iter_meta::<gst_video::VideoRegionOfInterestMeta>()
                .count(),
            0
        );
    }

    #[test]
    fn take_detections_ignores_rois_without_detection_param() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        // A plain ROI with no `detection` param must be left untouched.
        gst_video::VideoRegionOfInterestMeta::add(b, "face", (0, 0, 4, 4));
        let dets = take_detections(b);
        assert!(dets.is_empty());
        assert_eq!(
            b.iter_meta::<gst_video::VideoRegionOfInterestMeta>()
                .count(),
            1
        );
    }

    #[test]
    fn take_detections_on_empty_buffer_is_empty() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        assert!(take_detections(b).is_empty());
    }

    #[test]
    fn process_buffer_consumes_chars_and_attaches_one_line() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        char_roi(b, "A", 0, 0.9);
        char_roi(b, "B", 20, 0.8);
        let lines = process_buffer(b, 0.0, 256);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "AB");
        // Exactly one ROI remains: the assembled line (char ROIs consumed).
        let labels: Vec<String> = b
            .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .filter_map(|m| {
                m.params()
                    .find(|p| p.name() == "detection")
                    .and_then(|p| p.get::<String>("label").ok())
            })
            .collect();
        assert_eq!(labels, vec!["AB".to_string()]);
    }

    #[test]
    fn process_buffer_filters_below_min_confidence() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        char_roi(b, "A", 0, 0.9);
        char_roi(b, "B", 20, 0.8); // line confidence = min = 0.8
        let lines = process_buffer(b, 0.85, 256);
        assert!(lines.is_empty());
        // Chars consumed, nothing attached.
        assert_eq!(
            b.iter_meta::<gst_video::VideoRegionOfInterestMeta>()
                .count(),
            0
        );
    }
}
