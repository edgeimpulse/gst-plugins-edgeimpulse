//! Compose mode: recognize each detector region in place and rewrite the
//! frame's metadata with the recognized text.
//!
//! Classic crop-fed OCR runs as a *separate* element downstream of
//! `edgeimpulsecrop`, consuming one cropped buffer per detection that carries a
//! `CropOriginMeta`. That handoff breaks across plugin variants because
//! `CropOriginMeta` is registered under a variant-suffixed API name, so a
//! detector crop from one variant `.so` and a recognizer from another never
//! share the meta type — the recognizer silently drops every buffer.
//!
//! Compose mode sidesteps that entirely: the recognizer runs in place on the
//! full frame, reads the *standard* `VideoRegionOfInterestMeta` the detector
//! attached (a shared GType from `libgstvideo`, identical across variants),
//! recognizes each region, removes the detector's boxes, and attaches one text
//! box per recognized region so a downstream overlay renders the recognized
//! text instead of the detector's generic labels.

use crate::detection::{take_detections, Detection};
use crate::ocr::backend::OcrLine;
use crate::ocr::shaping::{attach_results, filter_and_truncate};
use gstreamer as gst;

/// Read and remove the detector's ROI metas from `buf`, recognize each region
/// via `recognize`, then attach one text ROI per surviving line back onto
/// `buf`. Returns the kept lines (already filtered by confidence and truncated
/// to `max_len` chars) so the caller can post `ocr` bus messages for them.
///
/// `recognize` maps a detection to `Some((text, confidence))` when the region
/// yields text, or `None` to drop it (empty read, failed dictionary gate, or a
/// throttled/duplicate region). Each produced line inherits the detection's
/// full-frame coordinates so the text box lands exactly where the object was.
pub fn recognize_and_rewrite<F>(
    buf: &mut gst::BufferRef,
    min_confidence: f32,
    max_len: usize,
    mut recognize: F,
) -> Vec<OcrLine>
where
    F: FnMut(&Detection) -> Option<(String, f32)>,
{
    let detections = take_detections(buf);
    let lines: Vec<OcrLine> = detections
        .iter()
        .filter_map(|det| {
            recognize(det).map(|(text, confidence)| OcrLine {
                text,
                confidence,
                x: det.x,
                y: det.y,
                w: det.width,
                h: det.height,
            })
        })
        .collect();
    let lines = filter_and_truncate(lines, min_confidence, max_len);
    attach_results(buf, &lines);
    lines
}

/// Extract the `(x, y, w, h)` sub-region from a tightly-packed RGB frame,
/// clamped to the frame bounds. Returns the region's pixels plus its actual
/// (clamped) width and height, which may be smaller than requested when the box
/// runs past an edge. A fully out-of-bounds box yields an empty `(vec, 0, 0)`.
pub(crate) fn crop_region(
    frame: &[u8],
    frame_w: u32,
    frame_h: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> (Vec<u8>, u32, u32) {
    let x0 = x.min(frame_w);
    let y0 = y.min(frame_h);
    let x1 = x.saturating_add(w).min(frame_w);
    let y1 = y.saturating_add(h).min(frame_h);
    let cw = x1.saturating_sub(x0);
    let ch = y1.saturating_sub(y0);
    let mut out = Vec::with_capacity((cw as usize) * (ch as usize) * 3);
    let stride = frame_w as usize * 3;
    for row in y0..y1 {
        let start = row as usize * stride + x0 as usize * 3;
        let end = start + cw as usize * 3;
        out.extend_from_slice(&frame[start..end]);
    }
    (out, cw, ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gstreamer_video as gst_video;

    fn add_detection(buf: &mut gst::BufferRef, rect: (u32, u32, u32, u32), label: &str, id: u64) {
        let mut roi = gst_video::VideoRegionOfInterestMeta::add(buf, label, rect);
        let s = gst::Structure::builder("detection")
            .field("label", label)
            .field("confidence", 0.99_f64)
            .field("object_id", id)
            .build();
        roi.add_param(s);
    }

    /// Collect `(roi_name, detection_label, confidence, rect)` for every ROI
    /// meta currently on the buffer.
    fn metas(buf: &gst::BufferRef) -> Vec<(String, String, f64, (u32, u32, u32, u32))> {
        buf.iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .map(|m| {
                let p = m.params().find(|p| p.name() == "detection");
                let label = p
                    .as_ref()
                    .and_then(|p| p.get::<String>("label").ok())
                    .unwrap_or_default();
                let conf = p
                    .as_ref()
                    .and_then(|p| p.get::<f64>("confidence").ok())
                    .unwrap_or_default();
                (m.roi_type().to_string(), label, conf, m.rect())
            })
            .collect()
    }

    #[test]
    fn rewrites_detector_boxes_with_recognized_text() {
        gst::init().unwrap();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        // Three detector boxes with the generic labels the overlay would show.
        add_detection(b, (10, 20, 30, 40), "class", 1);
        add_detection(b, (50, 60, 15, 25), "class", 2);
        add_detection(b, (5, 5, 8, 8), "unknown", 3);

        // Recognize text for the first and third regions; the middle yields none.
        let lines = recognize_and_rewrite(b, 0.0, 256, |det| match det.object_id {
            1 => Some(("HELLO".to_string(), 0.9)),
            3 => Some(("42".to_string(), 0.8)),
            _ => None,
        });

        // Returned lines carry the recognized text at the detection coordinates.
        assert_eq!(lines.len(), 2);
        let hello = lines.iter().find(|l| l.text == "HELLO").unwrap();
        assert_eq!((hello.x, hello.y, hello.w, hello.h), (10, 20, 30, 40));
        let forty_two = lines.iter().find(|l| l.text == "42").unwrap();
        assert_eq!(
            (forty_two.x, forty_two.y, forty_two.w, forty_two.h),
            (5, 5, 8, 8)
        );

        // The buffer now carries exactly the two text boxes and NONE of the
        // detector's original "class"/"unknown" labels.
        let got = metas(b);
        assert_eq!(
            got.len(),
            2,
            "detector boxes must be replaced, not appended"
        );
        assert!(
            !got.iter()
                .any(|(_, label, _, _)| label == "class" || label == "unknown"),
            "no detector label may survive compose"
        );
        assert!(got.iter().any(|(name, label, _, rect)| name == "HELLO"
            && label == "HELLO"
            && *rect == (10, 20, 30, 40)));
        assert!(got
            .iter()
            .any(|(name, label, _, rect)| name == "42" && label == "42" && *rect == (5, 5, 8, 8)));
    }

    #[test]
    fn drops_lines_below_min_confidence() {
        gst::init().unwrap();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        add_detection(b, (10, 20, 30, 40), "class", 1);
        add_detection(b, (5, 5, 8, 8), "class", 2);

        // "OK"@0.9 passes, "lo"@0.4 is below the 0.85 gate.
        let lines = recognize_and_rewrite(b, 0.85, 256, |det| match det.object_id {
            1 => Some(("OK".to_string(), 0.9)),
            2 => Some(("lo".to_string(), 0.4)),
            _ => None,
        });

        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "OK");
        let got = metas(b);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].1, "OK");
    }

    /// 4×4 RGB frame where each pixel's red channel encodes `row * 4 + col`.
    fn ramp_frame() -> Vec<u8> {
        let mut f = Vec::new();
        for row in 0..4u8 {
            for col in 0..4u8 {
                f.extend_from_slice(&[row * 4 + col, 0, 0]);
            }
        }
        f
    }

    #[test]
    fn crop_region_extracts_subrect() {
        let frame = ramp_frame();
        // Region (1,1) 2×2 -> reds {5,6, 9,10}.
        let (out, w, h) = crop_region(&frame, 4, 4, 1, 1, 2, 2);
        assert_eq!((w, h), (2, 2));
        let reds: Vec<u8> = out.chunks_exact(3).map(|p| p[0]).collect();
        assert_eq!(reds, vec![5, 6, 9, 10]);
    }

    #[test]
    fn crop_region_clamps_to_frame_bounds() {
        let frame = ramp_frame();
        // Region starts at (3,3) with size 4×4 but the frame is only 4×4, so it
        // clamps to a 1×1 crop of the bottom-right pixel (red = 15).
        let (out, w, h) = crop_region(&frame, 4, 4, 3, 3, 4, 4);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, vec![15, 0, 0]);

        // Fully out of bounds -> empty.
        let (out, w, h) = crop_region(&frame, 4, 4, 10, 10, 2, 2);
        assert_eq!((w, h), (0, 0));
        assert!(out.is_empty());
    }
}
