//! Pure, host-testable helpers: filtering results and turning an [`OcrLine`]
//! into the ROI metadata / bus message the rest of the stack consumes.
use crate::ocr::backend::OcrLine;
use gstreamer as gst;
use gstreamer_video as gst_video;

/// Drop empty / low-confidence lines and truncate over-long text (by chars).
pub fn filter_and_truncate(
    mut lines: Vec<OcrLine>,
    min_confidence: f32,
    max_len: usize,
) -> Vec<OcrLine> {
    lines.retain(|l| !l.text.trim().is_empty() && l.confidence >= min_confidence);
    for l in &mut lines {
        if l.text.chars().count() > max_len {
            l.text = l.text.chars().take(max_len).collect();
        }
    }
    lines
}

/// Attach one `VideoRegionOfInterestMeta` per line with a `detection` param
/// (`label`, `confidence`) — exactly what `edgeimpulseoverlay` renders.
pub fn attach_results(buf: &mut gst::BufferRef, lines: &[OcrLine]) {
    for line in lines {
        let mut roi = gst_video::VideoRegionOfInterestMeta::add(
            buf,
            line.text.as_str(),
            (line.x, line.y, line.w, line.h),
        );
        let s = gst::Structure::builder("detection")
            .field("label", line.text.as_str())
            .field("confidence", line.confidence as f64)
            .build();
        roi.add_param(s);
    }
}

/// The parent detection's box, in original-frame pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Where a recognized line was measured, and what it was read from.
///
/// `frame_width` / `frame_height` are the dimensions a consumer must divide the
/// line's box by to get resolution-independent coordinates. They are **not**
/// always the dimensions of the buffer passing through the element: on the
/// crop-fed path the buffer is the crop, and the frame is whatever
/// `CropOriginMeta` says the crop was taken from.
///
/// `parent` is set only when OCR ran on a crop of a parent detection. The crop
/// rect is that detection's box, which is what makes it a usable join key: it is
/// correct with or without an object tracker, unlike `object_id`, which stays 0
/// when the upstream detector does not track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OcrOrigin {
    pub frame_width: u32,
    pub frame_height: u32,
    pub parent: Option<ParentBox>,
}

/// Build an `ocr` element-message structure for one recognized line.
///
/// `parent_*` fields are present as a group or absent as a group, so a consumer
/// never sees a half-described parent.
pub fn build_ocr_message(line: &OcrLine, pts_ms: i64, origin: &OcrOrigin) -> gst::Structure {
    let mut builder = gst::Structure::builder("ocr")
        .field("text", line.text.as_str())
        .field("confidence", line.confidence as f64)
        .field("x", line.x as i32)
        .field("y", line.y as i32)
        .field("width", line.w as i32)
        .field("height", line.h as i32)
        .field("timestamp", pts_ms)
        .field("frame_width", origin.frame_width as i32)
        .field("frame_height", origin.frame_height as i32);
    if let Some(parent) = origin.parent {
        builder = builder
            .field("parent_x", parent.x as i32)
            .field("parent_y", parent.y as i32)
            .field("parent_width", parent.w as i32)
            .field("parent_height", parent.h as i32);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, conf: f32) -> OcrLine {
        OcrLine {
            text: text.into(),
            confidence: conf,
            x: 1,
            y: 2,
            w: 3,
            h: 4,
        }
    }

    #[test]
    fn filter_drops_empty_and_low_confidence() {
        let out = filter_and_truncate(
            vec![line("hi", 0.9), line("  ", 0.9), line("lo", 0.1)],
            0.5,
            10,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "hi");
    }

    #[test]
    fn filter_truncates_long_text() {
        let out = filter_and_truncate(vec![line("abcdef", 1.0)], 0.0, 3);
        assert_eq!(out[0].text, "abc");
    }

    #[test]
    fn filter_truncates_multibyte_by_chars() {
        // Truncation must count characters, not bytes: a byte-slice
        // reimplementation would split the 2-byte 'é' and return "á" or panic.
        let out = filter_and_truncate(vec![line("áéíóú", 1.0)], 0.0, 2);
        assert_eq!(out[0].text, "áé");
    }

    #[test]
    fn attaches_one_roi_meta_per_line() {
        gst::init().unwrap();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        attach_results(b, &[line("SN-42", 0.8), line("SN-99", 0.6)]);
        let metas: Vec<_> = b
            .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .collect();
        assert_eq!(metas.len(), 2);
        // Order-independent: each line must yield its own `detection` param with
        // the matching label and confidence (catches overwrite / early-return).
        let got: Vec<(String, f64)> = metas
            .iter()
            .map(|m| {
                let p = m.params().find(|p| p.name() == "detection").unwrap();
                (
                    p.get::<String>("label").unwrap(),
                    p.get::<f64>("confidence").unwrap(),
                )
            })
            .collect();
        assert!(got.contains(&("SN-42".to_string(), 0.8f32 as f64)));
        assert!(got.contains(&("SN-99".to_string(), 0.6f32 as f64)));
    }

    #[test]
    fn builds_ocr_message_fields() {
        gst::init().unwrap();
        let origin = OcrOrigin {
            frame_width: 0,
            frame_height: 0,
            parent: None,
        };
        let s = build_ocr_message(&line("hello", 0.7), 1234, &origin);
        assert_eq!(s.name(), "ocr");
        assert_eq!(s.get::<String>("text").unwrap(), "hello");
        assert_eq!(s.get::<f64>("confidence").unwrap(), 0.7f32 as f64);
        assert_eq!(s.get::<i32>("x").unwrap(), 1);
        assert_eq!(s.get::<i32>("y").unwrap(), 2);
        assert_eq!(s.get::<i32>("width").unwrap(), 3);
        assert_eq!(s.get::<i32>("height").unwrap(), 4);
        assert_eq!(s.get::<i64>("timestamp").unwrap(), 1234);
    }

    #[test]
    fn builds_ocr_message_with_frame_dimensions() {
        gst::init().unwrap();
        let origin = OcrOrigin {
            frame_width: 1920,
            frame_height: 1080,
            parent: None,
        };
        let s = build_ocr_message(&line("hello", 0.7), 1234, &origin);
        assert_eq!(s.get::<i32>("frame_width").unwrap(), 1920);
        assert_eq!(s.get::<i32>("frame_height").unwrap(), 1080);
        assert!(
            !s.has_field("parent_x"),
            "a full-frame read has no parent detection"
        );
    }

    #[test]
    fn builds_ocr_message_with_parent_lineage() {
        gst::init().unwrap();
        let origin = OcrOrigin {
            frame_width: 1920,
            frame_height: 1080,
            parent: Some(ParentBox {
                x: 40,
                y: 60,
                w: 120,
                h: 32,
            }),
        };
        let s = build_ocr_message(&line("SN-42", 0.9), 7, &origin);
        assert_eq!(s.get::<i32>("parent_x").unwrap(), 40);
        assert_eq!(s.get::<i32>("parent_y").unwrap(), 60);
        assert_eq!(s.get::<i32>("parent_width").unwrap(), 120);
        assert_eq!(s.get::<i32>("parent_height").unwrap(), 32);
    }
}
