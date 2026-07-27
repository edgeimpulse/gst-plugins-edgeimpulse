//! Shared detection extraction from `VideoRegionOfInterestMeta`.
//!
//! Edge Impulse detection elements (e.g. `edgeimpulsevideoinfer`) attach one
//! `VideoRegionOfInterestMeta` per detected object, carrying a `detection`
//! `GstStructure` param (`label`, `confidence`, `object_id`). Both the
//! `edgeimpulsecrop` element (which crops one buffer per detection) and the
//! `edgeimpulseocr` recognizer's compose mode (which recognizes each region in
//! place) read these, so the extraction lives here in one implementation.
//!
//! This module has no runner/inference dependency, so it builds regardless of
//! which backend features are enabled.

use gst::buffer::BufferMetaForeachAction;
use gstreamer as gst;
use gstreamer_video as gst_video;
use std::ops::ControlFlow;

/// A detection read from a `VideoRegionOfInterestMeta`. Coordinates are in the
/// full-frame pixel space the meta was attached in.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub label: String,
    pub confidence: f64,
    pub object_id: u64,
}

/// Read the `label`/`confidence`/`object_id` of a ROI's `detection` param, if
/// present. Missing fields default to empty / zero.
fn read_detection_param(roi: &gst_video::VideoRegionOfInterestMeta) -> (String, f64, u64) {
    let mut label = String::new();
    let mut confidence = 0.0_f64;
    let mut object_id = 0_u64;
    if let Some(param) = roi.params().find(|p| p.name() == "detection") {
        if let Ok(l) = param.get::<String>("label") {
            label = l;
        }
        if let Ok(c) = param.get::<f64>("confidence") {
            confidence = c;
        }
        if let Ok(id) = param.get::<u64>("object_id") {
            object_id = id;
        }
    }
    (label, confidence, object_id)
}

/// Read every detection bounding box from the buffer's
/// `VideoRegionOfInterestMeta` entries, without modifying the buffer.
pub fn extract_detections(buffer: &gst::BufferRef) -> Vec<Detection> {
    buffer
        .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
        .map(|roi| {
            let (x, y, width, height) = roi.rect();
            let (label, confidence, object_id) = read_detection_param(&roi);
            Detection {
                x,
                y,
                width,
                height,
                label,
                confidence,
                object_id,
            }
        })
        .collect()
}

/// Read every detection carrying a `detection` param **and remove those metas**
/// from the buffer. Used by compose mode, which replaces the detector's boxes
/// with recognized-text boxes on the same full-frame buffer. ROIs without a
/// `detection` param (foreign metadata) are left in place.
pub fn take_detections(buffer: &mut gst::BufferRef) -> Vec<Detection> {
    let mut out = Vec::new();
    buffer.foreach_meta_mut(|mut meta| {
        let action = match meta.downcast_ref::<gst_video::VideoRegionOfInterestMeta>() {
            Some(roi) if roi.params().any(|p| p.name() == "detection") => {
                let (x, y, width, height) = roi.rect();
                let (label, confidence, object_id) = read_detection_param(roi);
                out.push(Detection {
                    x,
                    y,
                    width,
                    height,
                    label,
                    confidence,
                    object_id,
                });
                BufferMetaForeachAction::Remove
            }
            _ => BufferMetaForeachAction::Keep,
        };
        ControlFlow::Continue(action)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_detection(
        buf: &mut gst::BufferRef,
        rect: (u32, u32, u32, u32),
        label: &str,
        confidence: f64,
        object_id: u64,
    ) {
        let mut roi = gst_video::VideoRegionOfInterestMeta::add(buf, label, rect);
        let s = gst::Structure::builder("detection")
            .field("label", label)
            .field("confidence", confidence)
            .field("object_id", object_id)
            .build();
        roi.add_param(s);
    }

    #[test]
    fn extract_reads_label_confidence_and_object_id() {
        gst::init().unwrap();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        add_detection(b, (10, 20, 30, 40), "cat", 0.9, 7);
        add_detection(b, (50, 60, 15, 25), "dog", 0.5, 8);

        let dets = extract_detections(b);
        assert_eq!(dets.len(), 2);
        let cat = dets.iter().find(|d| d.label == "cat").unwrap();
        assert_eq!(
            (
                cat.x,
                cat.y,
                cat.width,
                cat.height,
                cat.confidence,
                cat.object_id
            ),
            (10, 20, 30, 40, 0.9, 7)
        );
        let dog = dets.iter().find(|d| d.label == "dog").unwrap();
        assert_eq!(
            (
                dog.x,
                dog.y,
                dog.width,
                dog.height,
                dog.confidence,
                dog.object_id
            ),
            (50, 60, 15, 25, 0.5, 8)
        );
    }

    #[test]
    fn extract_defaults_when_detection_param_absent() {
        gst::init().unwrap();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        // A ROI with no `detection` param (foreign metadata).
        gst_video::VideoRegionOfInterestMeta::add(b, "roi", (1, 2, 3, 4));

        let dets = extract_detections(b);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].label, "");
        assert_eq!(dets[0].confidence, 0.0);
        assert_eq!(dets[0].object_id, 0);
        assert_eq!(
            (dets[0].x, dets[0].y, dets[0].width, dets[0].height),
            (1, 2, 3, 4)
        );
    }

    #[test]
    fn take_reads_and_consumes_only_detection_rois() {
        gst::init().unwrap();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        add_detection(b, (10, 20, 30, 40), "cat", 0.9, 7);
        // A foreign ROI without a `detection` param must be left in place.
        gst_video::VideoRegionOfInterestMeta::add(b, "keepme", (1, 1, 2, 2));

        let dets = take_detections(b);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].label, "cat");

        // Only the foreign ROI remains.
        let remaining: Vec<_> = b
            .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .map(|m| m.rect())
            .collect();
        assert_eq!(remaining, vec![(1, 1, 2, 2)]);
    }
}
