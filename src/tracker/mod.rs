//! A generic, temporal detection tracker.
//!
//! The tracker associates per-recognition [`Detection`]s to persistent tracks by
//! bounding-box IoU, ages tracks in and out over time, and consolidates each
//! track's recent reads into a single stable label plus mean confidence. It is
//! pure (no I/O, no GStreamer or OCR types) so it is host-testable and reusable
//! by any detection source.

use std::collections::VecDeque;

/// Axis-aligned bounding box in full-frame pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// One detection fed into the tracker for a single recognition.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub bbox: BBox,
}

/// Tuning for [`Tracker`].
#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    /// Minimum IoU for a detection to associate with an existing track.
    pub iou_threshold: f32,
    /// Number of recent reads kept per track for the mode/mean vote (min 1).
    pub window: usize,
    /// Recognitions a track must be seen before it is reported.
    pub min_hits: u32,
    /// Consecutive absences tolerated before a track is dropped.
    pub max_misses: u32,
}

/// A confirmed, consolidated track emitted by [`Tracker::confirmed`].
#[derive(Clone, Debug, PartialEq)]
pub struct ConfirmedTrack {
    pub id: u64,
    pub label: String,
    pub confidence: f32,
    pub bbox: BBox,
}

/// Intersection-over-union of two boxes. Returns `0.0` when either box is empty
/// (zero width or height) or the boxes do not overlap.
pub fn iou(a: BBox, b: BBox) -> f32 {
    if a.w <= 0.0 || a.h <= 0.0 || b.w <= 0.0 || b.h <= 0.0 {
        return 0.0;
    }
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = (a.x + a.w).min(b.x + b.w);
    let iy2 = (a.y + a.h).min(b.y + b.h);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let union = a.w * a.h + b.w * b.h - inter;
    inter / union
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x: f32, y: f32, w: f32, h: f32) -> BBox {
        BBox { x, y, w, h }
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        assert!((iou(b(0.0, 0.0, 10.0, 10.0), b(0.0, 0.0, 10.0, 10.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        assert_eq!(iou(b(0.0, 0.0, 10.0, 10.0), b(20.0, 20.0, 10.0, 10.0)), 0.0);
    }

    #[test]
    fn iou_half_overlap_is_one_third() {
        // Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150.
        let got = iou(b(0.0, 0.0, 10.0, 10.0), b(5.0, 0.0, 10.0, 10.0));
        assert!((got - (50.0 / 150.0)).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn iou_empty_box_is_zero() {
        assert_eq!(iou(b(0.0, 0.0, 0.0, 10.0), b(0.0, 0.0, 10.0, 10.0)), 0.0);
    }
}
