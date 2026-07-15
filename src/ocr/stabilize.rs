//! Glue between recognized OCR lines and the generic [`crate::tracker`].
//!
//! Maps `OcrLine`s into tracker detections, advances the tracker for one
//! recognition, and maps the confirmed tracks into [`StabilizedLine`]s carrying
//! a per-coordinate velocity. The OCR worker calls [`stabilize`] once per
//! recognition when the tracker is enabled and [`passthrough`] otherwise. The
//! element calls [`extrapolate_box`] per displayed frame. Pure and
//! host-testable.

use crate::ocr::backend::OcrLine;
use crate::tracker::{BBox, Detection, Tracker};

/// A recognized line after tracking: box as floats plus a per-coordinate
/// velocity (`[vx, vy, vw, vh]`, units/second) used for per-frame extrapolation.
/// `velocity` is `[0.0; 4]` when box prediction is off.
#[derive(Clone, Debug, PartialEq)]
pub struct StabilizedLine {
    pub text: String,
    pub confidence: f32,
    pub bbox: [f32; 4],
    pub velocity: [f32; 4],
}

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
/// seconds since the previous recognition) and return the stabilized lines.
/// Empty-text lines are dropped before tracking so they never spawn tracks.
/// Returned lines are ordered by track age (oldest first), not by input order.
pub fn stabilize(tracker: &mut Tracker, lines: Vec<OcrLine>, dt: f32) -> Vec<StabilizedLine> {
    let detections: Vec<Detection> = lines
        .iter()
        .filter(|l| !l.text.is_empty())
        .map(line_to_detection)
        .collect();
    tracker.update(&detections, dt);
    tracker
        .confirmed()
        .into_iter()
        .map(|c| StabilizedLine {
            text: c.label,
            confidence: c.confidence,
            bbox: [c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h],
            velocity: c.velocity,
        })
        .collect()
}

/// Map raw recognition lines straight to [`StabilizedLine`]s with zero velocity,
/// used when no tracker is active (both stabilization capabilities off).
pub fn passthrough(lines: Vec<OcrLine>) -> Vec<StabilizedLine> {
    lines
        .into_iter()
        .map(|l| StabilizedLine {
            text: l.text,
            confidence: l.confidence,
            bbox: [l.x as f32, l.y as f32, l.w as f32, l.h as f32],
            velocity: [0.0; 4],
        })
        .collect()
}

/// Linearly extrapolate a box to display time and clamp it to the frame.
///
/// `dt_s` is `(display_pts - reference_pts)` in seconds; it is clamped to
/// `[0, dt_cap]` so a stalled worker cannot fling the box off-screen. The box
/// origin is clamped to the frame and its extent is clamped so it never spills
/// past the right/bottom edges. With `velocity == [0; 4]` the result is the
/// rounded input box regardless of `dt_s`.
pub fn extrapolate_box(
    bbox: [f32; 4],
    velocity: [f32; 4],
    dt_s: f32,
    dt_cap: f32,
    width: u32,
    height: u32,
) -> (u32, u32, u32, u32) {
    let dt = dt_s.clamp(0.0, dt_cap);
    let x = (bbox[0] + velocity[0] * dt)
        .round()
        .clamp(0.0, width as f32) as u32;
    let y = (bbox[1] + velocity[1] * dt)
        .round()
        .clamp(0.0, height as f32) as u32;
    let w = (bbox[2] + velocity[2] * dt).round().max(0.0) as u32;
    let h = (bbox[3] + velocity[3] * dt).round().max(0.0) as u32;
    let w = w.min(width.saturating_sub(x));
    let h = h.min(height.saturating_sub(y));
    (x, y, w, h)
}

/// Tracker time step (seconds) from the previous and current recognized PTS
/// (milliseconds), plus the PTS to carry forward. A non-forward step — a
/// missing PTS (which upstream decodes to 0) or a back-dated one — yields
/// `dt = 0` and KEEPS the last good `prev`, so a dropout followed by recovery
/// cannot produce a huge catch-up dt. Genuine forward gaps are clamped to
/// `max_dt_s` to survive PTS discontinuities (e.g. seeks). Returns
/// `(dt, next_prev)`.
pub fn tracker_dt(prev_pts_ms: Option<i64>, pts_ms: i64, max_dt_s: f32) -> (f32, i64) {
    match prev_pts_ms {
        Some(prev) if pts_ms > prev => {
            let dt = ((pts_ms - prev) as f32 / 1000.0).min(max_dt_s);
            (dt, pts_ms)
        }
        Some(prev) => (0.0, prev),
        None => (0.0, pts_ms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::{KalmanConfig, TrackerConfig};

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
        assert_eq!(out[0].bbox, [0.0, 0.0, 20.0, 10.0]);
        assert_eq!(out[0].velocity, [0.0; 4]);
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

    fn kalman_tracker(responsiveness: f32) -> Tracker {
        Tracker::new(TrackerConfig {
            iou_threshold: 0.3,
            window: 10,
            min_hits: 1,
            max_misses: 5,
            kalman: Some(KalmanConfig::from_responsiveness(responsiveness)),
            consolidate_labels: true,
        })
    }

    #[test]
    fn velocity_propagates_from_tracker() {
        let mut t = kalman_tracker(1.0);
        let mut x = 0u32;
        for _ in 0..40 {
            stabilize(&mut t, vec![line("A", 0.9, x, 0, 10, 10)], 1.0);
            x += 4;
        }
        let out = stabilize(&mut t, vec![line("A", 0.9, x, 0, 10, 10)], 1.0);
        assert!(out[0].velocity[0] > 2.0, "vx {}", out[0].velocity[0]);
    }

    #[test]
    fn passthrough_has_zero_velocity_and_float_box() {
        let out = passthrough(vec![line("Hi", 0.7, 3, 4, 5, 6)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].bbox, [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(out[0].velocity, [0.0; 4]);
    }

    #[test]
    fn extrapolate_zero_velocity_returns_rounded_box() {
        // Fractional input so a `.trunc()`/`as u32` regression would be caught
        // (round-half-away-from-zero: .4→down, .5/.6→up).
        let got = extrapolate_box([10.4, 20.6, 30.5, 40.5], [0.0; 4], 5.0, 0.5, 640, 480);
        assert_eq!(got, (10, 21, 31, 41));
    }

    #[test]
    fn extrapolate_moves_box_by_velocity_times_dt() {
        let got = extrapolate_box(
            [10.0, 10.0, 20.0, 20.0],
            [4.0, 0.0, 0.0, 0.0],
            1.0,
            2.0,
            640,
            480,
        );
        assert_eq!(got, (14, 10, 20, 20));
    }

    #[test]
    fn extrapolate_caps_dt() {
        // dt_s huge but dt_cap=0.5 ⇒ only 0.5s of motion applied (4*0.5=2).
        let got = extrapolate_box(
            [10.0, 0.0, 20.0, 20.0],
            [4.0, 0.0, 0.0, 0.0],
            100.0,
            0.5,
            640,
            480,
        );
        assert_eq!(got.0, 12);
    }

    #[test]
    fn extrapolate_clamps_to_frame() {
        // x pushed past the right edge is clamped; width then clamps to 0.
        let got = extrapolate_box(
            [630.0, 0.0, 20.0, 20.0],
            [1000.0, 0.0, 0.0, 0.0],
            1.0,
            0.5,
            640,
            480,
        );
        assert_eq!(got.0, 640);
        assert_eq!(got.2, 0);
    }

    #[test]
    fn extrapolate_clamps_to_bottom_edge() {
        // Mirror of the right-edge case on the y axis: guards against a
        // transposed clamp that used `width`/`x` for the vertical extent.
        let got = extrapolate_box(
            [0.0, 470.0, 20.0, 20.0],
            [0.0, 1000.0, 0.0, 0.0],
            1.0,
            0.5,
            640,
            480,
        );
        assert_eq!(got.1, 480);
        assert_eq!(got.3, 0);
    }

    #[test]
    fn extrapolate_negative_dt_is_clamped_to_zero() {
        let got = extrapolate_box(
            [10.0, 10.0, 20.0, 20.0],
            [4.0, 4.0, 0.0, 0.0],
            -3.0,
            0.5,
            640,
            480,
        );
        assert_eq!(got, (10, 10, 20, 20));
    }

    #[test]
    fn tracker_dt_first_recognition_is_zero() {
        assert_eq!(tracker_dt(None, 5000, 1.0), (0.0, 5000));
    }

    #[test]
    fn tracker_dt_normal_forward_step() {
        let (dt, prev) = tracker_dt(Some(5000), 5040, 1.0);
        assert!((dt - 0.04).abs() < 1e-6, "dt {dt}");
        assert_eq!(prev, 5040);
    }

    #[test]
    fn tracker_dt_missing_pts_keeps_prev_and_zero_dt() {
        // A missing PTS decodes to 0 upstream; must not corrupt the baseline.
        assert_eq!(tracker_dt(Some(5000), 0, 1.0), (0.0, 5000));
    }

    #[test]
    fn tracker_dt_backdated_pts_keeps_prev_and_zero_dt() {
        // A non-zero but back-dated PTS travels the same non-forward arm.
        assert_eq!(tracker_dt(Some(5000), 4000, 1.0), (0.0, 5000));
    }

    #[test]
    fn tracker_dt_forward_discontinuity_is_clamped() {
        // 50s seek-style jump clamps to max_dt_s, baseline still advances.
        assert_eq!(tracker_dt(Some(5000), 55000, 1.0), (1.0, 55000));
    }
}
