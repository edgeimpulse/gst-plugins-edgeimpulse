//! A generic, temporal detection tracker.
//!
//! The tracker associates per-recognition [`Detection`]s to persistent tracks by
//! bounding-box IoU, ages tracks in and out over time, and consolidates each
//! track's recent reads into a single stable label plus mean confidence. It is
//! pure (no I/O, no GStreamer or OCR types) so it is host-testable and reusable
//! by any detection source.

mod kalman;
use kalman::Kalman1D;
pub use kalman::KalmanConfig;

use std::collections::VecDeque;

/// Axis-aligned bounding box in full-frame pixels. `(x, y)` is the top-left
/// corner; `w` and `h` are expected to be positive (a zero/negative extent
/// yields an IoU of 0).
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
    /// When `Some`, each track runs a per-coordinate constant-velocity Kalman
    /// filter: association uses the predicted box and the reported box/velocity
    /// are Kalman estimates. `None` ⇒ Phase-1 (raw last box, zero velocity).
    pub kalman: Option<KalmanConfig>,
    /// When `true`, `confirmed()` reports the consolidated mode label + mean
    /// confidence; when `false`, it reports the latest matched read. Lets box
    /// prediction run without forcing text consolidation.
    pub consolidate_labels: bool,
}

/// A confirmed, consolidated track emitted by [`Tracker::confirmed`].
#[derive(Clone, Debug, PartialEq)]
pub struct ConfirmedTrack {
    /// Stable per-object id assigned by the tracker. Part of the generic
    /// tracker contract so other consumers can correlate objects across
    /// frames; the OCR overlay consumer does not read it yet.
    #[allow(dead_code)]
    pub id: u64,
    pub label: String,
    pub confidence: f32,
    pub bbox: BBox,
    /// Per-coordinate velocity `[vx, vy, vw, vh]` in units/second from the
    /// Kalman filters, or `[0.0; 4]` when Kalman is disabled.
    pub velocity: [f32; 4],
}

struct Track {
    id: u64,
    bbox: BBox,
    history: VecDeque<(String, f32)>,
    hits: u32,
    misses: u32,
    /// Per-coordinate Kalman filters `[x, y, w, h]` when box prediction is on.
    filters: Option<[Kalman1D; 4]>,
}

/// Associates detections to tracks over time and consolidates their reads.
pub struct Tracker {
    config: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u64,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 0,
        }
    }

    /// Feed one recognition's detections: predict tracks forward by `dt`
    /// seconds (Kalman only), associate to tracks by IoU on the predicted box,
    /// spawn new tracks for unmatched detections, and age out tracks that were
    /// not seen. `dt` is the wall-clock gap since the previous recognition;
    /// `dt <= 0` skips prediction.
    pub fn update(&mut self, detections: &[Detection], dt: f32) {
        // 1. Predict Kalman tracks forward to "now" so association and the
        //    reported box use the predicted position.
        for track in &mut self.tracks {
            if let Some(filters) = track.filters.as_mut() {
                for f in filters.iter_mut() {
                    f.predict(dt);
                }
                track.bbox = bbox_from_filters(filters);
            }
        }

        // 2. Candidate (detection, track) pairs above the IoU threshold. With
        //    Kalman on, `track.bbox` is the predicted box (step 1); without, it
        //    is the last matched raw box (Phase 1).
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for (di, d) in detections.iter().enumerate() {
            for (ti, track) in self.tracks.iter().enumerate() {
                let score = iou(d.bbox, track.bbox);
                if score >= self.config.iou_threshold {
                    candidates.push((di, ti, score));
                }
            }
        }
        // Descending: highest IoU first.
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
        let mut det_taken = vec![false; detections.len()];
        let mut track_taken = vec![false; self.tracks.len()];
        let mut match_for_det: Vec<Option<usize>> = vec![None; detections.len()];
        for (di, ti, _score) in candidates {
            if det_taken[di] || track_taken[ti] {
                continue;
            }
            det_taken[di] = true;
            track_taken[ti] = true;
            match_for_det[di] = Some(ti);
        }

        // 3. Refresh matched tracks from their detection.
        let window = self.config.window.max(1);
        for (di, d) in detections.iter().enumerate() {
            if let Some(ti) = match_for_det[di] {
                let track = &mut self.tracks[ti];
                if let Some(filters) = track.filters.as_mut() {
                    filters[0].update(d.bbox.x);
                    filters[1].update(d.bbox.y);
                    filters[2].update(d.bbox.w);
                    filters[3].update(d.bbox.h);
                    track.bbox = bbox_from_filters(filters);
                } else {
                    track.bbox = d.bbox;
                }
                track.history.push_back((d.label.clone(), d.confidence));
                while track.history.len() > window {
                    track.history.pop_front();
                }
                track.hits = track.hits.saturating_add(1);
                track.misses = 0;
            }
        }

        // 4. Age unmatched tracks; drop those absent too long.
        for (ti, track) in self.tracks.iter_mut().enumerate() {
            if !track_taken[ti] {
                track.misses = track.misses.saturating_add(1);
            }
        }
        let max_misses = self.config.max_misses;
        self.tracks.retain(|t| t.misses <= max_misses);

        // 5. Spawn tracks for unmatched detections.
        for (di, d) in detections.iter().enumerate() {
            if !det_taken[di] {
                let mut history = VecDeque::new();
                history.push_back((d.label.clone(), d.confidence));
                let filters = self.config.kalman.map(|cfg| {
                    [
                        Kalman1D::new(d.bbox.x, cfg),
                        Kalman1D::new(d.bbox.y, cfg),
                        Kalman1D::new(d.bbox.w, cfg),
                        Kalman1D::new(d.bbox.h, cfg),
                    ]
                });
                self.tracks.push(Track {
                    id: self.next_id,
                    bbox: d.bbox,
                    history,
                    hits: 1,
                    misses: 0,
                    filters,
                });
                self.next_id = self.next_id.wrapping_add(1);
            }
        }
    }

    /// Snapshot of tracks seen at least `min_hits` times.
    pub fn confirmed(&self) -> Vec<ConfirmedTrack> {
        self.tracks
            .iter()
            .filter(|t| t.hits >= self.config.min_hits)
            .map(|t| {
                let (label, confidence) = if self.config.consolidate_labels {
                    consolidate(&t.history)
                } else {
                    latest_read(&t.history)
                };
                let velocity = match &t.filters {
                    Some(f) => velocity_from_filters(f),
                    None => [0.0; 4],
                };
                ConfirmedTrack {
                    id: t.id,
                    label,
                    confidence,
                    bbox: t.bbox,
                    velocity,
                }
            })
            .collect()
    }
}

/// Pick the most frequent label in `history` (ties broken by higher mean
/// confidence for that label, then the more recent occurrence) and the mean
/// confidence across the whole window. `history` is non-empty for a live track;
/// an empty history yields an empty label and `0.0`.
fn consolidate(history: &VecDeque<(String, f32)>) -> (String, f32) {
    if history.is_empty() {
        return (String::new(), 0.0);
    }
    let mean = history.iter().map(|(_, c)| *c).sum::<f32>() / history.len() as f32;

    let mut labels: Vec<&str> = Vec::with_capacity(history.len());
    let mut counts: Vec<u32> = Vec::with_capacity(history.len());
    let mut conf_sums: Vec<f32> = Vec::with_capacity(history.len());
    let mut last_index: Vec<usize> = Vec::with_capacity(history.len());
    for (i, (label, conf)) in history.iter().enumerate() {
        if let Some(pos) = labels.iter().position(|l| *l == label.as_str()) {
            counts[pos] += 1;
            conf_sums[pos] += *conf;
            last_index[pos] = i;
        } else {
            labels.push(label.as_str());
            counts.push(1);
            conf_sums.push(*conf);
            last_index.push(i);
        }
    }

    let mut best = 0usize;
    for i in 1..labels.len() {
        let mean_i = conf_sums[i] / counts[i] as f32;
        let mean_best = conf_sums[best] / counts[best] as f32;
        // Tie-break chain: higher count, then higher mean confidence, then more
        // recent. The `mean_i == mean_best` check uses exact float equality on
        // purpose — only identical bit patterns reach the last_index recency
        // tie-break; any difference in bit pattern is resolved directly by `>`.
        let better = counts[i] > counts[best]
            || (counts[i] == counts[best]
                && (mean_i > mean_best
                    || (mean_i == mean_best && last_index[i] > last_index[best])));
        if better {
            best = i;
        }
    }
    (labels[best].to_string(), mean)
}

/// The most recent `(label, confidence)` read for a track, or `("", 0.0)` for
/// an empty history. Used when label consolidation is disabled.
fn latest_read(history: &VecDeque<(String, f32)>) -> (String, f32) {
    match history.back() {
        Some((label, conf)) => (label.clone(), *conf),
        None => (String::new(), 0.0),
    }
}

/// Read the four filter positions as a box (top-left + size).
fn bbox_from_filters(f: &[Kalman1D; 4]) -> BBox {
    BBox {
        x: f[0].position(),
        y: f[1].position(),
        w: f[2].position(),
        h: f[3].position(),
    }
}

/// Read the four filter velocities as `[vx, vy, vw, vh]`.
fn velocity_from_filters(f: &[Kalman1D; 4]) -> [f32; 4] {
    [
        f[0].velocity(),
        f[1].velocity(),
        f[2].velocity(),
        f[3].velocity(),
    ]
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

    const DT: f32 = 1.0;

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

    #[test]
    fn iou_contained_box_is_area_ratio() {
        // 2x2 box fully inside a 4x4 box at origin: inter=4, union=16.
        let got = iou(b(0.0, 0.0, 4.0, 4.0), b(1.0, 1.0, 2.0, 2.0));
        assert!((got - (4.0 / 16.0)).abs() < 1e-6, "got {got}");
    }

    fn det(label: &str, conf: f32, bx: BBox) -> Detection {
        Detection {
            label: label.into(),
            confidence: conf,
            bbox: bx,
        }
    }

    fn cfg(iou_threshold: f32, window: usize, min_hits: u32, max_misses: u32) -> TrackerConfig {
        TrackerConfig {
            iou_threshold,
            window,
            min_hits,
            max_misses,
            kalman: None,
            consolidate_labels: true,
        }
    }

    #[test]
    fn track_reported_only_after_min_hits() {
        let mut t = Tracker::new(cfg(0.3, 10, 2, 5));
        let d = det("A", 0.5, b(0.0, 0.0, 10.0, 10.0));
        t.update(&[d.clone()], DT);
        assert_eq!(t.confirmed().len(), 0, "one hit < min_hits(2)");
        t.update(&[d], DT);
        assert_eq!(t.confirmed().len(), 1, "two hits reaches min_hits");
    }

    #[test]
    fn track_dropped_after_max_misses() {
        let mut t = Tracker::new(cfg(0.3, 10, 1, 2));
        t.update(&[det("A", 0.5, b(0.0, 0.0, 10.0, 10.0))], DT);
        assert_eq!(t.confirmed().len(), 1);
        t.update(&[], DT); // miss 1
        t.update(&[], DT); // miss 2 (== max_misses, still alive)
        assert_eq!(t.confirmed().len(), 1, "held through misses <= max_misses");
        t.update(&[], DT); // miss 3 (> max_misses -> dropped)
        assert_eq!(t.confirmed().len(), 0);
    }

    #[test]
    fn overlapping_detections_do_not_share_a_track() {
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        t.update(&[det("A", 0.5, b(0.0, 0.0, 10.0, 10.0))], DT);
        // Two detections both overlap the single existing track; only one may
        // claim it, the other must spawn a new track.
        t.update(
            &[
                det("A", 0.5, b(1.0, 0.0, 10.0, 10.0)),
                det("B", 0.5, b(2.0, 0.0, 10.0, 10.0)),
            ],
            DT,
        );
        assert_eq!(t.confirmed().len(), 2);
    }

    #[test]
    fn low_iou_detection_spawns_new_track() {
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        t.update(&[det("A", 0.5, b(0.0, 0.0, 10.0, 10.0))], DT);
        t.update(&[det("B", 0.5, b(100.0, 100.0, 10.0, 10.0))], DT);
        assert_eq!(t.confirmed().len(), 2);
    }

    #[test]
    fn confirmed_bbox_is_latest_detection() {
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        t.update(&[det("A", 0.5, b(0.0, 0.0, 10.0, 10.0))], DT);
        t.update(&[det("A", 0.5, b(3.0, 0.0, 10.0, 10.0))], DT);
        assert_eq!(t.confirmed()[0].bbox, b(3.0, 0.0, 10.0, 10.0));
    }

    #[test]
    fn label_is_most_frequent_not_highest_confidence() {
        // "full" recurs (count 2) at low confidence; "frag" appears once at high
        // confidence. Mode must pick the recurring "full".
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        let bx = b(0.0, 0.0, 10.0, 10.0);
        t.update(&[det("full", 0.30, bx)], DT);
        t.update(&[det("frag", 0.90, bx)], DT);
        t.update(&[det("full", 0.35, bx)], DT);
        assert_eq!(t.confirmed()[0].label, "full");
    }

    #[test]
    fn label_tie_broken_by_higher_mean_confidence() {
        // "A" and "B" each appear once (tie on count); "B" has higher confidence.
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        let bx = b(0.0, 0.0, 10.0, 10.0);
        t.update(&[det("A", 0.40, bx)], DT);
        t.update(&[det("B", 0.80, bx)], DT);
        assert_eq!(t.confirmed()[0].label, "B");
    }

    #[test]
    fn confidence_is_mean_of_window() {
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        let bx = b(0.0, 0.0, 10.0, 10.0);
        t.update(&[det("A", 0.20, bx)], DT);
        t.update(&[det("A", 0.60, bx)], DT);
        assert!((t.confirmed()[0].confidence - 0.40).abs() < 1e-6);
    }

    #[test]
    fn window_forgets_old_reads() {
        // window=2: after three "A","A","B" reads the history is ["A","B"], a
        // tie broken by "B" (more recent, equal confidence).
        let mut t = Tracker::new(cfg(0.3, 2, 1, 5));
        let bx = b(0.0, 0.0, 10.0, 10.0);
        t.update(&[det("A", 0.50, bx)], DT);
        t.update(&[det("A", 0.50, bx)], DT);
        t.update(&[det("B", 0.50, bx)], DT);
        assert_eq!(t.confirmed()[0].label, "B");
    }

    #[test]
    fn misses_reset_on_re_match() {
        // A track that accumulates misses and is then re-matched must reset its
        // miss counter to 0 (not keep counting from where it left off).
        let mut t = Tracker::new(cfg(0.3, 10, 1, 2));
        let bx = b(0.0, 0.0, 10.0, 10.0);
        t.update(&[det("A", 0.9, bx)], DT); // hits=1, misses=0
        t.update(&[], DT); // miss 1
        t.update(&[], DT); // miss 2 (still <= max_misses=2)
        t.update(&[det("A", 0.9, bx)], DT); // re-match: misses back to 0
        t.update(&[], DT); // miss 1 (would already be > max_misses if reset failed)
        t.update(&[], DT); // miss 2
        assert_eq!(t.confirmed().len(), 1, "re-match reset misses to 0");
        t.update(&[], DT); // miss 3 -> exceeds max_misses -> dropped
        assert_eq!(
            t.confirmed().len(),
            0,
            "dropped after misses exceed max post-reset"
        );
    }

    fn cfg_kalman(
        iou_threshold: f32,
        window: usize,
        min_hits: u32,
        max_misses: u32,
        responsiveness: f32,
    ) -> TrackerConfig {
        TrackerConfig {
            iou_threshold,
            window,
            min_hits,
            max_misses,
            kalman: Some(KalmanConfig::from_responsiveness(responsiveness)),
            consolidate_labels: true,
        }
    }

    #[test]
    fn phase1_velocity_is_zero() {
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        let bx = b(0.0, 0.0, 10.0, 10.0);
        t.update(&[det("A", 0.5, bx)], DT);
        t.update(&[det("A", 0.5, bx)], DT);
        assert_eq!(t.confirmed()[0].velocity, [0.0; 4]);
    }

    #[test]
    fn latest_read_used_when_not_consolidating() {
        // window sees A, A, B; consolidate picks mode "A", latest-read picks "B".
        let mut config = cfg(0.3, 10, 1, 5);
        config.consolidate_labels = false;
        let mut t = Tracker::new(config);
        let bx = b(0.0, 0.0, 10.0, 10.0);
        t.update(&[det("A", 0.10, bx)], DT);
        t.update(&[det("A", 0.20, bx)], DT);
        t.update(&[det("B", 0.90, bx)], DT);
        let c = &t.confirmed()[0];
        assert_eq!(c.label, "B");
        assert!((c.confidence - 0.90).abs() < 1e-6);
    }

    #[test]
    fn kalman_estimates_object_velocity() {
        // Object moves +4px/frame at dt=1 ⇒ true vx ~4/s.
        let mut t = Tracker::new(cfg_kalman(0.3, 10, 1, 5, 1.0));
        let mut x = 0.0f32;
        for _ in 0..40 {
            t.update(&[det("A", 0.9, b(x, 0.0, 10.0, 10.0))], DT);
            x += 4.0;
        }
        let v = t.confirmed()[0].velocity;
        assert!((v[0] - 4.0).abs() < 0.6, "vx {}", v[0]);
        assert!(v[1].abs() < 0.3, "vy {}", v[1]);
    }

    #[test]
    fn prediction_keeps_one_track_through_a_fast_jump() {
        // Establish a steady +4px/frame velocity (each step within the IoU
        // gate), then jump to 84 (last match 76): raw IoU with 76 is 0.11
        // (<0.3), but the predicted box (~80) is only ~4px away (IoU ~0.43), so
        // Kalman keeps a single track.
        let mut t = Tracker::new(cfg_kalman(0.3, 10, 1, 5, 1.0));
        let mut x = 0.0f32;
        for _ in 0..20 {
            t.update(&[det("A", 0.9, b(x, 0.0, 10.0, 10.0))], DT);
            x += 4.0;
        }
        assert_eq!(t.confirmed().len(), 1, "single track while establishing");
        t.update(&[det("A", 0.9, b(84.0, 0.0, 10.0, 10.0))], DT);
        assert_eq!(
            t.confirmed().len(),
            1,
            "predicted association survives jump"
        );
    }

    #[test]
    fn phase1_loses_track_on_the_same_fast_jump() {
        // The same sequence without Kalman: the last raw box (76) does not
        // overlap the jumped detection (84, IoU 0.11), so the track ghosts
        // (still within max_misses) and a new one spawns ⇒ two confirmed tracks.
        let mut t = Tracker::new(cfg(0.3, 10, 1, 5));
        let mut x = 0.0f32;
        for _ in 0..20 {
            t.update(&[det("A", 0.9, b(x, 0.0, 10.0, 10.0))], DT);
            x += 4.0;
        }
        t.update(&[det("A", 0.9, b(84.0, 0.0, 10.0, 10.0))], DT);
        assert_eq!(
            t.confirmed().len(),
            2,
            "raw association spawns a second track"
        );
    }
}
