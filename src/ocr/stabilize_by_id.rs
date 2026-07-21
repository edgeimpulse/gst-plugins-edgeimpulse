//! Per-`object_id` OCR read stabilizer.
//!
//! Upstream detection assigns a stable `object_id` to each tracked region, so
//! unlike the full-frame IoU tracker this stabilizer just keeps a bounded
//! history of reads per id, votes for the most frequent text and ages tracks on
//! frame boundaries. One "frame" is the group of crop buffers that share a PTS.

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
struct Track {
    reads: VecDeque<(String, f32)>,
    misses: u32,
}

/// Bounded-history voting stabilizer keyed by `object_id`.
#[derive(Debug)]
pub struct IdStabilizer {
    window: usize,
    min_hits: usize,
    max_misses: u32,
    tracks: HashMap<u64, Track>,
}

impl IdStabilizer {
    pub fn new(window: usize, min_hits: usize, max_misses: u32) -> Self {
        Self {
            window: window.max(1),
            min_hits: min_hits.max(1),
            max_misses,
            tracks: HashMap::new(),
        }
    }

    /// Record a read for `object_id`. Empty text is ignored (still counts as a
    /// frame appearance via `end_frame`'s `seen` set, but adds no vote).
    pub fn observe(&mut self, object_id: u64, text: String, confidence: f32) {
        if text.is_empty() {
            return;
        }
        let track = self.tracks.entry(object_id).or_insert_with(|| Track {
            reads: VecDeque::new(),
            misses: 0,
        });
        track.misses = 0;
        track.reads.push_back((text, confidence));
        while track.reads.len() > self.window {
            track.reads.pop_front();
        }
    }

    /// Consolidated `(text, confidence)` for `object_id`, or `None` when the
    /// track has fewer than `min_hits` reads.
    pub fn consolidated(&self, object_id: u64) -> Option<(String, f32)> {
        let track = self.tracks.get(&object_id)?;
        if track.reads.len() < self.min_hits {
            return None;
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for (t, _) in &track.reads {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        let winner = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(t, _)| t.to_string())?;
        let matching: Vec<f32> = track
            .reads
            .iter()
            .filter(|(t, _)| *t == winner)
            .map(|(_, c)| *c)
            .collect();
        let conf = matching.iter().sum::<f32>() / matching.len() as f32;
        Some((winner, conf))
    }

    /// Advance one frame boundary: any track not in `seen` gains a miss; tracks
    /// exceeding `max_misses` are expired.
    pub fn end_frame(&mut self, seen: &HashSet<u64>) {
        let mut expired = Vec::new();
        for (id, track) in self.tracks.iter_mut() {
            if !seen.contains(id) {
                track.misses += 1;
                if track.misses > self.max_misses {
                    expired.push(*id);
                }
            }
        }
        for id in expired {
            self.tracks.remove(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_min_hits_returns_none() {
        let mut s = IdStabilizer::new(5, 2, 3);
        s.observe(1, "AB".to_string(), 0.9);
        assert_eq!(s.consolidated(1), None);
    }

    #[test]
    fn reaches_min_hits_and_votes_majority() {
        let mut s = IdStabilizer::new(5, 2, 3);
        s.observe(1, "AB".to_string(), 0.9);
        s.observe(1, "AB".to_string(), 0.7);
        s.observe(1, "AC".to_string(), 0.5);
        let (text, conf) = s.consolidated(1).expect("should consolidate");
        assert_eq!(text, "AB");
        assert!((conf - 0.8).abs() < 1e-6, "conf was {conf}");
    }

    #[test]
    fn window_bounds_history() {
        let mut s = IdStabilizer::new(2, 1, 3);
        s.observe(1, "AA".to_string(), 0.1);
        s.observe(1, "BB".to_string(), 0.5);
        s.observe(1, "BB".to_string(), 0.5); // "AA" evicted by window=2
        assert_eq!(s.consolidated(1).unwrap().0, "BB");
    }

    #[test]
    fn empty_text_is_ignored() {
        let mut s = IdStabilizer::new(5, 1, 3);
        s.observe(1, String::new(), 0.9);
        assert_eq!(s.consolidated(1), None);
    }

    #[test]
    fn expires_after_max_misses() {
        let mut s = IdStabilizer::new(5, 1, 2);
        s.observe(7, "AB".to_string(), 0.9);
        let empty = HashSet::new();
        s.end_frame(&empty); // miss 1
        s.end_frame(&empty); // miss 2
        assert!(s.consolidated(7).is_some(), "still alive at max_misses");
        s.end_frame(&empty); // miss 3 > max_misses -> expired
        assert_eq!(s.consolidated(7), None);
    }

    #[test]
    fn seen_resets_misses() {
        let mut s = IdStabilizer::new(5, 1, 1);
        s.observe(7, "AB".to_string(), 0.9);
        let empty = HashSet::new();
        s.end_frame(&empty); // miss 1
        s.observe(7, "AB".to_string(), 0.9); // resets misses to 0
        let mut seen = HashSet::new();
        seen.insert(7u64);
        s.end_frame(&seen);
        assert!(s.consolidated(7).is_some());
    }
}
