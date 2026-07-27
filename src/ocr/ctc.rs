//! Greedy CTC decoding for CRNN sequence recognizers.
//!
//! Input logits are a flat row-major `[T, C]` slice: `T` timesteps, `C`
//! classes, class index 0 reserved as the CTC blank. Decoding takes the argmax
//! per timestep, collapses consecutive repeats, drops blanks and maps the
//! remaining class indices through `charset`. Confidence is softmaxed only for
//! genuine logits; rows that are already probability distributions (softmax
//! applied inside the model head) are used as-is.

/// Result of decoding a single crop.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub text: String,
    pub confidence: f32,
}

/// Parse the `charset` property string into per-class characters.
/// Index 0 is the CTC blank; its glyph is only a placeholder and never emitted.
pub fn parse_charset(s: &str) -> Vec<char> {
    s.chars().collect()
}

/// Numerically-stable softmax over a single timestep slice.
fn softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Heuristic: a row is already a probability distribution when every entry is a
/// finite value in `[0, 1]` (small tolerance) and the entries do not sum to much
/// more than 1. Such rows come from model heads that apply softmax internally
/// (e.g. the PaddleOCR CRNN recognizer, whose int8 output dequantises to
/// `[0, 0.996]`). Re-softmaxing them would collapse a confident peak toward
/// `1/C`, so we detect and use them as-is.
///
/// The sum is only bounded from ABOVE: int8 quantisation of a peaked softmax lets
/// the long tail underflow to zero, so a genuine probability row can sum well
/// below 1 (observed ~0.92 on device). An upper bound still rejects unnormalised
/// rows whose values happen to sit in `[0, 1]` but sum far above 1.
fn is_probability_row(row: &[f32]) -> bool {
    let mut sum = 0.0f32;
    for &v in row {
        if !v.is_finite() || !(-1e-3..=1.0 + 1e-3).contains(&v) {
            return false;
        }
        sum += v;
    }
    sum <= 1.0 + 5e-2
}

/// Greedy CTC decode. Returns empty text with confidence 0.0 when nothing
/// survives collapsing (all-blank), or when `num_classes` is inconsistent.
pub fn ctc_greedy_decode(logits: &[f32], num_classes: usize, charset: &[char]) -> Decoded {
    if num_classes == 0 || charset.len() != num_classes || logits.len() < num_classes {
        return Decoded {
            text: String::new(),
            confidence: 0.0,
        };
    }
    let timesteps = logits.len() / num_classes;
    let mut text = String::new();
    let mut kept_probs: Vec<f32> = Vec::new();
    let mut prev_idx: Option<usize> = None;

    for t in 0..timesteps {
        let row = &logits[t * num_classes..(t + 1) * num_classes];
        // argmax is invariant under softmax (monotonic), so read it from the raw
        // row and only pay for softmax on the chosen class' confidence below.
        let (idx, &raw_max) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let is_repeat = prev_idx == Some(idx);
        prev_idx = Some(idx);
        if idx == 0 || is_repeat {
            continue; // blank or collapsed repeat
        }
        text.push(charset[idx]);
        // Confidence is the probability of the chosen class. If the head already
        // emitted probabilities, use the raw value; only softmax genuine logits.
        let prob = if is_probability_row(row) {
            raw_max
        } else {
            softmax(row)[idx]
        };
        kept_probs.push(prob);
    }

    let confidence = if kept_probs.is_empty() {
        0.0
    } else {
        kept_probs.iter().sum::<f32>() / kept_probs.len() as f32
    };
    Decoded { text, confidence }
}

/// Dictionary allowlist gate. Empty dictionary allows everything.
pub fn passes_dictionary(text: &str, dictionary: &[String]) -> bool {
    dictionary.is_empty() || dictionary.iter().any(|w| w == text)
}

/// Parse the comma-separated `dictionary` property into trimmed, non-empty entries.
pub fn parse_dictionary(s: &str) -> Vec<String> {
    s.split(',')
        .map(|w| w.trim().to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cs() -> Vec<char> {
        parse_charset("_AB")
    }

    #[test]
    fn collapses_consecutive_repeats() {
        // A, A(repeat->collapsed), B  => "AB"
        let logits = vec![
            0.0, 10.0, 0.0, // A
            0.0, 10.0, 0.0, // A (repeat)
            0.0, 0.0, 10.0, // B
        ];
        let d = ctc_greedy_decode(&logits, 3, &cs());
        assert_eq!(d.text, "AB");
        assert!(d.confidence > 0.9, "confidence was {}", d.confidence);
    }

    #[test]
    fn blank_separates_repeated_chars() {
        // A, blank, A => "AA" (blank breaks the repeat collapse)
        let logits = vec![
            0.0, 10.0, 0.0, // A
            10.0, 0.0, 0.0, // blank
            0.0, 10.0, 0.0, // A
        ];
        let d = ctc_greedy_decode(&logits, 3, &cs());
        assert_eq!(d.text, "AA");
    }

    #[test]
    fn all_blank_yields_empty() {
        let logits = vec![10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let d = ctc_greedy_decode(&logits, 3, &cs());
        assert_eq!(d.text, "");
        assert_eq!(d.confidence, 0.0);
    }

    #[test]
    fn rejects_inconsistent_charset() {
        let logits = vec![0.0, 10.0, 0.0];
        let d = ctc_greedy_decode(&logits, 3, &parse_charset("_A")); // len 2 != 3
        assert_eq!(
            d,
            Decoded {
                text: String::new(),
                confidence: 0.0
            }
        );
    }

    #[test]
    fn already_softmaxed_rows_are_not_resoftmaxed() {
        // PaddleOCR CRNN heads emit post-softmax probabilities. Re-softmaxing them
        // squashes a ~0.99 peak toward uniform (the on-device double-softmax bug
        // that reported ~0.006 confidence). A row that already sums to 1 must be
        // used as-is, so the reported confidence stays close to the true peak.
        let logits = vec![
            0.005, 0.990, 0.005, // A  (already a probability distribution)
            0.005, 0.005, 0.990, // B
        ];
        let d = ctc_greedy_decode(&logits, 3, &cs());
        assert_eq!(d.text, "AB");
        assert!(
            (d.confidence - 0.990).abs() < 0.02,
            "confidence was {} (expected ~0.99; re-softmax bug would give ~0.57)",
            d.confidence
        );
    }

    #[test]
    fn int8_underflowed_probability_row_is_detected() {
        // int8-quantised softmax (scale 1/256): the 437-class tail underflows to
        // zero, so a real probability row sums to LESS than 1 (EDGE99 on-device:
        // peak ~0.92, row sum ~0.92). A too-tight `|sum-1|` test rejects these and
        // re-softmaxes them back to ~0.006. Such sub-unit rows must still count as
        // probabilities.
        let logits = vec![
            0.0, 0.92, 0.0, // A: valid prob peak 0.92, tail underflowed (sum 0.92)
        ];
        let d = ctc_greedy_decode(&logits, 3, &cs());
        assert_eq!(d.text, "A");
        assert!(
            (d.confidence - 0.92).abs() < 0.02,
            "confidence was {} (expected ~0.92; sum<1 must not trigger re-softmax)",
            d.confidence
        );
    }

    #[test]
    fn unnormalised_row_in_unit_range_is_still_softmaxed() {
        // Values all within [0,1] but summing well above 1 are not a probability
        // distribution (e.g. an unnormalised head). The sum upper bound must reject
        // them so they are softmaxed rather than reported raw.
        let logits = vec![0.6, 0.8, 0.7]; // argmax = "A" (0.8), sum 2.1
        let d = ctc_greedy_decode(&logits, 3, &cs());
        assert_eq!(d.text, "A");
        assert!(
            d.confidence < 0.5,
            "confidence was {} (expected softmax ~0.37, not the raw 0.8)",
            d.confidence
        );
    }

    #[test]
    fn raw_logits_are_softmaxed_for_confidence() {
        // Genuine (unnormalised) logits must still be softmaxed so a dominant class
        // yields high confidence.
        let logits = vec![
            0.0, 12.0, 0.0, // A (raw logits, not a distribution)
        ];
        let d = ctc_greedy_decode(&logits, 3, &cs());
        assert_eq!(d.text, "A");
        assert!(d.confidence > 0.9, "confidence was {}", d.confidence);
    }

    #[test]
    fn dictionary_gates_membership() {
        assert!(passes_dictionary("ABC", &[])); // empty = allow all
        assert!(passes_dictionary("ABC", &vec!["ABC".to_string()]));
        assert!(!passes_dictionary("ABC", &vec!["XYZ".to_string()]));
    }

    #[test]
    fn parse_dictionary_trims_and_drops_empty() {
        assert_eq!(
            parse_dictionary("ABC, XYZ ,, 123"),
            vec!["ABC".to_string(), "XYZ".to_string(), "123".to_string()]
        );
    }
}
