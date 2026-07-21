//! Greedy CTC decoding for CRNN sequence recognizers.
//!
//! Input logits are a flat row-major `[T, C]` slice: `T` timesteps, `C`
//! classes, class index 0 reserved as the CTC blank. Decoding softmaxes each
//! timestep, takes the argmax, collapses consecutive repeats, drops blanks and
//! maps the remaining class indices through `charset`.

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

/// Greedy CTC decode. Returns empty text with confidence 0.0 when nothing
/// survives collapsing (all-blank), or when `num_classes` is inconsistent.
pub fn ctc_greedy_decode(logits: &[f32], num_classes: usize, charset: &[char]) -> Decoded {
    if num_classes == 0 || charset.len() != num_classes || logits.len() < num_classes {
        return Decoded { text: String::new(), confidence: 0.0 };
    }
    let timesteps = logits.len() / num_classes;
    let mut text = String::new();
    let mut kept_probs: Vec<f32> = Vec::new();
    let mut prev_idx: Option<usize> = None;

    for t in 0..timesteps {
        let row = &logits[t * num_classes..(t + 1) * num_classes];
        let probs = softmax(row);
        let (idx, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, p)| (i, p))
            .unwrap_or((0, &0.0));

        let is_repeat = prev_idx == Some(idx);
        prev_idx = Some(idx);
        if idx == 0 || is_repeat {
            continue; // blank or collapsed repeat
        }
        text.push(charset[idx]);
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
        assert_eq!(d, Decoded { text: String::new(), confidence: 0.0 });
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
