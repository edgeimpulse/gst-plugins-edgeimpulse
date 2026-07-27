//! Post-decode text normalization for recognized OCR text.
//!
//! A recognizer's charset can be much larger than a given solution cares about
//! (e.g. the pretrained PaddleOCR head emits full English incl. lowercase and
//! punctuation, while a solution may only read uppercase alphanumeric codes).
//! Normalization is a generic, model-agnostic text transform applied after CTC
//! decoding; the *choice* of mode lives in the caller/workflow, not here.

/// How to normalize recognized text before it is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalize {
    /// Emit the decoded text unchanged.
    None,
    /// Uppercase the text.
    Upper,
    /// Uppercase and keep only ASCII alphanumeric characters (`[A-Z0-9]`).
    UpperAlnum,
}

impl Normalize {
    /// Parse the `normalize` property string. Unknown values fall back to `None`
    /// so a hand-edited pipeline can never panic on an unexpected token.
    pub fn parse(s: &str) -> Self {
        match s {
            "upper" => Normalize::Upper,
            "upper-alnum" => Normalize::UpperAlnum,
            _ => Normalize::None,
        }
    }
}

/// Apply the normalization mode to a decoded string.
pub fn normalize_text(text: &str, mode: Normalize) -> String {
    match mode {
        Normalize::None => text.to_string(),
        Normalize::Upper => text.to_uppercase(),
        Normalize::UpperAlnum => text
            .to_uppercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_leaves_text_unchanged() {
        assert_eq!(normalize_text("aB-c 1!", Normalize::None), "aB-c 1!");
    }

    #[test]
    fn upper_uppercases_but_keeps_other_chars() {
        assert_eq!(normalize_text("aB-c 1!", Normalize::Upper), "AB-C 1!");
    }

    #[test]
    fn upper_alnum_uppercases_and_drops_non_alphanumeric() {
        assert_eq!(normalize_text("aB-c 1!", Normalize::UpperAlnum), "ABC1");
    }

    #[test]
    fn upper_alnum_drops_non_ascii_letters() {
        // `to_uppercase()` yields `ÁÉ`, neither of which is ASCII alphanumeric.
        assert_eq!(normalize_text("áé9", Normalize::UpperAlnum), "9");
    }

    #[test]
    fn parse_maps_known_modes_and_defaults_to_none() {
        assert_eq!(Normalize::parse("upper"), Normalize::Upper);
        assert_eq!(Normalize::parse("upper-alnum"), Normalize::UpperAlnum);
        assert_eq!(Normalize::parse("none"), Normalize::None);
        assert_eq!(Normalize::parse("bogus"), Normalize::None);
    }
}
