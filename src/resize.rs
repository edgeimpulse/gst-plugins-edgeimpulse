//! Shared image-resize helpers used by the crop and video-inference elements.
//!
//! These are pure (no GStreamer or runner dependencies) so they can be used by
//! elements that build without an inference backend (e.g. `edgeimpulsecrop`).

/// How an image is fitted to target dimensions during preprocessing.
///
/// Shared by `edgeimpulsecrop` and `edgeimpulsevideoinfer` so both match how a
/// model was trained.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ResizeMode {
    /// Stretch to the target, ignoring aspect ratio (legacy behavior).
    #[default]
    Squash,
    /// Scale preserving aspect ratio to fit within the target, then center on a
    /// zero-padded (black) canvas. Mirrors Edge Impulse's
    /// `EI_CLASSIFIER_RESIZE_FIT_LONGEST` preprocessing so recognizer/classifier
    /// input matches how the model was trained.
    FitLongest,
}

impl ResizeMode {
    /// Parse from a property string; unknown values fall back to `Squash`.
    pub fn from_property(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "fit-longest" | "fit" | "longest" => ResizeMode::FitLongest,
            _ => ResizeMode::Squash,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ResizeMode::Squash => "squash",
            ResizeMode::FitLongest => "fit-longest",
        }
    }
}

/// Dimensions to scale a `src_w`×`src_h` image to so it fits within
/// `dst_w`×`dst_h` while preserving aspect ratio (FIT_LONGEST). The result is
/// clamped to at least 1px and never exceeds the target on either axis.
pub fn fit_longest_dims(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return (dst_w.max(1), dst_h.max(1));
    }
    let src_aspect = src_w as f32 / src_h as f32;
    let dst_aspect = dst_w as f32 / dst_h as f32;
    if src_aspect > dst_aspect {
        // Wider than target: fill width, shrink height.
        (
            dst_w,
            ((dst_w as f32 / src_aspect) as u32).max(1).min(dst_h),
        )
    } else {
        // Taller than (or equal to) target: fill height, shrink width.
        (
            ((dst_h as f32 * src_aspect) as u32).max(1).min(dst_w),
            dst_h,
        )
    }
}

/// Dimensions to scale a `src_w`×`src_h` image to so it *fills* `dst_w`×`dst_h`
/// while preserving aspect ratio (FIT_SHORTEST). At least as large as the target
/// on both axes; the overflow is meant to be center-cropped by [`crop_center`].
pub fn fit_shortest_dims(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return (dst_w.max(1), dst_h.max(1));
    }
    let s = (dst_w as f32 / src_w as f32).max(dst_h as f32 / src_h as f32);
    (
        ((src_w as f32 * s).round() as u32).max(dst_w),
        ((src_h as f32 * s).round() as u32).max(dst_h),
    )
}

/// Center a `src_w`×`src_h` image (`channels` bytes/pixel, tightly packed) onto
/// a black `dst_w`×`dst_h` canvas. Returns exactly `dst_w*dst_h*channels` bytes.
pub fn pad_center(
    scaled: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: usize,
) -> Vec<u8> {
    let mut canvas = vec![0u8; dst_w as usize * dst_h as usize * channels];
    if src_w == 0 || src_h == 0 || src_w > dst_w || src_h > dst_h {
        return canvas;
    }
    let start_x = (dst_w - src_w) / 2;
    let start_y = (dst_h - src_h) / 2;
    let dst_row = dst_w as usize * channels;
    let src_row = src_w as usize * channels;
    if scaled.len() < src_row * src_h as usize {
        return canvas;
    }
    for row in 0..src_h as usize {
        let s = row * src_row;
        let d = (start_y as usize + row) * dst_row + start_x as usize * channels;
        canvas[d..d + src_row].copy_from_slice(&scaled[s..s + src_row]);
    }
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_property_parses_fit_longest_and_defaults_to_squash() {
        assert_eq!(
            ResizeMode::from_property("fit-longest"),
            ResizeMode::FitLongest
        );
        assert_eq!(
            ResizeMode::from_property("FIT_LONGEST"),
            ResizeMode::FitLongest
        );
        assert_eq!(ResizeMode::from_property("squash"), ResizeMode::Squash);
        assert_eq!(ResizeMode::from_property("nonsense"), ResizeMode::Squash);
    }

    #[test]
    fn fit_longest_dims_scales_tall_source_to_height() {
        // src taller than target: limited by height, width shrinks (padded sides).
        assert_eq!(fit_longest_dims(10, 100, 320, 48), (4, 48));
    }

    #[test]
    fn fit_longest_dims_scales_wide_source_to_width() {
        // src wider than target: limited by width, height shrinks (padded top/bottom).
        assert_eq!(fit_longest_dims(100, 10, 32, 48), (32, 3));
    }

    #[test]
    fn fit_longest_dims_matching_aspect_fills_target() {
        assert_eq!(fit_longest_dims(200, 30, 320, 48), (320, 48));
    }

    #[test]
    fn pad_center_rgb_centers_on_black_canvas() {
        // 2x2 red centered on 4x4 -> corners black, center red.
        let red = vec![255u8, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
        let out = pad_center(&red, 2, 2, 4, 4, 3);
        assert_eq!(out.len(), 4 * 4 * 3);
        assert_eq!(&out[0..3], &[0, 0, 0], "corner must be black");
        let idx = (4 + 1) * 3; // (row 1, col 1)
        assert_eq!(&out[idx..idx + 3], &[255, 0, 0], "center must be red");
    }

    #[test]
    fn pad_center_gray_centers_on_black_canvas() {
        let gray = vec![200u8, 200, 200, 200]; // 2x2, 1 channel
        let out = pad_center(&gray, 2, 2, 4, 4, 1);
        assert_eq!(out.len(), 16);
        assert_eq!(out[0], 0, "corner must be black");
        assert_eq!(out[4 + 1], 200, "center must be gray value");
    }

    #[test]
    fn fit_shortest_dims_fills_and_overflows() {
        // Equal aspect: exact fill.
        assert_eq!(fit_shortest_dims(200, 30, 320, 48), (320, 48));
        // Tall source: width fills, height overflows.
        assert_eq!(fit_shortest_dims(100, 400, 320, 48), (320, 1280));
        // Wide source: height-limited scale, width overflows target.
        assert_eq!(fit_shortest_dims(400, 100, 320, 48), (320, 80));
    }
}
