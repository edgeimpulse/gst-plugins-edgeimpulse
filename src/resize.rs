//! Shared image-resize helpers used by the crop, video-inference and OCR
//! recognizer elements.
//!
//! These have no GStreamer or runner dependencies (only the `image` crate) so
//! they can be used by elements that build without an inference backend (e.g.
//! `edgeimpulsecrop`).

use image::imageops::FilterType;
use image::{ImageBuffer, RgbImage};

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
    /// Scale preserving aspect ratio to *fill* the target, then center-crop the
    /// overflow. Mirrors Edge Impulse's `EI_CLASSIFIER_RESIZE_FIT_SHORTEST`.
    FitShortest,
}

impl ResizeMode {
    /// Parse a property string into a canonical mode, or `None` if it is not one
    /// of the recognized values. Callers interpreting *model metadata* (rather
    /// than a user-supplied property) should use this to detect — and log — an
    /// unexpected mode instead of silently squashing.
    pub fn try_from_property(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "fit-longest" | "fit" | "longest" => Some(ResizeMode::FitLongest),
            "fit-shortest" | "shortest" => Some(ResizeMode::FitShortest),
            "squash" => Some(ResizeMode::Squash),
            _ => None,
        }
    }

    /// Parse from a property string; unknown values fall back to `Squash`.
    pub fn from_property(s: &str) -> Self {
        Self::try_from_property(s).unwrap_or(ResizeMode::Squash)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ResizeMode::Squash => "squash",
            ResizeMode::FitLongest => "fit-longest",
            ResizeMode::FitShortest => "fit-shortest",
        }
    }
}

/// User-facing resize-mode selection for `edgeimpulsevideoinfer`. `Auto` defers
/// to the model's declared mode (`ModelParameters.image_resize_mode`);
/// `Explicit` overrides it (e.g. for testing or when metadata is wrong).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ResizeModeSetting {
    #[default]
    Auto,
    Explicit(ResizeMode),
}

impl ResizeModeSetting {
    /// Parse from a property string. `"auto"` (case-insensitive) selects `Auto`;
    /// anything else is an explicit [`ResizeMode`].
    pub fn from_property(s: &str) -> Self {
        if s.trim().eq_ignore_ascii_case("auto") {
            ResizeModeSetting::Auto
        } else {
            ResizeModeSetting::Explicit(ResizeMode::from_property(s))
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ResizeModeSetting::Auto => "auto",
            ResizeModeSetting::Explicit(m) => m.as_str(),
        }
    }

    /// Resolve to a concrete [`ResizeMode`], reading the model's declared mode
    /// string when set to `Auto`.
    pub fn resolve(self, model_mode: &str) -> ResizeMode {
        match self {
            ResizeModeSetting::Auto => ResizeMode::from_property(model_mode),
            ResizeModeSetting::Explicit(m) => m,
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
    // Round (fit_longest_dims truncates): fill semantics need the scaled image to
    // reach the target on both axes; the trailing `.max(dst_*)` enforces it even
    // when rounding lands just under.
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

/// Center-crop a `src_w`×`src_h` image (`channels` bytes/pixel, tightly packed,
/// with `src_w >= dst_w` and `src_h >= dst_h`) down to `dst_w`×`dst_h`. Returns
/// exactly `dst_w*dst_h*channels` bytes. Invalid input yields a black canvas.
pub fn crop_center(
    scaled: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: usize,
) -> Vec<u8> {
    debug_assert!(
        dst_w == 0 || dst_h == 0 || (src_w >= dst_w && src_h >= dst_h),
        "crop_center expects src >= dst on both axes (got {src_w}x{src_h} -> {dst_w}x{dst_h})"
    );
    let mut out = vec![0u8; dst_w as usize * dst_h as usize * channels];
    if dst_w == 0 || dst_h == 0 || src_w < dst_w || src_h < dst_h {
        return out;
    }
    let start_x = (src_w - dst_w) / 2;
    let start_y = (src_h - dst_h) / 2;
    let src_row = src_w as usize * channels;
    let dst_row = dst_w as usize * channels;
    if scaled.len() < src_row * src_h as usize {
        return out;
    }
    for row in 0..dst_h as usize {
        let s = (start_y as usize + row) * src_row + start_x as usize * channels;
        let d = row * dst_row;
        out[d..d + dst_row].copy_from_slice(&scaled[s..s + dst_row]);
    }
    out
}

/// Resize `src` RGB (`src_w`×`src_h`, tightly packed) to `dst_w`×`dst_h` using
/// `mode`. Returns exactly `dst_w*dst_h*3` bytes. Invalid input yields a black
/// canvas of the target size.
///
/// Shared by `edgeimpulsecrop` (crop-then-resize) and the `edgeimpulseocr`
/// recognizer's compose mode (resize each detector region to the model input),
/// so both preprocess pixels identically to how the model was trained.
pub fn resize_rgb(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    mode: ResizeMode,
) -> Vec<u8> {
    let black = || vec![0u8; (dst_w as usize) * (dst_h as usize) * 3];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return black();
    }
    let img: RgbImage = match ImageBuffer::from_raw(src_w, src_h, src.to_vec()) {
        Some(i) => i,
        None => return black(),
    };

    match mode {
        ResizeMode::Squash => {
            image::imageops::resize(&img, dst_w, dst_h, FilterType::Triangle).into_raw()
        }
        ResizeMode::FitLongest => {
            // Scale by the limiting axis (aspect-preserving), then center on a
            // zero-padded canvas — Edge Impulse's FIT_LONGEST preprocessing.
            let (resize_w, resize_h) = fit_longest_dims(src_w, src_h, dst_w, dst_h);
            let scaled =
                image::imageops::resize(&img, resize_w, resize_h, FilterType::Triangle).into_raw();
            pad_center(&scaled, resize_w, resize_h, dst_w, dst_h, 3)
        }
        ResizeMode::FitShortest => {
            // Scale to fill (aspect-preserving), then center-crop the overflow —
            // Edge Impulse's FIT_SHORTEST preprocessing.
            let (resize_w, resize_h) = fit_shortest_dims(src_w, src_h, dst_w, dst_h);
            let scaled =
                image::imageops::resize(&img, resize_w, resize_h, FilterType::Triangle).into_raw();
            crop_center(&scaled, resize_w, resize_h, dst_w, dst_h, 3)
        }
    }
}

/// Forward mapping from original-image pixel coordinates to model-input pixel
/// coordinates for a [`ResizeMode`]: `model = orig * scale + offset`. Primarily
/// used to invert detection bounding boxes back to original-image space.
///
/// This uses a continuous float scale, whereas the pixel resize path
/// ([`fit_longest_dims`]/[`fit_shortest_dims`] + [`pad_center`]/[`crop_center`])
/// rounds to integer scaled dimensions and integer center offsets. The two agree
/// to within ~1px at realistic model input sizes; the gap only grows meaningful
/// at very small target dimensions (tens of px). If exact parity is ever needed,
/// derive the transform from the same integer scaled dims the pixel path uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResizeTransform {
    pub scale_x: f32,
    pub scale_y: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

impl ResizeTransform {
    /// Build the transform mapping a `src_w`×`src_h` image onto a `dst_w`×`dst_h`
    /// model input under `mode`.
    pub fn for_mode(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, mode: ResizeMode) -> Self {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Self {
                scale_x: 1.0,
                scale_y: 1.0,
                offset_x: 0.0,
                offset_y: 0.0,
            };
        }
        let (sw, sh) = (src_w as f32, src_h as f32);
        let (dw, dh) = (dst_w as f32, dst_h as f32);
        match mode {
            ResizeMode::Squash => Self {
                scale_x: dw / sw,
                scale_y: dh / sh,
                offset_x: 0.0,
                offset_y: 0.0,
            },
            ResizeMode::FitLongest | ResizeMode::FitShortest => {
                let s = if mode == ResizeMode::FitLongest {
                    (dw / sw).min(dh / sh)
                } else {
                    (dw / sw).max(dh / sh)
                };
                Self {
                    scale_x: s,
                    scale_y: s,
                    offset_x: (dw - sw * s) / 2.0,
                    offset_y: (dh - sh * s) / 2.0,
                }
            }
        }
    }

    /// Map a model-input point back to original-image coordinates.
    pub fn inverse_point(&self, mx: f32, my: f32) -> (f32, f32) {
        (
            (mx - self.offset_x) / self.scale_x,
            (my - self.offset_y) / self.scale_y,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: (u8, u8, u8)) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            v.push(rgb.0);
            v.push(rgb.1);
            v.push(rgb.2);
        }
        v
    }

    fn px(buf: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8) {
        let o = ((y * w + x) * 3) as usize;
        (buf[o], buf[o + 1], buf[o + 2])
    }

    #[test]
    fn resize_rgb_fit_longest_pads_narrow_source_with_black_sides() {
        // 48x48 white (aspect 1) into 320x48 (aspect ~6.67): limited by height,
        // resized to 48x48 centered at x=136, black on both sides.
        let src = solid(48, 48, (255, 255, 255));
        let out = resize_rgb(&src, 48, 48, 320, 48, ResizeMode::FitLongest);
        assert_eq!(out.len(), 320 * 48 * 3);
        assert_eq!(
            px(&out, 320, 0, 24),
            (0, 0, 0),
            "left padding must be black"
        );
        assert_eq!(
            px(&out, 320, 160, 24),
            (255, 255, 255),
            "center must hold the image"
        );
        assert_eq!(
            px(&out, 320, 315, 24),
            (0, 0, 0),
            "right padding must be black"
        );
    }

    #[test]
    fn resize_rgb_fit_longest_pads_wide_source_with_black_top_bottom() {
        // 640x48 white (aspect ~13.3) into 320x48: limited by width, resized to
        // 320x24 centered at y=12, black top and bottom.
        let src = solid(640, 48, (255, 255, 255));
        let out = resize_rgb(&src, 640, 48, 320, 48, ResizeMode::FitLongest);
        assert_eq!(
            px(&out, 320, 160, 0),
            (0, 0, 0),
            "top padding must be black"
        );
        assert_eq!(
            px(&out, 320, 160, 24),
            (255, 255, 255),
            "center must hold the image"
        );
        assert_eq!(
            px(&out, 320, 160, 47),
            (0, 0, 0),
            "bottom padding must be black"
        );
    }

    #[test]
    fn resize_rgb_fit_longest_matching_aspect_fills_without_padding() {
        // 640x96 (aspect ~6.67) into 320x48 (same aspect): fills entirely.
        let src = solid(640, 96, (200, 100, 50));
        let out = resize_rgb(&src, 640, 96, 320, 48, ResizeMode::FitLongest);
        assert_eq!(
            px(&out, 320, 0, 24),
            (200, 100, 50),
            "left edge filled (no padding)"
        );
        assert_eq!(
            px(&out, 320, 319, 24),
            (200, 100, 50),
            "right edge filled (no padding)"
        );
    }

    #[test]
    fn resize_rgb_squash_fills_entire_target_ignoring_aspect() {
        // Squash stretches a square to fill the wide target — no black padding.
        let src = solid(48, 48, (255, 255, 255));
        let out = resize_rgb(&src, 48, 48, 320, 48, ResizeMode::Squash);
        assert_eq!(px(&out, 320, 0, 24), (255, 255, 255));
        assert_eq!(px(&out, 320, 315, 24), (255, 255, 255));
    }

    #[test]
    fn resize_rgb_fit_shortest_crops_vertical_center() {
        // 4x8 with the middle 4 rows white; fit-shortest to 4x4 (s=1.0) crops the
        // top/bottom 2 rows, leaving an all-white 4x4 — a squash would blend grey.
        let (w, h) = (4u32, 8u32);
        let mut src = vec![0u8; (w * h) as usize * 3];
        for row in 2..6usize {
            for col in 0..4usize {
                let i = (row * 4 + col) * 3;
                src[i] = 255;
                src[i + 1] = 255;
                src[i + 2] = 255;
            }
        }
        let out = resize_rgb(&src, w, h, 4, 4, ResizeMode::FitShortest);
        assert_eq!(out.len(), 4 * 4 * 3);
        assert!(out.iter().all(|&b| b == 255), "center band fills the crop");
    }

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

    #[test]
    fn crop_center_rgb_takes_middle() {
        // 4x4 with a 2x2 red center on black; crop to 2x2 -> all red.
        let mut src = vec![0u8; 4 * 4 * 3];
        for (r, c) in [(1usize, 1usize), (1, 2), (2, 1), (2, 2)] {
            let i = (r * 4 + c) * 3;
            src[i] = 255;
        }
        let out = crop_center(&src, 4, 4, 2, 2, 3);
        assert_eq!(out.len(), 2 * 2 * 3);
        assert_eq!(&out[0..3], &[255, 0, 0], "top-left of crop is red");
        assert_eq!(&out[9..12], &[255, 0, 0], "bottom-right of crop is red");
    }

    #[test]
    fn crop_center_gray_takes_middle() {
        let mut src = vec![0u8; 16];
        for (r, c) in [(1usize, 1usize), (1, 2), (2, 1), (2, 2)] {
            src[r * 4 + c] = 200;
        }
        let out = crop_center(&src, 4, 4, 2, 2, 1);
        assert_eq!(out, vec![200, 200, 200, 200]);
    }

    #[test]
    fn crop_center_offsets_on_both_axes() {
        // 5x5 gray where each pixel encodes row*10 + col; crop 3x3 -> start (1,1).
        let mut src = vec![0u8; 25];
        for r in 0..5u8 {
            for c in 0..5u8 {
                src[(r as usize) * 5 + c as usize] = r * 10 + c;
            }
        }
        let out = crop_center(&src, 5, 5, 3, 3, 1);
        // Both start_x and start_y are (5-3)/2 = 1, so top-left copied pixel is (1,1).
        assert_eq!(out.len(), 9);
        assert_eq!(out[0], 11, "crop top-left = src (row 1, col 1)");
        assert_eq!(out[2], 13, "crop top-right = src (row 1, col 3)");
        assert_eq!(out[8], 33, "crop bottom-right = src (row 3, col 3)");
    }

    #[test]
    fn from_property_parses_fit_shortest() {
        assert_eq!(
            ResizeMode::from_property("fit-shortest"),
            ResizeMode::FitShortest
        );
        assert_eq!(
            ResizeMode::from_property("FIT_SHORTEST"),
            ResizeMode::FitShortest
        );
        assert_eq!(ResizeMode::FitShortest.as_str(), "fit-shortest");
    }

    #[test]
    fn try_from_property_distinguishes_none_and_unknown() {
        // Canonical strings parse to Some(..).
        assert_eq!(
            ResizeMode::try_from_property("squash"),
            Some(ResizeMode::Squash)
        );
        assert_eq!(
            ResizeMode::try_from_property("fit-longest"),
            Some(ResizeMode::FitLongest)
        );
        // "none" and typos are NOT canonical -> None, so a metadata consumer can
        // detect and log them instead of silently squashing.
        assert_eq!(ResizeMode::try_from_property("none"), None);
        assert_eq!(ResizeMode::try_from_property("fit_shrotest"), None);
        // The lenient wrapper still falls back to Squash for those.
        assert_eq!(ResizeMode::from_property("none"), ResizeMode::Squash);
        assert_eq!(
            ResizeMode::from_property("fit_shrotest"),
            ResizeMode::Squash
        );
    }

    #[test]
    fn transform_squash_inverts_per_axis() {
        let t = ResizeTransform::for_mode(100, 50, 50, 50, ResizeMode::Squash);
        assert_eq!((t.scale_x, t.scale_y), (0.5, 1.0));
        assert_eq!(t.inverse_point(25.0, 25.0), (50.0, 25.0));
    }

    #[test]
    fn transform_fit_longest_letterboxes_and_inverts() {
        // 100x50 -> 50x50: s=min(0.5,1.0)=0.5, scaled 50x25, pad_y=12.5.
        let t = ResizeTransform::for_mode(100, 50, 50, 50, ResizeMode::FitLongest);
        assert_eq!(
            (t.scale_x, t.scale_y, t.offset_x, t.offset_y),
            (0.5, 0.5, 0.0, 12.5)
        );
        assert_eq!(t.inverse_point(25.0, 25.0), (50.0, 25.0));
    }

    #[test]
    fn transform_fit_shortest_fills_and_inverts() {
        // 100x50 -> 50x50: s=max(0.5,1.0)=1.0, scaled 100x50, offset_x=-25.
        let t = ResizeTransform::for_mode(100, 50, 50, 50, ResizeMode::FitShortest);
        assert_eq!(
            (t.scale_x, t.scale_y, t.offset_x, t.offset_y),
            (1.0, 1.0, -25.0, 0.0)
        );
        assert_eq!(t.inverse_point(0.0, 25.0), (25.0, 25.0));
    }

    #[test]
    fn setting_from_property_auto_and_explicit() {
        assert_eq!(
            ResizeModeSetting::from_property("auto"),
            ResizeModeSetting::Auto
        );
        assert_eq!(
            ResizeModeSetting::from_property("fit-shortest"),
            ResizeModeSetting::Explicit(ResizeMode::FitShortest)
        );
        assert_eq!(ResizeModeSetting::Auto.as_str(), "auto");
        assert_eq!(
            ResizeModeSetting::Explicit(ResizeMode::FitShortest).as_str(),
            "fit-shortest"
        );
    }

    #[test]
    fn setting_resolve_uses_model_mode_only_when_auto() {
        assert_eq!(
            ResizeModeSetting::Auto.resolve("fit-longest"),
            ResizeMode::FitLongest
        );
        assert_eq!(
            ResizeModeSetting::Auto.resolve("fit-shortest"),
            ResizeMode::FitShortest
        );
        assert_eq!(
            ResizeModeSetting::Explicit(ResizeMode::Squash).resolve("fit-longest"),
            ResizeMode::Squash
        );
    }
}
