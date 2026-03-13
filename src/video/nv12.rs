/// NV12 to RGB conversion utilities for inference feature extraction.
///
/// NV12 memory layout:
/// ```text
/// Offset 0:              Y plane (width × height bytes, one luma byte per pixel)
/// Offset width × height: UV plane (width × height/2 bytes, interleaved U,V pairs)
///                         Each U,V pair covers a 2×2 pixel block
/// ```
///
/// The conversion uses BT.601 coefficients (standard for camera output):
///   R = Y + 1.402 * (V - 128)
///   G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
///   B = Y + 1.772 * (U - 128)

/// Convert NV12 frame data to RGB888 byte vector.
///
/// # Arguments
/// * `nv12_data` - Raw NV12 bytes: Y plane followed by UV plane
/// * `width` - Frame width in pixels (must be even)
/// * `height` - Frame height in pixels (must be even)
/// * `stride` - Stride (bytes per row) of the Y plane. If 0 or equal to width, no padding.
///
/// # Returns
/// RGB888 byte vector of length width * height * 3
pub fn nv12_to_rgb(nv12_data: &[u8], width: usize, height: usize, stride: usize) -> Vec<u8> {
    let stride = if stride == 0 { width } else { stride };
    let mut rgb = vec![0u8; width * height * 3];

    let y_plane = &nv12_data[..stride * height];
    let uv_plane = &nv12_data[stride * height..];

    for row in 0..height {
        for col in 0..width {
            let y_idx = row * stride + col;
            let uv_row = row / 2;
            let uv_col = (col / 2) * 2;
            let uv_idx = uv_row * stride + uv_col;

            let y = y_plane[y_idx] as f32;
            let u = uv_plane[uv_idx] as f32;
            let v = uv_plane[uv_idx + 1] as f32;

            let r = (y + 1.402 * (v - 128.0)).clamp(0.0, 255.0) as u8;
            let g = (y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)).clamp(0.0, 255.0) as u8;
            let b = (y + 1.772 * (u - 128.0)).clamp(0.0, 255.0) as u8;

            let rgb_idx = (row * width + col) * 3;
            rgb[rgb_idx] = r;
            rgb[rgb_idx + 1] = g;
            rgb[rgb_idx + 2] = b;
        }
    }

    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nv12_to_rgb_black_frame() {
        let width = 4;
        let height = 4;
        let mut nv12 = vec![0u8; width * height + width * height / 2];
        let uv_offset = width * height;
        for i in 0..(width * height / 2) {
            nv12[uv_offset + i] = 128;
        }
        let rgb = nv12_to_rgb(&nv12, width, height, 0);
        assert_eq!(rgb.len(), width * height * 3);
        for pixel in rgb.chunks(3) {
            assert_eq!(pixel, &[0, 0, 0], "Expected black pixel");
        }
    }

    #[test]
    fn test_nv12_to_rgb_white_frame() {
        let width = 4;
        let height = 4;
        let mut nv12 = vec![235u8; width * height + width * height / 2];
        let uv_offset = width * height;
        for i in 0..(width * height / 2) {
            nv12[uv_offset + i] = 128;
        }
        let rgb = nv12_to_rgb(&nv12, width, height, 0);
        for pixel in rgb.chunks(3) {
            assert_eq!(pixel[0], 235);
            assert_eq!(pixel[1], 235);
            assert_eq!(pixel[2], 235);
        }
    }

    #[test]
    fn test_nv12_to_rgb_red_pixel() {
        let width = 2;
        let height = 2;
        let mut nv12 = vec![0u8; width * height + width * height / 2];
        for i in 0..4 {
            nv12[i] = 82;
        }
        nv12[4] = 90;
        nv12[5] = 240;
        let rgb = nv12_to_rgb(&nv12, width, height, 0);
        for pixel in rgb.chunks(3) {
            assert!(pixel[0] > 200, "R={} should be > 200", pixel[0]);
            assert!(pixel[1] < 30, "G={} should be < 30", pixel[1]);
            assert!(pixel[2] < 30, "B={} should be < 30", pixel[2]);
        }
    }

    #[test]
    fn test_nv12_to_rgb_with_stride() {
        let width = 4;
        let height = 2;
        let stride = 8;
        let y_plane_size = stride * height;
        let uv_plane_size = stride * (height / 2);
        let mut nv12 = vec![0u8; y_plane_size + uv_plane_size];
        for row in 0..height {
            for col in 0..width {
                nv12[row * stride + col] = 100;
            }
            for col in width..stride {
                nv12[row * stride + col] = 255;
            }
        }
        let uv_offset = y_plane_size;
        for row in 0..(height / 2) {
            for col in 0..width {
                nv12[uv_offset + row * stride + col] = 128;
            }
        }
        let rgb = nv12_to_rgb(&nv12, width, height, stride);
        assert_eq!(rgb.len(), width * height * 3);
        for pixel in rgb.chunks(3) {
            assert_eq!(pixel, &[100, 100, 100]);
        }
    }

    #[test]
    fn test_nv12_to_rgb_output_size() {
        let width = 640;
        let height = 480;
        let nv12_size = width * height + width * height / 2;
        let mut nv12 = vec![128u8; nv12_size];
        let uv_offset = width * height;
        for i in 0..(width * height / 2) {
            nv12[uv_offset + i] = 128;
        }
        let rgb = nv12_to_rgb(&nv12, width, height, 0);
        assert_eq!(rgb.len(), width * height * 3);
    }
}
