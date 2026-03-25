# edgeimpulseoverlay

Video overlay element that visualizes inference results by drawing bounding boxes and labels on video frames.

## Element Details

- Long name: Edge Impulse Overlay
- Class: Filter/Effect/Video
- Description: Draws bounding boxes on video frames based on ROI metadata, including object tracking IDs

## Pad Templates

- Sink/Source pads (Always available):
  ```
  video/x-raw
    format: { RGB, BGR, RGBA, BGRA, UYVY, YUY2, YVYU, NV12, NV21, I420, YV12 }
    width: [ 1, 2147483647 ]
    height: [ 1, 2147483647 ]
  ```

## Key Features

- Draws bounding boxes for object detection and visual anomaly detection results (from [`VideoRegionOfInterestMeta`](../README.md#videoregionofinterestmeta))
- Displays class labels with confidence scores and object tracking IDs when available
- Supports wide range of video formats
- Automatic text sizing: Calculates optimal font size based on frame dimensions and bounding box sizes
- Text scale control: Use `text-scale-ratio` property to fine-tune text size (0.1x to 5.0x scaling)

## Properties

1. `stroke-width` (integer):
   - Width of the bounding box lines in pixels
   - Range: 1 - 100
   - Default: 2

2. `text-color` (unsigned integer):
   - Color of the text in RGB format
   - Range: 0 - 4294967295
   - Default: white (0xFFFFFF)
   - If set to default white, automatic brightness-based color selection is used

3. `background-color` (unsigned integer):
   - Color of the text background in RGB format
   - Range: 0 - 4294967295
   - Default: black (0x000000)

4. `text-font` (string):
   - Font family to use for text rendering
   - Default: "Sans"

5. `text-scale-ratio` (double):
   - Scale factor for text size. Values > 1.0 make text larger, < 1.0 make text smaller
   - Range: 0.1 - 5.0
   - Default: 1.0 (no scaling)
   - **Note:** This property overrides the automatic text sizing. The element calculates optimal font size based on frame/bbox dimensions, then applies this scale factor.

## Example Pipelines

```bash
# FFI mode (default)
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  edgeimpulsevideoinfer ! \
  edgeimpulseoverlay stroke-width=3 text-scale-ratio=1.5 text-color=0x00FF00 background-color=0x000000 ! \
  autovideosink sync=false

# EIM mode
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  edgeimpulsevideoinfer model-path=<path-to-model> ! \
  edgeimpulseoverlay stroke-width=3 text-scale-ratio=1.5 text-color=0x00FF00 background-color=0x000000 ! \
  autovideosink sync=false
```

The overlay element automatically processes `VideoRegionOfInterestMeta` from upstream elements (like `edgeimpulsevideoinfer`) and visualizes them with configurable styles.

## Debugging

```bash
GST_DEBUG=edgeimpulseoverlay:4 gst-launch-1.0 ...
```
