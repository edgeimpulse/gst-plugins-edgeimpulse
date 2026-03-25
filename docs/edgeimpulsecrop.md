# edgeimpulsecrop

Dynamic crop element that extracts detected object regions from video frames based on upstream inference metadata. For each bounding box detected by an upstream `edgeimpulsevideoinfer`, the element produces a separate cropped buffer — enabling multi-stage pipelines where a detection model finds objects and a classification model analyzes each one individually.

## Element Details

- Long name: Edge Impulse Dynamic Crop
- Class: Filter/Video
- Description: Crops detected object regions from video frames based on upstream inference metadata

## Pad Templates

- Sink pad (Always available):
  ```
  video/x-raw
    format: RGB
    width: [ 1, 2147483647 ]
    height: [ 1, 2147483647 ]
  ```
- Source pad (Always available):
  ```
  video/x-raw
    format: RGB
    width: [ 1, 2147483647 ]
    height: [ 1, 2147483647 ]
  ```

## Properties

1. `padding` (integer):
   - Extra pixels around each bounding box crop
   - Range: 0 -- 1000
   - Default: 0
   - Flags: readable, writable, changeable in PLAYING state

2. `target-width` (integer):
   - Resize all crops to this width. Setting a fixed target avoids GStreamer caps renegotiation per-crop.
   - Range: 0 -- 4096 (0 = keep natural crop size)
   - Default: 0
   - Flags: readable, writable

3. `target-height` (integer):
   - Resize all crops to this height. Setting a fixed target avoids GStreamer caps renegotiation per-crop.
   - Range: 0 -- 4096 (0 = keep natural crop size)
   - Default: 0
   - Flags: readable, writable

## How It Works

1. Reads [`VideoRegionOfInterestMeta`](../README.md#videoregionofinterestmeta) attached by upstream `edgeimpulsevideoinfer`
2. For each detection bounding box:
   - Extracts the crop region (with optional padding, clamped to frame bounds)
   - Optionally resizes to `target-width` x `target-height`
   - Attaches a [`CropOriginMeta`](../README.md#4-croporiginmeta) with source coordinates, original frame dimensions, object_id, label, and confidence
   - Pushes the crop buffer downstream
3. If no detections are present, the full frame is passed through unchanged

This is a **1-to-N element**: one input buffer may produce N output buffers (one per detected object). The downstream element (e.g., a classification `edgeimpulsevideoinfer`) processes each crop independently.

## Example Pipelines

```bash
# Two-stage pipeline: detect objects, then classify each crop
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulsevideoinfer_detection ! tee name=t \
    t. ! queue ! edgeimpulseoverlay_detection ! autovideosink \
    t. ! queue ! \
        edgeimpulsecontinueif condition="detection_count >= 1" ! \
        edgeimpulsecrop padding=10 target-width=96 target-height=96 ! \
        edgeimpulsevideoinfer_classification ! \
        fakesink

# Crop with padding and fixed output size
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulsevideoinfer ! \
    edgeimpulsecrop padding=20 target-width=320 target-height=320 ! \
    edgeimpulsevideoinfer_classification ! \
    fakesink
```

## Debugging

```bash
GST_DEBUG=edgeimpulsecrop:4 gst-launch-1.0 ...
```
