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
   - Resize all crops to this width. Takes effect only when `target-height` is also set. A fixed size avoids renegotiating caps for every crop.
   - Range: 0 -- 4096 (0 = keep natural crop size)
   - Default: 0
   - Flags: readable, writable

3. `target-height` (integer):
   - Resize all crops to this height. Takes effect only when `target-width` is also set. A fixed size avoids renegotiating caps for every crop.
   - Range: 0 -- 4096 (0 = keep natural crop size)
   - Default: 0
   - Flags: readable, writable

4. `drop-empty` (boolean):
   - Drop frames without detections instead of passing the full frame through, so downstream elements only see crops
   - Default: false
   - Flags: readable, writable, changeable in PLAYING state

## How It Works

1. Reads [`VideoRegionOfInterestMeta`](../README.md#videoregionofinterestmeta) attached by upstream `edgeimpulsevideoinfer`
2. For each detection bounding box:
   - Extracts the crop region (with optional padding, clamped to frame bounds)
   - Optionally resizes to `target-width` x `target-height`
   - Attaches a [`CropOriginMeta`](../README.md#4-croporiginmeta) with padded, frame-clamped source coordinates, the unpadded detection rect, original frame dimensions, object_id, label, and confidence
   - Pushes the crop buffer downstream
3. If no detections are present, the full frame is passed through (resized to the target size when one is set), or dropped when `drop-empty` is true

Without a target size each crop keeps its natural size, so the output size changes from buffer to buffer. The element pushes a new caps event before every buffer whose size differs from the previous one, and lays each buffer out with the row stride its caps imply, so any downstream element can read it. Elements that only cope with one input size should get a fixed `target-width` and `target-height`, normally the downstream model's input size.

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
