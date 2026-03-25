# edgeimpulsevideoinfer

Video inference element that processes video frames through Edge Impulse models. The element automatically handles frame resizing to match model input requirements and scales detection results back to the original resolution.

## Element Details

- Long name: Edge Impulse Video Inference
- Class: Filter/Video/AI
- Description: Runs video inference on Edge Impulse models (EIM)

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

1. `model-path` (string):
   - Path to Edge Impulse model file
   - Default: null
   - Flags: readable, writable

2. `threshold` (string):
   - Format: `blockId.type=value`
   - Types:
     - `min_score`: For object detection blocks (confidence threshold)
     - `min_anomaly_score`: For anomaly detection blocks
   - Examples:
     - `5.min_score=0.6`: Set minimum confidence score for block 5
     - `4.min_anomaly_score=0.35`: Set minimum anomaly score for block 4
   - Default: ""
   - Flags: readable, writable

## Key Features

- Accepts RGB video frames of any resolution
- Passes frames through unchanged
- Performs inference when model is loaded
- Supports classification, object detection and anomaly detection models
- Emits inference results as bus messages (see [Public API](../README.md#public-api-inference-and-ingestion-output))
- Attaches [`VideoRegionOfInterestMeta`](../README.md#videoregionofinterestmeta) to each video frame
- Attaches [`VideoClassificationMeta`](../README.md#videoclassificationmeta) for classification results
- Attaches [`VideoAnomalyMeta`](../README.md#videoanomalymeta) for anomaly detection results
- Attaches [`InferenceResultMeta`](../README.md#3-inferenceresultmeta-convenience-layer) to each buffer

## Example Pipelines

Basic pipeline with built-in overlay:
```bash
# FFI mode (default)
gst-launch-1.0 avfvideosrc ! \
  queue max-size-buffers=2 leaky=downstream ! \
  videoconvert n-threads=4 ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  queue max-size-buffers=2 leaky=downstream ! \
  edgeimpulsevideoinfer ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false

# EIM mode
gst-launch-1.0 avfvideosrc ! \
  queue max-size-buffers=2 leaky=downstream ! \
  videoconvert n-threads=4 ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  queue max-size-buffers=2 leaky=downstream ! \
  edgeimpulsevideoinfer model-path=<path-to-model> ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false
```

Pipeline with threshold settings:
```bash
# FFI mode (default) - Set object detection threshold
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  edgeimpulsevideoinfer threshold="5.min_score=0.6" ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false

# FFI mode (default) - Set multiple thresholds
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  edgeimpulsevideoinfer \
    threshold="5.min_score=0.6" \
    threshold="4.min_anomaly_score=0.35" ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false

# EIM mode - Set object detection threshold
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  edgeimpulsevideoinfer model-path=<path-to-model> threshold="5.min_score=0.6" ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false

# EIM mode - Set multiple thresholds
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  video/x-raw,format=RGB,width=1920,height=1080 ! \
  edgeimpulsevideoinfer model-path=<path-to-model> \
    threshold="5.min_score=0.6" \
    threshold="4.min_anomaly_score=0.35" ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false
```

## Debugging

```bash
GST_DEBUG=edgeimpulsevideoinfer:4 gst-launch-1.0 ...
```
