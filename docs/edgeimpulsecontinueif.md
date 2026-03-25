# edgeimpulsecontinueif

Conditional gate element that passes or drops buffers based on upstream inference metadata. Works with both audio and video pipelines — any buffer carrying `InferenceResultMeta` can be filtered. Place it after `edgeimpulsevideoinfer` or `edgeimpulseaudioinfer` (or after a `tee`) to skip downstream processing when a condition is not met — for example, to avoid running a classification model when no objects were detected.

## Element Details

- Long name: Edge Impulse Continue If
- Class: Filter/Video
- Description: Conditionally passes or drops buffers based on upstream inference metadata

## Pad Templates

- Sink pad (Always available):
  ```
  ANY
  ```
- Source pad (Always available):
  ```
  ANY
  ```

## Properties

1. `condition` (string):
   - Expression evaluated per buffer to decide pass/drop
   - Default: "" (empty = pass all)
   - Flags: readable, writable, changeable in PLAYING state

2. `drop` (boolean):
   - When true, unconditionally drop all buffers (manual override)
   - Default: false
   - Flags: readable, writable, changeable in PLAYING state

3. `rules` (string):
   - JSON array of ordered rules for conditional metadata output
   - First matching rule wins; its `metadata` key-value pairs are posted as a `edge-impulse-continue-if-metadata` bus message
   - Default: "" (empty = no rules)
   - Flags: readable, writable, changeable in PLAYING state
   - Example:
     ```json
     [
       {"condition": "detection_count > 4",  "metadata": {"severity": "critical", "color": "purple"}},
       {"condition": "detection_count >= 1", "metadata": {"severity": "warning", "color": "red"}},
       {"condition": "detection_count == 0", "metadata": {"severity": "ok", "color": "green"}}
     ]
     ```

## Condition Variables

Available condition variables (extracted from [`InferenceResultMeta`](../README.md#3-inferenceresultmeta-convenience-layer), with fallback to video-specific metadata):

| Variable | Type | Description |
|---|---|---|
| `detection_count` | number | Number of detected objects |
| `max_confidence` | number | Highest confidence score across all detections |
| `has_class("name")` | function | True if any detection or classification matches the class name |
| `classification` | string | Top classification label |
| `classification_confidence` | number | Top classification confidence score |
| `anomaly_score` | number | Anomaly score |
| `visual_anomaly_max` | number | Visual anomaly max score |

## Condition Syntax

```
detection_count >= 1
max_confidence > 0.8
has_class("crack")
classification == "defect"
anomaly_score > 0.5
```

Supported operators: `>=`, `<=`, `>`, `<`, `==`, `!=`

## How It Works

- Reads [`InferenceResultMeta`](../README.md#3-inferenceresultmeta-convenience-layer) attached to each buffer by upstream `edgeimpulsevideoinfer` or `edgeimpulseaudioinfer` (falls back to video-specific metadata if `InferenceResultMeta` is not present)
- Evaluates the condition expression against the extracted values
- If the condition is true, the buffer passes through unchanged (zero-copy)
- If the condition is false, the buffer is marked with `GAP | DROPPABLE` flags
- If `rules` are configured, the first matching rule posts a `edge-impulse-continue-if-metadata` bus message with its key-value pairs

## Example Pipelines

```bash
# Skip classification when no objects are detected
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulsevideoinfer ! tee name=t \
    t. ! queue ! edgeimpulseoverlay ! autovideosink \
    t. ! queue ! edgeimpulsecontinueif condition="detection_count >= 1" ! \
        edgeimpulsecrop ! edgeimpulsevideoinfer_classification ! fakesink

# Only process high-confidence detections
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulsevideoinfer ! \
    edgeimpulsecontinueif condition="max_confidence > 0.9" ! \
    edgeimpulseoverlay ! autovideosink

# Gate on specific class
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulsevideoinfer ! \
    edgeimpulsecontinueif condition='has_class("crack")' ! \
    edgeimpulseoverlay ! autovideosink
```

## Debugging

```bash
GST_DEBUG=edgeimpulsecontinueif:4 gst-launch-1.0 ...
```
