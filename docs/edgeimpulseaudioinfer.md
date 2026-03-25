# edgeimpulseaudioinfer

Audio inference element that processes audio streams through Edge Impulse models.

## Element Details

- Long name: Edge Impulse Audio Inference
- Class: Filter/Audio/AI
- Description: Runs audio inference on Edge Impulse models (EIM)

## Pad Templates

- Sink pad (Always available):
  ```
  audio/x-raw
    format: S16LE
    rate: [ 8000, 48000 ]
    channels: 1
    layout: interleaved
  ```
- Source pad (Always available):
  ```
  audio/x-raw
    format: S16LE
    rate: [ 8000, 48000 ]
    channels: 1
    layout: interleaved
  ```

## Properties

1. `model-path` (string):
   - Path to Edge Impulse model file
   - Default: null
   - Flags: readable, writable

2. `threshold` (string):
   - Format: `blockId.type=value`
   - Types:
     - `min_score`: For object detection blocks
     - `min_anomaly_score`: For anomaly detection blocks
   - Examples:
     - `5.min_score=0.6`: Set minimum confidence score for block 5
     - `4.min_anomaly_score=0.35`: Set minimum anomaly score for block 4
   - Default: ""
   - Flags: readable, writable

## Key Features

- Accepts S16LE mono audio at 8-48kHz
- Passes audio through unchanged
- Performs inference when model is loaded
- Attaches [`InferenceResultMeta`](../README.md#3-inferenceresultmeta-convenience-layer) to each buffer
- Emits inference results as bus messages (see [Public API](../README.md#public-api-inference-and-ingestion-output))

## Example Pipelines

```bash
# FFI mode (default)
gst-launch-1.0 autoaudiosrc ! \
    capsfilter caps="audio/x-raw,format=F32LE" ! \
    audioconvert ! \
    audioresample ! \
    capsfilter caps="audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved" ! \
    edgeimpulseaudioinfer ! \
    audioconvert ! \
    audioresample ! \
    capsfilter caps="audio/x-raw,format=F32LE,channels=2,rate=44100" ! \
    autoaudiosink

# EIM mode (legacy)
gst-launch-1.0 autoaudiosrc ! \
    capsfilter caps="audio/x-raw,format=F32LE" ! \
    audioconvert ! \
    audioresample ! \
    capsfilter caps="audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved" ! \
    edgeimpulseaudioinfer model-path=<path-to-model> ! \
    audioconvert ! \
    audioresample ! \
    capsfilter caps="audio/x-raw,format=F32LE,channels=2,rate=44100" ! \
    autoaudiosink
```

## Debugging

```bash
GST_DEBUG=edgeimpulseaudioinfer:4 gst-launch-1.0 ...
```
