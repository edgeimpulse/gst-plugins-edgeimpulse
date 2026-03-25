# edgeimpulsesink

Sink element that uploads audio or video buffers to Edge Impulse using the ingestion API.

## Element Details

- Long name: Edge Impulse Ingestion Sink
- Class: Sink/AI
- Description: Uploads audio or video buffers to Edge Impulse ingestion API (WAV for audio, PNG for video)

## Pad Templates

- Sink pad (Always available):
  ```
  audio/x-raw
    format: S16LE
    channels: 1
    rate: 16000
  video/x-raw
    format: { RGB, RGBA }
    width: [ 1, 2147483647 ]
    height: [ 1, 2147483647 ]
  ```

## Properties

1. `api-key` (string, required):
   - Edge Impulse API key
   - Flags: readable, writable
2. `hmac-key` (string, optional):
   - Optional HMAC key for signing requests
   - Flags: readable, writable
3. `label` (string, optional):
   - Optional label for the sample
   - Flags: readable, writable
4. `category` (string, default: "training"):
   - Category for the sample (training, testing, anomaly)
   - Flags: readable, writable
5. `upload-interval-ms` (u32, default: 0):
   - Minimum interval in milliseconds between uploads (0 = every buffer)
   - Flags: readable, writable

## Key Features

- Supports both audio (WAV) and video (PNG) ingestion
- Batches and uploads buffers at a configurable interval
- Emits bus messages for ingestion results and errors (see [Public API](../README.md#public-api-inference-and-ingestion-output))
- Can be used in pipelines for automated dataset collection

## Example Pipeline

```bash
gst-launch-1.0 autoaudiosrc ! audioconvert ! audioresample ! \
    audio/x-raw,format=S16LE,channels=1,rate=16000 ! \
    edgeimpulsesink api-key=<your-api-key> upload-interval-ms=1000 category=training
```

See `examples/audio_ingestion.rs` for a full example with bus message handling.

## Debugging

```bash
GST_DEBUG=edgeimpulsesink:4 gst-launch-1.0 ...
```
