# edgeimpulseocr

Reads text from video frames (optical character recognition) and attaches each recognized line to the buffer as inference metadata. For every line of text found, the element attaches a `VideoRegionOfInterestMeta` — so downstream elements such as `edgeimpulseoverlay` render the text exactly like any other detection — and, optionally, posts an `ocr` message on the bus.

Recognition runs on a background worker thread, fully decoupled from the streaming thread, so a slow model never stalls the pipeline: frames pass through immediately and results attach to a slightly later frame.

## Element Details

- Long name: Edge Impulse OCR
- Class: Filter/Analyzer/Video
- Description: Reads text from video frames and attaches it as ROI metadata

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

1. `backend` (string):
   - OCR engine to use. `ocrs` performs recognition in-process with embedded [rten](https://github.com/robertknight/rten) models; `edge-impulse` (decode recognition results from an upstream Edge Impulse model) is planned and currently recognizes nothing.
   - Default: `ocrs`
   - Flags: readable, writable, changeable in the NULL or READY state

2. `detection-model` (string):
   - Path to a text-detection `.rten` model. Empty uses the model embedded in the plugin.
   - Default: `""` (embedded model)
   - Flags: readable, writable, changeable in the NULL or READY state

3. `recognition-model` (string):
   - Path to a text-recognition `.rten` model. Empty uses the model embedded in the plugin.
   - Default: `""` (embedded model)
   - Flags: readable, writable, changeable in the NULL or READY state

4. `min-confidence` (double):
   - Drop recognized lines below this confidence. Note: the `ocrs` backend does not expose a per-line confidence (it reports `1.0`), so this acts as a no-op for that backend today.
   - Range: 0.0 -- 1.0
   - Default: 0.0
   - Flags: readable, writable, changeable in PLAYING state

5. `max-text-length` (unsigned integer):
   - Truncate each recognized line to at most this many characters.
   - Range: 1 -- 4294967295
   - Default: 256
   - Flags: readable, writable, changeable in PLAYING state

6. `post-message` (boolean):
   - Post an `ocr` element message on the bus for each recognized line.
   - Default: true
   - Flags: readable, writable, changeable in PLAYING state

7. `interval` (unsigned integer):
   - Run recognition on one frame out of every N (higher values reduce load).
   - Range: 1 -- 4294967295
   - Default: 1
   - Flags: readable, writable, changeable in PLAYING state

## How It Works

1. Every `interval`-th input frame is copied and handed to a background worker thread; all other frames pass straight through.
2. The worker runs the configured backend (text detection followed by recognition) off the streaming thread.
3. The most recent results are attached to passing buffers as `VideoRegionOfInterestMeta` — one per line, each with a `detection` param carrying the recognized text as `label` and the confidence — so `edgeimpulseoverlay` (or any ROI consumer) can render them.
4. When `post-message` is true, each recognized line is also emitted as an `ocr` bus message.

Because recognition is asynchronous, results attach to a slightly later frame than the one they were computed from. The `ocr` message's `timestamp` field carries the PTS (in milliseconds) of the source frame the text was read from.

### `ocr` Bus Message

```
ocr, text=(string), confidence=(double),
     x=(int), y=(int), width=(int), height=(int), timestamp=(gint64)
```

`x`, `y`, `width`, and `height` are the line's bounding box in full-frame pixels; `timestamp` is the source-frame PTS in milliseconds.

## Backends

- **`ocrs`** (default): Pure-Rust text detection and recognition using [ocrs](https://github.com/robertknight/ocrs). The default detection and recognition models are embedded in the plugin, so no external files are required. Override them with `detection-model` / `recognition-model` to use your own; the standard models can also be fetched separately with [`examples/download-ocr-models.sh`](../examples/download-ocr-models.sh). This backend is compiled only when the plugin is built with the `ocrs` cargo feature (enabled by default).
- **`edge-impulse`** (planned): Will decode text from an upstream Edge Impulse recognition model, letting you train and deploy custom OCR models via Edge Impulse Studio. Not yet implemented — selecting it currently recognizes nothing.

> **Build in release mode.** The recognition models are large; debug builds run recognition far too slowly to be usable. Always build and run with `--release`.

## Example Pipelines

```bash
# Recognize text from a camera stream and overlay it on the video
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulseocr backend=ocrs ! edgeimpulseoverlay ! autovideosink

# Recognize text from a single still image (run with -m to print the `ocr`
# bus messages; imagefreeze never ends, so stop with Ctrl+C — or use the
# self-terminating examples/ocr_inference.rs below).
gst-launch-1.0 -m filesrc location=text.png ! decodebin ! imagefreeze ! \
    videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulseocr backend=ocrs ! fakesink

# Use your own rten models instead of the embedded defaults (Ctrl+C to stop)
gst-launch-1.0 -m filesrc location=text.png ! decodebin ! imagefreeze ! \
    videoconvert ! video/x-raw,format=RGB ! \
    edgeimpulseocr backend=ocrs \
        detection-model=text-detection.rten \
        recognition-model=text-recognition.rten ! \
    fakesink
```

An end-to-end example that prints recognized text is available at [`examples/ocr_inference.rs`](../examples/ocr_inference.rs):

```bash
export GST_PLUGIN_PATH="$(pwd)/target/release:$GST_PLUGIN_PATH"
cargo run --release --no-default-features --features "eim ocrs" \
    --example ocr_inference -- --image text.png
```
