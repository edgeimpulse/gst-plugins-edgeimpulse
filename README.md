# GStreamer Edge Impulse Plugin
[![CI](https://github.com/edgeimpulse/gst-plugins-edgeimpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/edgeimpulse/gst-plugins-edgeimpulse/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://edgeimpulse.github.io/gst-plugins-edgeimpulse/)

A GStreamer plugin that enables real-time machine learning inference using Edge Impulse models. The plugin provides three elements for audio and video inference and visualization.

## Dependencies
This plugin depends on:
* GStreamer 1.20 or newer
* [edge-impulse-runner-rs](https://github.com/edgeimpulse/edge-impulse-runner-rs) - Rust bindings for Edge Impulse Linux SDK
* A trained Edge Impulse model file (.eim)
* Rust toolchain (via rustup)

## Installation

### 1. Install Rust
First, install the Rust toolchain using rustup:

```bash
# On Unix-like OS (Linux, macOS)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts to complete the installation. After installation, restart your terminal to ensure the Rust tools are in your PATH.

### 2. Install GStreamer
Download and install GStreamer from the official binaries:

#### macOS
Download and install both packages:
- [Runtime installer](https://gstreamer.freedesktop.org/data/pkg/osx/1.24.12/gstreamer-1.0-1.24.12-universal.pkg)
- [Development installer](https://gstreamer.freedesktop.org/data/pkg/osx/1.24.12/gstreamer-1.0-devel-1.24.12-universal.pkg)

#### Linux
Install from your distribution's package manager. For example:

Ubuntu/Debian:
```bash
sudo apt-get install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio
```

### 3. Build the Plugin

Clone and build the plugin:
```bash
git clone https://github.com/edgeimpulse/gst-plugins-edgeimpulse.git
cd gst-plugins-edgeimpulse
cargo build --release
```

#### Cross-compilation for ARM64
To build a Linux ARM64-compatible .so file from macOS, you'll need Docker.

You'll also need to provide your SSH key for accessing private repositories.
The build process supports different SSH key types (id_rsa, id_ed25519, etc.).
Replace `id_ed25519` in the following commands with your SSH key name if different:

First, build the Docker image:

```bash
docker build -t gst-plugins-edgeimpulse-builder .
```

Then, add the aarch64 target to your Rust toolchain:
```bash
rustup target add aarch64-unknown-linux-gnu
```

Finally, run the build. Replace `id_ed25519` with your SSH key name if different:
```bash
docker run -it \
    -v $(pwd):/app \
    -v $HOME/.ssh/id_ed25519:/root/.ssh/id_ed25519 \
    -e SSH_KEY_NAME=id_ed25519 \
    gst-plugins-edgeimpulse-builder \
    cargo build --release --target aarch64-unknown-linux-gnu
```

The compiled .so file will be available in `target/aarch64-unknown-linux-gnu/release/libgstedgeimpulse.so`.

## Elements

### edgeimpulseaudioinfer
Audio inference element that processes audio streams through Edge Impulse models.

Element Details:
- Long name: Edge Impulse Audio Inference
- Class: Filter/Audio/AI
- Description: Runs audio inference on Edge Impulse models (EIM)

Pad Templates:
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

Properties:
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

Key features:
- Accepts S16LE mono audio at 8-48kHz
- Passes audio through unchanged
- Performs inference when model is loaded
- Emits inference results as messages

Example pipeline:
```bash
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

### edgeimpulsevideoinfer
Video inference element that processes video frames through Edge Impulse models.

Element Details:
- Long name: Edge Impulse Video Inference
- Class: Filter/Video/AI
- Description: Runs video inference on Edge Impulse models (EIM)

Pad Templates:
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

Properties:
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

Key features:
- Accepts RGB video frames of any resolution
- Passes frames through unchanged
- Performs inference when model is loaded
- Supports both classification and object detection models
- Emits inference results as messages
- Configurable thresholds for object detection and anomaly detection

For object detection models, the element provides two mechanisms to consume results:

1. Bus Messages:
   - Sends element messages on the GStreamer bus
   - Messages contain raw JSON results and timing information
   - Useful for custom applications that want to process detection results

2. QC IM SDK Compatible Metadata:
   - Attaches VideoRegionOfInterestMeta to each video frame
   - Compatible with Qualcomm IM SDK `qtioverlay` element
   - Enables automatic visualization in QC IM SDK pipelines
   - Each ROI includes bounding box coordinates, label and confidence

Example pipelines:

Basic pipeline with built-in overlay:
```bash
gst-launch-1.0  avfvideosrc ! \
  queue max-size-buffers=2 leaky=downstream ! \
  videoconvert n-threads=4 ! \
  videoscale method=nearest-neighbour ! \
  video/x-raw,format=RGB,width=384,height=384 ! \
  queue max-size-buffers=2 leaky=downstream ! \
  edgeimpulsevideoinfer model-path=<path-to-model> ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false
```

Pipeline with threshold settings:
```bash
# Set object detection threshold
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB,width=384,height=384 ! \
  edgeimpulsevideoinfer model-path=<path-to-model> threshold="5.min_score=0.6" ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false

# Set multiple thresholds
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB,width=384,height=384 ! \
  edgeimpulsevideoinfer model-path=<path-to-model> \
    threshold="5.min_score=0.6" \
    threshold="4.min_anomaly_score=0.35" ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false
```

### edgeimpulseoverlay
Video overlay element that visualizes inference results from edgeimpulsevideoinfer.

Key features:
- Draws bounding boxes for object detection results
- Displays class labels with confidence scores
- Customizable visualization options

Example pipeline:
```bash
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB,width=384,height=384 ! \
  edgeimpulsevideoinfer model-path=<path-to-model> ! \
  edgeimpulseoverlay ! \
  autovideosink sync=false
```

## Examples

The repository includes examples demonstrating both audio and video inference. These examples have been tested on MacOS.

### Audio Classification
Run the audio classification example:
```bash
# Basic usage
cargo run --example audio_classify -- --model path/to/your/model.eim

# With threshold settings
cargo run --example audio_classify -- --model path/to/your/model.eim \
    --threshold "5.min_score=0.6" \
    --threshold "4.min_anomaly_score=0.35"

# With audio file input
cargo run --example audio_classify -- --model path/to/your/model.eim \
    --audio input.wav \
    --threshold "5.min_score=0.6"
```

This will capture audio from the default microphone (or audio file if specified) and display inference results:

For classification:
```
Got element message with name: edge-impulse-inference-result
Message structure: edge-impulse-inference-result {
    timestamp: (guint64) 9498000000,
    type: "classification",
    result: {
        "classification": {
            "no": 0.015625,
            "noise": 0.968750,
            "yes": 0.019531
        }
    }
}
Detected: noise (96.9%)
```

For anomaly detection:
```
Got element message with name: edge-impulse-inference-result
Message structure: edge-impulse-inference-result {
    timestamp: (guint64) 9498000000,
    type: "anomaly",
    result: {
        "anomaly": 0.35,
        "classification": {
            "normal": 0.85,
            "anomalous": 0.15
        }
    }
}
Detected: normal (85.0%)
Anomaly score: 35.0%
```

The element will automatically detect the model type and emit appropriate messages. Thresholds can be set for both object detection (`min_score`) and anomaly detection (`min_anomaly_score`) blocks.

### Video Classification/Detection
Run the video classification example:
```bash
# Basic usage
cargo run --example video_classify -- --model path/to/your/model.eim

# With threshold settings
cargo run --example video_classify -- --model path/to/your/model.eim \
    --threshold "5.min_score=0.6" \
    --threshold "4.min_anomaly_score=0.35"
```

This will capture video from your camera and display inference results with visualization. Example outputs:

For object detection:
```
Got element message with name: edge-impulse-inference-result
Message structure: edge-impulse-inference-result {
    timestamp: (guint64) 1234567890,
    type: "object-detection",
    result: {
        "bounding_boxes": [
            {
                "label": "person",
                "value": 0.95,
                "x": 24,
                "y": 145,
                "width": 352,
                "height": 239
            }
        ]
    }
}
Detected: person (95.0%)
```

For classification:
```
Got element message with name: edge-impulse-inference-result
Message structure: edge-impulse-inference-result {
    timestamp: (guint64) 1234567890,
    type: "classification",
    result: {
        "classification": [
            {"label": "cat", "value": 0.85},
            {"label": "dog", "value": 0.15}
        ]
    }
}
Detected: cat (85.0%)
```

The element will automatically detect the model type (classification or object detection) and emit appropriate messages. For object detection models, bounding boxes will be visualized on the video output when using the `edgeimpulseoverlay` element.

## Message Format
The elements emit "edge-impulse-inference-result" messages containing:
- timestamp: Buffer presentation timestamp
- type: "classification" or "object-detection" (video only)
- result: JSON string with model output

## Debugging
Enable debug output with:
```bash
GST_DEBUG=edgeimpulseaudioinfer:4 # for audio element
GST_DEBUG=edgeimpulsevideoinfer:4 # for video element
GST_DEBUG=edgeimpulseoverlay:4 # for overlay element
```

## Acknowledgments
This crate is designed to work with Edge Impulse's machine learning models. For more information about Edge Impulse and their ML deployment solutions, visit [Edge Impulse](https://edgeimpulse.com/).

## License
This project is licensed under the BSD 3-Clause Clear License - see the LICENSE file for details.