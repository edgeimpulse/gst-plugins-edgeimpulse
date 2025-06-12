# GStreamer Edge Impulse Plugin
[![CI](https://github.com/edgeimpulse/gst-plugins-edgeimpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/edgeimpulse/gst-plugins-edgeimpulse/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://edgeimpulse.github.io/gst-plugins-edgeimpulse/)

A GStreamer plugin that enables real-time machine learning inference using Edge Impulse models. The plugin provides three elements for audio and video inference and visualization.

## Public API: Inference Output

The plugin exposes inference results through two standardized mechanisms:

1. **GStreamer Bus Messages**
   - All inference elements emit structured messages on the GStreamer bus with the name `edge-impulse-inference-result`.
   - These messages contain the inference result as JSON, along with buffer timestamp and type.
   - Available for both audio and video elements.

2. **Video Frame Metadata (VideoRegionOfInterestMeta)**
   - For video inference, results are also attached as metadata to each video frame using `VideoRegionOfInterestMeta`.
   - This enables downstream elements (e.g., overlays, SDKs) to consume and visualize results directly.
   - Available for video elements only.

   **VideoRegionOfInterestMeta Content:**

   Each `VideoRegionOfInterestMeta` attached to a video frame contains:
   - `x` (u32): X coordinate of the top-left corner of the region (in pixels)
   - `y` (u32): Y coordinate of the top-left corner of the region (in pixels)
   - `width` (u32): Width of the region (in pixels)
   - `height` (u32): Height of the region (in pixels)
   - `label` (String): Class label or description for the region

   For object detection, each detected object is represented as a separate ROI. For classification, a single ROI may cover the whole frame with the top label. For visual anomaly detection, the ROI may include anomaly scores and grid data as additional metadata.

### Supported Model Types and Output Formats

#### 1. Object Detection
- **Bus Message Example:**
  ```json
  {
    "timestamp": 1234567890,
    "type": "object-detection",
    "result": {
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
  ```
- **Video Metadata:**
  - Each detected object is attached as a `VideoRegionOfInterestMeta` with bounding box coordinates, label, and confidence.

#### 2. Classification
- **Bus Message Example:**
  ```json
  {
    "timestamp": 1234567890,
    "type": "classification",
    "result": {
      "classification": {
        "cat": 0.85,
        "dog": 0.15
      }
    }
  }
  ```
- **Video Metadata:**
  - For video, the top classification result may be attached as a single ROI covering the frame, with label and confidence.

#### 3. Visual Anomaly Detection
- **Bus Message Example:**
  ```json
  {
    "timestamp": 1234567890,
    "type": "anomaly-detection",
    "result": {
      "anomaly": 0.35,
      "classification": {
        "normal": 0.85,
        "anomalous": 0.15
      },
      "visual_anomaly_max": 0.42,
      "visual_anomaly_mean": 0.21,
      "visual_anomaly_grid": [
        { "x": 0, "y": 0, "width": 32, "height": 32, "score": 0.12 },
        { "x": 32, "y": 0, "width": 32, "height": 32, "score": 0.18 }
        // ... more grid cells ...
      ]
    }
  }
  ```
- **Video Metadata:**
  - The frame will have a `VideoAnomalyMeta` attached, containing:
    - `anomaly`: The overall anomaly score for the frame
    - `visual_anomaly_max`: The maximum anomaly score in the grid
    - `visual_anomaly_mean`: The mean anomaly score in the grid
    - `visual_anomaly_grid`: A list of grid cells, each with its own region (`x`, `y`, `width`, `height`) and anomaly `score`
  - Optionally, each grid cell may also be represented as a `VideoRegionOfInterestMeta` with the anomaly score as the label or as additional metadata, enabling visualization overlays.

> **Note:** Audio elements only emit bus messages; video elements emit both bus messages and metadata.

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
- Emits inference results as messages (see [Public API](#public-api-inference-output))

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
- Emits inference results as messages (see [Public API](#public-api-inference-output))
- Attaches VideoRegionOfInterestMeta to each video frame (see [Public API](#public-api-inference-output))

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
Video overlay element that visualizes inference results by drawing bounding boxes and labels on video frames.

Element Details:
- Long name: Edge Impulse Overlay
- Class: Filter/Effect/Video
- Description: Draws bounding boxes on video frames based on ROI metadata

Pad Templates:
- Sink/Source pads (Always available):
  ```
  video/x-raw
    format: { RGB, BGR, RGBA, BGRA, UYVY, YUY2, YVYU, NV12, NV21, I420, YV12 }
    width: [ 1, 2147483647 ]
    height: [ 1, 2147483647 ]
  ```

Key features:
- Draws bounding boxes for object detection results (from VideoRegionOfInterestMeta)
- Displays class labels with confidence scores
- Supports wide range of video formats

Properties:
1. `stroke-width` (integer):
   - Width of the bounding box lines in pixels
   - Range: 1 - 100
   - Default: 2

2. `text-color` (unsigned integer):
   - Color of the text in RGB format
   - Range: 0 - 4294967295
   - Default: white (0xFFFFFF)

3. `text-font` (string):
   - Font family to use for text rendering
   - Default: "Sans"

4. `text-font-size` (unsigned integer):
   - Size of the text font in pixels
   - Range: 0 - 4294967295
   - Default: 14

Example pipeline:
```bash
gst-launch-1.0 avfvideosrc ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB,width=384,height=384 ! \
  edgeimpulsevideoinfer model-path=<path-to-model> ! \
  edgeimpulseoverlay stroke-width=3 text-font-size=20 text-color=0x00FF00 ! \
  autovideosink sync=false
```

The overlay element automatically processes VideoRegionOfInterestMeta from upstream elements (like edgeimpulsevideoinfer) and visualizes them with configurable styles.

## Examples

The repository includes examples demonstrating both audio and video inference. These examples have been tested on MacOS.

### Audio Inference
Run the audio inference example:
```bash
# Basic usage
cargo run --example audio_inference -- --model path/to/your/model.eim

# With threshold settings
cargo run --example audio_inference -- --model path/to/your/model.eim \
    --threshold "5.min_score=0.6" \
    --threshold "4.min_anomaly_score=0.35"

# With audio file input
cargo run --example audio_inference -- --model path/to/your/model.eim \
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

### Video Inference
Run the video inference example:
```bash
# Basic usage
cargo run --example video_inference -- --model path/to/your/model.eim

# With threshold settings
cargo run --example video_inference -- --model path/to/your/model.eim \
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
        "classification": {
            "cat": 0.85,
            "dog": 0.15
        }
    }
}
Detected: cat (85.0%)
```

For visual anomaly detection:
```
Got element message with name: edge-impulse-inference-result
Message structure: edge-impulse-inference-result {
    timestamp: (guint64) 1234567890,
    type: "anomaly-detection",
    result: {
        "anomaly": 0.35,
        "classification": {
            "normal": 0.85,
            "anomalous": 0.15
        },
        "visual_anomaly_max": 0.42,
        "visual_anomaly_mean": 0.21,
        "visual_anomaly_grid": [
            { "x": 0, "y": 0, "width": 32, "height": 32, "score": 0.12 },
            { "x": 32, "y": 0, "width": 32, "height": 32, "score": 0.18 }
            // ... more grid cells ...
        ]
    }
}
Detected: normal (85.0%)
Anomaly score: 35.0%
Max grid score: 42.0%
Mean grid score: 21.0%
Grid cells:
  Cell at (0, 0) size 32x32: score 12.0%
  Cell at (32, 0) size 32x32: score 18.0%
  ...
```

The element will automatically detect the model type and emit appropriate messages. Thresholds can be set for both object detection (`min_score`) and anomaly detection (`min_anomaly_score`) blocks. See [Public API](#public-api-inference-output) for output details.

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