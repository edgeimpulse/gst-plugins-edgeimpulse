# GStreamer Edge Impulse Plugin
A GStreamer plugin that enables real-time machine learning inference using Edge Impulse models. The plugin provides two elements for audio and video inference.

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
git clone https://github.com/edgeimpulse/gst-plugin-edgeimpulse.git
cd gst-plugin-edgeimpulse
cargo build --release
```

## Elements

### edgeimpulseaudioinfer
Audio inference element that processes audio streams through Edge Impulse models.

Key features:
- Accepts S16LE mono audio at 16kHz
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

Key features:
- Accepts RGB video frames
- Passes frames through unchanged
- Performs inference when model is loaded
- Supports both classification and object detection models
- Emits inference results as messages

Example pipeline:
```bash
gst-launch-1.0  avfvideosrc ! \
  queue max-size-buffers=2 leaky=downstream ! \
  videoconvert n-threads=4 ! \
  videoscale method=nearest-neighbour ! \
  video/x-raw,format=RGB,width=384,height=384 ! \
  queue max-size-buffers=2 leaky=downstream ! \
  edgeimpulsevideoinfer model-path=<path-to-model> ! \
  queue max-size-buffers=2 leaky=downstream ! \
  videoscale method=nearest-neighbour ! \
  video/x-raw,width=480,height=480 ! \
  videoconvert n-threads=4 ! \
  autovideosink sync=false
```

## Examples

The repository includes examples demonstrating both audio and video inference. These examples have been tested on MacOS.

### Audio Classification
Run the audio classification example:
```bash
cargo run --example audio_classify -- --model path/to/your/model.eim
```

This will capture audio from the default microphone and display classification results:
```
Got element message with name: edge-impulse-inference-result
Got inference result message
Message structure: edge-impulse-inference-result {
    timestamp: (guint64) 9498000000,
    classification: Structure(classification {
        no: (gfloat) 0.015625,
        noise: (gfloat) 0.968750,
        yes: (gfloat) 0.019531
    })
}
Detected: noise (96.9%)
```

### Video Classification/Detection
Run the video classification example:
```bash
cargo run --example video_classify -- --model path/to/your/model.eim
```

This will capture video from your camera and display inference results:
```
Got element message with name: edge-impulse-inference-result
Message structure: edge-impulse-inference-result {
    timestamp: (guint64) 1234567890,
    type: "classification",
    result: "{\"classification\":[{\"label\":\"cat\",\"value\":0.95},{\"label\":\"dog\",\"value\":0.05}]}"
}
Detected: cat (95.0%)
```

## Message Format
Both elements emit "edge-impulse-inference-result" messages containing:
- timestamp: Buffer presentation timestamp
- type: "classification" or "object-detection" (video only)
- result: JSON string with model output

## Debugging
Enable debug output with:
```bash
GST_DEBUG=edgeimpulseaudioinfer:4 # for audio element
GST_DEBUG=edgeimpulsevideoinfer:4 # for video element
```

## Acknowledgments
This crate is designed to work with Edge Impulse's machine learning models. For more information about Edge Impulse and their ML deployment solutions, visit [Edge Impulse](https://edgeimpulse.com/).

## License
This project is licensed under the BSD 3-Clause Clear License - see the LICENSE file for details.