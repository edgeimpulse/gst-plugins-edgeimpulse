# GStreamer Edge Impulse Plugin
A GStreamer plugin that enables real-time machine learning inference using Edge Impulse models.

## Dependencies
This plugin depends on:
* GStreamer 1.20 or newer
* edge-impulse-runner-rs - Rust bindings for Edge Impulse Linux SDK
* A trained Edge Impulse model file (.eim)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/edgeimpulse/gstreamer-edge-impulse.git
```
2. Build the plugin:
```bash
cargo build --release
```

## Examples

All examples are located in the `examples` directory.

These examples have only been tested on MacOS.

### Audio Classification

The repository includes an example that demonstrates audio classification using Edge Impulse models within a GStreamer pipeline.

To run the audio classification example:
```bash
cargo run --example audio_classify -- --model path/to/your/model.eim
```

This will start the GStreamer pipeline capturing audio from the default microphone and display the classification results in the console.

The example will print the detected class with the highest confidence:
```
Got element message with name: edge-impulse-inference-result
Got inference result message
Message structure: edge-impulse-inference-result { timestamp: (guint64) 9498000000, classification: Structure(classification { no: (gfloat) 0.015625, noise: (gfloat) 0.968750, yes: (gfloat) 0.019531 }) }
Detected: noise (96.9%)
```


## Acknowledgments
This crate is designed to work with Edge Impulse's machine learning models. For more information about Edge Impulse and their ML deployment solutions, visit [Edge Impulse](https://edgeimpulse.com/).


## License
This project is licensed under the BSD 3-Clause Clear License - see the LICENSE file for details.