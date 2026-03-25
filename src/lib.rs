//! # GStreamer Edge Impulse Plugin
//!
//! A GStreamer plugin for real-time machine learning inference and data ingestion
//! using [Edge Impulse](https://edgeimpulse.com/) models and APIs.
//!
//! ## Elements
//!
//! The plugin provides six elements:
//!
//! | Element | Description |
//! |---------|-------------|
//! | [`video::EdgeImpulseVideoInfer`] | Video inference (classification, detection, anomaly) |
//! | [`audio::EdgeImpulseAudioInfer`] | Audio inference (classification, keyword spotting) |
//! | [`overlay::EdgeImpulseOverlay`] | Draws bounding boxes and labels on video frames |
//! | [`sink::EdgeImpulseSink`] | Uploads audio/video to Edge Impulse ingestion API |
//! | [`filter::EdgeImpulseContinueIf`] | Conditional gate — passes or drops buffers based on inference metadata |
//! | [`crop::EdgeImpulseCrop`] | Dynamic crop — extracts per-detection regions from video frames |
//!
//! ## Metadata types
//!
//! Inference elements attach metadata to every buffer they process. There are
//! two layers:
//!
//! ### Primary API — video-specific metadata
//!
//! These are the main interface for all downstream consumers, including
//! `edgeimpulseoverlay`, `edgeimpulsecrop`, and external elements such as
//! Qualcomm IM SDK's `qtioverlay`:
//!
//! - **`VideoRegionOfInterestMeta`** (GStreamer built-in) — One per detected object with
//!   bounding box coordinates and label. Attached by [`video::EdgeImpulseVideoInfer`].
//!
//! - **`VideoClassificationMeta`** / **`VideoAnomalyMeta`** — Top classification label
//!   and anomaly scores. Attached by [`video::EdgeImpulseVideoInfer`].
//!
//! ### Convenience layer — media-agnostic summary
//!
//! - **[`meta::InferenceResultMeta`]** — Pre-computed summary (detection count, top class,
//!   confidence, anomaly scores). Attached by both audio and video inference elements.
//!   Read by [`filter::EdgeImpulseContinueIf`] to evaluate gate conditions without
//!   parsing video-specific metadata. Does **not** replace the primary API above.
//!
//! ### Crop metadata
//!
//! - **[`crop::meta::CropOriginMeta`]** — Attached to each cropped buffer by
//!   [`crop::EdgeImpulseCrop`], recording the source region in the original frame
//!   so downstream results can be mapped back to full-frame coordinates.
//!
//! ## Common pipeline patterns
//!
//! **Single-stage video inference with overlay:**
//!
//! ```text
//! camera → videoconvert → capsfilter(RGB) → edgeimpulsevideoinfer → edgeimpulseoverlay → display
//! ```
//!
//! **Two-stage detection → classification with crop:**
//!
//! ```text
//! camera → videoconvert → edgeimpulsevideoinfer(detection) → tee
//!   ├── edgeimpulseoverlay → display
//!   └── edgeimpulsecontinueif(detection_count >= 1)
//!         → edgeimpulsecrop → edgeimpulsevideoinfer(classification) → sink
//! ```
//!
//! **Audio inference:**
//!
//! ```text
//! autoaudiosrc → audioconvert → audioresample → capsfilter(S16LE, 16kHz, mono)
//!   → edgeimpulseaudioinfer → audioconvert → autoaudiosink
//! ```

use gstreamer as gst;
use gstreamer::glib;

mod audio;
mod common;
mod crop;
mod filter;
pub mod meta;
mod overlay;
pub mod sink;
pub mod video;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    video::register(plugin)?;
    audio::register(plugin)?;
    overlay::register(plugin)?;
    sink::register(plugin)?;
    filter::register(plugin)?;
    crop::register(plugin)?;
    Ok(())
}

// Include the dynamically generated plugin definition from build.rs
// This ensures the registration function name matches what GStreamer expects
include!(concat!(env!("OUT_DIR"), "/plugin_define.rs"));
