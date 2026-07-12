# edgeimpulseocr `edge-impulse` Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `edgeimpulseocr backend=edge-impulse` decode the per-character detections that an upstream `edgeimpulsevideoinfer` element attaches to each buffer into lines of text, replacing the current no-op placeholder.

**Architecture:** The `edge-impulse` backend runs no model, so it decodes inline in `transform_ip` (bypassing the worker thread used by pixel backends like `ocrs`). A new pure module `src/ocr/decode.rs` reads the buffer's `VideoRegionOfInterestMeta` character boxes (attached by upstream, already in full-frame pixels with `label`+`confidence` in a `detection` param), **consumes** them, groups them into rows, concatenates each row into a line, and re-uses the existing `shaping` helpers to attach one line ROI per row and post the `ocr` bus message. The `ocr` bus schema and downstream consumers (VIS firmware, `edgeimpulseoverlay`) are unchanged.

**Tech Stack:** Rust, `gstreamer` 0.23 / `gstreamer-video` 0.23 / `gstreamer-base` 0.23, `gst_base::BaseTransform` subclass. Build/test via Cargo with the Homebrew GStreamer toolchain.

---

## Prerequisites (every Cargo command in this plan)

The plugin builds against Homebrew GStreamer. Export these once per shell before any `cargo` command:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig"
```

Feature set matches CI: `--no-default-features --features eim` (the `edge-impulse` OCR path needs neither the `ocrs` nor `ffi` features; `decode.rs` uses only `gstreamer` + `gstreamer-video`, which are always available).

- **Lib unit tests** (Tasks 1–2, also run in CI): `cargo test --no-default-features --features eim --lib <filter>`
- **Integration tests** (Tasks 3–4, local only): the crate is a `cdylib` with no rlib, so `cargo test --test ocr` does **not** build the plugin. You must build it first and point `GST_PLUGIN_PATH` at the artifact:
  ```bash
  cargo build --no-default-features --features eim
  GST_PLUGIN_PATH="$(pwd)/target/debug" cargo test --no-default-features --features eim --test ocr
  ```

## File Structure

- **Create** `src/ocr/decode.rs` — the entire `edge-impulse` decode path: `Detection` struct, pure `assemble_lines`, meta-consuming `take_detections`, and the orchestrating `process_buffer`. Split from `imp.rs` so the meaty logic is host-testable in lib unit tests (CI-covered) rather than only through integration tests.
- **Modify** `src/ocr/mod.rs` — declare `mod decode;` (not feature-gated).
- **Modify** `src/ocr/imp.rs` — skip the worker for `edge-impulse` in `start()`; branch to a new inline `transform_ip_edge_impulse` in `transform_ip`; add an `ei_frame_count` field for message throttling.
- **Modify** `tests/ocr.rs` — retarget the existing `edge-impulse` test (it now decodes) and add happy-path + throttle integration tests.
- **Modify** `docs/edgeimpulseocr.md` — document the now-implemented backend, an example pipeline, and per-property applicability.

No changes to `src/ocr/backend.rs` (the `Backend::EdgeImpulse` variant and `OcrLine` already exist) or `src/ocr/shaping.rs` (its `filter_and_truncate` / `attach_results` / `build_ocr_message` are reused verbatim).

---

## Task 1: `decode.rs` — `Detection` + pure `assemble_lines`

**Files:**
- Create: `src/ocr/decode.rs`
- Modify: `src/ocr/mod.rs:7-9` (add `mod decode;`)
- Test: `src/ocr/decode.rs` (`#[cfg(test)]` module, pure — no `gst::init`)

- [ ] **Step 1: Declare the module**

In `src/ocr/mod.rs`, add `mod decode;` next to the other non-feature-gated module declarations. Change:

```rust
mod backend;
mod imp;
mod shaping;
```

to:

```rust
mod backend;
mod decode;
mod imp;
mod shaping;
```

- [ ] **Step 2: Write `decode.rs` with `Detection` + `assemble_lines` and the tests**

Create `src/ocr/decode.rs` with exactly this content:

```rust
//! `edge-impulse` OCR backend: decode the per-character detections an upstream
//! Edge Impulse object-detection model (run by `edgeimpulsevideoinfer`) attached
//! to the buffer into lines of text, reusing the shared `shaping` output path.
//!
//! Unlike the `ocrs` backend this evaluates no model: the character boxes are
//! already on the buffer as `VideoRegionOfInterestMeta` (in full-frame pixels,
//! with `label` + `confidence` in a `detection` param). We group them into rows,
//! concatenate each row into a line, and hand the lines to `shaping`.

use crate::ocr::backend::OcrLine;
use gstreamer as gst;
use gstreamer_video as gst_video;

/// One upstream character detection, in full-frame pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Overlap length of the 1-D intervals `[a0, a1)` and `[b0, b1)` (0 if disjoint).
fn overlap_1d(a0: u32, a1: u32, b0: u32, b1: u32) -> u32 {
    a1.min(b1).saturating_sub(a0.max(b0))
}

/// Group character detections into text lines: rows by vertical overlap,
/// left-to-right within a row, one `OcrLine` per row, ordered top-to-bottom.
///
/// Detections with an empty label are dropped. A box joins the current row when
/// its vertical span overlaps the row's span by at least half the shorter of the
/// two heights; otherwise it starts a new row. Line text is the row's labels
/// concatenated in x order (no spaces inserted); line confidence is the minimum
/// member confidence; line bbox is the union of member boxes.
pub fn assemble_lines(dets: &[Detection]) -> Vec<OcrLine> {
    let mut items: Vec<&Detection> = dets.iter().filter(|d| !d.label.is_empty()).collect();
    if items.is_empty() {
        return Vec::new();
    }
    // Sort top-to-bottom (tie-break left) so a single greedy pass can grow rows:
    // each box either extends the current (last) row or starts a new one.
    items.sort_by(|a, b| (a.y, a.x).cmp(&(b.y, b.x)));

    let mut rows: Vec<Vec<&Detection>> = Vec::new();
    for d in items {
        let joins_last = rows.last().is_some_and(|row| {
            let top = row.iter().map(|r| r.y).min().unwrap();
            let bottom = row.iter().map(|r| r.y + r.h).max().unwrap();
            let shorter = (bottom - top).min(d.h);
            shorter > 0 && overlap_1d(d.y, d.y + d.h, top, bottom) * 2 >= shorter
        });
        if joins_last {
            rows.last_mut().unwrap().push(d);
        } else {
            rows.push(vec![d]);
        }
    }

    let mut lines: Vec<OcrLine> = rows
        .into_iter()
        .map(|mut row| {
            row.sort_by(|a, b| (a.x, a.y).cmp(&(b.x, b.y)));
            let text: String = row.iter().map(|d| d.label.as_str()).collect();
            let confidence = row
                .iter()
                .map(|d| d.confidence)
                .fold(f32::INFINITY, f32::min);
            let x = row.iter().map(|d| d.x).min().unwrap();
            let y = row.iter().map(|d| d.y).min().unwrap();
            let right = row.iter().map(|d| d.x + d.w).max().unwrap();
            let bottom = row.iter().map(|d| d.y + d.h).max().unwrap();
            OcrLine {
                text,
                confidence,
                x,
                y,
                w: right - x,
                h: bottom - y,
            }
        })
        .collect();
    lines.sort_by(|a, b| (a.y, a.x).cmp(&(b.y, b.x)));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(label: &str, conf: f32, x: u32, y: u32, w: u32, h: u32) -> Detection {
        Detection {
            label: label.into(),
            confidence: conf,
            x,
            y,
            w,
            h,
        }
    }

    #[test]
    fn empty_input_yields_no_lines() {
        assert!(assemble_lines(&[]).is_empty());
    }

    #[test]
    fn single_detection_is_one_line() {
        let lines = assemble_lines(&[det("A", 0.9, 5, 10, 8, 20)]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "A");
        assert_eq!((lines[0].x, lines[0].y, lines[0].w, lines[0].h), (5, 10, 8, 20));
    }

    #[test]
    fn same_row_orders_left_to_right() {
        // Fed out of order; must come back "AB".
        let lines = assemble_lines(&[
            det("B", 0.8, 20, 10, 10, 20),
            det("A", 0.9, 0, 10, 10, 20),
        ]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "AB");
    }

    #[test]
    fn union_bbox_covers_all_members() {
        let lines = assemble_lines(&[
            det("A", 0.9, 0, 10, 10, 20),
            det("B", 0.8, 20, 10, 10, 24),
        ]);
        // x: 0..30, y: 10, bottom: max(30, 34) = 34 -> h = 24.
        assert_eq!((lines[0].x, lines[0].y, lines[0].w, lines[0].h), (0, 10, 30, 24));
    }

    #[test]
    fn line_confidence_is_minimum_member() {
        let lines = assemble_lines(&[
            det("A", 0.9, 0, 10, 10, 20),
            det("B", 0.3, 20, 10, 10, 20),
        ]);
        assert!((lines[0].confidence - 0.3).abs() < 1e-6);
    }

    #[test]
    fn two_rows_are_separate_and_top_first() {
        let lines = assemble_lines(&[
            det("X", 0.9, 0, 100, 10, 20),
            det("Y", 0.9, 20, 100, 10, 20),
            det("A", 0.9, 0, 0, 10, 20),
            det("B", 0.9, 20, 0, 10, 20),
        ]);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "AB");
        assert_eq!(lines[1].text, "XY");
    }

    #[test]
    fn empty_labels_are_dropped() {
        let lines = assemble_lines(&[
            det("A", 0.9, 0, 10, 10, 20),
            det("", 0.9, 20, 10, 10, 20),
        ]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "A");
    }

    #[test]
    fn slight_vertical_offset_still_one_row() {
        // 18/20 overlap of the shorter height -> same row.
        let lines = assemble_lines(&[
            det("A", 0.9, 0, 10, 10, 20),
            det("B", 0.9, 20, 12, 10, 20),
        ]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "AB");
    }
}
```

- [ ] **Step 3: Run the tests to verify they pass**

Run:
```bash
cargo test --no-default-features --features eim --lib decode::tests
```
Expected: PASS — `test result: ok. 8 passed`.

> Note: this is not the classic red-then-green loop because `assemble_lines` and its tests are added together; the module did not exist before, so there is nothing to fail against first. The tests still fully pin the behavior. (Task 2 and the integration tests use a proper failing-first step.)

- [ ] **Step 4: Confirm the module compiles under the default feature set too**

Run:
```bash
cargo build --lib
```
Expected: builds clean (verifies `decode.rs` compiles with `ocrs` enabled as well, since `mod decode;` is not feature-gated).

- [ ] **Step 5: Commit**

```bash
git add src/ocr/mod.rs src/ocr/decode.rs
git commit -m "feat(ocr): add Detection + assemble_lines for edge-impulse decode"
```

---

## Task 2: `decode.rs` — `take_detections` (consume metas) + `process_buffer`

**Files:**
- Modify: `src/ocr/decode.rs` (add two functions + tests that require `gst::init`)
- Test: `src/ocr/decode.rs` (`#[cfg(test)]`)

- [ ] **Step 1: Write the failing tests**

Add these tests inside the existing `mod tests` block in `src/ocr/decode.rs` (after `slight_vertical_offset_still_one_row`). They reference `take_detections` and `process_buffer`, which do not exist yet, so the module will fail to compile — that is the intended failure.

```rust
    fn init() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| gst::init().expect("gst init"));
    }

    fn char_roi(buf: &mut gst::BufferRef, label: &str, x: u32, conf: f64) {
        let mut roi =
            gst_video::VideoRegionOfInterestMeta::add(buf, label, (x, 10, 10, 20));
        roi.add_param(
            gst::Structure::builder("detection")
                .field("label", label)
                .field("confidence", conf)
                .build(),
        );
    }

    #[test]
    fn take_detections_reads_and_consumes_detection_rois() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        char_roi(b, "A", 0, 0.9);
        char_roi(b, "B", 20, 0.8);
        let dets = take_detections(b);
        assert_eq!(dets.len(), 2);
        assert!(dets.iter().any(|d| d.label == "A" && (d.confidence - 0.9).abs() < 1e-6));
        assert!(dets.iter().any(|d| d.label == "B"));
        // The character ROIs are consumed.
        assert_eq!(
            b.iter_meta::<gst_video::VideoRegionOfInterestMeta>().count(),
            0
        );
    }

    #[test]
    fn take_detections_ignores_rois_without_detection_param() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        // A plain ROI with no `detection` param must be left untouched.
        gst_video::VideoRegionOfInterestMeta::add(b, "face", (0, 0, 4, 4));
        let dets = take_detections(b);
        assert!(dets.is_empty());
        assert_eq!(
            b.iter_meta::<gst_video::VideoRegionOfInterestMeta>().count(),
            1
        );
    }

    #[test]
    fn take_detections_on_empty_buffer_is_empty() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        assert!(take_detections(b).is_empty());
    }

    #[test]
    fn process_buffer_consumes_chars_and_attaches_one_line() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        char_roi(b, "A", 0, 0.9);
        char_roi(b, "B", 20, 0.8);
        let lines = process_buffer(b, 0.0, 256);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "AB");
        // Exactly one ROI remains: the assembled line (char ROIs consumed).
        let labels: Vec<String> = b
            .iter_meta::<gst_video::VideoRegionOfInterestMeta>()
            .filter_map(|m| {
                m.params()
                    .find(|p| p.name() == "detection")
                    .and_then(|p| p.get::<String>("label").ok())
            })
            .collect();
        assert_eq!(labels, vec!["AB".to_string()]);
    }

    #[test]
    fn process_buffer_filters_below_min_confidence() {
        init();
        let mut buf = gst::Buffer::with_size(64).unwrap();
        let b = buf.get_mut().unwrap();
        char_roi(b, "A", 0, 0.9);
        char_roi(b, "B", 20, 0.8); // line confidence = min = 0.8
        let lines = process_buffer(b, 0.85, 256);
        assert!(lines.is_empty());
        // Chars consumed, nothing attached.
        assert_eq!(
            b.iter_meta::<gst_video::VideoRegionOfInterestMeta>().count(),
            0
        );
    }
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
cargo test --no-default-features --features eim --lib decode::tests
```
Expected: FAIL — compile error, `cannot find function take_detections` / `process_buffer` in this scope.

- [ ] **Step 3: Implement `take_detections` + `process_buffer`**

Add these imports at the top of `src/ocr/decode.rs` (below the existing `use` lines):

```rust
use gst::buffer::BufferMetaForeachAction;
use std::ops::ControlFlow;
```

Add these two functions to `src/ocr/decode.rs`, after `assemble_lines` and before the `#[cfg(test)]` module:

```rust
/// Read every character detection from the buffer's `VideoRegionOfInterestMeta`
/// entries **and remove those metas**. We consume the per-character boxes so
/// downstream renders only the assembled line, not the raw glyphs. Only ROIs
/// carrying a `detection` param (the Edge Impulse convention) are taken; any
/// other ROI meta is left in place.
pub fn take_detections(buf: &mut gst::BufferRef) -> Vec<Detection> {
    let mut out = Vec::new();
    buf.foreach_meta_mut(|mut meta| {
        let action = match meta.downcast_ref::<gst_video::VideoRegionOfInterestMeta>() {
            Some(roi) => match roi.params().find(|p| p.name() == "detection") {
                Some(p) => {
                    let (x, y, w, h) = roi.rect();
                    let label = p.get::<String>("label").unwrap_or_default();
                    let confidence = p.get::<f64>("confidence").unwrap_or(0.0) as f32;
                    out.push(Detection {
                        label,
                        confidence,
                        x,
                        y,
                        w,
                        h,
                    });
                    BufferMetaForeachAction::Remove
                }
                None => BufferMetaForeachAction::Keep,
            },
            None => BufferMetaForeachAction::Keep,
        };
        ControlFlow::Continue(action)
    });
    out
}

/// Full edge-impulse transform for one buffer: consume the upstream character
/// detections, assemble them into lines, filter by confidence / length, attach
/// one ROI meta per line, and return the lines so the caller can post `ocr`
/// bus messages.
pub fn process_buffer(
    buf: &mut gst::BufferRef,
    min_confidence: f32,
    max_len: usize,
) -> Vec<OcrLine> {
    let dets = take_detections(buf);
    let lines = assemble_lines(&dets);
    let lines = crate::ocr::shaping::filter_and_truncate(lines, min_confidence, max_len);
    crate::ocr::shaping::attach_results(buf, &lines);
    lines
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cargo test --no-default-features --features eim --lib decode::tests
```
Expected: PASS — `test result: ok. 13 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ocr/decode.rs
git commit -m "feat(ocr): consume upstream detection metas and assemble lines"
```

---

## Task 3: Wire `imp.rs` — inline `edge-impulse` path + integration test

**Files:**
- Modify: `src/ocr/imp.rs:227-258` (add `transform_ip_edge_impulse` to the inherent impl)
- Modify: `src/ocr/imp.rs:266-270` (skip worker in `start()`)
- Modify: `src/ocr/imp.rs:332-344` (branch in `transform_ip`)
- Modify: `tests/ocr.rs:91-149` (retarget the existing `edge-impulse` test)
- Test: `tests/ocr.rs` (new `edge_impulse_decodes_upstream_detections`)

- [ ] **Step 1: Write the failing integration test**

In `tests/ocr.rs`, first **replace** the existing `noop_backend_attaches_no_metas` test (currently at lines 91–149) — its comment claims a no-op backend, which is no longer true. Replace the whole `#[test] fn noop_backend_attaches_no_metas() { ... }` function with:

```rust
#[test]
fn edge_impulse_without_upstream_detections_attaches_no_metas() {
    init();
    let got_meta = Arc::new(Mutex::new(false));
    let gm = got_meta.clone();
    // With no upstream detections on the buffers (plain videotestsrc), the
    // edge-impulse backend has nothing to decode, so it must attach no ROI metas.
    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=2 ! video/x-raw,format=RGB,width=80,height=48 ! \
         videoconvert ! {} backend=edge-impulse interval=1 ! appsink name=sink",
        ocr_element_name()
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gstreamer_app::AppSink>()
        .unwrap();
    sink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |s| {
                let sample = s.pull_sample().unwrap();
                if let Some(buf) = sample.buffer() {
                    if buf
                        .iter_meta::<gstreamer_video::VideoRegionOfInterestMeta>()
                        .count()
                        > 0
                    {
                        *gm.lock().unwrap() = true;
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    pipeline.set_state(gst::State::Playing).unwrap();
    for msg in pipeline
        .bus()
        .unwrap()
        .iter_timed(gst::ClockTime::from_seconds(10))
    {
        use gst::MessageView::*;
        match msg.view() {
            Eos(..) => break,
            Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
    assert!(
        !*got_meta.lock().unwrap(),
        "edge-impulse must not attach metas when there are no upstream detections"
    );
}
```

Then **add** this new test immediately after it:

```rust
#[test]
fn edge_impulse_decodes_upstream_detections() {
    init();
    if gst::ElementFactory::find(&ocr_element_name()).is_none() {
        panic!(
            "edgeimpulseocr not found — build the plugin and set \
             GST_PLUGIN_PATH=\"$(pwd)/target/debug\""
        );
    }

    // Labels of ROI metas seen on output buffers, read from their `detection`
    // param (matching how downstream consumers read them).
    let out_labels = Arc::new(Mutex::new(Vec::<String>::new()));
    let ol = out_labels.clone();

    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=80,height=48 ! \
         videoconvert ! {elem} name=ocr backend=edge-impulse interval=1 ! \
         appsink name=sink",
        elem = ocr_element_name(),
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();

    // Mimic an upstream edgeimpulsevideoinfer: inject two character detections
    // ("A" at x=0, "B" at x=20, same row) on the ocr element's sink pad.
    let ocr = pipeline.by_name("ocr").unwrap();
    let sinkpad = ocr.static_pad("sink").unwrap();
    sinkpad.add_probe(gst::PadProbeType::BUFFER, |_pad, info| {
        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
            let buf = buffer.make_mut();
            for (label, x, conf) in [("A", 0u32, 0.9f64), ("B", 20u32, 0.8f64)] {
                let mut roi = gstreamer_video::VideoRegionOfInterestMeta::add(
                    buf,
                    label,
                    (x, 10, 10, 20),
                );
                roi.add_param(
                    gst::Structure::builder("detection")
                        .field("label", label)
                        .field("confidence", conf)
                        .build(),
                );
            }
        }
        gst::PadProbeReturn::Ok
    });

    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gstreamer_app::AppSink>()
        .unwrap();
    sink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |s| {
                let sample = s.pull_sample().unwrap();
                if let Some(buf) = sample.buffer() {
                    for m in buf.iter_meta::<gstreamer_video::VideoRegionOfInterestMeta>() {
                        if let Some(p) = m.params().find(|p| p.name() == "detection") {
                            if let Ok(l) = p.get::<String>("label") {
                                ol.lock().unwrap().push(l);
                            }
                        }
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    let ocr_texts = Arc::new(Mutex::new(Vec::<String>::new()));
    let ot = ocr_texts.clone();
    pipeline.set_state(gst::State::Playing).unwrap();
    for msg in pipeline
        .bus()
        .unwrap()
        .iter_timed(gst::ClockTime::from_seconds(10))
    {
        use gst::MessageView::*;
        match msg.view() {
            Element(e) => {
                if let Some(st) = e.structure() {
                    if st.name() == "ocr" {
                        ot.lock().unwrap().push(st.get::<String>("text").unwrap());
                    }
                }
            }
            Eos(..) => break,
            Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();

    assert_eq!(
        *ocr_texts.lock().unwrap(),
        vec!["AB".to_string()],
        "expected exactly one ocr message with the assembled text"
    );
    assert_eq!(
        *out_labels.lock().unwrap(),
        vec!["AB".to_string()],
        "character ROIs must be consumed and replaced by a single line ROI"
    );
}
```

- [ ] **Step 2: Run to verify it fails**

Build the current plugin (still the no-op edge-impulse path) and run the new test:
```bash
cargo build --no-default-features --features eim
GST_PLUGIN_PATH="$(pwd)/target/debug" cargo test --no-default-features --features eim --test ocr edge_impulse_decodes_upstream_detections
```
Expected: FAIL — asserts `["AB"]` but the placeholder decodes nothing: `ocr_texts` is `[]` and `out_labels` is `["A", "B"]` (the injected char ROIs pass through untouched).

- [ ] **Step 3: Skip the worker for `edge-impulse` in `start()`**

In `src/ocr/imp.rs`, at the top of `fn start`, change:

```rust
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings.lock().unwrap().clone();
        self.worker_gone_logged.store(false, Ordering::Relaxed);
```

to:

```rust
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings.lock().unwrap().clone();
        // The edge-impulse backend decodes upstream detection metas inline in
        // transform_ip; it evaluates no model and needs no worker thread.
        if matches!(Backend::parse(&settings.backend), Some(Backend::EdgeImpulse)) {
            return self.parent_start();
        }
        self.worker_gone_logged.store(false, Ordering::Relaxed);
```

- [ ] **Step 4: Add the inline `transform_ip_edge_impulse` method**

In `src/ocr/imp.rs`, inside the inherent `impl EdgeImpulseOcr { ... }` block that contains `build_backend`/`build_ocrs` (ends around line 258), add this method right before the closing `}` of that impl block (after the `#[cfg(not(feature = "ocrs"))] fn build_ocrs`):

```rust
    /// Decode the character detections an upstream `edgeimpulsevideoinfer`
    /// attached to this buffer into lines of text: consume the per-character ROI
    /// metas, assemble them into lines, attach one line ROI, and post one `ocr`
    /// message per line. Runs inline (no worker) because there is no model.
    fn transform_ip_edge_impulse(
        &self,
        buf: &mut gst::BufferRef,
        min_confidence: f64,
        max_text_length: u32,
        post_message: bool,
        _interval: u32,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let pts_ms = buf.pts().map(|t| t.mseconds() as i64).unwrap_or(0);
        let lines = crate::ocr::decode::process_buffer(
            buf,
            min_confidence as f32,
            max_text_length as usize,
        );
        if post_message {
            for line in &lines {
                let s = build_ocr_message(line, pts_ms);
                let _ = self
                    .obj()
                    .post_message(gst::message::Element::builder(s).src(&*self.obj()).build());
            }
        }
        Ok(gst::FlowSuccess::Ok)
    }
```

- [ ] **Step 5: Branch to it in `transform_ip`**

In `src/ocr/imp.rs`, at the top of `fn transform_ip`, change:

```rust
        let (interval, min_confidence, max_text_length, post_message) = {
            let settings = self.settings.lock().unwrap();
            (
                // max(1) keeps the modulo below divide-by-zero-safe independent
                // of the GObject minimum, so a direct Settings construction
                // (e.g. in a unit test) can never panic here.
                settings.interval.max(1),
                settings.min_confidence,
                settings.max_text_length,
                settings.post_message,
            )
        };
```

to:

```rust
        let (backend, interval, min_confidence, max_text_length, post_message) = {
            let settings = self.settings.lock().unwrap();
            (
                settings.backend.clone(),
                // max(1) keeps the modulo below divide-by-zero-safe independent
                // of the GObject minimum, so a direct Settings construction
                // (e.g. in a unit test) can never panic here.
                settings.interval.max(1),
                settings.min_confidence,
                settings.max_text_length,
                settings.post_message,
            )
        };

        // The edge-impulse backend decodes upstream detection metas synchronously
        // (no pixels, no worker); handle it before the caps/worker path below.
        if matches!(Backend::parse(&backend), Some(Backend::EdgeImpulse)) {
            return self.transform_ip_edge_impulse(
                buf,
                min_confidence,
                max_text_length,
                post_message,
                interval,
            );
        }
```

- [ ] **Step 6: Rebuild and run the whole integration suite**

```bash
cargo build --no-default-features --features eim
GST_PLUGIN_PATH="$(pwd)/target/debug" cargo test --no-default-features --features eim --test ocr
```
Expected: PASS — `passes_buffers_through_unchanged`, `exposes_configurable_properties`, `edge_impulse_without_upstream_detections_attaches_no_metas`, and `edge_impulse_decodes_upstream_detections` all pass; `worker_recognizes_text_and_attaches_results` prints `skipping:` and returns.

- [ ] **Step 7: Run the lib tests to confirm no regression**

```bash
cargo test --no-default-features --features eim --lib
```
Expected: PASS (Task 1–2 decode tests + existing backend/shaping tests).

- [ ] **Step 8: Commit**

```bash
git add src/ocr/imp.rs tests/ocr.rs
git commit -m "feat(ocr): implement edge-impulse backend decoding in transform_ip"
```

---

## Task 4: `interval` throttling of `ocr` messages

**Files:**
- Modify: `src/ocr/imp.rs:11` (import `AtomicU64`)
- Modify: `src/ocr/imp.rs:82-90` (add `ei_frame_count` field)
- Modify: `src/ocr/imp.rs` `start()` edge-impulse branch (reset counter)
- Modify: `src/ocr/imp.rs` `transform_ip_edge_impulse` (throttle posting)
- Test: `tests/ocr.rs` (new `edge_impulse_interval_throttles_ocr_messages`)

- [ ] **Step 1: Write the failing test**

In `tests/ocr.rs`, add after `edge_impulse_decodes_upstream_detections`:

```rust
#[test]
fn edge_impulse_interval_throttles_ocr_messages() {
    init();
    if gst::ElementFactory::find(&ocr_element_name()).is_none() {
        panic!(
            "edgeimpulseocr not found — build the plugin and set \
             GST_PLUGIN_PATH=\"$(pwd)/target/debug\""
        );
    }

    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=80,height=48 ! \
         videoconvert ! {elem} name=ocr backend=edge-impulse interval=5 ! \
         appsink name=sink",
        elem = ocr_element_name(),
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();

    // Inject one decodable detection on every buffer.
    let ocr = pipeline.by_name("ocr").unwrap();
    let sinkpad = ocr.static_pad("sink").unwrap();
    sinkpad.add_probe(gst::PadProbeType::BUFFER, |_pad, info| {
        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
            let buf = buffer.make_mut();
            let mut roi =
                gstreamer_video::VideoRegionOfInterestMeta::add(buf, "X", (0, 10, 10, 20));
            roi.add_param(
                gst::Structure::builder("detection")
                    .field("label", "X")
                    .field("confidence", 0.9f64)
                    .build(),
            );
        }
        gst::PadProbeReturn::Ok
    });

    // Drain output buffers so the pipeline runs to EOS.
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gstreamer_app::AppSink>()
        .unwrap();
    sink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |s| {
                let _ = s.pull_sample().unwrap();
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    let ocr_count = Arc::new(Mutex::new(0usize));
    let oc = ocr_count.clone();
    pipeline.set_state(gst::State::Playing).unwrap();
    for msg in pipeline
        .bus()
        .unwrap()
        .iter_timed(gst::ClockTime::from_seconds(10))
    {
        use gst::MessageView::*;
        match msg.view() {
            Element(e) => {
                if e.structure().map_or(false, |st| st.name() == "ocr") {
                    *oc.lock().unwrap() += 1;
                }
            }
            Eos(..) => break,
            Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();

    assert_eq!(
        *ocr_count.lock().unwrap(),
        1,
        "interval=5 over 5 buffers must post exactly one ocr message"
    );
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo build --no-default-features --features eim
GST_PLUGIN_PATH="$(pwd)/target/debug" cargo test --no-default-features --features eim --test ocr edge_impulse_interval_throttles_ocr_messages
```
Expected: FAIL — asserts `1` but gets `5` (Task 3 posts a message every frame).

- [ ] **Step 3: Import `AtomicU64`**

In `src/ocr/imp.rs`, change:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
```

to:

```rust
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
```

- [ ] **Step 4: Add the `ei_frame_count` field**

In `src/ocr/imp.rs`, change the struct:

```rust
#[derive(Default)]
pub struct EdgeImpulseOcr {
    pub(crate) settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
    info: Mutex<Option<gst_video::VideoInfo>>,
    /// Set once the worker thread is observed gone, so we warn only once
    /// instead of on every subsequently dropped frame.
    worker_gone_logged: AtomicBool,
}
```

to:

```rust
#[derive(Default)]
pub struct EdgeImpulseOcr {
    pub(crate) settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
    info: Mutex<Option<gst_video::VideoInfo>>,
    /// Set once the worker thread is observed gone, so we warn only once
    /// instead of on every subsequently dropped frame.
    worker_gone_logged: AtomicBool,
    /// Frames seen by the edge-impulse path, used to post an `ocr` message only
    /// once per `interval` frames (reset in `start`).
    ei_frame_count: AtomicU64,
}
```

- [ ] **Step 5: Reset the counter in `start()`**

In `src/ocr/imp.rs`, change the edge-impulse branch added in Task 3:

```rust
        if matches!(Backend::parse(&settings.backend), Some(Backend::EdgeImpulse)) {
            return self.parent_start();
        }
```

to:

```rust
        if matches!(Backend::parse(&settings.backend), Some(Backend::EdgeImpulse)) {
            self.ei_frame_count.store(0, Ordering::Relaxed);
            return self.parent_start();
        }
```

- [ ] **Step 6: Throttle posting in `transform_ip_edge_impulse`**

In `src/ocr/imp.rs`, replace the whole `transform_ip_edge_impulse` method from Task 3 with:

```rust
    /// Decode the character detections an upstream `edgeimpulsevideoinfer`
    /// attached to this buffer into lines of text: consume the per-character ROI
    /// metas, assemble them into lines, attach one line ROI, and post one `ocr`
    /// message per line. Runs inline (no worker) because there is no model.
    fn transform_ip_edge_impulse(
        &self,
        buf: &mut gst::BufferRef,
        min_confidence: f64,
        max_text_length: u32,
        post_message: bool,
        interval: u32,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let pts_ms = buf.pts().map(|t| t.mseconds() as i64).unwrap_or(0);
        let lines = crate::ocr::decode::process_buffer(
            buf,
            min_confidence as f32,
            max_text_length as usize,
        );
        // Detections are consumed and lines attached every frame (so the overlay
        // stays stable); `interval` only throttles how often we *post* messages.
        let n = self.ei_frame_count.fetch_add(1, Ordering::Relaxed) + 1;
        let due = n % (interval.max(1) as u64) == 0;
        if post_message && due {
            for line in &lines {
                let s = build_ocr_message(line, pts_ms);
                let _ = self
                    .obj()
                    .post_message(gst::message::Element::builder(s).src(&*self.obj()).build());
            }
        }
        Ok(gst::FlowSuccess::Ok)
    }
```

- [ ] **Step 7: Rebuild and run the integration suite**

```bash
cargo build --no-default-features --features eim
GST_PLUGIN_PATH="$(pwd)/target/debug" cargo test --no-default-features --features eim --test ocr
```
Expected: PASS — including `edge_impulse_interval_throttles_ocr_messages` (1 message) and `edge_impulse_decodes_upstream_detections` (interval=1 → still 1 message).

- [ ] **Step 8: Commit**

```bash
git add src/ocr/imp.rs tests/ocr.rs
git commit -m "feat(ocr): throttle edge-impulse ocr messages by interval"
```

---

## Task 5: Documentation

**Files:**
- Modify: `docs/edgeimpulseocr.md` (backend description, example pipeline, property applicability)
- Modify: `src/ocr/imp.rs` (`backend` property `blurb`)

- [ ] **Step 1: Update the `backend` property blurb**

In `src/ocr/imp.rs`, in the `properties()` list, change the `backend` ParamSpec blurb:

```rust
                glib::ParamSpecString::builder("backend")
                    .nick("Backend")
                    .blurb("OCR backend implementation to use")
                    .default_value(Some("ocrs"))
                    .mutable_ready()
                    .build(),
```

to:

```rust
                glib::ParamSpecString::builder("backend")
                    .nick("Backend")
                    .blurb(
                        "OCR backend: 'ocrs' runs a built-in detection+recognition \
                         model on the RGB frame; 'edge-impulse' decodes per-character \
                         detections from an upstream edgeimpulsevideoinfer element",
                    )
                    .default_value(Some("ocrs"))
                    .mutable_ready()
                    .build(),
```

- [ ] **Step 2: Read the current backend descriptions in `docs/edgeimpulseocr.md`**

Run:
```bash
grep -n "edge-impulse\|no-op\|placeholder\|planned" docs/edgeimpulseocr.md
```
This surfaces the two spots (around lines 33 and 91) that describe `edge-impulse` as planned/no-op. Note the surrounding heading levels so replacements match the file's style.

- [ ] **Step 3: Update the backend description (near line 33)**

Replace the `edge-impulse` bullet/description that says it is planned or a no-op placeholder with:

```markdown
- `edge-impulse`: decodes the per-character object-detection results of an
  **upstream** `edgeimpulsevideoinfer` element into lines of text. It evaluates
  no model itself: it reads the `VideoRegionOfInterestMeta` character boxes
  (label + confidence in a `detection` param, in full-frame pixels), groups them
  into rows by vertical overlap, and concatenates each row into a line. The
  per-character ROIs are consumed and replaced by one ROI per assembled line, so
  `edgeimpulseoverlay` renders the lines, not the raw glyphs.
```

- [ ] **Step 4: Replace the second "planned/no-op" mention (near line 91) with an example + applicability table**

Replace that paragraph with (adjust `##`/`###` levels to match the surrounding document):

````markdown
### Example: `edge-impulse` backend

Run an Edge Impulse per-character object-detection model upstream and let
`edgeimpulseocr` assemble the characters into text lines:

```bash
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
  edgeimpulsevideoinfer model-path=/path/to/model ! \
  edgeimpulseocr backend=edge-impulse ! \
  edgeimpulseoverlay ! videoconvert ! autovideosink
```

Compare with the built-in model backend, which needs no upstream inference:

```bash
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGB ! \
  edgeimpulseocr backend=ocrs ! \
  edgeimpulseoverlay ! videoconvert ! autovideosink
```

#### Property applicability

| Property            | `ocrs`                    | `edge-impulse`                                |
| ------------------- | ------------------------- | --------------------------------------------- |
| `detection-model`   | Path to detection model   | Ignored (detection happens upstream)          |
| `recognition-model` | Path to recognition model | Ignored (recognition happens upstream)        |
| `min-confidence`    | Filters recognized lines  | Filters lines by their weakest character      |
| `max-text-length`   | Truncates line text       | Truncates line text                           |
| `post-message`      | Posts `ocr` bus messages  | Posts `ocr` bus messages                      |
| `interval`          | 1-in-N frames recognized  | Every frame decoded; throttles `ocr` messages |
````

- [ ] **Step 5: Verify the build still succeeds after the blurb edit**

Docs are not compiled, but confirm the `blurb` edit did not break the build:
```bash
cargo build --no-default-features --features eim
```
Expected: builds clean.

- [ ] **Step 6: Commit**

```bash
git add docs/edgeimpulseocr.md src/ocr/imp.rs
git commit -m "docs(ocr): document implemented edge-impulse backend"
```

---

## Final verification

- [ ] **Formatting** (CI runs `cargo fmt -- --check`):

```bash
cargo fmt -- --check
```
Expected: no output (clean). If it reports diffs, run `cargo fmt` and amend the relevant commit.

- [ ] **Full CI-equivalent build + lib tests:**

```bash
cargo build --no-default-features --features eim --verbose
cargo test --no-default-features --features eim --verbose --lib
```
Expected: build succeeds; all lib tests pass.

- [ ] **Full integration suite (local):**

```bash
cargo build --no-default-features --features eim
GST_PLUGIN_PATH="$(pwd)/target/debug" cargo test --no-default-features --features eim --test ocr
```
Expected: all `tests/ocr.rs` tests pass (the models-gated `worker_recognizes_text_and_attaches_results` prints `skipping:` and returns).

- [ ] **Default-feature build** (ensures the `ocrs` build path still compiles with the new module):

```bash
cargo build
```
Expected: builds clean.
