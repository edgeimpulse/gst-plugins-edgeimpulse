#!/usr/bin/env bash
#
# benchmark_rb3.sh — Compare system-memory vs GBM zero-copy inference pipelines
#
# Run on Qualcomm RB3 board with:
#   ./examples/benchmark_rb3.sh [--duration 30] [--width 1920] [--height 1080]
#
# Prerequisites:
#   - libgstedgeimpulse.so in GST_PLUGIN_PATH
#   - Edge Impulse model deployed via FFI (or specify --model-path)
#   - qtiqmmfsrc available (Qualcomm IM SDK installed)
#   - gst-launch-1.0 available
#
# Output: CSV file with per-run metrics and a summary comparison.

set -euo pipefail

# Defaults
DURATION=30
WIDTH=1920
HEIGHT=1080
MODEL_PATH=""
NUM_FRAMES=""
OUTPUT_DIR="/tmp/ei-benchmark-$(date +%Y%m%d-%H%M%S)"
CAMERA_ID=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --duration SEC     Duration of each test run (default: $DURATION)
    --width W          Capture width (default: $WIDTH)
    --height H         Capture height (default: $HEIGHT)
    --model-path PATH  Path to .eim model (optional, uses FFI mode if omitted)
    --num-frames N     Stop after N frames instead of duration
    --camera-id ID     Camera ID for qtiqmmfsrc (default: $CAMERA_ID)
    --output-dir DIR   Output directory for results (default: auto-generated in /tmp)
    -h, --help         Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --num-frames) NUM_FRAMES="$2"; shift 2 ;;
        --camera-id) CAMERA_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Build the model property string
MODEL_PROP=""
if [[ -n "$MODEL_PATH" ]]; then
    MODEL_PROP="model-path=$MODEL_PATH"
fi

# Determine how to limit the run
if [[ -n "$NUM_FRAMES" ]]; then
    LIMIT="num-buffers=$NUM_FRAMES"
    echo "Running each test for $NUM_FRAMES frames"
else
    LIMIT=""
    echo "Running each test for $DURATION seconds"
fi

# Function to run a pipeline and extract metrics
run_pipeline() {
    local name="$1"
    local pipeline="$2"
    local logfile="$OUTPUT_DIR/${name}.log"
    local cpufile="$OUTPUT_DIR/${name}_cpu.log"

    echo ""
    echo "===== Running: $name ====="
    echo "Pipeline: $pipeline"
    echo ""

    # Start CPU monitoring in background
    (
        while true; do
            top -bn1 | head -3 | tail -1 >> "$cpufile" 2>/dev/null || true
            sleep 1
        done
    ) &
    local cpu_pid=$!

    # Run the pipeline
    local start_time
    start_time=$(date +%s%N)

    if [[ -n "$NUM_FRAMES" ]]; then
        GST_DEBUG="edgeimpulsevideoinfer:4,fpsdisplaysink:4" \
            timeout $((DURATION + 30)) \
            gst-launch-1.0 -e $pipeline 2>&1 | tee "$logfile" || true
    else
        GST_DEBUG="edgeimpulsevideoinfer:4,fpsdisplaysink:4" \
            timeout "$DURATION" \
            gst-launch-1.0 -e $pipeline 2>&1 | tee "$logfile" || true
    fi

    local end_time
    end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    # Stop CPU monitoring
    kill "$cpu_pid" 2>/dev/null || true
    wait "$cpu_pid" 2>/dev/null || true

    # Extract metrics from log
    local fps
    fps=$(grep -oE 'fps:[[:space:]]*[0-9.]+' "$logfile" | tail -1 | sed 's/fps:[[:space:]]*//' || echo "N/A")

    local inference_times
    inference_times=$(grep -oE 'timing_ms=[0-9]+' "$logfile" | sed 's/timing_ms=//' || true)
    local avg_inference="N/A"
    if [[ -n "$inference_times" ]]; then
        avg_inference=$(echo "$inference_times" | awk '{ total += $1; count++ } END { if (count > 0) printf "%.1f", total/count; else print "N/A" }')
    fi

    local frame_count
    frame_count=$(grep -c "Transform" "$logfile" 2>/dev/null || echo "0")

    # Extract average CPU from top output
    local avg_cpu="N/A"
    if [[ -f "$cpufile" ]]; then
        avg_cpu=$(grep -oE '[0-9.]+[[:space:]]*id' "$cpufile" | awk '{ total += (100 - $1); count++ } END { if (count > 0) printf "%.1f", total/count; else print "N/A" }' || echo "N/A")
    fi

    echo ""
    echo "--- Results: $name ---"
    echo "  Duration:        ${elapsed_ms}ms"
    echo "  Frames:          $frame_count"
    echo "  FPS:             $fps"
    echo "  Avg inference:   ${avg_inference}ms"
    echo "  Avg CPU usage:   ${avg_cpu}%"
    echo ""

    # Write CSV row
    echo "$name,$elapsed_ms,$frame_count,$fps,$avg_inference,$avg_cpu" >> "$OUTPUT_DIR/results.csv"
}

# CSV header
echo "test,duration_ms,frames,fps,avg_inference_ms,avg_cpu_pct" > "$OUTPUT_DIR/results.csv"

# ============================================================
# Test 1: System Memory Pipeline (baseline)
# ============================================================

PIPELINE_SYSMEM="qtiqmmfsrc camera=$CAMERA_ID $LIMIT ! \
    video/x-raw\(memory:GBM\),format=NV12,width=$WIDTH,height=$HEIGHT,framerate=30/1 ! \
    queue max-size-buffers=2 leaky=downstream ! \
    videoconvert ! \
    video/x-raw,format=RGB ! \
    queue max-size-buffers=2 leaky=downstream ! \
    edgeimpulsevideoinfer $MODEL_PROP ! \
    fpsdisplaysink video-sink=fakesink text-overlay=false signal-fps-measurements=true sync=false"

run_pipeline "sysmem_rgb" "$PIPELINE_SYSMEM"

# ============================================================
# Test 2: GBM Zero-Copy Pipeline
# ============================================================

PIPELINE_GBM="qtiqmmfsrc camera=$CAMERA_ID $LIMIT ! \
    video/x-raw\(memory:GBM\),format=NV12,width=$WIDTH,height=$HEIGHT,framerate=30/1 ! \
    queue max-size-buffers=2 leaky=downstream ! \
    edgeimpulsevideoinfer $MODEL_PROP ! \
    fpsdisplaysink video-sink=fakesink text-overlay=false signal-fps-measurements=true sync=false"

run_pipeline "gbm_zerocopy" "$PIPELINE_GBM"

# ============================================================
# Test 3: GBM Zero-Copy with Overlay + Display
# ============================================================

PIPELINE_GBM_DISPLAY="qtiqmmfsrc camera=$CAMERA_ID $LIMIT ! \
    video/x-raw\(memory:GBM\),format=NV12,width=$WIDTH,height=$HEIGHT,framerate=30/1 ! \
    queue max-size-buffers=2 leaky=downstream ! \
    edgeimpulsevideoinfer $MODEL_PROP ! \
    queue max-size-buffers=2 leaky=downstream ! \
    edgeimpulseoverlay ! \
    queue max-size-buffers=2 leaky=downstream ! \
    waylandsink sync=false"

run_pipeline "gbm_display" "$PIPELINE_GBM_DISPLAY"

# ============================================================
# Test 4: System Memory with Overlay + Display (baseline)
# ============================================================

PIPELINE_SYSMEM_DISPLAY="qtiqmmfsrc camera=$CAMERA_ID $LIMIT ! \
    video/x-raw\(memory:GBM\),format=NV12,width=$WIDTH,height=$HEIGHT,framerate=30/1 ! \
    queue max-size-buffers=2 leaky=downstream ! \
    videoconvert ! \
    video/x-raw,format=RGB ! \
    queue max-size-buffers=2 leaky=downstream ! \
    edgeimpulsevideoinfer $MODEL_PROP ! \
    queue max-size-buffers=2 leaky=downstream ! \
    edgeimpulseoverlay ! \
    queue max-size-buffers=2 leaky=downstream ! \
    videoconvert ! \
    waylandsink sync=false"

run_pipeline "sysmem_display" "$PIPELINE_SYSMEM_DISPLAY"

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================="
echo "  BENCHMARK SUMMARY"
echo "========================================="
echo ""
column -t -s',' "$OUTPUT_DIR/results.csv"
echo ""
echo "Full results saved to: $OUTPUT_DIR/results.csv"
echo "Logs saved to: $OUTPUT_DIR/*.log"
