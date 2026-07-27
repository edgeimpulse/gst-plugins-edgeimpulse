#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$0")"
DEST="$HERE/assets"          # used by the models-gated integration test (explicit paths)
VENDOR="$HERE/../models"     # embedded into the .so via include_bytes! (Step 3b)
mkdir -p "$DEST" "$VENDOR"
BASE="https://ocrs-models.s3-accelerate.amazonaws.com"

# Pinned SHA-256 of the exact model bytes vendored under ../models and embedded
# into the plugin binary. The download is verified against these so a corrupted
# or tampered file can never silently become part of the built .so.
SHA_DETECTION="f15cfb56bd02c4bf478a20343986504a1f01e1665c2b3a0ad66340f054b1b5ca"
SHA_RECOGNITION="e484866d4cce403175bd8d00b128feb08ab42e208de30e42cd9889d8f1735a6e"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify() {
  local file="$1" expected="$2" actual
  actual="$(sha256_of "$file")"
  if [ "$actual" != "$expected" ]; then
    echo "ERROR: checksum mismatch for $file" >&2
    echo "  expected $expected" >&2
    echo "  actual   $actual" >&2
    exit 1
  fi
}

curl -fsSL "$BASE/text-detection.rten"   -o "$DEST/text-detection.rten"
curl -fsSL "$BASE/text-recognition.rten" -o "$DEST/text-recognition.rten"
verify "$DEST/text-detection.rten"   "$SHA_DETECTION"
verify "$DEST/text-recognition.rten" "$SHA_RECOGNITION"
cp "$DEST/text-detection.rten" "$DEST/text-recognition.rten" "$VENDOR/"
echo "Downloaded and verified ocrs models in $DEST and vendored them into $VENDOR"
