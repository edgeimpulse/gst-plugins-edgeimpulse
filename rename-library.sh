#!/bin/bash
# Rename the output library to include PLUGIN_VARIANT suffix
# This allows multiple plugin variants to coexist in the same GStreamer installation
#
# Usage:
#   PLUGIN_VARIANT=variantX ./rename-library.sh
#   Or after building: cargo build --release && PLUGIN_VARIANT=variantX ./rename-library.sh

set -e

if [ -z "$PLUGIN_VARIANT" ]; then
    echo "Error: PLUGIN_VARIANT environment variable is not set"
    echo "Usage: PLUGIN_VARIANT=variantX ./rename-library.sh"
    exit 1
fi

# Determine the library extension based on the platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_EXT="so"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    LIB_EXT="dll"
else
    LIB_EXT="so" # Default fallback
fi

# Try both debug and release directories
for PROFILE in debug release; do
    BUILD_DIR="target/$PROFILE"
    ORIGINAL_LIB="libgstedgeimpulse.${LIB_EXT}"
    NEW_LIB="libgstedgeimpulse_${PLUGIN_VARIANT}.${LIB_EXT}"
    ORIGINAL_D="libgstedgeimpulse.d"
    NEW_D="libgstedgeimpulse_${PLUGIN_VARIANT}.d"

    if [ -f "$BUILD_DIR/$ORIGINAL_LIB" ]; then
        echo "Renaming $ORIGINAL_LIB to $NEW_LIB in $BUILD_DIR"
        mv "$BUILD_DIR/$ORIGINAL_LIB" "$BUILD_DIR/$NEW_LIB"

        if [ -f "$BUILD_DIR/$ORIGINAL_D" ]; then
            mv "$BUILD_DIR/$ORIGINAL_D" "$BUILD_DIR/$NEW_D"
        fi
        echo "Successfully renamed library to $NEW_LIB"
        exit 0
    fi
done

echo "Error: Library libgstedgeimpulse.${LIB_EXT} not found in target/debug or target/release"
echo "Please run 'cargo build' or 'cargo build --release' first"
exit 1

