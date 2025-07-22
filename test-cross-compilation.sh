#!/bin/bash

# Simple test script to verify cross compilation setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "=== Testing Cross Compilation Setup ==="

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed or not in PATH"
    exit 1
fi

print_success "Docker and docker-compose are available"

# Check if environment variables are set
if [ -z "$EI_PROJECT_ID" ] || [ -z "$EI_API_KEY" ]; then
    print_warning "EI_PROJECT_ID and/or EI_API_KEY not set"
    print_warning "This will prevent FFI mode from working, but EIM mode can still be tested"
    print_warning "Set these variables to test FFI mode:"
    print_warning "  export EI_PROJECT_ID='your_project_id'"
    print_warning "  export EI_API_KEY='your_api_key'"
fi

# Create test assets
print_status "Creating test assets..."
mkdir -p examples/assets

# Create a test image if it doesn't exist
if [ ! -f "examples/assets/test_image.png" ]; then
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" | base64 -d > examples/assets/test_image.png
    print_success "Created test image"
fi

# Create a test audio file if it doesn't exist
if [ ! -f "examples/assets/test_audio.wav" ]; then
    # Create a minimal WAV file
    echo "RIFF" > examples/assets/test_audio.wav
    echo -n -e "\x24\x00\x00\x00" >> examples/assets/test_audio.wav
    echo "WAVE" >> examples/assets/test_audio.wav
    print_success "Created test audio file"
fi

# Test Docker build
print_status "Testing Docker build..."
if docker-compose build; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Test cross compilation build
print_status "Testing cross compilation build..."
if docker-compose run --rm aarch64-build bash -c "
    cargo build --target aarch64-unknown-linux-gnu --features ffi --release
"; then
    print_success "Cross compilation build successful"
else
    print_error "Cross compilation build failed"
    exit 1
fi

# Check if output files exist
print_status "Checking output files..."
if docker-compose run --rm aarch64-build bash -c "test -f './target/aarch64-unknown-linux-gnu/release/libgstedgeimpulse.so'"; then
    print_success "Plugin binary created successfully"
    docker-compose run --rm aarch64-build bash -c "ls -la target/aarch64-unknown-linux-gnu/release/libgstedgeimpulse.so"
else
    print_error "Plugin binary not found"
    exit 1
fi

# Test examples build
print_status "Testing examples build..."
if docker-compose run --rm aarch64-build bash -c "
    cargo build --target aarch64-unknown-linux-gnu --features ffi --release --examples
"; then
    print_success "Examples built successfully"
else
    print_error "Examples build failed"
    exit 1
fi

# Check if examples exist
print_status "Checking examples..."
if docker-compose run --rm aarch64-build bash -c "test -d './target/aarch64-unknown-linux-gnu/release/examples'"; then
    print_success "Examples directory created"
    docker-compose run --rm aarch64-build bash -c "ls -la target/aarch64-unknown-linux-gnu/release/examples/"
else
    print_error "Examples directory not found"
    exit 1
fi

print_success "=== Cross Compilation Setup Test Completed Successfully ==="
print_status "You can now use ./test-aarch64-examples.sh to test the examples"
print_status "Or run docker-compose run --rm aarch64-build for manual testing"