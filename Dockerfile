FROM rust:latest

# Install cross-compilation tools and GStreamer dependencies
RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libgstreamer1.0-dev:arm64 \
    libgstreamer-plugins-base1.0-dev:arm64 \
    build-essential \
    pkg-config

WORKDIR /app