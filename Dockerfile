FROM rust:latest

# Add ARM64 architecture and update package lists
RUN dpkg --add-architecture arm64 && \
    apt-get update

# Install cross-compilation tools and dependencies
RUN apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libssl-dev:arm64 \
    libgstreamer1.0-dev:arm64 \
    libgstreamer-plugins-base1.0-dev:arm64 \
    libdrm-dev:arm64 \
    libpango1.0-dev:arm64 \
    libcairo2-dev:arm64 \
    libpangocairo-1.0-0:arm64 \
    pkg-config

# Set up pkg-config for cross-compilation
ENV PKG_CONFIG_ALLOW_CROSS=1
ENV PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR=/

# Set up OpenSSL for cross-compilation
ENV OPENSSL_DIR=/usr/include/aarch64-linux-gnu
ENV OPENSSL_LIB_DIR=/usr/lib/aarch64-linux-gnu
ENV OPENSSL_INCLUDE_DIR=/usr/include/aarch64-linux-gnu

WORKDIR /app

# Set up SSH for private repository access
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]