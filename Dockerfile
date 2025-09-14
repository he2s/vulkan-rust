# syntax=docker/dockerfile:1

# --- Builder stage: compiles a static MUSL binary for host arch ---
FROM rust:1.89-alpine AS builder

# Build dependencies for shaderc (cmake, ninja, python3), C toolchain, MUSL, etc.
RUN apk add --no-cache \
    build-base musl-dev cmake ninja python3 git pkgconfig alsa-lib-dev

# Workdir and source
WORKDIR /app
COPY . .

# Detect Docker's target arch and map to a MUSL triple
ARG TARGETARCH
RUN set -eux; \
  case "${TARGETARCH:-amd64}" in \
    amd64)  export TARGET_TRIPLE=x86_64-unknown-linux-musl ;; \
    arm64)  export TARGET_TRIPLE=aarch64-unknown-linux-musl ;; \
    *) echo "Unsupported TARGETARCH: ${TARGETARCH}"; exit 1 ;; \
  esac; \
  echo "${TARGET_TRIPLE}" > /app/.build_target; \
  rustup target add "${TARGET_TRIPLE}"; \
  # Speed up incremental compiles in CI by priming deps (optional)
  cargo fetch

# Build the binary in release for the selected MUSL target.
# BIN_NAME can be overridden at build time; defaults to crate name.
ARG BIN_NAME=vulkan-midi-visualizer
ARG RUSTFLAGS
ENV RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+crt-static"
RUN set -eux; \
  TARGET_TRIPLE="$(cat /app/.build_target)"; \
  cargo build --release --target "${TARGET_TRIPLE}"; \
  # Stash paths for export convenience
  mkdir -p /out; \
  cp "target/${TARGET_TRIPLE}/release/${BIN_NAME}" "/out/${BIN_NAME}"

# --- (Optional) minimal runtime stage if you want to run inside a container ---
FROM scratch AS runtime
# (Not used by the script; here for completeness)
COPY --from=builder /out/ /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/vulkan-midi-visualizer"]
