#!/usr/bin/env bash
set -euo pipefail

# Binary name (defaults to your crate/bin name)
BIN_NAME="${BIN_NAME:-vulkan-midi-visualizer}"
IMAGE_TAG="${IMAGE_TAG:-vmv-musl-build}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

# Build the image; Docker will set TARGETARCH automatically.
echo "==> Building static MUSL binary in Docker (bin: ${BIN_NAME})"
docker build \
  --build-arg BIN_NAME="${BIN_NAME}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" .

# Create a throwaway container and copy artifacts
CID="$(docker create "${IMAGE_TAG}")"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

# Read the target triple recorded during build (for nice output paths)
TARGET_TRIPLE="$(docker cp "${CID}":/app/.build_target - | cat)"
TARGET_TRIPLE="${TARGET_TRIPLE:-unknown-musl}"

mkdir -p "dist/${TARGET_TRIPLE}"
docker cp "${CID}":/out/"${BIN_NAME}" "dist/${TARGET_TRIPLE}/"

echo "==> Artifact written to: dist/${TARGET_TRIPLE}/${BIN_NAME}"
echo "    (static MUSL; copy it to your target machine and run)"
