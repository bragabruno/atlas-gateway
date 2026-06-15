#!/usr/bin/env bash
# docker.sh — container build verification.
# DISCOVERED GAP: this service ships a deploy/ Helm chart but has no Dockerfile,
# so nothing builds the image the chart references. Add a Dockerfile to enable.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

image="${ATLAS_IMAGE:-atlas-gateway:dev}"
if [[ ! -f Dockerfile ]]; then
  skip "docker build" "no Dockerfile (GAP: deploy/ chart references an unbuilt image)"
  exit 0
fi
require_cmd docker "Docker Desktop / a running daemon"
if ! docker info >/dev/null 2>&1; then
  skip "docker build" "docker daemon not running"
  exit 0
fi
run "docker build ${image}" docker build -t "$image" .
log_ok "image built: ${image}"
