#!/usr/bin/env bash
# ci.sh — the single source of truth. Devs run this locally; CI runs the same
# stage scripts (parallelized per stage in bitbucket-pipelines.yml). Stages that
# are N/A or whose tools are absent skip cleanly, so the same command works on a
# laptop and in CI. See atlas-docs/07-build-system.md.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

log_step "atlas-gateway — full CI gate"
start=$(date +%s)
scripts/lint.sh
scripts/test.sh
scripts/coverage.sh
scripts/build.sh
scripts/infra.sh
scripts/docker.sh
scripts/security.sh
log_ok "CI passed ($(( $(date +%s) - start ))s)"
