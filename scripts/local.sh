#!/usr/bin/env bash
# local.sh — run the gateway locally with the dev (Mock) provider. PORT overrides
# the listen port (default 8000).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

require_cmd uvicorn "pip install -e .[dev]"
port="${PORT:-8000}"
log_info "atlas-gateway → http://127.0.0.1:${port} (MockProvider; Ctrl-C to stop)"
exec uvicorn app.main:app --reload --port "$port"
