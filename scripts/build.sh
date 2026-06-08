#!/usr/bin/env bash
# build.sh — build verification: the package imports, and the OpenAPI contract
# (cross-repo source of truth, ADR-014) is regenerated and not stale. Publishes
# nothing.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

require_cmd python "pip install -e .[dev]"
run "import smoke (app.main)" python -c "import app.main"

if [[ -f scripts/export_openapi.py ]]; then
  run "export OpenAPI spec" python scripts/export_openapi.py
  if has_cmd git && [[ -f openapi.json ]] && ! git diff --quiet -- openapi.json; then
    log_error "openapi.json is stale — re-run scripts/export_openapi.py and commit (ADR-014 contract drift)"
    exit 1
  fi
fi
log_ok "build verification passed"
