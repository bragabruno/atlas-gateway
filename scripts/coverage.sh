#!/usr/bin/env bash
# coverage.sh — coverage gate (recommended). pytest-cov is optional; when it is
# not installed this is a documented no-op. ATLAS_COV_MIN sets the fail-under %.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

cov_min="${ATLAS_COV_MIN:-0}"
require_cmd python
if ! python -c "import pytest_cov" >/dev/null 2>&1; then
  skip "coverage" "pytest-cov not installed (recommended gate: add to [dev], set ATLAS_COV_MIN)"
  exit 0
fi
run "pytest --cov (fail-under ${cov_min}%)" \
  pytest -q --cov=app --cov-report=term-missing "--cov-fail-under=${cov_min}"
log_ok "coverage gate passed"
