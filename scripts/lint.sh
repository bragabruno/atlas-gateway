#!/usr/bin/env bash
# lint.sh — Gate 1 (static correctness): supply-chain age audit + Trunk(ruff
# lint+format) + pyright (strict). The single lint entrypoint for dev and CI.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

require_cmd python "install Python 3.12 / activate the venv"
run "dependency age audit (XCUT-4)" python scripts/dep_audit.py --min-age-days 14

require_cmd trunk "https://get.trunk.io"
# --ci in CI (machine output + caching); --no-progress for a clean local run.
trunk_flag="--no-progress"
[[ -n "${CI:-}" ]] && trunk_flag="--ci"
run "trunk check (ruff lint + format)" trunk check --all "$trunk_flag"

require_cmd pyright "pip install -e .[dev]"
run "pyright (strict)" pyright

log_ok "lint: all static checks passed"
