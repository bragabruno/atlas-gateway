#!/usr/bin/env bash
# security.sh — security scans. Each scanner is optional and ADVISORY by default
# (missing tools skip; findings warn). Set ATLAS_SECURITY_STRICT=1 to make
# findings fail the gate.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

strict="${ATLAS_SECURITY_STRICT:-0}"
failures=0

# soft <label> <cmd...> — run a scanner; under non-strict a failure only warns.
soft() {
  local label="$1"; shift
  if run "$label" "$@"; then
    return 0
  fi
  if [[ "$strict" == "1" ]]; then
    failures=$((failures + 1))
  else
    log_warn "${label}: findings (advisory; set ATLAS_SECURITY_STRICT=1 to enforce)"
  fi
}

if has_cmd trunk; then
  soft "secret scan (trunk → gitleaks)" trunk check --all --no-progress --filter=gitleaks
else
  skip "secret scan" "trunk not installed (recommended: gitleaks via Trunk)"
fi

if has_cmd pip-audit; then
  soft "pip-audit (dependency CVEs)" pip-audit --progress-spinner=off
else
  skip "pip-audit" "not installed (recommended: dependency CVE scan)"
fi

if has_cmd trivy; then
  soft "trivy fs (vulnerabilities)" trivy fs --quiet --scanners vuln .
else
  skip "trivy" "not installed (recommended: filesystem + image scan)"
fi

if [[ "$failures" -gt 0 ]]; then
  log_error "security: ${failures} scanner(s) reported findings (strict mode)"
  exit 1
fi
log_ok "security stage complete"
