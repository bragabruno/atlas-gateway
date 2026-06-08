#!/usr/bin/env bash
# common.sh — shared helpers for the Atlas build scripts (logging, timing,
# command checks, error trap). Source AFTER colors.sh:
#   source scripts/lib/colors.sh
#   source scripts/lib/common.sh
# Sourced, not executed. Targets bash 3.2+ (Linux + macOS).

# Inert color fallbacks so this file is self-consistent (colors.sh, sourced
# first, supplies the real values; these := are no-ops when already set).
: "${C_RESET:=}" "${C_RED:=}" "${C_GREEN:=}" "${C_YELLOW:=}" "${C_BLUE:=}" "${C_CYAN:=}" "${C_BOLD:=}" "${C_DIM:=}"

# Repo root = the dir that contains scripts/ (this file is scripts/lib/common.sh).
ATLAS_REPO_ROOT="${ATLAS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ATLAS_REPO_ROOT

# Prefer the project virtualenv's tools locally; in CI they're already on PATH
# from `pip install -e .[dev]`. No-op for non-Python repos (no .venv).
if [[ -x "${ATLAS_REPO_ROOT}/.venv/bin/python" ]]; then
  PATH="${ATLAS_REPO_ROOT}/.venv/bin:${PATH}"
  export PATH
fi

log_info()  { printf '%s\n' "${C_DIM}•${C_RESET} $*"; }
log_step()  { printf '%s\n' "${C_BLUE}${C_BOLD}▸ $*${C_RESET}"; }
log_ok()    { printf '%s\n' "${C_GREEN}✔${C_RESET} $*"; }
log_warn()  { printf '%s\n' "${C_YELLOW}⚠ $*${C_RESET}" >&2; }
log_error() { printf '%s\n' "${C_RED}✖ $*${C_RESET}" >&2; }

# skip <stage> <reason> — uniform no-op for stages N/A to this repo.
skip() { printf '%s\n' "${C_DIM}↷ skip ${1} — ${2}${C_RESET}"; }

# require_cmd <cmd> [hint] — fail fast (rc 127) when a required tool is missing.
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "required command not found: $1${2:+ — $2}"
    return 127
  fi
}

# has_cmd <cmd> — true when the command is present (optional/advisory stages).
has_cmd() { command -v "$1" >/dev/null 2>&1; }

# run <label> <cmd...> — execute with timing and clear pass/fail reporting.
run() {
  local label="$1"; shift
  local start end rc=0
  log_step "$label"
  start=$(date +%s)
  "$@" || rc=$?
  end=$(date +%s)
  if [[ "$rc" -eq 0 ]]; then
    log_ok "${label} ($((end - start))s)"
  else
    log_error "${label} failed (rc=${rc}, $((end - start))s)"
    return "$rc"
  fi
}

# on_err <line> <rc> — error-trap reporter for the entry scripts.
on_err() { log_error "aborted (line ${1:-?}, rc ${2:-?})"; }
