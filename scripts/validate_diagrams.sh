#!/usr/bin/env bash
#
# validate_diagrams.sh — Mermaid + PlantUML diagram validation (XCUT-6).
#
# Validates that every diagram in the given roots renders/parses without error:
#   * Mermaid  — fenced ```mermaid blocks inside .md files, plus standalone .mmd
#                files, validated with mermaid-cli (mmdc).
#   * PlantUML — .puml files, syntax-checked with the PlantUML jar (-checkonly).
#
# A single broken diagram fails CI (non-zero exit). Pinned tool versions
# (see Tooling below) keep the check reproducible per the supply-chain policy.
#
# Usage
# -----
#   scripts/validate_diagrams.sh [ROOT ...]
#
#   ROOT   One or more directories to scan recursively. Defaults to "diagrams"
#          and "docs/diagrams" if they exist (covers atlas-docs and per-repo
#          docs/diagrams/ layouts).
#
# Tooling (pinned — see atlas-docs/02 §2 supply-chain policy)
# ----------------------------------------------------------
#   MERMAID_CLI_VERSION   @mermaid-js/mermaid-cli  (npx, run via MMDC)
#   PLANTUML_VERSION      PlantUML jar version (PLANTUML_JAR or PATH `plantuml`)
#
# Environment overrides
# ---------------------
#   MMDC          mermaid-cli invocation (default: npx -y @mermaid-js/mermaid-cli@<ver>)
#   PLANTUML_JAR  path to plantuml.jar (otherwise a `plantuml` binary on PATH is used)
#
set -euo pipefail

MERMAID_CLI_VERSION="11.14.0"
PLANTUML_VERSION="1.2026.3"

MMDC_DEFAULT="npx -y @mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}"
MMDC="${MMDC:-$MMDC_DEFAULT}"

PUPPETEER_CONFIG="${PUPPETEER_CONFIG:-}"

fail_count=0
checked_count=0

log()  { printf '%s\n' "$*"; }
err()  { printf 'ERROR: %s\n' "$*" >&2; }

# Resolve scan roots: explicit args, else conventional defaults that exist.
roots=()
if [ "$#" -gt 0 ]; then
  roots=("$@")
else
  for candidate in diagrams docs/diagrams; do
    [ -d "$candidate" ] && roots+=("$candidate")
  done
fi

if [ "${#roots[@]}" -eq 0 ]; then
  log "validate_diagrams.sh: no diagram roots found (looked for diagrams/, docs/diagrams/); nothing to validate."
  exit 0
fi

log "validate_diagrams.sh: scanning roots: ${roots[*]}"

# --- Mermaid: a temp puppeteer config keeps mmdc headless/sandbox-safe in CI ---
mmdc_args=()
if [ -n "$PUPPETEER_CONFIG" ]; then
  mmdc_args+=(--puppeteerConfigFile "$PUPPETEER_CONFIG")
fi

validate_mermaid() {
  local file="$1"
  checked_count=$((checked_count + 1))
  # mmdc parses .md (extracting ```mermaid blocks) and .mmd directly; render to
  # an SVG in a throwaway dir. A parse error makes mmdc exit non-zero.
  local out
  out="$(mktemp -d)"
  if $MMDC "${mmdc_args[@]}" --input "$file" --output "$out/out.svg" >/dev/null 2>"$out/err.log"; then
    log "  OK  (mermaid) $file"
  else
    err "mermaid validation failed: $file"
    sed 's/^/      /' "$out/err.log" >&2 || true
    fail_count=$((fail_count + 1))
  fi
  rm -rf "$out"
}

validate_plantuml() {
  local file="$1"
  checked_count=$((checked_count + 1))
  local rc=0
  if [ -n "${PLANTUML_JAR:-}" ]; then
    java -jar "$PLANTUML_JAR" -checkonly -failfast2 "$file" || rc=$?
  elif command -v plantuml >/dev/null 2>&1; then
    plantuml -checkonly -failfast2 "$file" || rc=$?
  else
    err "PlantUML not available (set PLANTUML_JAR or install 'plantuml' v${PLANTUML_VERSION}); cannot validate $file"
    fail_count=$((fail_count + 1))
    return
  fi
  if [ "$rc" -eq 0 ]; then
    log "  OK  (plantuml) $file"
  else
    err "plantuml validation failed: $file"
    fail_count=$((fail_count + 1))
  fi
}

for root in "${roots[@]}"; do
  if [ ! -d "$root" ]; then
    err "root not found: $root"
    fail_count=$((fail_count + 1))
    continue
  fi

  # Mermaid in .md (only files that actually contain a mermaid fence) and .mmd.
  while IFS= read -r md; do
    if grep -q '```mermaid' "$md"; then
      validate_mermaid "$md"
    fi
  done < <(find "$root" -type f -name '*.md')

  while IFS= read -r mmd; do
    validate_mermaid "$mmd"
  done < <(find "$root" -type f -name '*.mmd')

  # PlantUML .puml files.
  while IFS= read -r puml; do
    validate_plantuml "$puml"
  done < <(find "$root" -type f -name '*.puml')
done

log ""
log "validate_diagrams.sh: checked ${checked_count} diagram source file(s), ${fail_count} failure(s)."

if [ "$fail_count" -gt 0 ]; then
  exit 1
fi
exit 0
