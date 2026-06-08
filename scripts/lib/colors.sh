#!/usr/bin/env bash
# colors.sh — ANSI color definitions for the Atlas build scripts.
# Sourced (not executed). Colors are disabled when stdout is not a TTY or when
# NO_COLOR is set (https://no-color.org). Consumed by lib/common.sh + siblings.
# shellcheck disable=SC2034  # these vars are used by sibling scripts that source this file.

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
else
  C_RESET=''
  C_RED=''
  C_GREEN=''
  C_YELLOW=''
  C_BLUE=''
  C_CYAN=''
  C_BOLD=''
  C_DIM=''
fi
