#!/usr/bin/env bash
# afterFileEdit: rustfmt edited .rs files (fail open).
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  exit 0
fi

if ! command -v rustfmt >/dev/null 2>&1; then
  exit 0
fi

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.file_path // empty')"

if [[ -z "$file_path" || "$file_path" != *.rs ]]; then
  exit 0
fi

if [[ ! -f "$file_path" ]]; then
  exit 0
fi

# Fail open: formatting errors must not block the agent.
rustfmt --edition 2021 "$file_path" || true
exit 0