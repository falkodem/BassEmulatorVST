#!/usr/bin/env bash
set -euo pipefail

# Codex hook: re-render RESEARCH.html after RESEARCH.md edits.
# The hook receives event JSON on stdin. If the event shape changes, fall back
# to a no-op rather than blocking the agent.

payload="$(cat || true)"

if command -v jq >/dev/null 2>&1; then
  file_path="$(
    printf '%s' "$payload" | jq -r '
      .tool_input.file_path //
      .tool_input.path //
      .input.file_path //
      .input.path //
      .file_path //
      ""
    ' 2>/dev/null || true
  )"
else
  file_path=""
fi

case "$file_path" in
  *RESEARCH.md) ;;
  *) exit 0 ;;
esac

cd /home/falkodem/Documents/Projects/BassEmulatorVST

if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

python3 scripts/render_docs.py
