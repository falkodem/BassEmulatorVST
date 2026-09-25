#!/usr/bin/env bash
set -euo pipefail

# Codex передаёт событие PostToolUse в JSON через stdin.
payload="$(cat)"
changed="$(printf '%s' "$payload" | python3 -c '
import json
import re
import sys

try:
    event = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    raise SystemExit(0)

tool_input = event.get("tool_input") or {}
path = str(tool_input.get("file_path") or tool_input.get("path") or "")
patch = str(tool_input.get("command") or "")
if path.replace("\\", "/").split("/")[-1] == "RESEARCH.md" or re.search(
    r"(?m)^\*\*\* (?:Add|Update) File: (?:.*/)?RESEARCH\.md\s*$", patch
):
    print("yes")
')"

[[ "$changed" == "yes" ]] || exit 0

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

if ! python3 -c 'import markdown, pygments' >/dev/null 2>&1; then
  echo "RESEARCH.html не обновлён: текущему Python нужны markdown и pygments (docs-зависимости в pyproject.toml)." >&2
  exit 1
fi

python3 scripts/render_docs.py
