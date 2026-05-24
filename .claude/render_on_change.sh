#!/usr/bin/env bash
# Hook: re-render RESEARCH.html after edits to RESEARCH.md

file_path=$(jq -r '.tool_input.file_path // ""')

echo "$(date) hook fired for: $file_path" >> /tmp/hook-test.txt

echo "$file_path" | grep -q 'RESEARCH\.md$' || exit 0

cd /home/falkodem/Documents/Projects/BassEmulatorVST
source venv/bin/activate
python3 scripts/render_docs.py
