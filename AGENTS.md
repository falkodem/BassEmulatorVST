# BassEmulatorVST Agent Guide

## Project Context

BassEmulatorVST is a JUCE VST3 plugin for real-time monophonic guitar-to-bass conversion.
The target workflow is recording bass parts from a guitar in Reaper while avoiding inaccurate
intonation at note attacks.

Current focus: Phase 2A, integrating PESTO pitch detection into the plugin through ANIRA and
ONNX Runtime. Phase 1 DSP remains a baseline: onset detection, YIN, envelope following,
sawtooth oscillator, LadderFilter, and APVTS parameters.

Guitar is detected in the E2-E4 range. Bass synthesis uses `F0(guitar) / 2`, producing E1-E3.

## Read First

- `CLAUDE.md` is the detailed project memory and remains authoritative for architecture,
  build commands, ML commands, and current status.
- `ROADMAP.md` owns phase planning and backlog.
- `RESEARCH.md` owns research notes; it renders to `RESEARCH.html`.
- `REVIEW.md` is a historical review of `RESEARCH.md`; some findings may already be applied.

## C++ / JUCE Rules

- DSP runs in `src/PluginProcessor.*`; GUI runs in `src/PluginEditor.*`.
- Keep small DSP helpers header-only in `src/*.h` unless there is a concrete reason otherwise.
- Do not allocate, lock, open files, throw exceptions, or do string-heavy work in `processBlock`.
- Allocate buffers and model state in `prepareToPlay`.
- Read APVTS parameters in audio code through `getRawParameterValue(...)->load()`.
- When adding or changing plugin parameters, update `CLAUDE.md`.
- After C++ changes, run:

```bash
cmake --build build --config Release
```

Do not deploy the `.vst3` into the Reaper plugin directory unless explicitly asked.

## Python / ML Rules

The project-local Poetry executable lives inside `venv`. Activate the venv before Poetry commands:

```bash
source venv/bin/activate
```

Common commands:

```bash
poetry run python ml/import_dataset.py
poetry run python ml/slice_dataset.py
poetry run python ml/train.py
poetry run python ml/process_audio.py --run runs/v0/YYYYMMDD_HHMMSS --input data/v0/guitar/
```

Network can be unreliable. Do not install packages automatically unless explicitly asked. Prefer
giving the exact install command to the user.

Do not run long training jobs unless explicitly requested. Use short smoke checks when needed.

## Documentation Rules

- If `RESEARCH.md` changes, regenerate `RESEARCH.html` with:

```bash
source venv/bin/activate
python3 scripts/render_docs.py
```

- Keep `RESEARCH.md` readable as plain Markdown. Do not replace Markdown tables/lists with HTML
  just for styling.
- Use `research-editor` conventions from `.claude/agents/research-editor.md` when editing
  `RESEARCH.md`.
- Use `pm` conventions from `.claude/agents/pm.md` when editing `ROADMAP.md`.

## Role Boundaries

These role files are inherited project guidance:

- `.claude/agents/developer.md` for code changes.
- `.claude/agents/pm.md` for roadmap planning.
- `.claude/agents/researcher.md` for research drafts.
- `.claude/agents/research-editor.md` for `RESEARCH.md` edits.

Codex subagents are not the same mechanism as Claude Code agents. When delegating work, give the
subagent the relevant role file to read and a narrow, self-contained task.

## Git / Worktree

The worktree may contain user changes. Do not revert or overwrite unrelated changes. Do not commit
unless explicitly asked.
