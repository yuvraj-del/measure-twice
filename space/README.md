---
title: Measure Twice
emoji: 📏
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Measure Twice

Qwen3-1.7B fine-tuned (SFT, then GRPO) to make fewer, valid and safe tool calls.
Pick a test task; the same task runs on the base and the fine-tuned model side by side.
Each step is labelled: gold, duplicate, unnecessary, unsafe, invalid, final_ok, final_bad.

## Files

`app.py` imports the project's core modules, so this Space repo must also hold copies of them,
taken unchanged from the main repo:

- `sandbox/` (`tools.py`, `tasks.py`, `__init__.py` if present)
- `reward.py`
- `rollout.py`

Re-copy them whenever they change in the main repo, or the demo labels will drift from the eval.

## Configuration (Space variables)

| var | default | meaning |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen3-1.7B` | left column |
| `TUNED_MODEL` | `BASE_MODEL` | right column base weights |
| `TUNED_ADAPTER` | unset | LoRA adapter (HF id or path) applied to `TUNED_MODEL` |
| `DEMO_ORACLE` | unset | `1` = no GPU: left is a scripted sloppy baseline (repeats a call, skips the check before destructive actions), right is the oracle |

Running two 1.7B models needs a GPU Space; on CPU hardware use `DEMO_ORACLE=1`.
