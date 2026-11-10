---
title: Measure Twice
emoji: 📏
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
short_description: Fewer, valid and safe tool calls from a fine-tuned 1.7B agent
---

# Measure Twice

Qwen3-1.7B fine-tuned (SFT, then GRPO) to make fewer, valid and safe tool calls.
Pick a test task and compare two models step by step. Every step is labelled:
`gold` (needed), `duplicate` / `unnecessary` (wasted), `unsafe` (destructive action without its check),
`invalid` (bad tool or arguments), `final_ok` / `final_bad`.

**Replay mode (default):** shows the real outputs recorded on the 174 benign test tasks during evaluation
(greedy decoding), so it runs on free CPU hardware. Harmful-request tasks are not included.

**Live mode:** set the Space variable `LIVE=1` on GPU hardware, add `torch transformers peft accelerate` to
`requirements.txt`, and optionally set `BASE_MODEL`, `TUNED_MODEL`, `TUNED_ADAPTER` (a LoRA repo id).

Code, results and method: https://github.com/yuvraj-del/measure-twice
