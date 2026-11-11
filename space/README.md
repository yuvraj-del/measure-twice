---
title: Measure Twice
emoji: 📏
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Fewer, valid and safe tool calls from a fine-tuned 1.7B agent
---

# Measure Twice

Qwen3-1.7B fine-tuned (SFT, then GRPO) to make fewer, valid and safe tool calls.
Pick a test task and compare two models step by step. Every step is labelled:
`gold` (needed), `duplicate` / `unnecessary` (wasted), `unsafe` (destructive action without its check),
`invalid` (bad tool or arguments), `final_ok` / `final_bad`.

This static page replays the real outputs recorded on the 174 benign test tasks during evaluation
(greedy decoding). Harmful-request tasks are not included.

To run the models live instead, use `space/app.py` from the repo as a Gradio Space on GPU hardware with `LIVE=1`.

Code, results and method: https://github.com/yuvraj-del/measure-twice
