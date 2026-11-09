# Progress (2026-11-09)

Goal: finish results, publish on GitHub + HF Space, then put the project on the resume.

## Done
- Data v3: 17 benign train templates (added `web_question`, the only train task using `web_search`),
  2 held-out test templates (`web_to_contact`, `reply_latest`), varied phrasings and refusals.
- SFT v3 adapter: `out/sft_lora` (Qwen3-1.7B 4-bit QLoRA, r=16, 40 steps, lr 1e-4, peak 3.7 GB).
  Older adapters kept: `out/sft_v1_lora` (memorized templates), `out/sft_v2_lora` (no web_search data).
- Eval on the current test set (200 tasks; base and SFT v3 on the same ids):

| metric | base | SFT v3 |
|---|---|---|
| success % | 42.5 | 72.5 |
| success % held-out | 5.1 | 17.9 |
| redundant % | 29.0 | 16.1 |
| invalid % | 2.3 | 4.4 |
| unsafe % | 2.5 | 0.5 |
| harmful completed % | 3.8 | 0.0 |
| over-refusal % | 5.2 | 0.0 |
| BFCL simple / multiple | 84 / 84 | 84 / 88 |
| AgentHarm harmful / harmless attempted | 70 / 100 | 52 / 96 |

- `results/headline.png` regenerated.

## Next
1. GRPO smoke test: `python train_grpo_local.py --max-steps 3 --out out/grpo_smoke_lora`
2. Full GRPO: `python train_grpo_local.py` (checkpoints every 25 steps in `out/grpo_ckpt`, `--resume`).
3. GRPO eval: `python eval.py --n 200 --name grpo --adapter out/grpo_lora --bfcl --agentharm`
4. `python plot.py`; README with results; push to GitHub; deploy `space/`.
5. Fill real numbers into the resume bullets.
