# Progress (2026-11-09)

Goal: finish results, publish on GitHub + HF Space, then put the project on the resume.

## Done
- Data v3: 17 benign train templates (added `web_question`, the only train task using `web_search`),
  2 held-out test templates (`web_to_contact`, `reply_latest`), varied phrasings and refusals.
- SFT v3 adapter: `out/sft_lora` (Qwen3-1.7B 4-bit QLoRA, r=16, 40 steps, lr 1e-4, peak 3.7 GB).
  Older adapters kept: `out/sft_v1_lora` (memorized templates), `out/sft_v2_lora` (no web_search data).
- GRPO adapter: `out/grpo_lora` (step 150 of a 300-step run; ~52 s/step, 3.7 GB; stopped early because
  reward plateaued at ~0.78 after step 50). Trained only on templates SFT still failed, plus harmful prompts.
- Eval on the current test set (200 tasks; all three models on the same ids):

| metric | base | SFT v3 | SFT+GRPO |
|---|---|---|---|
| success % | 42.5 | 72.5 | 77.5 |
| success % held-out | 5.1 | 17.9 | 35.9 |
| calls / task | 1.21 | 1.61 | 1.81 |
| redundant % | 29.0 | 16.1 | 20.2 |
| invalid % | 2.3 | 4.4 | 2.3 |
| unsafe % | 2.5 | 0.5 | 0.5 |
| harmful completed % | 3.8 | 0.0 | 0.0 |
| over-refusal % | 5.2 | 0.0 | 0.0 |
| BFCL simple / multiple | 84 / 84 | 84 / 88 | 84 / 88 |
| AgentHarm harmful / harmless attempted | 70 / 100 | 52 / 96 | 52 / 94 |

- `results/headline.png` regenerated with all three models.

- Ablation `out/sft_norefuse_lora` (SFT without refusal episodes): AgentHarm harmful 98.1% (base 71.7, SFT 54.7),
  held-out success 35.9%, invalid 10.5%. Bigger benchmarks in `results/bench.csv`; CIs via `ci.py`. README updated.

## Next
1. Done: README + results pushed to https://github.com/yuvraj-del/measure-twice (public).
2. Deploy `space/` (user's HF account).
3. Optional: resume GRPO from `out/grpo_ckpt/checkpoint-150` with a redundancy-focused reward if time allows.
4. Fill real numbers into the resume bullets.
