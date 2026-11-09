# Progress (2026-11-08)

Goal: finish results, publish on GitHub + HF Space, then put the project on the resume.

## Done
- SFT adapter trained locally: `out/sft_lora` (Qwen3-1.7B, LoRA r=16, 40 steps, peak 3.7 GB VRAM).
- Base eval complete (200 test tasks + BFCL + AgentHarm): `results/summary.csv`, `results/base.jsonl`.
  Success 44.5% (held-out templates 5.1%), redundant 32.1% (held-out 59.6%), unsafe 2.0%,
  BFCL simple/multiple 84/84, AgentHarm harmful attempt 70%.
- `train_grpo_local.py` written; CPU dry run passed (7231 prompts, reward check OK). Not yet run on GPU.
- Git repo initialized (local only, not pushed). Gated AgentHarm data and transcripts are gitignored.
- `dist/measure-twice.zip`: code + data + SFT adapter, ready for a private Kaggle dataset if needed.

## Next
1. SFT eval (was stopped at 16/200; eval.py has no resume, so rerun from scratch, ~2-3 h):
   `python eval.py --n 200 --name sft --adapter out/sft_lora --bfcl --agentharm > results/sft_log.txt 2>&1`
2. GRPO smoke test, check VRAM and s/step: `python train_grpo_local.py --max-steps 3 --out out/grpo_smoke_lora`
3. Full GRPO: `python train_grpo_local.py` (400 steps; checkpoints every 25 in `out/grpo_ckpt`, continue with `--resume`).
4. GRPO eval: `python eval.py --n 200 --name grpo --adapter out/grpo_lora --bfcl --agentharm`
5. `python plot.py` -> `results/headline.png`; README with the results table; push to GitHub; deploy `space/`.
6. Fill real numbers into the resume bullets.
