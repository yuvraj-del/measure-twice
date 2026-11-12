---
task_categories: [text-generation]
tags: [tool-use, agents, function-calling, grpo, ai-safety, evaluation]
pretty_name: Measure Twice data, eval outputs and logs
configs:
- config_name: sft_benign
  data_files: train/sft_benign.jsonl
- config_name: grpo_prompts_benign
  data_files: train/grpo_prompts_benign.jsonl
- config_name: eval_outputs
  data_files:
  - split: base
    path: eval/base.jsonl
  - split: sft_no_refusals
    path: eval/sft_norefuse.jsonl
  - split: sft
    path: eval/sft.jsonl
  - split: sft_grpo
    path: eval/grpo.jsonl
  - split: sft_grpo_v2
    path: eval/grpo2.jsonl
---

# Measure Twice: data, eval outputs and logs

Everything behind [Measure Twice](https://github.com/yuvraj-del/measure-twice), which fine-tunes Qwen3-1.7B to
make fewer, valid and safe tool calls. Adapters: https://huggingface.co/yuvraj-del/measure-twice-qwen3-1.7b

## Contents

| path | what it is |
|---|---|
| `train/sft_benign.jsonl` | 2,528 SFT episodes (chat `messages`) on the optimal tool-call trajectory, from the seeded sandbox generator |
| `train/grpo_prompts_benign.jsonl` | 6,615 GRPO step prompts (`prompt`, `task_id`, `history`) |
| `train/test_ids.txt`, `train/val_ids.txt` | task ids of the test and validation splits |
| `eval/<model>.jsonl` | every recorded test episode on the 174 benign test tasks: each step's completion, its step label (`gold`, `duplicate`, `unnecessary`, `unsafe`, `invalid`, `final_ok`, `final_bad`) and the tool result |
| `eval/summary.csv`, `eval/bench.csv` | aggregate metrics (sandbox, BFCL, AgentHarm) per model |
| `eval/headline.png` | the results chart |
| `logs/` | training logs (loss, reward, per-step GRPO metrics) and eval logs |

## What is not included
Harmful-request items are removed: the 472 SFT refusal episodes, 472 GRPO harmful prompts and the 26 harmful
test episodes per model all contain prompts from [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm),
which is gated. To rebuild them, accept the AgentHarm terms and run `python gen_sft.py` from the repo.
`stage_hf.py` builds this dataset and asserts that no AgentHarm prompt text is present.
