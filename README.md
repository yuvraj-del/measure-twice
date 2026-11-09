# Measure Twice

Fine-tuning **Qwen3-1.7B** to be a better tool-using agent: finish the task, skip redundant calls,
check before destructive actions, and refuse harmful requests without over-refusing benign ones.
Trained with 4-bit QLoRA on a 6 GB laptop GPU.

![results](results/headline.png)

## Results

200 test tasks in the sandbox, same task ids for both models, greedy decoding, thinking disabled.

| metric | base | SFT |
|---|---:|---:|
| task success % | 42.5 | **72.5** |
| success % on held-out task types | 5.1 | **17.9** |
| redundant calls % | 29.0 | **16.1** |
| unsafe calls % (destructive action without its check) | 2.5 | **0.5** |
| harmful tasks completed % | 3.8 | **0.0** |
| over-refusal % (benign task refused) | 5.2 | **0.0** |
| invalid calls % | **2.3** | 4.4 |
| BFCL simple / multiple (exact match) | 84 / 84 | 84 / **88** |
| AgentHarm: harmful / benign prompts attempted % | 70 / 100 | **52** / 96 |

Honest notes: calls per task rise (1.2 to 1.6) because the base model often stops before finishing;
held-out generalization is the weakest result; invalid calls went up slightly. GRPO on top of SFT is in progress.

## How it works

- **Sandbox** (`sandbox/`): 12 mock stateful tools (files, email, calendar, contacts, banking, web search),
  pure Python, no external services. Destructive tools declare the check that must run first
  (e.g. `read_file` before `delete_file`).
- **Tasks** (`sandbox/tasks.py`): a seeded generator with 17 training templates and 2 templates
  (`web_to_contact`, `reply_latest`) that only appear in the test split. Harmful prompts come from
  [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm), with about 30% of them held out for test.
- **Reward** (`reward.py`): scores each assistant turn given the calls already made, with one of
  `gold, duplicate, unnecessary, unsafe, invalid, final_ok, final_bad, harmful_act, harmful_probe`.
  The weights in `R` are the tuning knob.
- **Rollout** (`rollout.py`): the multi-turn agent loop shared by data generation, eval and the demo.
- **Training**: SFT on oracle episodes (`train_local.py`, loss only on the assistant turn), then GRPO with the
  step reward (`train_grpo_local.py`, TRL). Kaggle notebooks (`train_sft.ipynb`, `train_grpo.ipynb`) do the same with Unsloth.
- **Eval** (`eval.py`): sandbox metrics split by in-distribution and held-out templates, plus BFCL and AgentHarm.
- **Demo** (`space/`): Gradio app that runs base and fine-tuned models side by side on a test task, step labels included.

## Reproduce

```bash
pip install torch transformers peft trl datasets bitsandbytes accelerate matplotlib
python reward.py                       # reward self-checks
python gen_sft.py                      # data/sft.jsonl, data/grpo.jsonl (AgentHarm is gated: accept its terms, set HF_TOKEN)
python train_local.py --load-4bit --episodes 3000 --lr 1e-4 --max-steps 40
python eval.py --n 200 --name base --bfcl --agentharm
python eval.py --n 200 --name sft --adapter out/sft_lora --bfcl --agentharm
python train_grpo_local.py
python plot.py
```

Generated data and eval transcripts are not committed, because they contain AgentHarm prompts.
