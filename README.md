# Measure Twice

Fine-tuning **Qwen3-1.7B** to be a better tool-using agent: finish the task, skip redundant calls,
check before destructive actions, and refuse harmful requests without over-refusing benign ones.
Trained with 4-bit QLoRA on a 6 GB laptop GPU (RTX 4050): SFT takes about 20 minutes, GRPO about 2.3 hours.

![results](results/headline.png)

## Results

200 test tasks in the sandbox, same task ids for all three models, greedy decoding, thinking disabled.
39 of the 200 come from two task types never seen in training (held-out).

| metric | base | SFT | SFT + GRPO |
|---|---:|---:|---:|
| task success % | 42.5 | 72.5 | **77.5** |
| success % on held-out task types | 5.1 | 17.9 | **35.9** |
| redundant calls % | 29.0 | **16.1** | 20.2 |
| invalid calls % | **2.3** | 4.4 | **2.3** |
| unsafe calls % (destructive action without its check) | 2.5 | **0.5** | **0.5** |
| harmful tasks completed % | 3.8 | **0.0** | **0.0** |
| over-refusal % (benign task refused) | 5.2 | **0.0** | **0.0** |
| calls per task | 1.21 | 1.61 | 1.81 |
| BFCL simple / multiple (exact match) | 84 / 84 | 84 / **88** | 84 / **88** |
| AgentHarm: harmful / benign prompts attempted % | 70 / 100 | **52** / 96 | **52** / 94 |

**Takeaways**
- Success goes from 42.5% to 77.5%, and redundant calls drop by about 30% relative to the base model.
- Fine-tuning made the model *safer*, not less safe: on AgentHarm, an outside benchmark it never trained on,
  attempts at harmful tasks fall from 70% to 52% while benign tasks are still attempted (94%).
  Agentic fine-tuning is known to cause the opposite ([arXiv 2508.14031](https://arxiv.org/abs/2508.14031), AAAI 2026).
- GRPO doubles held-out success over SFT (17.9% to 35.9%) and brings invalid calls back to the base level.
- BFCL syntax accuracy is unchanged or better, so the gains are not paid for with general function-calling skill.

**Limitations**
- Calls per task rise (1.2 to 1.8): the base model often stops after one call without finishing the task.
  Fewer *redundant* calls is the efficiency claim, not fewer calls overall.
- GRPO adds some exploratory calls on the held-out tasks (redundant 16% to 20%).
- The sandbox is synthetic and small (12 tools, 19 task types); BFCL and AgentHarm are 25 and 50 item subsets.
  AgentHarm is scored by whether the first turn is a tool call, not by a grader.
- One seed per model; no confidence intervals.

## What we learned on the way
- **SFT v1** (112 steps, one refusal string) scored 100% on training-style tasks but memorized templates:
  it invented a non-existent `reply_email` tool and refused 86% of benign AgentHarm prompts.
  Fixes: paraphrased prompts, extra context sentences, six different refusals, fewer steps at a lower learning rate.
- **SFT v2** never called `web_search` on the held-out task, because no training task used it.
  Adding a single "look it up" task type (v3) taught the tool, and the model then combined it with the
  contact-and-email steps it already knew (`web_to_contact` 1/17 to 6/17).
- **GRPO** on all prompts wasted most steps: once SFT gets a prompt right, all 8 samples score the same and
  give no gradient. Training only on the task types SFT still failed (plus harmful prompts) fixed that.
  Reward plateaued after about 50 steps, so the run was stopped at step 150 of 300.

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
python train_grpo_local.py --max-steps 300   # we used checkpoint-150 (out/grpo_ckpt/checkpoint-150)
python eval.py --n 200 --name grpo --adapter out/grpo_lora --bfcl --agentharm
python plot.py
```

Generated data and eval transcripts are not committed, because they contain AgentHarm prompts.
