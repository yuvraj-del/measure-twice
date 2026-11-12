---
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
tags: [lora, qlora, grpo, trl, tool-use, agents, function-calling, ai-safety]
---

# Measure Twice: Qwen3-1.7B LoRA adapters for efficient, safe tool calls

LoRA adapters that make Qwen3-1.7B a better tool-using agent: finish the task, skip redundant calls, check before
destructive actions, and refuse harmful requests. Trained with 4-bit QLoRA on a 6 GB laptop GPU.
Built for the Lexsi AI Safety & Alignment Hackathon (Track 3: Agentic Fine-tuning).

- Code, method and reproduce guide: https://github.com/yuvraj-del/measure-twice
- Demo: https://huggingface.co/spaces/yuvraj-del/measure-twice
- Data, eval outputs and logs: https://huggingface.co/datasets/yuvraj-del/measure-twice-data

## Adapters in this repo

| path | what it is |
|---|---|
| `.` (root) | **SFT + GRPO**, the headline model |
| `sft/` | SFT on optimal trajectories plus varied refusals for harmful requests |
| `sft-no-refusals/` | Ablation: the same SFT data without refusals. **Less safe than the base model**; published only to reproduce the safety regression. Do not deploy. |
| `grpo-v2/` | A second GRPO round with harsher waste penalties (negative result, not better than the root adapter) |

## Results (200 sandbox test tasks; AgentHarm: 53 held-out harmful prompts)

| | base | SFT, no refusals | SFT | SFT + GRPO |
|---|---:|---:|---:|---:|
| task success % | 42.5 | 68.0 | 72.5 | **77.5** |
| held-out task types % | 5.1 | 35.9 | 17.9 | **35.9** |
| redundant calls % | 29.0 | 20.2 | **16.1** | 20.2 |
| invalid turns % | **2.3** | 10.5 | 4.4 | **2.3** |
| AgentHarm harmful attempted % | 71.7 | 98.1 | **54.7** | **54.7** |
| BFCL simple / multiple % | 90 / 86 | 88 / 90 | 90 / 90 | 90 / 90 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("yuvraj-del/measure-twice-qwen3-1.7b")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto")
model = PeftModel.from_pretrained(model, "yuvraj-del/measure-twice-qwen3-1.7b")  # subfolder="sft" etc. for others
```

Prompts use Qwen3's Hermes-style `<tool_call>` format with `enable_thinking=False`; see `rollout.py` in the repo.

## Limitations
Trained and evaluated on a synthetic 12-tool sandbox; one seed per model; AgentHarm scored by first-turn
tool-call attempts. Calls per task rise versus the base model, because the base model often gives up early.
