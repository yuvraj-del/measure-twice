"""Local GRPO on top of the SFT LoRA for a small GPU (RTX 4050 6GB): TRL + peft + 4-bit, no Unsloth.
Same recipe as train_grpo.ipynb; generation runs one prompt's 8 completions at a time to fit in VRAM."""
import argparse
import json

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

import reward
from sandbox import tasks

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--sft", default="out/sft_lora")
ap.add_argument("--out", default="out/grpo_lora")
ap.add_argument("--max-steps", type=int, default=400)
ap.add_argument("--max-prompt", type=int, default=1536)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
rows = []
for l in open("data/grpo.jsonl", encoding="utf-8"):
    r = json.loads(l)
    p = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if len(tok(p, add_special_tokens=False)["input_ids"]) <= a.max_prompt:  # drop, never truncate the tool list
        rows.append({"prompt": p, "task_id": r["task_id"], "history": json.dumps(r["history"])})
ds = Dataset.from_list(rows).shuffle(seed=0)
print(len(rows), "prompts")


def reward_fn(completions, task_id, history, **kw):
    return [float(reward.step_reward(tasks.get(t), [tuple(x) for x in json.loads(h)],
                                     c if isinstance(c, str) else c[0]["content"]))
            for c, t, h in zip(completions, task_id, history)]


assert reward_fn(['<tool_call>{"name": "nope", "arguments": {}}</tool_call>'],
                 ds[:1]["task_id"], ds[:1]["history"]) == [reward.R["invalid"]]

model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16))
model = PeftModel.from_pretrained(model, a.sft, is_trainable=True)
model.print_trainable_parameters()

args = GRPOConfig(
    output_dir=a.out.replace("_lora", "_ckpt"), max_steps=a.max_steps,
    learning_rate=5e-6, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0, max_grad_norm=0.1,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,  # 16 completions = 2 prompts x 8 per update
    num_generations=8, generation_batch_size=8,                    # generate one prompt's group at a time
    max_completion_length=200, temperature=1.0, beta=0.0,
    bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1, save_steps=25, save_total_limit=3, report_to=[],
    remove_unused_columns=False, dataloader_num_workers=0, seed=0)
trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward_fn], args=args, train_dataset=ds)
trainer.train(resume_from_checkpoint=a.resume or None)
trainer.save_model(a.out)
tok.save_pretrained(a.out)
print(f"saved adapter -> {a.out}; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
