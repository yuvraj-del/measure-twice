"""Local LoRA SFT for a small GPU (RTX 4050 6GB): plain transformers + peft, no Unsloth.
One example per assistant turn; loss only on that turn, prompt rendered exactly like rollout.hf_generator."""
import argparse
import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--episodes", type=int, default=1500, help="oracle episodes from data/sft.jsonl to use")
ap.add_argument("--out", default="out/sft_lora")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--max-steps", type=int, default=-1)
ap.add_argument("--load-4bit", action="store_true", help="QLoRA, if bf16 does not fit")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--no-refusals", action="store_true", help="ablation: drop harmful-refusal episodes")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open("data/sft.jsonl", encoding="utf-8")]
random.Random(0).shuffle(rows)
ex = []
from sandbox.tasks import REFUSALS
for r in rows[:a.episodes]:
    m = r["messages"]
    if a.no_refusals and m[-1]["content"] in REFUSALS:
        continue
    for i, x in enumerate(m):
        if x["role"] != "assistant":
            continue
        p = tok(tok.apply_chat_template(m[:i], tokenize=False, add_generation_prompt=True, enable_thinking=False),
                add_special_tokens=False)["input_ids"]
        t = tok(x["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) + len(t) <= a.max_len:
            ex.append({"input_ids": p + t, "attention_mask": [1] * (len(p) + len(t)), "labels": [-100] * len(p) + t})
print(f"{len(ex)} turn examples from {min(a.episodes, len(rows))} episodes; "
      f"mean len {sum(len(e['input_ids']) for e in ex) / len(ex):.0f} tokens")

kw = {"dtype": torch.bfloat16, "device_map": {"": 0}}
if a.load_4bit:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                         "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=a.out + "_ckpt", per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
    num_train_epochs=1, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine",
    warmup_steps=max(1, int(0.03 * (a.max_steps if a.max_steps > 0 else len(ex) // a.grad_accum))),
    bf16=True, logging_steps=5, save_steps=20, save_total_limit=2, report_to=[], dataloader_num_workers=0,
    gradient_checkpointing_kwargs={"use_reentrant": False})
Trainer(model=model, args=args, train_dataset=Dataset.from_list(ex),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train(
    resume_from_checkpoint=a.resume or None)
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
print(f"saved adapter -> {a.out}; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
