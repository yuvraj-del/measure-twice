"""Build dist/hf_model (LoRA adapters) and dist/hf_dataset (benign data, eval outputs, logs) for HF upload.
AgentHarm is gated, so every harmful-prompt item (training refusals, harmful test episodes) is dropped.
Run from the repo root: python stage_hf.py"""
import json
import shutil
from pathlib import Path

from sandbox.tasks import REFUSALS

D = Path("dist")
M, DS = D / "hf_model", D / "hf_dataset"
for p in (M, DS):
    shutil.rmtree(p, ignore_errors=True)

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
                 "tokenizer.json", "tokenizer_config.json"]
for src, dst in [("grpo_lora", "."), ("sft_lora", "sft"), ("sft_norefuse_lora", "sft-no-refusals"),
                 ("grpo2_lora", "grpo-v2")]:
    (M / dst).mkdir(parents=True, exist_ok=True)
    for f in ADAPTER_FILES:
        shutil.copy(Path("out") / src / f, M / dst / f)


def jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return len(rows)


harmful_prompts = {r["prompt"] for r in jsonl("data/harmful.jsonl")} if Path("data/harmful.jsonl").exists() else set()
sft = [e for e in jsonl("data/sft.jsonl") if e["messages"][-1]["content"] not in REFUSALS]
grpo = [e for e in jsonl("data/grpo.jsonl") if "-harmful-" not in e["task_id"]]
counts = {"train/sft_benign.jsonl": write(DS / "train/sft_benign.jsonl", sft),
          "train/grpo_prompts_benign.jsonl": write(DS / "train/grpo_prompts_benign.jsonl", grpo)}
for m in ["base", "sft_norefuse", "sft", "grpo", "grpo2"]:
    counts[f"eval/{m}.jsonl"] = write(DS / f"eval/{m}.jsonl", [e for e in jsonl(f"results/{m}.jsonl") if not e["harmful"]])
for f in ["summary.csv", "bench.csv", "headline.png"]:
    shutil.copy(Path("results") / f, DS / "eval" / f)
for f in ["data/test_ids.txt", "data/val_ids.txt"]:
    shutil.copy(f, DS / "train" / Path(f).name)
(DS / "logs").mkdir()
for f in [*Path("out").glob("*_log*.txt"), *Path("results").glob("*_log.txt")]:
    shutil.copy(f, DS / "logs" / f"{f.parent.name}_{f.name}")

# safety check: no gated prompt text anywhere in the staged dataset
blob = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in DS.rglob("*") if p.suffix in {".jsonl", ".txt", ".csv"})
assert not any(h[:80] in blob for h in harmful_prompts if len(h) > 40), "AgentHarm prompt text leaked"
train = "".join(p.read_text(encoding="utf-8") for p in (DS / "train").glob("*.jsonl"))
assert not any(r in train for r in REFUSALS), "refusal episodes (harmful prompts) leaked"  # eval may hold benign over-refusals
shutil.copy("hf/model_card.md", M / "README.md")
shutil.copy("hf/dataset_card.md", DS / "README.md")
print(json.dumps(counts, indent=1))
