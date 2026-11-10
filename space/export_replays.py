"""results/{base,sft,grpo}.jsonl -> space/replays.json: recorded test episodes for the no-GPU demo.
Benign tasks only: harmful tasks carry AgentHarm prompts (gated) and possibly harmful model output.
Run from the repo root: python space/export_replays.py"""
import json
import os

MODELS = {"base": "Base Qwen3-1.7B", "sft": "SFT", "grpo": "SFT + GRPO"}
KEYS = ("success", "n_calls", "n_redundant", "n_unsafe", "n_invalid")

eps = {m: {e["task_id"]: e for e in map(json.loads, open(f"results/{m}.jsonl", encoding="utf-8"))} for m in MODELS}
out = {}
for tid, e in eps["base"].items():
    if e["harmful"] or not all(tid in eps[m] for m in MODELS):
        continue
    out[tid] = {"prompt": e["messages"][1]["content"], "template": e["template"], "heldout": e["heldout"],
                "runs": {MODELS[m]: {**{k: eps[m][tid][k] for k in KEYS},
                                     "steps": [{k: s[k] for k in ("completion", "label", "result")}
                                               for s in eps[m][tid]["steps"]]}
                         for m in MODELS}}
assert out and not any(v["template"] == "harmful" for v in out.values())
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replays.json")
json.dump(out, open(path, "w", encoding="utf-8"))
print(f"{len(out)} benign tasks -> {path} ({os.path.getsize(path) // 1024} KB)")
