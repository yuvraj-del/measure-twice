"""Write data/sft.jsonl, data/grpo.jsonl and val/test id lists from oracle episodes."""
import argparse
import json
from pathlib import Path

from rollout import oracle, run_episode
from sandbox.tasks import HARMFUL_PATH, fetch_harmful, generate

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=3000)
ap.add_argument("--n-eval", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-harmful", action="store_true", help="skip fetching benchmark prompts")
a = ap.parse_args()

if not HARMFUL_PATH.exists() and not a.no_harmful:
    fetch_harmful()
out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)
n_sft = n_grpo = n_harm = 0
with (out / "sft.jsonl").open("w", encoding="utf-8") as sft, (out / "grpo.jsonl").open("w", encoding="utf-8") as grpo:
    for t in generate("train", a.n_train, a.seed):
        ep = run_episode(t, oracle(t))
        msgs, hist = ep["messages"], []
        sft.write(json.dumps({"messages": msgs}) + "\n")
        n_sft += 1
        n_harm += t.harmful
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            grpo.write(json.dumps({"prompt": msgs[:i], "task_id": t.id, "history": [list(h) for h in hist]}) + "\n")
            n_grpo += 1
            if len(hist) < len(t.gold):
                hist.append(t.gold[len(hist)])
for split in ("val", "test"):
    (out / f"{split}_ids.txt").write_text("\n".join(t.id for t in generate(split, a.n_eval, a.seed)) + "\n")
print(f"sft={n_sft} episodes ({n_harm} harmful), grpo={n_grpo} step prompts, val/test={a.n_eval} ids -> {out}")
