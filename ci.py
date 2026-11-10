"""95% bootstrap CIs for sandbox metrics, plus paired differences between models (same task ids).
Usage: python ci.py base sft grpo"""
import json
import random
import sys

METRICS = {  # name -> (episode filter, numerator, denominator)
    "success %": (lambda e: True, lambda e: e["success"], lambda e: 1),
    "success % held-out": (lambda e: e["heldout"], lambda e: e["success"], lambda e: 1),
    "redundant % of calls": (lambda e: True, lambda e: e["n_redundant"], lambda e: e["n_calls"]),
    "invalid % of turns": (lambda e: True, lambda e: e["n_invalid"], lambda e: len(e["steps"])),
}


def stat(eps, ids, m):
    keep, num, den = METRICS[m]
    sel = [eps[i] for i in ids if keep(eps[i])]
    d = sum(den(e) for e in sel)
    return 100 * sum(num(e) for e in sel) / d if d else float("nan")


def ci(vals):
    v = sorted(x for x in vals if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def main(names, B=2000):
    runs = {n: {e["task_id"]: e for e in map(json.loads, open(f"results/{n}.jsonl", encoding="utf-8"))} for n in names}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    rng = random.Random(0)
    boots = [[rng.choice(ids) for _ in ids] for _ in range(B)]
    for m in METRICS:
        print(f"\n{m}")
        for n in names:
            lo, hi = ci(stat(runs[n], b, m) for b in boots)
            print(f"  {n:14} {stat(runs[n], ids, m):5.1f}  [{lo:5.1f}, {hi:5.1f}]")
        for a, b in zip(names, names[1:]):
            diffs = [stat(runs[b], s, m) - stat(runs[a], s, m) for s in boots]
            lo, hi = ci(diffs)
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  {b} - {a}: {stat(runs[b], ids, m) - stat(runs[a], ids, m):+5.1f}  [{lo:+5.1f}, {hi:+5.1f}]{sig}")


if __name__ == "__main__":
    assert ci(range(1000)) == (25, 974)
    main(sys.argv[1:] or ["base", "sft", "grpo"])
