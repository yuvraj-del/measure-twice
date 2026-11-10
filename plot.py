"""results/summary.csv (+ results/bench.csv if present) -> results/headline.png (grouped bars, one group per metric, one bar per model)."""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [("success", "task success %", True), ("success_ho", "held-out task success %", True),
           ("redundant", "redundant calls %", False), ("invalid", "invalid turns %", False),
           ("agentharm_harmful_attempt", "AgentHarm harmful attempted %", False)]
ORDER = {"base": "base", "sft_norefuse": "SFT, no refusals", "sft": "SFT", "grpo": "SFT + GRPO"}
# Okabe-Ito colorblind-safe palette
COLORS = ["#999999", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def main(src="results/summary.csv", dst="results/headline.png"):
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    try:  # larger benchmark subsets override the small ones in summary.csv
        with open("results/bench.csv", newline="") as f:
            bench = {r["name"]: r for r in csv.DictReader(f)}
        rows = [{**r, **bench.get(r["name"], {})} for r in rows]
    except FileNotFoundError:
        pass
    rows = sorted((r for r in rows if r["name"] in ORDER), key=lambda r: list(ORDER).index(r["name"]))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 4.6))
    for ax, (key, title, up) in zip(axes, METRICS):
        vals = [float(r.get(key) or "nan") for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=COLORS[:len(rows)])
        ax.bar_label(bars, labels=[f"{v:.1f}" if v == v else "n/a" for v in vals], fontsize=9, padding=2)
        ax.set_title(title + ("\n(higher better)" if up else "\n(lower better)"), fontsize=10)
        ax.set_xticks(range(len(rows)), [ORDER[r["name"]] for r in rows], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, max([v for v in vals if v == v] + [1]) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Measure Twice: base vs fine-tuned Qwen3-1.7B (200 sandbox tasks; AgentHarm: 53 held-out prompts)", fontsize=13)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
