"""results/summary.csv -> results/headline.png (grouped bars, one group per metric, one bar per model)."""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [("calls", "calls / task"), ("redundant", "redundant %"), ("unsafe", "unsafe %"),
           ("harmful_compl", "harmful completion %"), ("success", "success %")]
# Okabe-Ito colorblind-safe palette
COLORS = ["#999999", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def main(src="results/summary.csv", dst="results/headline.png"):
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 4.2))
    for ax, (key, title) in zip(axes, METRICS):
        vals = [float(r.get(key) or "nan") for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=COLORS[:len(rows)])
        ax.bar_label(bars, labels=[f"{v:.1f}" if v == v else "n/a" for v in vals], fontsize=9, padding=2)
        ax.set_title(title + ("  (higher better)" if key == "success" else "  (lower better)"), fontsize=10)
        ax.set_xticks(range(len(rows)), [r["name"] for r in rows], rotation=20, fontsize=9)
        ax.set_ylim(0, max([v for v in vals if v == v] + [1]) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Measure Twice: fewer, valid, safe tool calls (sandbox test split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
