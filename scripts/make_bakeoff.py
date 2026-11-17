"""Bake-off table (README) and figure from results/bakeoff/<checkpoint>/<run>.json.

    uv run --python 3.12 python -I scripts/make_bakeoff.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("heldout-w030", "0139 w030"), ("heldout-w045", "0139 w045"),
        ("heldout-0841-w00", "0841 w00"), ("heldout-0841-ag1", "0841 ag1"), ("heldout-0841-ag2", "0841 ag2")]
MODELS = [("s42-20k", "seed42 step 20k"), ("s42-75k", "seed42 step 75k"), ("s43-20k", "seed43 step 20k"),
          ("s43-75k", "seed43 step 75k"), ("ens-s42s43-75k", "mean of seed42+43 step 75k")]
START, END = "<!-- results:bakeoff:start -->", "<!-- results:bakeoff:end -->"


def auc(model: str, run: str):
    if model == "s42-20k":   # the first held-out pass used this checkpoint
        p = ROOT / "results" / run / "eval_ink.json"
    else:
        p = ROOT / "results" / "bakeoff" / model / f"{run}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["labels"]["metrics_in_mask"]["roc_auc"]


def main() -> None:
    grid = {m: [auc(m, r) for r, _ in RUNS] for m, _ in MODELS}
    head = "| Checkpoint | " + " | ".join(n for _, n in RUNS) + " | mean, unseen scroll (0841) |"
    lines = [head, "|---" * (len(RUNS) + 2) + "|"]
    for m, label in MODELS:
        v = grid[m]
        cells = ["n/a" if x is None else f"{x:.3f}" for x in v]
        unseen = [x for x in v[2:] if x is not None]
        mean = f"{sum(unseen) / len(unseen):.3f}" if len(unseen) == 3 else "n/a"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")
    md = "\n".join(lines)

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for m, label in MODELS:
        ys = grid[m]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [ys[i] for i in xs], marker="o", label=label, lw=2.2 if m.startswith("ens") else 1.2)
    ax.axhline(0.85, color="#999", ls="--", lw=0.9)
    ax.text(len(RUNS) - 0.6, 0.853, "gate 0.85 before screening", fontsize=8, color="#666", ha="right")
    ax.axvspan(1.5, len(RUNS) - 0.5, color="#f2e3df", zorder=0)
    ax.text(3, 0.96, "unseen scroll PHerc0841", ha="center", fontsize=9, color="#8a3b2b")
    ax.set_xticks(range(len(RUNS)), [n for _, n in RUNS])
    ax.set_ylabel("ROC AUC inside supervision mask")
    ax.set_ylim(0.65, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.savefig(ROOT / "figures" / "bakeoff_auc.png", dpi=110)

    readme = ROOT / "README.md"
    t = readme.read_text()
    if START in t:
        pre, rest = t.split(START, 1)
        _, post = rest.split(END, 1)
        readme.write_text(pre + START + "\n" + md + "\n" + END + post)
    print(md)


if __name__ == "__main__":
    main()
