"""Build the held-out results table (markdown, injected into README) and figure from results/*/eval_*.json.

    uv run --python 3.12 python -I scripts/make_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [  # (results dir, label, group)
    ("w035-train", "PHerc0139 w035", "in training set"),
    ("heldout-w030", "PHerc0139 w030", "held out, same scroll"),
    ("heldout-w045", "PHerc0139 w045", "held out, same scroll"),
    ("heldout-0841-w00", "PHerc0841 w00", "held out, unseen scroll"),
    ("heldout-0841-ag1", "PHerc0841 auto-grown 1", "held out, unseen scroll"),
    ("heldout-0841-ag2", "PHerc0841 auto-grown 2", "held out, unseen scroll"),
]
COLORS = {"in training set": "#8c8c8c", "held out, same scroll": "#2f6db5", "held out, unseen scroll": "#c8553d"}
START, END = "<!-- results:heldout:start -->", "<!-- results:heldout:end -->"


def load(d: str) -> dict:
    ink = json.loads((ROOT / "results" / d / "eval_ink.json").read_text())
    ren = json.loads((ROOT / "results" / d / "eval_render.json").read_text())
    lab = ink["labels"]
    return {
        "m": lab["metrics_in_mask"],
        "align": lab.get("texture_alignment"),
        "render_exact": ren["agreement"]["exact_fraction"],
        "render_within1": ren["agreement"]["within1_fraction"],
    }


def table(rows) -> str:
    out = ["| Segment | Set | Scored px | ROC AUC | AP | best F0.5 | Label grid → ours (dy, dx) | Render identical / within 1 |",
           "|---|---|---|---|---|---|---|---|"]
    for (d, label, group), r in rows:
        m, a = r["m"], r["align"]
        al = "native 9 µm" if a is None else f"({a['dy']}, {a['dx']}) r={a['r']:.2f}"
        out.append(f"| {label} | {group} | {m['n_pixels']:,} | {m['roc_auc']:.3f} | {m['average_precision']:.3f} | "
                   f"{m['best_f05']:.3f} | {al} | {100*r['render_exact']:.2f} % / {100*r['render_within1']:.0f} % |")
    return "\n".join(out)


def figure(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    names = [label for (_, label, _), _ in rows]
    for ax, key, title in zip(axes, ["roc_auc", "average_precision"], ["ROC AUC", "Average precision"]):
        vals = [r["m"][key] for _, r in rows]
        cols = [COLORS[g] for (_, _, g), _ in rows]
        y = range(len(rows))[::-1]
        ax.barh(list(y), vals, color=cols)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
        ax.set_yticks(list(y), names, fontsize=9)
        ax.set_xlim(0.4 if key == "roc_auc" else 0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "roc_auc":
            ax.axvline(0.5, color="#bbbbbb", lw=0.8, ls="--")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, COLORS.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("scrollprize/ink_9um seed42 step-20000 on CPU, scored inside each segment's supervision mask", fontsize=10)
    fig.savefig(ROOT / "figures" / "heldout_metrics.png", dpi=110, bbox_inches="tight")


def main() -> None:
    rows = [(r, load(r[0])) for r in RUNS]
    md = table(rows)
    figure(rows)
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        pre, rest = text.split(START, 1)
        _, post = rest.split(END, 1)
        readme.write_text(pre + START + "\n" + md + "\n" + END + post, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
