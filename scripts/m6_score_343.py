"""Score our PHerc 343 First Letters image against the published one (M6 Arm A) and draw a small side-by-side figure.

    uv run --python 3.12 python -I scripts/m6_score_343.py --published PUB.png --ours OURS.png \
        --sidecar PUB.json --letter-boxes LETTER_BOXES.json [--map-sums map_sha256.txt] --out results/m6-pherc343

Ink opacity a = (R - B) / 135 is read back from both images over the data panel given in the published sidecar;
letter boxes (raw render frame) are moved to the panel frame with the sidecar's area and orientation.
Writes <out>/ink_metrics.json and <out>/ink_compare.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vpipe import compare

R_TARGET = 0.999


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def block_reduce(x: np.ndarray, f: int, how: str) -> np.ndarray:
    h, w = (x.shape[0] // f) * f, (x.shape[1] // f) * f
    b = x[:h, :w].reshape(h // f, f, w // f, f, *x.shape[2:])
    return getattr(b, how)(axis=(1, 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--letter-boxes", type=Path, required=True)
    ap.add_argument("--map-sums", type=Path, help="sha256sum list of our merged reading-side maps")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--downsample", type=int, default=6)
    a = ap.parse_args()

    side = json.loads(a.sidecar.read_text())
    dp = side["outputs"]["image"]["data_panel"]
    panel = (dp["x"], dp["y"], dp["width"], dp["height"])
    ar = side["area"]["raw_render_px"]
    area = (ar["col0"], ar["row0"], ar["col1"], ar["row1"])
    ori = side["orientation"]
    letters = json.loads(a.letter_boxes.read_text())
    assert letters["frame"] == "raw", letters["frame"]
    boxes_area = compare.boxes_raw_to_area([e["box"] for e in letters["letters"]], area,
                                           flip_lr=ori["flip_lr"], flip_ud=ori["flip_ud"])
    for b, ref in zip(boxes_area, side["letters"]["letters_area_px"]):  # same frame as the author's sidecar
        assert np.allclose(b, ref["box"], atol=1e-6), (b, ref)
    boxes = [dict(id=e["id"], box=b) for e, b in zip(letters["letters"], boxes_area)]

    P = np.asarray(Image.open(a.published).convert("RGB"))
    O = np.asarray(Image.open(a.ours).convert("RGB"))
    m = compare.ink_overlay_agreement(P, O, panel, boxes)

    sums = {}
    if a.map_sums:
        for line in a.map_sums.read_text().splitlines():
            h, f = line.split(maxsplit=1)
            sums[f.strip()] = h
    maps = [dict(file=e["file"], model=e["model"], published_sha256=e["sha256"],
                 ours_sha256=sums.get(e["file"]), equal=sums.get(e["file"]) == e["sha256"])
            for e in side["ink"]["maps"]] if sums else None

    hp, ho = sha256(a.published), sha256(a.ours)
    res = {
        "what": "PHerc 343 First Letters image, M6 Arm A: our ink maps at d=+1 with the published sidecar statistics",
        "opacity_rule": "a = (R - B) / 135 over the data panel; pixels pure black in both images and annotation "
                        "(row-line) pixels excluded",
        "published_png_sha256": hp,
        "ours_png_sha256": ho,
        "png_sha256_equal": hp == ho,
        "pearson_r_target": R_TARGET,
        "pearson_r_pass": bool(m["pearson_r"] >= R_TARGET),
        "boxes_all_pass": m["boxes_ratio_pass"] == m["boxes_total"],
        "agreement": m,
        "ink_maps_vs_sidecar": maps,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ink_metrics.json").write_text(json.dumps(res, indent=1) + "\n")

    x, y, w, h = panel
    f = a.downsample
    pub = block_reduce(P[y:y + h, x:x + w].astype(np.float32), f, "mean").astype(np.uint8)
    our = block_reduce(O[y:y + h, x:x + w].astype(np.float32), f, "mean").astype(np.uint8)
    ap_, okp = compare.overlay_opacity(P[y:y + h, x:x + w])
    ao_, oko = compare.overlay_opacity(O[y:y + h, x:x + w])
    d = np.where(okp & oko, np.abs(ap_ - ao_), 0.0)
    dmax = block_reduce(d, f, "max")
    fig, axs = plt.subplots(3, 1, figsize=(9, 9.6), constrained_layout=True)
    axs[0].imshow(pub)
    axs[0].set_title("published (Nieuwlaar, CC BY-NC 4.0)")
    axs[1].imshow(our)
    axs[1].set_title(f"ours: CPU, villa 2d3fb38d, sidecar statistics (r = {m['pearson_r']:.5f})")
    im = axs[2].imshow(dmax, cmap="magma", vmin=0, vmax=max(0.05, float(np.percentile(dmax, 99.9))))
    axs[2].set_title(f"|delta opacity|, max per {f}x{f} block (mean {m['mean_abs_diff']:.4f})")
    fig.colorbar(im, ax=axs[2], location="bottom", shrink=0.5, aspect=40, label="|a_ours - a_published|")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(a.out / "ink_compare.png", dpi=110)
    print(json.dumps({k: v for k, v in res.items() if k not in ("agreement", "ink_maps_vs_sidecar")}, indent=1))
    print("r", m["pearson_r"], "r boxes", m["pearson_r_letter_boxes"], "boxes pass",
          m["boxes_ratio_pass"], "/", m["boxes_total"])


if __name__ == "__main__":
    main()
