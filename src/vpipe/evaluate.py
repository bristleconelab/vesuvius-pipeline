"""Evaluate pipeline outputs against published reference products and write JSON results.

    python -m vpipe.evaluate render  OURS.zarr  REF.zarr  -o results/render.json
    python -m vpipe.evaluate ink     PRED.tif   --labels L.zarr --ref-pred R.tif --ref-image T.webp -o results/ink.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from vpipe import compare


def _open_level0(path: str) -> np.ndarray:
    import zarr

    z = zarr.open(path, mode="r")
    arr = z["0"] if hasattr(z, "array_keys") and "0" in list(z.array_keys()) else z
    return np.asarray(arr[...])


def eval_render(ours: str, ref: str) -> dict:
    a, b = _open_level0(ours), _open_level0(ref)
    out = {"ours": ours, "reference": ref}
    if a.shape != b.shape:
        out["error"] = f"shape mismatch {a.shape} vs {b.shape}"
        return out
    out["agreement"] = compare.volume_agreement(a, b)
    # also check the reversed depth order, in case normals were flipped the other way
    rev = compare.volume_agreement(a[::-1], b)
    out["agreement_depth_reversed_mean_r"] = float(np.nanmean(rev["pearson_per_layer"]))
    return out


def _read_image(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        img = tifffile.imread(path)
    elif p.suffix.lower() == ".zarr" or p.is_dir():
        img = _open_level0(path)
    else:
        from PIL import Image

        img = np.asarray(Image.open(path).convert("L"))
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"{path}: expected a 2D image, got {img.shape}")
    return img


def _pearson(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if mask is not None:
        x, y = x[mask], y[mask]
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def _load_on_grid(path: str, shape, level: int = 0) -> tuple[np.ndarray, list]:
    """Load a label-like image (zarr level or image file) and bring it onto `shape`."""
    if Path(path).suffix.lower() == ".zarr":
        import zarr

        a = np.asarray(zarr.open(path, mode="r")[str(level)][...])
    else:
        a = _read_image(path)
    src = list(a.shape)
    if a.shape != tuple(shape):
        a = compare.resize_to(a, shape)
    return a, src


def _depth_mean(path: str, level: int) -> np.ndarray:
    """Mean over axis 0 of a (depth, H, W) zarr level, as float32.

    Published surface volumes are chunked over the full depth (e.g. 109 x 128 x 128), so reading one
    depth slice at a time decodes every chunk once per slice. Reading full-depth bands one chunk row
    tall decodes each chunk exactly once and keeps RAM at depth x chunk_rows x W. The per-band sum
    adds slices in the same order as a slice-by-slice float64 accumulator, so the result is identical.
    """
    import zarr

    a = zarr.open(path, mode="r")[str(level)]
    depth, H = a.shape[0], a.shape[1]
    rows = (getattr(a, "shards", None) or a.chunks)[1]
    acc = np.zeros(a.shape[1:], np.float64)
    for y0 in range(0, H, rows):
        band = a[:, y0:y0 + rows]
        out = acc[y0:y0 + rows]
        for i in range(depth):
            out += band[i]
    return (acc / depth).astype(np.float32)


def eval_ink(pred_path: str, labels: str | None, ref_pred: str | None, ref_image: str | None,
             label_level: int = 0, mask: str | None = None,
             align_ref: str | None = None, align_ref_level: int = 2, align_ours: str | None = None) -> dict:
    from vpipe import align

    pred = _read_image(pred_path)
    out: dict = {"prediction": pred_path, "shape": list(pred.shape)}
    valid = np.ones(pred.shape, bool)

    if labels:
        lab, lab_src = _load_on_grid(labels, pred.shape, label_level)
        y = lab > 0.5 * float(lab.max())
        if mask:
            m, m_src = _load_on_grid(mask, pred.shape, label_level)
            sup = m > 0.5 * float(m.max())
        else:
            sup, m_src = valid, None
        shift = None
        if align_ref and align_ours:
            shift = align.texture_shift(_depth_mean(align_ref, align_ref_level), _depth_mean(align_ours, 0))
            y = align.shift_image(y, shift["dy"], shift["dx"])
            sup = align.shift_image(sup, shift["dy"], shift["dx"])
        out["labels"] = {
            "path": labels,
            "mask": mask,
            "level": label_level,
            "source_shape": lab_src,
            "mask_source_shape": m_src,
            "texture_alignment": shift,
            "metrics_in_mask": compare.ink_metrics(pred, y, sup),
        }
        if shift:
            out["labels"]["metrics_in_mask_unaligned"] = compare.ink_metrics(
                pred, align.shift_image(y, -shift["dy"], -shift["dx"]),
                align.shift_image(sup, -shift["dy"], -shift["dx"]))

    if ref_pred:
        r = _read_image(ref_pred)
        r_r = compare.resize_to(r, pred.shape)
        m = valid & (r_r > 0)
        out["reference_prediction"] = {
            "path": ref_pred,
            "source_shape": list(r.shape),
            "pearson_valid": _pearson(pred, r_r, m),
            "alignment_shift_px": list(compare.estimate_shift(r_r, pred.astype(np.float32))),
        }

    if ref_image:
        t = _read_image(ref_image).astype(np.float32)
        # tutorial images are rescaled for display: (p - 0.25) / 0.5 clipped to [0, 1]
        disp = np.clip((pred.astype(np.float32) / 255.0 - 0.25) / 0.5, 0, 1)
        small = compare.resize_to(disp, t.shape)
        m = compare.resize_to(valid.astype(np.float32), t.shape) > 0.99
        out["reference_image"] = {
            "path": ref_image,
            "shape": list(t.shape),
            "pearson_valid": _pearson(small, t, m),
            "alignment_shift_px": list(compare.estimate_shift(t, small)),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vpipe.evaluate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("ours")
    r.add_argument("reference")
    i = sub.add_parser("ink")
    i.add_argument("prediction")
    i.add_argument("--labels")
    i.add_argument("--label-level", type=int, default=0)
    i.add_argument("--mask", help="supervision mask: only these pixels are scored against --labels")
    i.add_argument("--align-ref", help="published surface volume on the label grid (zarr), for texture alignment")
    i.add_argument("--align-ref-level", type=int, default=2)
    i.add_argument("--align-ours", help="our rendered surface volume (zarr) on the prediction grid")
    i.add_argument("--ref-pred")
    i.add_argument("--ref-image")
    for p in (r, i):
        p.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "render":
        res = eval_render(a.ours, a.reference)
    else:
        res = eval_ink(a.prediction, a.labels, a.ref_pred, a.ref_image, a.label_level, a.mask,
                       a.align_ref, a.align_ref_level, a.align_ours)
    res["_meta"] = {"seconds": round(time.time() - t0, 1), "host": os.environ.get("VPIPE_HOST_LABEL", "unlabelled"),
                    "python": platform.python_version(), "numpy": np.__version__}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:4000])


if __name__ == "__main__":
    main()
