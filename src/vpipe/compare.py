"""Numeric comparisons between pipeline outputs and published reference products.

All functions are pure numpy/scipy so they run on CPU and are unit-tested on synthetic data.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def volume_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement of two uint8 surface volumes shaped (layers, H, W).

    Returns exact-match and |diff|<=1 fractions, mean absolute difference and Pearson r per layer.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per_layer = []
    for i in range(a.shape[0]):
        x = a[i].ravel().astype(np.float64)
        y = b[i].ravel().astype(np.float64)
        if x.std() == 0 or y.std() == 0:
            per_layer.append(1.0 if np.array_equal(x, y) else float("nan"))
        else:
            per_layer.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "shape": list(a.shape),
        "exact_fraction": float((d == 0).mean()),
        "within1_fraction": float((d <= 1).mean()),
        "mae": float(d.mean()),
        "max_abs_diff": int(d.max()),
        "pearson_per_layer": per_layer,
    }


def _to_uint8(pred: np.ndarray) -> np.ndarray:
    if pred.dtype == np.uint8:
        return pred
    p = np.asarray(pred, dtype=np.float64)
    if p.max() > 1.0:
        p = p / 255.0
    return np.clip(np.round(p * 255), 0, 255).astype(np.uint8)


def ink_metrics(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None,
                beta: float = 0.5) -> dict:
    """Threshold-free and best-threshold metrics of an ink prediction against binary labels.

    pred: uint8 (0-255) or float in [0,1]. labels: truthy = ink. mask: truthy = pixel counts.
    ROC AUC counts ties as 1/2 (Mann-Whitney). AP is the step-wise precision-recall area.
    best_f05 is the maximum F-beta (beta=0.5, the 2023 Kaggle ink metric) over all 256 thresholds.
    """
    p = _to_uint8(pred).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    if mask is not None:
        m = np.asarray(mask).astype(bool).ravel()
        p, y = p[m], y[m]
    pos = np.bincount(p[y], minlength=256).astype(np.float64)
    neg = np.bincount(p[~y], minlength=256).astype(np.float64)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        raise ValueError("labels must contain both classes inside the mask")
    # cumulative counts with prediction >= t, for t = 255..0
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    pos_above = np.concatenate([[0.0], tp[:-1]])[::-1]  # count of positives strictly above level l
    auc = float(np.sum(neg * (pos_above + 0.5 * pos)) / (P * N))
    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - rec_prev) * precision))
    b2 = beta * beta
    f = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)
    k = int(np.argmax(f))
    return {
        "n_pixels": int(P + N),
        "ink_fraction": float(P / (P + N)),
        "roc_auc": auc,
        "average_precision": ap,
        "best_f05": float(f[k]),
        "best_threshold": int(255 - k),
        "precision_at_best": float(precision[k]),
        "recall_at_best": float(recall[k]),
    }


def estimate_shift(ref: np.ndarray, moved: np.ndarray) -> tuple[int, int]:
    """Integer (dy, dx) such that moved ~= np.roll(ref, (dy, dx)), by phase correlation."""
    if ref.shape != moved.shape:
        raise ValueError("shape mismatch")
    a = np.asarray(ref, np.float64)
    b = np.asarray(moved, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    R = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.ifft2(R).real
    dy, dx = np.unravel_index(int(np.argmax(c)), c.shape)
    H, W = c.shape
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return int(dy), int(dx)


def resize_to(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a 2D image to an exact shape (anti-aliased when shrinking). Returns float32."""
    x = np.asarray(img, np.float32)
    f = (shape[0] / x.shape[0], shape[1] / x.shape[1])
    sig = [max(0.0, (1.0 / fi - 1.0) / 2.0) for fi in f]
    if any(s > 0 for s in sig):
        x = ndimage.gaussian_filter(x, sig)
    out = ndimage.zoom(x, f, order=1, grid_mode=True, mode="nearest")
    if out.shape != tuple(shape):  # zoom rounds; pad/crop by at most one pixel
        out = out[: shape[0], : shape[1]]
        out = np.pad(out, ((0, shape[0] - out.shape[0]), (0, shape[1] - out.shape[1])), mode="edge")
    return out.astype(np.float32)


# --- First Letters overlay images ----------------------------------------------------------------------------------
# The PHerc 343 image (dimmed-ct style) is pixel = CT grey * 0.45 * (1 - op) + (255, 235, 120) * op, so R - B = 135 op
# and G - B = 115 op exactly before 8-bit rounding: the ink opacity can be read back from either PNG.

def overlay_opacity(rgb: np.ndarray, ink=(255, 235, 120), tol: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Ink opacity a = (R - B) / (ink_R - ink_B) of an overlay image (H, W, 3), and a mask of pixels that are a
    grey/ink mixture (R >= B and G - B consistent with the ink colour within `tol` levels). Row lines (cyan) and
    other annotation colours fall outside the mask."""
    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError("expected an (H, W, 3) image")
    R, G, B = (x[..., i].astype(np.float64) for i in range(3))
    span_rb = float(ink[0] - ink[2])
    k = float(ink[1] - ink[2]) / span_rb
    d = R - B
    ok = (d >= 0) & (np.abs((G - B) - k * d) <= tol)
    return d / span_rb, ok


def boxes_raw_to_area(boxes, area, flip_lr: bool, flip_ud: bool) -> list[list[float]]:
    """Boxes [x0, y0, x1, y1] in raw render pixels -> pixels of the area (x0, y0, x1, y1 exclusive) as shown,
    i.e. mirrored in x and/or y when the image is in reading orientation."""
    c0, r0, c1, r1 = area
    out = []
    for x0, y0, x1, y1 in boxes:
        xs = (c1 - x1, c1 - x0) if flip_lr else (x0 - c0, x1 - c0)
        ys = (r1 - y1, r1 - y0) if flip_ud else (y0 - r0, y1 - r0)
        out.append([float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1])])
    return out


def box_slices(box, shape) -> tuple[slice, slice]:
    """Pixels whose centre (i + 0.5) lies in [x0, x1) x [y0, y1), clipped to `shape` (H, W)."""
    x0, y0, x1, y1 = box

    def rng(lo, hi, n):
        a = int(np.clip(np.ceil(lo - 0.5), 0, n))
        b = int(np.clip(np.ceil(hi - 0.5), 0, n))
        return slice(a, max(a, b))

    return rng(y0, y1, shape[0]), rng(x0, x1, shape[1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ink_overlay_agreement(published: np.ndarray, ours: np.ndarray, panel, boxes,
                          ink_level: float = 0.0, ratio_range=(0.8, 1.25), ink=(255, 235, 120)) -> dict:
    """Agreement of two overlay images over their data panel (x, y, width, height).

    Pixels compared: inside the panel, a grey/ink mixture in both images (row lines excluded) and not pure black in
    both (invalid papyrus). Reports the Pearson r of the ink opacity, its absolute differences, and per letter box
    (dicts with "id" and "box" in panel pixels): r and the ink fraction (opacity > ink_level) of both images and
    their ratio ours / published; a box passes if the ratio lies within ratio_range (inclusive).
    """
    P = np.asarray(published)
    O = np.asarray(ours)
    if P.shape != O.shape:
        raise ValueError(f"shape mismatch {P.shape} vs {O.shape}")
    x, y, w, h = panel
    P = P[y:y + h, x:x + w, :3]
    O = O[y:y + h, x:x + w, :3]
    ap, okp = overlay_opacity(P, ink)
    ao, oko = overlay_opacity(O, ink)
    black = (P == 0).all(-1) & (O == 0).all(-1)
    m = okp & oko & ~black
    dif = np.abs(ap - ao)[m]
    lo, hi = ratio_range
    inkp, inko = ap > ink_level, ao > ink_level
    out_boxes = []
    union = np.zeros_like(m)
    for b in boxes:
        sl = box_slices(b["box"], m.shape)
        mb = m[sl]
        union[sl] |= mb
        n = int(mb.sum())
        fp = float(inkp[sl][mb].mean()) if n else float("nan")
        fo = float(inko[sl][mb].mean()) if n else float("nan")
        ratio = fo / fp if fp > 0 else (1.0 if fo == 0 else float("inf"))
        out_boxes.append(dict(id=b["id"], n_pixels=n, pearson_r=_pearson(ap[sl][mb], ao[sl][mb]),
                              ink_fraction_published=fp, ink_fraction_ours=fo, ink_fraction_ratio=ratio,
                              **{"pass": bool(lo <= ratio <= hi)}))
    return {
        "panel": [int(v) for v in panel],
        "n_pixels": int(m.sum()),
        "excluded_annotation_pixels": int((~(okp & oko) & ~black).sum()),
        "pearson_r": _pearson(ap[m], ao[m]),
        "mean_abs_diff": float(dif.mean()) if dif.size else float("nan"),
        "p99_abs_diff": float(np.percentile(dif, 99)) if dif.size else float("nan"),
        "max_abs_diff": float(dif.max()) if dif.size else float("nan"),
        "ink_level": float(ink_level),
        "ink_fraction_published": float(inkp[m].mean()) if m.any() else float("nan"),
        "ink_fraction_ours": float(inko[m].mean()) if m.any() else float("nan"),
        "pearson_r_letter_boxes": _pearson(ap[union], ao[union]),
        "n_pixels_letter_boxes": int(union.sum()),
        "ratio_range": [float(lo), float(hi)],
        "boxes_total": len(out_boxes),
        "boxes_ratio_pass": int(sum(b["pass"] for b in out_boxes)),
        "boxes": out_boxes,
    }
