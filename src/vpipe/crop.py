"""Crop a render to the labelled region and paste a model output back onto the full render grid.

Slow models (Hecate on CPU: ~220 s per Mpx) only need to see the part of a segment that is scored. The crop is the
bounding box of the supervision mask on the render grid plus a margin that covers the texture-alignment shift.

    python -m vpipe.crop bbox  RENDER.zarr MASK.zarr [--margin 128]          -> prints "y0 x0 h w"
    python -m vpipe.crop paste MODEL.png RENDER.zarr Y0 X0 H W OUT.tif        -> full-size uint8 tif, 0 outside
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from vpipe.compare import resize_to


def mask_bbox(mask: np.ndarray, shape: tuple[int, int], margin: int = 128) -> tuple[int, int, int, int]:
    """(y0, x0, h, w) of the truthy part of `mask` resized onto `shape`, padded by `margin` and clipped."""
    m = np.asarray(mask)
    if m.shape != tuple(shape):
        m = resize_to(m.astype(np.float32), shape) > 0.5 * float(m.max())
    else:
        m = m > 0.5 * float(m.max())
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        raise ValueError("mask is empty")
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + 1 + margin, shape[0])
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + 1 + margin, shape[1])
    return y0, x0, y1 - y0, x1 - x0


def paste(pred: np.ndarray, full_shape: tuple[int, int], y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Resize a model output computed on the crop (any resolution) to (h, w) and place it in a zero image."""
    p = np.asarray(pred, np.float32)
    if p.ndim == 3:
        p = p[..., 0]
    if p.shape != (h, w):
        p = resize_to(p, (h, w))
    out = np.zeros(full_shape, np.uint8)
    out[y0:y0 + h, x0:x0 + w] = np.clip(np.rint(p), 0, 255).astype(np.uint8)
    return out


def _level0(path):
    import zarr

    return zarr.open(path, mode="r")["0"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vpipe.crop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bbox")
    b.add_argument("render")
    b.add_argument("mask")
    b.add_argument("--margin", type=int, default=128)
    p = sub.add_parser("paste")
    p.add_argument("pred")
    p.add_argument("render")
    for k in ("y0", "x0", "h", "w"):
        p.add_argument(k, type=int)
    p.add_argument("out")
    a = ap.parse_args(argv)
    shape = tuple(_level0(a.render).shape[1:])
    if a.cmd == "bbox":
        y0, x0, h, w = mask_bbox(np.asarray(_level0(a.mask)[...]), shape, a.margin)
        sys.stdout.write(f"{y0} {x0} {h} {w}\n")
    else:
        import tifffile
        from PIL import Image

        pred = np.asarray(Image.open(a.pred)) if a.pred.endswith(".png") else tifffile.imread(a.pred)
        tifffile.imwrite(a.out, paste(pred, shape, a.y0, a.x0, a.h, a.w))


if __name__ == "__main__":
    main()
