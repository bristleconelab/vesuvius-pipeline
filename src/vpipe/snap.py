"""Wrap-consistent snapping of a traced surface onto one papyrus sheet of a binary surface prediction.

A GrowPatch surface on the thresholded m7 prediction follows the right region but drifts slowly across neighbouring
wraps (PHerc0139 w035: median 11 voxels deep; PHerc 343 control: 9.5 voxels). CT-peak refinement cannot pull a vertex
back from half a wrap away (it cannot tell which sheet is "ours"). Here we never decide per vertex: starting at the seed,
we grow over the grid and give each vertex the band centre (along its normal) closest to the offset its already-assigned
neighbours chose. A slow drift of the traced surface then shows up as a slowly changing offset of the SAME band,
like phase unwrapping; a jump larger than `max_step` stops the growth there (vertex dropped) instead of switching wrap.

    python -m vpipe.snap IN.tifxyz OUT.tifxyz --pred URL_OR_PATH --seed X Y Z [--search 30 --max-step 4]
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from vpipe import surface_compare as sc


def band_centres(profile: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Offsets (in t units) of the middles of the positive runs of a boolean profile."""
    p = np.asarray(profile, bool)
    if not p.any():
        return np.empty(0)
    d = np.diff(np.r_[0, p.astype(np.int8), 0])
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
    return (ts[starts] + ts[ends]) / 2.0


def snap_offsets(centres: list, seed_rc: tuple[int, int], valid: np.ndarray, max_step: float = 4.0,
                 seed_offset: float = 0.0) -> np.ndarray:
    """Best-first growth over the grid: each vertex takes the band centre nearest to its assigned neighbour's offset.

    centres[r][c]: array of candidate offsets for vertex (r, c). Returns an offset grid (NaN = not reached / no band
    within max_step). Neighbours are 4-connected; the queue is ordered by the step size so smooth continuations win.
    """
    H, W = valid.shape
    off = np.full((H, W), np.nan)
    r0, c0 = seed_rc
    cand = centres[r0][c0]
    if not valid[r0, c0] or len(cand) == 0:
        raise ValueError("seed vertex has no band")
    off[r0, c0] = cand[np.argmin(np.abs(cand - seed_offset))]
    heap = []

    def push(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and valid[rr, cc] and np.isnan(off[rr, cc]) and len(centres[rr][cc]):
                k = centres[rr][cc]
                j = np.argmin(np.abs(k - off[r, c]))
                step = abs(k[j] - off[r, c])
                if step <= max_step:
                    heapq.heappush(heap, (step, rr, cc, float(k[j])))

    push(r0, c0)
    while heap:
        _, r, c, o = heapq.heappop(heap)
        if not np.isnan(off[r, c]):
            continue
        off[r, c] = o
        push(r, c)
    return off


def profiles(xyz: np.ndarray, nrm: np.ndarray, ok: np.ndarray, sample, search: float, step: float = 1.0):
    """Boolean prediction profiles along each normal; sample(zyx (3, M)) -> values."""
    ts = np.arange(-search, search + 1e-9, step)
    pts, nn = xyz[ok], nrm[ok]
    q = pts[:, None, :] + ts[None, :, None] * nn[:, None, :]
    v = sample(np.stack([q[..., 2].ravel(), q[..., 1].ravel(), q[..., 0].ravel()])).reshape(len(pts), len(ts))
    return ts, v > 0


def snap_surface(xyz, valid, sample, seed_xyz, search=30.0, max_step=4.0, smooth=1.0):
    """Return (snapped xyz, kept mask, offset grid, info)."""
    nrm = sc.grid_normals(xyz, valid)
    ok = valid & np.isfinite(nrm).all(-1)
    ts, prof = profiles(xyz, nrm, ok, sample, search)
    idx = np.argwhere(ok)
    centres = [[np.empty(0)] * valid.shape[1] for _ in range(valid.shape[0])]
    for (r, c), p in zip(idx, prof):
        centres[r][c] = band_centres(p, ts)
    d = np.linalg.norm(xyz[ok] - np.asarray(seed_xyz, float), axis=1)
    r0, c0 = idx[np.argmin(d)]
    off = snap_offsets(centres, (int(r0), int(c0)), ok, max_step)
    kept = np.isfinite(off)
    if smooth > 0:
        num = ndimage.gaussian_filter(np.where(kept, off, 0.0), smooth)
        den = ndimage.gaussian_filter(kept.astype(float), smooth)
        with np.errstate(invalid="ignore", divide="ignore"):
            off_s = np.where(kept, num / den, np.nan)
    else:
        off_s = off
    out = xyz.copy()
    out[kept] = xyz[kept] + off_s[kept][:, None] * nrm[kept]
    info = {"vertices": int(ok.sum()), "kept": int(kept.sum()), "kept_fraction": float(kept.sum() / max(ok.sum(), 1)),
            "median_abs_offset": float(np.nanmedian(np.abs(off))), "p90_abs_offset": float(np.nanpercentile(np.abs(off), 90)),
            "seed_vertex": [int(r0), int(c0)]}
    return out, kept, off, info


class ZarrSampler:
    """Nearest-voxel sampler over a (compressed) zarr array; loads the bounding box of each request."""

    def __init__(self, array):
        self.a = array

    def __call__(self, zyx):
        q = np.rint(zyx).astype(int)
        lo = np.maximum(q.min(1), 0)
        hi = np.minimum(q.max(1) + 1, self.a.shape)
        sub = np.asarray(self.a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        qq = np.clip(q - lo[:, None], 0, np.array(sub.shape)[:, None] - 1)
        inside = ((q >= 0) & (q < np.array(self.a.shape)[:, None])).all(0)
        v = sub[qq[0], qq[1], qq[2]]
        return np.where(inside, v, 0)


def snap_tiled(xyz, valid, array, seed_xyz, tile=24, **kw):
    """Profiles are sampled tile by tile (bounded memory); the growth itself runs over the whole grid."""
    samp = ZarrSampler(array)
    nrm = sc.grid_normals(xyz, valid)
    ok = valid & np.isfinite(nrm).all(-1)
    search = kw.get("search", 30.0)
    ts = np.arange(-search, search + 1e-9, 1.0)
    centres = [[np.empty(0)] * valid.shape[1] for _ in range(valid.shape[0])]
    H, W = valid.shape
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            sl = (slice(r0, r0 + tile), slice(c0, c0 + tile))
            m = ok[sl]
            if not m.any():
                continue
            _, prof = profiles(xyz[sl], nrm[sl], m, samp, search)
            for (r, c), p in zip(np.argwhere(m), prof):
                centres[r0 + r][c0 + c] = band_centres(p, ts)
    idx = np.argwhere(ok)
    d = np.linalg.norm(xyz[ok] - np.asarray(seed_xyz, float), axis=1)
    rs, cs = idx[np.argmin(d)]
    off = snap_offsets(centres, (int(rs), int(cs)), ok, kw.get("max_step", 4.0))
    kept = np.isfinite(off)
    out = xyz.copy()
    sm = kw.get("smooth", 1.0)
    num = ndimage.gaussian_filter(np.where(kept, off, 0.0), sm)
    den = ndimage.gaussian_filter(kept.astype(float), sm)
    with np.errstate(invalid="ignore", divide="ignore"):
        off_s = np.where(kept, num / den, np.nan)
    out[kept] = xyz[kept] + off_s[kept][:, None] * nrm[kept]
    info = {"vertices": int(ok.sum()), "kept": int(kept.sum()), "kept_fraction": float(kept.sum() / max(ok.sum(), 1)),
            "median_abs_offset": float(np.nanmedian(np.abs(off))), "p90_abs_offset": float(np.nanpercentile(np.abs(off), 90)),
            "seed_vertex": [int(rs), int(cs)]}
    return out, kept, off, info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vpipe.snap")
    ap.add_argument("surface")
    ap.add_argument("output")
    ap.add_argument("--pred", required=True, help="binary surface prediction OME-Zarr (path or http URL)")
    ap.add_argument("--level", default="0")
    ap.add_argument("--seed", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--search", type=float, default=30.0)
    ap.add_argument("--max-step", type=float, default=4.0)
    ap.add_argument("--smooth", type=float, default=1.0)
    a = ap.parse_args(argv)
    import zarr

    if a.pred.startswith("http"):
        import fsspec

        arr = zarr.open_group(fsspec.get_mapper(a.pred), mode="r")[a.level]
    else:
        arr = zarr.open_group(a.pred, mode="r")[a.level]
    xyz, valid = sc.load_tifxyz(a.surface)
    out, kept, off, info = snap_tiled(xyz, valid, arr, a.seed, search=a.search, max_step=a.max_step, smooth=a.smooth)
    from vpipe.refine import write_tifxyz

    write_tifxyz(Path(a.surface), Path(a.output), out, kept, {"snap": info, "max_step": a.max_step, "search": a.search})
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
