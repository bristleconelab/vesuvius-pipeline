"""Snap a traced surface onto the papyrus sheet it is following, using CT intensity along each vertex normal.

On PHerc0139 at 9.362 um the published w035 mesh sits on the CT intensity peak of the sheet (mean profile maximum at
offset 0, sheet ~6 voxels thick, next wrap ~16 voxels away), while surfaces grown on the thresholded m7 prediction
follow the right wrap on average but wander 2-15 voxels around it. Refinement: for every vertex, sample the CT along
its normal within +-radius (less than half the wrap spacing, so it cannot jump wraps), take the smoothed profile's
maximum, smooth those offsets over the grid, move the vertices, recompute normals, repeat.

    python -m vpipe.refine IN.tifxyz OUT.tifxyz --volume URL_OR_PATH [--radius 5 --iterations 3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage

from vpipe import surface_compare as sc


def nan_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs (normalised convolution)."""
    v = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(v, a, 0.0), sigma)
    den = ndimage.gaussian_filter(v.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~v] = np.nan
    return out


def profile_offsets(sample, pts: np.ndarray, nrm: np.ndarray, radius: float, step: float = 0.5,
                    profile_sigma: float = 1.0) -> np.ndarray:
    """Offset (voxels along the normal) of the smoothed CT maximum within [-radius, radius] for each point.

    sample(coords_zyx: (3, M) float array) -> (M,) intensities. Returns NaN where the profile is flat.
    """
    ts = np.arange(-radius, radius + 1e-9, step)
    q = pts[:, None, :] + ts[None, :, None] * nrm[:, None, :]          # (n, T, xyz)
    vals = sample(np.stack([q[..., 2].ravel(), q[..., 1].ravel(), q[..., 0].ravel()])).reshape(len(pts), len(ts))
    vals = ndimage.gaussian_filter1d(vals.astype(np.float64), profile_sigma / step, axis=1)
    k = np.argmax(vals, axis=1)
    off = ts[k]
    flat = (vals.max(1) - vals.min(1)) < 1e-6
    edge = (k == 0) | (k == len(ts) - 1)          # no peak inside the window: leave the vertex where it is
    off[flat | edge] = np.nan
    return off


def refine(xyz: np.ndarray, valid: np.ndarray, sample, radius: float = 5.0, iterations: int = 3,
           grid_sigma: float = 1.5) -> tuple[np.ndarray, list[dict]]:
    x = xyz.copy()
    log = []
    for it in range(iterations):
        n = sc.grid_normals(x, valid)
        ok = valid & np.isfinite(n).all(-1)
        off = np.full(valid.shape, np.nan)
        off[ok] = profile_offsets(sample, x[ok], n[ok], radius)
        sm = nan_gaussian(off, grid_sigma)
        move = ok & np.isfinite(sm)
        x[move] += sm[move][:, None] * n[move]
        log.append({"iteration": it, "vertices": int(move.sum()),
                    "median_abs_shift": float(np.nanmedian(np.abs(sm[move]))),
                    "p90_abs_shift": float(np.nanpercentile(np.abs(sm[move]), 90))})
    return x, log


class MirrorArray:
    """Read-only view of one level of an uncompressed zarr v2 volume served over HTTP, with an on-disk chunk mirror.

    Chunks are fetched once (in parallel, with retries) into cache_dir/<level>/<cz>/<cy>/<cx> and read from disk after
    that, so repeated passes over the same region cost no network. Missing chunks (HTTP 404) read as fill_value.
    """

    def __init__(self, base_url: str, level: str, cache_dir: str | Path, fetch=None, workers: int = 16,
                 read_only_dirs=()):
        """read_only_dirs: extra chunk stores laid out as <dir>/level_<L>/<cz>/<cy>/<cx>.bin (VC3D's remote_cache),
        checked before downloading; identical raw chunks, so a VC3D render's cache is reused as is."""
        import json as _json

        self.base, self.level, self.cache = base_url.rstrip("/"), str(level), Path(cache_dir)
        self.ro = [Path(d) for d in read_only_dirs]
        self.allow_fetch = True   # set False after prefetch(): chunks outside the prefetched set read as fill
        self.fetch = fetch or self._http_get
        self.workers = workers
        meta_path = self.cache / self.level / ".zarray"
        if not meta_path.exists():
            raw = self.fetch(f"{self.base}/{self.level}/.zarray")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_bytes(raw)
        m = _json.loads(meta_path.read_text())
        if m.get("compressor") is not None or m.get("filters") or m.get("dimension_separator", ".") != "/":
            raise ValueError("MirrorArray supports uncompressed zarr v2 with '/' separators only")
        self.shape, self.chunks = tuple(m["shape"]), tuple(m["chunks"])
        self.dtype, self.fill = np.dtype(m["dtype"]), m.get("fill_value") or 0

    @staticmethod
    def _http_get(url: str):
        import time
        import urllib.error
        import urllib.request

        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise IOError(f"failed to fetch {url}")

    def _chunk_path(self, c) -> Path:
        for d in self.ro:
            p = d / f"level_{self.level}" / str(c[0]) / str(c[1]) / f"{c[2]}.bin"
            if p.exists():
                return p
        return self.cache / self.level / str(c[0]) / str(c[1]) / str(c[2])

    def prefetch(self, keys) -> int:
        """Download every missing chunk in one parallel batch; returns how many were fetched."""
        keys = list(set(keys))
        n = sum(not self._chunk_path(c).exists() for c in keys)
        self._ensure(keys)
        self.allow_fetch = False
        return n

    def _ensure(self, keys) -> None:
        from concurrent.futures import ThreadPoolExecutor

        todo = [c for c in keys if not self._chunk_path(c).exists()]

        def get(c):
            raw = self.fetch(f"{self.base}/{self.level}/{c[0]}/{c[1]}/{c[2]}")
            p = self.cache / self.level / str(c[0]) / str(c[1]) / str(c[2])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".part")
            tmp.write_bytes(raw if raw is not None else b"")
            tmp.rename(p)

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(get, todo))

    def __getitem__(self, sl):
        lo = [s.start or 0 for s in sl]
        hi = [s.stop if s.stop is not None else n for s, n in zip(sl, self.shape)]
        out = np.full([h - l for l, h in zip(lo, hi)], self.fill, self.dtype)
        rng = [range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, self.chunks)]
        keys = [(a, b, d) for a in rng[0] for b in rng[1] for d in rng[2]]
        if self.allow_fetch:
            self._ensure(keys)
        for k in keys:
            p = self._chunk_path(k)
            if not p.exists():
                continue                      # not prefetched: outside the region any sample can reach
            raw = p.read_bytes()
            if not raw:
                continue
            ch = np.frombuffer(raw, self.dtype).reshape(self.chunks)
            c0 = [k[i] * self.chunks[i] for i in range(3)]
            a = [max(lo[i], c0[i]) for i in range(3)]
            b = [min(hi[i], c0[i] + self.chunks[i]) for i in range(3)]
            out[tuple(slice(a[i] - lo[i], b[i] - lo[i]) for i in range(3))] = \
                ch[tuple(slice(a[i] - c0[i], b[i] - c0[i]) for i in range(3))]
        return out


class TiledSampler:
    """Trilinear sampler over an OME-Zarr level that loads (and caches) the bounding box of each request."""

    def __init__(self, array, pad: int = 4):
        self.a, self.pad = array, pad

    def __call__(self, zyx: np.ndarray) -> np.ndarray:
        lo = np.maximum(np.floor(zyx.min(1)).astype(int) - self.pad, 0)
        hi = np.minimum(np.ceil(zyx.max(1)).astype(int) + self.pad + 1, self.a.shape)
        sub = np.asarray(self.a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
        return ndimage.map_coordinates(sub, zyx - lo[:, None], order=1, mode="nearest")


def refine_tiled(xyz, valid, array, tile: int = 16, **kw):
    """Run refine() tile by tile over the grid so only a small CT box is in memory at a time (1-node overlap)."""
    out = xyz.copy()
    logs = []
    H, W = valid.shape
    if hasattr(array, "prefetch"):
        pad = kw.get("radius", 5.0) + 8
        pts = xyz[valid]
        keys = set()
        for d in np.array(np.meshgrid([-pad, 0, pad], [-pad, 0, pad], [-pad, 0, pad])).T.reshape(-1, 3):
            q = np.clip(np.floor((pts + d)[:, ::-1] / np.array(array.chunks)).astype(int), 0,
                        np.array(array.shape) // np.array(array.chunks))
            keys.update(map(tuple, np.unique(q, axis=0)))
        array.prefetch(keys)
    samp = TiledSampler(array)
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            rs, cs = slice(max(r0 - 2, 0), min(r0 + tile + 2, H)), slice(max(c0 - 2, 0), min(c0 + tile + 2, W))
            v = valid[rs, cs]
            if v.sum() < 9:
                continue
            x_t, log = refine(xyz[rs, cs], v, samp, **kw)
            ir, ic = r0 - rs.start, c0 - cs.start
            core = (slice(ir, ir + min(tile, H - r0)), slice(ic, ic + min(tile, W - c0)))
            out[r0:r0 + tile, c0:c0 + tile][v[core]] = x_t[core][v[core]]
            logs.append(log[-1])
    return out, logs


def write_tifxyz(src: Path, dst: Path, xyz: np.ndarray, valid: np.ndarray, extra_meta: dict) -> None:
    import tifffile

    dst.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate("xyz"):
        ch = np.where(valid, xyz[..., i], -1.0).astype(np.float32)
        tifffile.imwrite(dst / f"{a}.tif", ch)
    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    meta.update({"uuid": dst.name, "name": dst.name, "refined": extra_meta})
    pts = xyz[valid]
    meta["bbox"] = [pts.min(0).tolist(), pts.max(0).tolist()]
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    for f in src.glob("*.tif"):
        if f.stem not in "xyz" and not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vpipe.refine")
    ap.add_argument("surface")
    ap.add_argument("output")
    ap.add_argument("--volume", required=True, help="OME-Zarr path or http(s) URL of the CT volume")
    ap.add_argument("--level", default="0")
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--grid-sigma", type=float, default=1.5)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--cache-dir", default="/data/vesuvius/cache/mirror", help="on-disk chunk mirror for HTTP volumes")
    ap.add_argument("--vc3d-cache", default="/data/vesuvius/vc3d-home-shared/.VC3D/remote_cache",
                    help="VC3D remote_cache to reuse (read only)")
    a = ap.parse_args(argv)
    if a.volume.startswith("http"):
        name = a.volume.rstrip("/").split("/")[-1]
        ro = [Path(a.vc3d_cache) / d for d in (os.listdir(a.vc3d_cache) if a.vc3d_cache and os.path.isdir(a.vc3d_cache) else [])
              if d.startswith(name + "-")]
        arr = MirrorArray(a.volume, a.level, Path(a.cache_dir) / name, workers=32, read_only_dirs=ro)
    else:
        import zarr

        arr = zarr.open_group(a.volume, mode="r")[a.level]
    xyz, valid = sc.load_tifxyz(a.surface)
    out, logs = refine_tiled(xyz, valid, arr, tile=a.tile, radius=a.radius, iterations=a.iterations,
                             grid_sigma=a.grid_sigma)
    shift = np.linalg.norm(out[valid] - xyz[valid], axis=1)
    summary = {"radius": a.radius, "iterations": a.iterations, "grid_sigma": a.grid_sigma, "tiles": len(logs),
               "median_total_shift": float(np.median(shift)), "p90_total_shift": float(np.percentile(shift, 90))}
    write_tifxyz(Path(a.surface), Path(a.output), out, valid, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
