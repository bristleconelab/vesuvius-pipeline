"""Score an automatically traced surface (tifxyz) against a reference surface of the same papyrus sheet.

tifxyz = x.tif / y.tif / z.tif float32 grids of volume coordinates; invalid vertices are -1 (or non-finite).
All distances are in voxels of the volume the meshes live on (multiply by the voxel size for um).

    python -m vpipe.surface_compare GROWN.tifxyz REFERENCE.tifxyz --voxel-um 9.362 -o out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

THRESHOLDS = (2, 6, 10)


def load_tifxyz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz grid H x W x 3 float64, valid mask H x W)."""
    import tifffile

    p = Path(path)
    xyz = np.stack([tifffile.imread(p / f"{a}.tif") for a in "xyz"], axis=-1).astype(np.float64)
    valid = np.isfinite(xyz).all(-1) & (xyz[..., 0] != -1) & (xyz[..., 2] > 0)
    return xyz, valid


def grid_normals(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unit normals from the cross product of the grid's row/column tangents (central differences)."""
    g = np.where(valid[..., None], xyz, np.nan)
    du = np.gradient(g, axis=1)
    dv = np.gradient(g, axis=0)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / norm
    return n


def triangle_area(xyz: np.ndarray, valid: np.ndarray) -> float:
    """Surface area (voxel^2) of the quads whose four corners are valid, split into two triangles."""
    a, b, c, d = xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, :-1], xyz[1:, 1:]
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    t1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
    t2 = 0.5 * np.linalg.norm(np.cross(b - d, c - d), axis=-1)
    return float(np.nansum((t1 + t2)[q]))


def densify(xyz: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    """Points on each valid quad at `factor` x `factor` bilinear samples, so coarse grids are compared fairly."""
    if factor <= 1:
        return xyz[valid]
    a, b, c, d = xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, :-1], xyz[1:, 1:]
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    A, B, C, D = a[q], b[q], c[q], d[q]
    s = (np.arange(factor) + 0.5) / factor
    u, v = np.meshgrid(s, s, indexing="ij")
    u, v = u.ravel()[None, :, None], v.ravel()[None, :, None]
    pts = (1 - u) * (1 - v) * A[:, None] + (1 - u) * v * B[:, None] + u * (1 - v) * C[:, None] + u * v * D[:, None]
    return pts.reshape(-1, 3)


def compare(grown: tuple, ref: tuple, densify_factor: int = 4, sheet_band: tuple = (4.0, 40.0)) -> dict:
    gx, gv = grown
    rx, rv = ref
    gp = densify(gx, gv, densify_factor)
    rp = densify(rx, rv, densify_factor)
    rn = grid_normals(rx, rv)
    rpts, rnorm = rx[rv], rn[rv]
    ok = np.isfinite(rnorm).all(-1)
    rpts, rnorm = rpts[ok], rnorm[ok]

    d_g2r, _ = cKDTree(rp).query(gp, workers=-1)      # grown -> reference: is what we traced on the sheet?
    d_r2g, _ = cKDTree(gp).query(rp, workers=-1)      # reference -> grown: how much of the sheet did we get?
    # signed offset of grown points along the reference normal, from the nearest reference vertex
    _, idx = cKDTree(rpts).query(gp, workers=-1)
    signed = np.einsum("ij,ij->i", gp - rpts[idx], rnorm[idx])
    lo, hi = sheet_band
    off_sheet = (np.abs(signed) >= lo) & (np.abs(signed) <= hi)

    out = {
        "grown_area_vx2": triangle_area(gx, gv),
        "reference_area_vx2": triangle_area(rx, rv),
        "grown_vertices": int(gv.sum()),
        "reference_vertices": int(rv.sum()),
        "grown_to_ref": {"median": float(np.median(d_g2r)), "p90": float(np.percentile(d_g2r, 90))},
        "ref_to_grown": {"median": float(np.median(d_r2g)), "p90": float(np.percentile(d_r2g, 90))},
        "precision_within": {str(t): float((d_g2r <= t).mean()) for t in THRESHOLDS},
        "coverage_within": {str(t): float((d_r2g <= t).mean()) for t in THRESHOLDS},
        "signed_normal_offset": {
            "median": float(np.median(signed)),
            "abs_p90": float(np.percentile(np.abs(signed), 90)),
            "off_sheet_fraction": float(off_sheet.mean()),
            "band_vx": [lo, hi],
        },
    }
    out["area_ratio"] = out["grown_area_vx2"] / max(out["reference_area_vx2"], 1e-9)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vpipe.surface_compare")
    ap.add_argument("grown")
    ap.add_argument("reference")
    ap.add_argument("--voxel-um", type=float, default=9.362)
    ap.add_argument("--densify", type=int, default=4)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    res = compare(load_tifxyz(a.grown), load_tifxyz(a.reference), a.densify)
    cm2 = (a.voxel_um * 1e-4) ** 2
    res["grown_area_cm2"] = res["grown_area_vx2"] * cm2
    res["reference_area_cm2"] = res["reference_area_vx2"] * cm2
    res["voxel_um"] = a.voxel_um
    res["inputs"] = {"grown": a.grown, "reference": a.reference}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
