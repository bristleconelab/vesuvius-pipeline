"""Model-free trust map for automatically traced surfaces: agreement between independent growths.

Two tracer runs from different seeds that land on the same papyrus sheet produce surfaces that coincide where both
are right; where one has jumped to a neighbouring wrap they separate by about one sheet spacing. Marking each vertex
of surface A as trusted when surface B passes within `tol` voxels of it needs no reference mesh, so it can run on an
unread scroll. Agreement does not prove correctness (both runs can make the same jump); it is a filter, scored here
against a reference where one exists.

    python -m vpipe.consensus A.tifxyz B.tifxyz [C.tifxyz ...] --tol 4 -o trust.json [--reference REF.tifxyz]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from vpipe import surface_compare as sc


def _surface_points(xyz: np.ndarray, valid: np.ndarray, factor: int = 8) -> np.ndarray:
    """Vertices plus a dense bilinear sampling of every quad (step 20 vx -> ~2.5 vx spacing), so point-to-point
    distance approximates point-to-surface distance to well under the tolerance."""
    return np.concatenate([xyz[valid], sc.densify(xyz, valid, factor)])


def agreement(a: tuple, others: list[tuple], tol: float = 4.0, min_votes: int = 1) -> np.ndarray:
    """Boolean grid over surface A: True where at least `min_votes` other surfaces pass within `tol` voxels.

    Vertices of A outside every other surface's footprint get no vote (False): unknown is not trusted.
    """
    ax, av = a
    pts = ax[av]
    votes = np.zeros(len(pts), int)
    for ox, ov in others:
        d, _ = cKDTree(_surface_points(ox, ov)).query(pts, workers=-1)
        votes += d <= tol
    out = np.zeros(av.shape, bool)
    out[av] = votes >= min_votes
    return out


def score_against_reference(a: tuple, trusted: np.ndarray, ref: tuple, on_sheet: float = 6.0) -> dict:
    """How well the trust map separates on-sheet from off-sheet vertices of A, using a reference mesh."""
    ax, av = a
    rp = _surface_points(*ref)
    d, _ = cKDTree(rp).query(ax[av], workers=-1)
    good = d <= on_sheet
    t = trusted[av]
    tp, fp = int((t & good).sum()), int((t & ~good).sum())
    return {
        "on_sheet_tol_vx": on_sheet,
        "all_vertices_on_sheet": float(good.mean()),
        "trusted_fraction": float(t.mean()),
        "precision_of_trusted": tp / max(tp + fp, 1),
        "recall_of_on_sheet": tp / max(int(good.sum()), 1),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vpipe.consensus")
    ap.add_argument("surfaces", nargs="+")
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--min-votes", type=int, default=1)
    ap.add_argument("--reference")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    surfs = [sc.load_tifxyz(s) for s in a.surfaces]
    res = {"tol_vx": a.tol, "min_votes": a.min_votes, "surfaces": {}}
    ref = sc.load_tifxyz(a.reference) if a.reference else None
    for i, s in enumerate(surfs):
        trusted = agreement(s, surfs[:i] + surfs[i + 1:], a.tol, a.min_votes)
        r = {"vertices": int(s[1].sum()), "trusted": int(trusted.sum())}
        if ref is not None:
            r.update(score_against_reference(s, trusted, ref))
        res["surfaces"][a.surfaces[i]] = r
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
