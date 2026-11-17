import numpy as np
import pytest

from vpipe import snap


def stacked_sheets(shape=(120, 96, 96), spacing=12, thick=3):
    """Binary prediction: horizontal sheets (constant z) every `spacing` voxels, `thick` voxels thick."""
    z = np.arange(shape[0])[:, None, None]
    return np.broadcast_to(((z % spacing) < thick), shape).astype(np.uint8)


def drifting_surface(h=40, w=40, z0=37.0, drift=0.5, step=1.5, o=4.0):
    """A traced surface that starts on a sheet and drifts `drift` voxels per grid column (crosses wraps)."""
    r, c = np.mgrid[0:h, 0:w].astype(float)
    xyz = np.stack([o + c * step, o + r * step, z0 + drift * c], -1)
    return xyz, np.ones((h, w), bool)


def test_band_centres():
    ts = np.arange(-5, 6, 1.0)
    p = np.zeros_like(ts, bool)
    p[[2, 3, 4]] = True          # t = -3..-1 -> centre -2
    p[[8, 9]] = True             # t = 3..4 -> centre 3.5
    assert list(snap.band_centres(p, ts)) == [-2.0, 3.5]
    assert snap.band_centres(np.zeros(5, bool), np.arange(5.0)).size == 0


def test_snap_follows_one_sheet_through_a_drift():
    vol = stacked_sheets()
    sample = snap.ZarrSampler(vol)
    xyz, v = drifting_surface()
    seed = xyz[20, 0]                       # seed on the sheet at z = 37 (band 36..38 -> centre 37)
    out, kept, off, info = snap.snap_surface(xyz, v, sample, seed, search=30, max_step=4, smooth=0)
    z = out[kept][:, 2]
    assert info["kept_fraction"] > 0.95
    assert np.allclose(z, 37.0, atol=1.01)  # every kept vertex is back on the seed's sheet, not a neighbour
    raw = xyz[v][:, 2]
    assert (np.abs(raw - 37.0) > 6).mean() > 0.4   # the input really drifted across wraps


def test_snap_drops_vertices_after_a_jump_bigger_than_max_step():
    vol = stacked_sheets()
    sample = snap.ZarrSampler(vol)
    xyz, v = drifting_surface(drift=0.0)
    xyz[:, 20:, 2] += 13.0                  # a hard wrap jump at column 20 (one spacing, offset changes by 13)
    seed = xyz[20, 0]
    out, kept, off, info = snap.snap_surface(xyz, v, sample, seed, search=30, max_step=4, smooth=0)
    assert np.allclose(out[kept][:, 2], 37.0, atol=1.01)
    # beyond the jump the same sheet is 13 voxels away along the normal; the growth may follow it only if the
    # step is gradual, so here the right half is either dropped or correctly snapped back to z = 37
    right = kept[:, 20:]
    assert (not right.any()) or np.allclose(out[:, 20:][right][:, 2], 37.0, atol=1.01)


def test_seed_without_band_raises():
    vol = np.zeros((40, 20, 20), np.uint8)
    xyz, v = drifting_surface(h=5, w=5, z0=10, drift=0)
    with pytest.raises(ValueError):
        snap.snap_surface(xyz, v, snap.ZarrSampler(vol), xyz[0, 0], search=5)
