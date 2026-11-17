import numpy as np
import pytest
import tifffile

from vpipe import surface_compare as sc


def plane(h, w, step=20.0, z0=100.0, ox=0.0, oy=0.0):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    xyz = np.stack([ox + c * step, oy + r * step, np.full_like(r, z0)], -1)
    return xyz, np.ones((h, w), bool)


def test_identical_surfaces_score_perfectly():
    ref = plane(20, 30)
    r = sc.compare(ref, ref)
    assert r["grown_to_ref"]["median"] == pytest.approx(0, abs=1e-9)
    assert r["coverage_within"]["2"] == 1.0 and r["precision_within"]["2"] == 1.0
    assert r["area_ratio"] == pytest.approx(1.0)
    assert r["signed_normal_offset"]["off_sheet_fraction"] == 0.0


def test_area_of_plane():
    xyz, v = plane(11, 21, step=10.0)
    assert sc.triangle_area(xyz, v) == pytest.approx(100 * 200)


def test_partial_coverage_and_parallel_sheet():
    ref = plane(20, 40)
    half = plane(20, 20)                       # left half only, on the sheet
    r = sc.compare(half, ref)
    assert r["precision_within"]["2"] == 1.0
    assert 0.45 < r["coverage_within"]["10"] < 0.6
    wrong = plane(20, 40, z0=115.0)            # parallel sheet 15 voxels away = wrong wrap
    r2 = sc.compare(wrong, ref)
    assert r2["precision_within"]["10"] == 0.0
    assert r2["signed_normal_offset"]["off_sheet_fraction"] == pytest.approx(1.0)
    assert abs(r2["signed_normal_offset"]["median"]) == pytest.approx(15.0, abs=1e-6)


def test_invalid_vertices_ignored(tmp_path):
    xyz, v = plane(10, 10)
    xyz[0, 0] = -1
    for i, a in enumerate("xyz"):
        tifffile.imwrite(tmp_path / f"{a}.tif", xyz[..., i].astype(np.float32))
    g, valid = sc.load_tifxyz(tmp_path)
    assert valid.sum() == 99 and not valid[0, 0]
