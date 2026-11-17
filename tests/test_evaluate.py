import numpy as np
import pytest
import tifffile
import zarr

from vpipe import evaluate


def _zarr_level0(path, arr):
    g = zarr.open_group(str(path), mode="w")
    g.create_array("0", data=arr)


def test_eval_ink_scores_only_inside_mask(tmp_path):
    lab = np.zeros((64, 64), np.uint8)
    lab[10:20, 10:40] = 255
    sup = np.zeros_like(lab)
    sup[0:32, :] = 255
    pred = np.where(lab > 0, 220, 40).astype(np.uint8)
    pred[50:60, 50:60] = 255  # outside the supervision mask: must not hurt the score
    tifffile.imwrite(tmp_path / "p.tif", pred)
    _zarr_level0(tmp_path / "lab.zarr", lab)
    _zarr_level0(tmp_path / "sup.zarr", sup)
    r = evaluate.eval_ink(str(tmp_path / "p.tif"), str(tmp_path / "lab.zarr"), None, None,
                          mask=str(tmp_path / "sup.zarr"))
    m = r["labels"]["metrics_in_mask"]
    assert m["roc_auc"] == 1.0
    assert m["n_pixels"] == 32 * 64
    assert r["labels"]["texture_alignment"] is None


def test_eval_render_identical(tmp_path):
    v = np.random.default_rng(0).integers(0, 256, (3, 16, 16), dtype=np.uint8)
    _zarr_level0(tmp_path / "a.zarr", v)
    _zarr_level0(tmp_path / "b.zarr", v)
    r = evaluate.eval_render(str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr"))
    assert r["agreement"]["exact_fraction"] == 1.0


def _depth_mean_slice_by_slice(path, level):
    a = zarr.open(path, mode="r")[str(level)]
    acc = np.zeros(a.shape[1:], np.float64)
    for i in range(a.shape[0]):
        acc += a[i]
    return (acc / a.shape[0]).astype(np.float32)


@pytest.mark.parametrize("fmt,chunks", [(2, (13, 16, 16)), (2, (5, 7, 32)), (3, (13, 9, 11)), (3, (1, 40, 40))])
def test_depth_mean_identical_to_slice_by_slice(tmp_path, fmt, chunks):
    rng = np.random.default_rng(7)
    v = rng.integers(0, 256, (13, 37, 45), dtype=np.uint8)
    g = zarr.open_group(str(tmp_path / "v.zarr"), mode="w", zarr_format=fmt)
    g.create_array("2", data=v, chunks=chunks)
    got = evaluate._depth_mean(str(tmp_path / "v.zarr"), 2)
    want = _depth_mean_slice_by_slice(str(tmp_path / "v.zarr"), 2)
    assert got.dtype == np.float32 and got.shape == (37, 45)
    assert np.array_equal(got, want)
    assert np.allclose(got, v.mean(0), atol=1e-4)


def test_eval_ink_window_limits_scoring(tmp_path):
    lab = np.zeros((40, 40), np.uint8)
    lab[5:10, 5:10] = 255
    lab[25:35, 25:35] = 255
    sup = np.full_like(lab, 255)
    pred = np.where(lab > 0, 220, 30).astype(np.uint8)
    pred[25:35, 25:35] = 30                 # ink outside the window is missed (the model never ran there)
    tifffile.imwrite(tmp_path / "p.tif", pred)
    _zarr_level0(tmp_path / "lab.zarr", lab)
    _zarr_level0(tmp_path / "sup.zarr", sup)
    full = evaluate.eval_ink(str(tmp_path / "p.tif"), str(tmp_path / "lab.zarr"), None, None, mask=str(tmp_path / "sup.zarr"))
    win = evaluate.eval_ink(str(tmp_path / "p.tif"), str(tmp_path / "lab.zarr"), None, None, mask=str(tmp_path / "sup.zarr"),
                            window=(0, 0, 20, 20))
    assert full["labels"]["metrics_in_mask"]["roc_auc"] < 1.0
    assert win["labels"]["metrics_in_mask"]["roc_auc"] == 1.0
    assert win["labels"]["metrics_in_mask"]["n_pixels"] == 400
