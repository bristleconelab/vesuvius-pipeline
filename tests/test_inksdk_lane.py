"""The analysis helpers of the ink-sdk Kaggle lane (pure numpy; the Kaggle plumbing is not run here)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inksdk_lane", ROOT / "kaggle" / "inksdk" / "lane.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


def test_pool_ignores_nan_and_drops_remainder():
    a = np.array([[1, np.nan, 5], [3, np.nan, 7], [9, 9, 9]], np.float32)
    p = lane.pool(a, 2)
    assert p.shape == (1, 1) and p[0, 0] == 2.0
    assert np.isnan(lane.pool(np.full((2, 2), np.nan), 2)[0, 0])


def test_pearson_is_nan_safe():
    a = np.arange(10, dtype=np.float32)
    b = 2 * a + 1
    b[3] = np.nan
    assert abs(lane.pearson(a, b) - 1.0) < 1e-9
    assert np.isnan(lane.pearson(a, np.ones(10)))


def test_stretch_and_zscore_respect_valid():
    a = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    valid = np.ones_like(a, bool)
    valid[0] = False
    s = lane.stretch(a, valid)
    assert np.isnan(s[0]).all() and np.nanmin(s) == 0 and np.nanmax(s) == 1
    z = lane.zscore(a, valid)
    assert abs(np.nanmean(z)) < 1e-6 and abs(np.nanstd(z) - 1) < 1e-5


def test_triage_prefers_line_texture_and_skips_empty_windows():
    rng = np.random.default_rng(0)
    H, W = 768, 1152
    norm = rng.uniform(0, 0.5, (H, W)).astype(np.float32)
    lines = norm.copy()
    lines[::24] = 1.0                                     # text-like rows of ink in the right third
    lines[1::24] = 1.0
    norm[:, 768:] = lines[:, 768:]
    valid = np.ones((H, W), bool)
    valid[:, :192] = False                                # an empty strip: windows there are dropped
    win = lane.triage(norm, valid, top=5)
    assert win and win[0]['x'] >= 768 - 192 and win[0]['score'] > 0.2
    assert all(w['valid_frac'] >= 0.5 for w in win)
    for i, a in enumerate(win):                           # no chosen pair overlaps by more than half a window
        for b in win[i + 1:]:
            oy = max(0, min(a['y'] + 384, b['y'] + 384) - max(a['y'], b['y']))
            ox = max(0, min(a['x'] + 384, b['x'] + 384) - max(a['x'], b['x']))
            assert oy * ox <= 0.5 * 384 * 384


def test_window_score_needs_enough_valid_lines():
    ink = np.ones((10, 10), bool)
    valid = np.zeros((10, 10), bool)
    valid[0] = True
    assert lane.window_score(ink, valid) is None


def test_generated_scripts_match_lane():
    mk = importlib.util.spec_from_file_location("inksdk_make", ROOT / "kaggle" / "inksdk" / "make_script.py")
    make = importlib.util.module_from_spec(mk)
    mk.loader.exec_module(make)
    for target in ("306b", "0358", "hx0358top"):
        script = (ROOT / "kaggle" / "inksdk" / target / "script.py").read_text()
        assert script == make.build(target)
        assert f"TARGET = '{target}'" in script
        meta = json.loads((ROOT / "kaggle" / "inksdk" / target / "kernel-metadata.json").read_text())
        assert meta["id"] == f"KAGGLE_USER/vesuvius-inksdk-{target}" and meta["is_private"] is True
        assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_internet"] is True


def test_hx_shelf_pack_is_even_and_disjoint():
    spec2 = importlib.util.spec_from_file_location("inksdk_hx", ROOT / "kaggle" / "inksdk" / "hx0358top" / "script.py")
    hx = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(hx)
    sizes = [(301, 517), (120, 99), (77, 640), (301, 33), (5, 5)]
    pos, (H, W) = hx.shelf_pack(sizes, 700, 64)
    boxes = [(pos[i][0], pos[i][1], pos[i][0] + h, pos[i][1] + w) for i, (h, w) in enumerate(sizes)]
    assert all(y % 2 == 0 and x % 2 == 0 for y, x in pos.values())
    assert all(b[2] <= H and b[3] <= W for b in boxes)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
