import numpy as np
import pytest

from vpipe import compare


def test_volume_agreement_identical():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(4, 32, 32), dtype=np.uint8)
    r = compare.volume_agreement(a, a.copy())
    assert r["exact_fraction"] == 1.0
    assert r["within1_fraction"] == 1.0
    assert r["mae"] == 0.0
    assert all(abs(x - 1.0) < 1e-9 for x in r["pearson_per_layer"])


def test_volume_agreement_off_by_one():
    a = np.full((2, 8, 8), 100, np.uint8)
    b = a.copy()
    b[0, 0, 0] = 101
    b[1, 0, 0] = 110
    r = compare.volume_agreement(a, b)
    assert r["exact_fraction"] == pytest.approx(1 - 2 / 128)
    assert r["within1_fraction"] == pytest.approx(1 - 1 / 128)


def test_volume_agreement_shape_mismatch():
    with pytest.raises(ValueError):
        compare.volume_agreement(np.zeros((1, 2, 2), np.uint8), np.zeros((1, 2, 3), np.uint8))


def test_ink_metrics_perfect_and_random():
    rng = np.random.default_rng(1)
    labels = rng.random((64, 64)) > 0.8
    perfect = np.where(labels, 255, 0).astype(np.uint8)
    m = compare.ink_metrics(perfect, labels)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["average_precision"] == pytest.approx(1.0)
    assert m["best_f05"] == pytest.approx(1.0)
    noise = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    m2 = compare.ink_metrics(noise, labels)
    assert 0.4 < m2["roc_auc"] < 0.6
    assert m2["average_precision"] == pytest.approx(labels.mean(), abs=0.07)


def test_ink_metrics_auc_matches_rank_formula():
    rng = np.random.default_rng(2)
    labels = rng.random(2000) > 0.7
    pred = np.clip(rng.normal(100 + 40 * labels, 30), 0, 255).astype(np.uint8)
    # Mann-Whitney AUC with ties counted 1/2
    pos, neg = pred[labels].astype(int), pred[~labels].astype(int)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    assert compare.ink_metrics(pred, labels)["roc_auc"] == pytest.approx(gt + 0.5 * eq, abs=1e-9)


def test_ink_metrics_respects_mask():
    labels = np.zeros((10, 10), bool)
    labels[:5] = True
    pred = np.where(labels, 200, 10).astype(np.uint8)
    pred[9, 9] = 255  # wrong, but masked out
    mask = np.ones_like(labels)
    mask[9, 9] = False
    assert compare.ink_metrics(pred, labels, mask)["roc_auc"] == pytest.approx(1.0)


def test_estimate_shift_recovers_known_offset():
    rng = np.random.default_rng(3)
    base = rng.random((128, 160))
    moved = np.roll(base, shift=(5, -7), axis=(0, 1))
    dy, dx = compare.estimate_shift(base, moved)
    assert (dy, dx) == (5, -7)


def test_resize_to_shape():
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = compare.resize_to(img, (2, 2))
    assert out.shape == (2, 2)
    assert out.dtype == np.float32


# --- First Letters overlay images (ink opacity from RGB) ---------------------------------------------------------

INK = np.array([255, 235, 120], np.float32)


def _overlay(grey, op, dim=0.45):
    """Synthetic dimmed-ct overlay pixel(s): grey * dim * (1 - op) + INK * op, rounded to 8 bit."""
    g = np.asarray(grey, np.float32)[..., None] * np.float32(dim)
    o = np.asarray(op, np.float32)[..., None]
    return np.rint(g * (1 - o) + INK * o).astype(np.uint8)


def test_overlay_opacity_recovers_opacity_and_masks_lines():
    rng = np.random.default_rng(1)
    grey = rng.integers(0, 256, size=(40, 50))
    op = np.clip(rng.normal(0.3, 0.3, size=(40, 50)), 0, 0.9)
    img = _overlay(grey, op)
    img[5, :] = (0, 255, 255)                    # a cyan row line
    a, ok = compare.overlay_opacity(img)
    assert not ok[5].any()
    assert ok[np.arange(40) != 5].all()
    assert np.abs(a[ok] - op[ok]).max() <= 1.0 / 135 + 1e-6
    assert (a[~op.astype(bool) & ok] == 0).all()   # no ink -> exactly grey -> opacity 0


def test_letter_boxes_raw_to_area_matches_published_sidecar():
    # first letter box of PHerc 343: raw frame -> area frame (reading orientation), as in the author's sidecar
    out = compare.boxes_raw_to_area([[4992.5, 684.0, 5441.7, 1054.8]], area=(380, 105, 6021, 2560),
                                    flip_lr=True, flip_ud=True)
    assert out[0] == pytest.approx([579.3, 1505.2, 1028.5, 1876.0])
    same = compare.boxes_raw_to_area([[10, 20, 30, 40]], area=(5, 5, 100, 100), flip_lr=False, flip_ud=False)
    assert same[0] == pytest.approx([5, 15, 25, 35])


def test_box_pixels_uses_pixel_centres():
    sl = compare.box_slices([1.4, 2.6, 4.5, 5.0], shape=(10, 10))
    assert sl == (slice(3, 5), slice(1, 4))      # rows with centre in [2.6, 5.0): 3, 4; cols in [1.4, 4.5): 1, 2, 3


def _pair(seed=2):
    rng = np.random.default_rng(seed)
    grey = rng.integers(30, 200, size=(60, 80))
    op = np.zeros((60, 80))
    op[10:20, 10:30] = rng.uniform(0.2, 0.9, size=(10, 20))
    op[35:50, 40:70] = rng.uniform(0.2, 0.9, size=(15, 30))
    op[30:60, 0:5] = 0                            # leave a black (invalid) strip below
    pub = _overlay(grey, op)
    pub[55:, :] = 0                               # invalid pixels: black in both images
    return pub, grey, op


def test_ink_overlay_agreement_identical_images():
    pub, _, _ = _pair()
    boxes = [dict(id=1, box=[8, 8, 32, 22]), dict(id=2, box=[38, 33, 72, 52])]
    r = compare.ink_overlay_agreement(pub, pub.copy(), panel=(0, 0, 80, 60), boxes=boxes)
    assert r["pearson_r"] == pytest.approx(1.0)
    assert r["n_pixels"] == 55 * 80              # black-in-both rows excluded
    assert r["pearson_r_letter_boxes"] == pytest.approx(1.0)
    assert r["boxes_ratio_pass"] == 2 and r["boxes_total"] == 2
    assert all(b["ink_fraction_ratio"] == pytest.approx(1.0) for b in r["boxes"])
    assert r["max_abs_diff"] == 0.0


def test_ink_overlay_agreement_detects_missing_ink_in_one_box():
    pub, grey, op = _pair()
    op2 = op.copy()
    op2[35:50, 40:70] *= 0.0                      # our image lost the ink of box 2
    ours = _overlay(grey, op2)
    ours[55:, :] = 0
    boxes = [dict(id=1, box=[8, 8, 32, 22]), dict(id=2, box=[38, 33, 72, 52])]
    r = compare.ink_overlay_agreement(pub, ours, panel=(0, 0, 80, 60), boxes=boxes)
    assert r["pearson_r"] < 0.9
    b1, b2 = r["boxes"]
    assert b1["pass"] and b1["pearson_r"] == pytest.approx(1.0)
    assert not b2["pass"] and b2["ink_fraction_ratio"] == 0.0
    assert r["boxes_ratio_pass"] == 1


def test_ink_overlay_agreement_uses_panel_offset():
    pub, _, _ = _pair()
    big = np.zeros((70, 100, 3), np.uint8)
    big[5:65, 7:87] = pub
    big[0, :] = (255, 255, 255)                    # caption area outside the panel is ignored
    boxes = [dict(id=1, box=[8, 8, 32, 22])]
    r = compare.ink_overlay_agreement(big, big.copy(), panel=(7, 5, 80, 60), boxes=boxes)
    ref = compare.ink_overlay_agreement(pub, pub.copy(), panel=(0, 0, 80, 60), boxes=boxes)
    assert r["n_pixels"] == ref["n_pixels"]
    assert r["boxes"][0]["ink_fraction_ours"] == ref["boxes"][0]["ink_fraction_ours"]


def test_ink_overlay_agreement_shape_mismatch():
    with pytest.raises(ValueError):
        compare.ink_overlay_agreement(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8),
                                      panel=(0, 0, 4, 4), boxes=[])
