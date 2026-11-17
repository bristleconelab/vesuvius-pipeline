import numpy as np
import pytest
from scipy import ndimage

from vpipe import align


def _texture(rng, shape, sigma=2.0):
    return ndimage.gaussian_filter(rng.random(shape).astype(np.float32), sigma) + 0.1


def _case(seed):
    """Random ref/ours pair: ref on a finer grid, ours = resized ref, rolled, noisy, with a ragged
    zero (unsupported) border on both images, like a real segment's footprint."""
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(70, 130)), int(rng.integers(70, 130))
    scale = float(rng.uniform(1.0, 4.0))
    big = _texture(rng, (int(h * scale), int(w * scale)), sigma=float(rng.uniform(1.0, 3.0)))
    big[: int(rng.integers(0, 6 * scale)), :] = 0
    big[:, -int(rng.integers(1, 6 * scale)):] = 0
    shift = (int(rng.integers(-5, 6)), int(rng.integers(-5, 6)))
    ours = np.roll(align.resize_to(big, (h, w)), shift, (0, 1))
    ours += rng.normal(0, float(rng.uniform(0.0, 0.05)), ours.shape).astype(np.float32)
    ours[-int(rng.integers(1, 8)):, :] = 0
    ours[:, : int(rng.integers(0, 8))] = 0
    ours[ours < 0] = 0
    return big, ours, shift


def test_texture_shift_recovers_offset_across_scales():
    rng = np.random.default_rng(4)
    big = _texture(rng, (400, 360))                           # "2.4 um" texture
    small = align.resize_to(big, (100, 90))                   # "9 um" grid, 4x coarser
    ours = np.roll(small, (3, -2), (0, 1))
    for fn in (align.texture_shift, lambda a, b, search: align.texture_shift_bruteforce(a, b, search, step=1)):
        r = fn(big, ours, search=6)
        assert (r["dy"], r["dx"]) == (3, -2)
        assert not r["on_search_edge"]


@pytest.mark.parametrize("seed", range(8))
def test_texture_shift_matches_bruteforce(seed):
    big, ours, true_shift = _case(seed)
    fast = align.texture_shift(big, ours, search=7)
    exact = align.texture_shift_bruteforce(big, ours, search=7, step=1)
    sub = align.texture_shift_bruteforce(big, ours, search=7)          # default step=5, as before
    assert set(fast) == set(exact) == {"dy", "dx", "r", "r_second", "on_search_edge"}
    assert (fast["dy"], fast["dx"]) == (exact["dy"], exact["dx"]) == (sub["dy"], sub["dx"])
    # the synthetic resize + ragged border can move the optimum by a pixel; both methods must agree on it
    assert abs(fast["dy"] - true_shift[0]) <= 1 and abs(fast["dx"] - true_shift[1]) <= 1
    assert fast["on_search_edge"] == exact["on_search_edge"]
    # same score as the full-overlap brute force, to floating-point precision
    assert fast["r"] == pytest.approx(exact["r"], abs=1e-9)
    assert fast["r_second"] == pytest.approx(exact["r_second"], abs=1e-9)
    assert isinstance(fast["dy"], int) and isinstance(fast["dx"], int)


def test_masked_shift_correlation_equals_corrcoef_at_every_shift():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 50)).astype(np.float32)
    Y = np.roll(X, (2, -3), (0, 1)) + rng.normal(0, 0.5, X.shape).astype(np.float32)
    m = rng.random(X.shape) > 0.2
    s = 4
    r, n = align.masked_shift_correlation(X, Y, m, s)
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            mm = m & np.roll(m, (dy, dx), (0, 1))
            a, b = np.roll(X, (dy, dx), (0, 1))[mm], Y[mm]
            assert n[dy + s, dx + s] == mm.sum()
            assert r[dy + s, dx + s] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-10)


def test_texture_shift_reports_search_edge():
    rng = np.random.default_rng(5)
    big = _texture(rng, (200, 180))
    shifted = np.roll(align.resize_to(big, (100, 90)), (6, 0), (0, 1))   # true shift outside +-3
    fast = align.texture_shift(big, shifted, search=3)
    exact = align.texture_shift_bruteforce(big, shifted, search=3, step=1)
    assert (fast["dy"], fast["dx"], fast["on_search_edge"]) == (exact["dy"], exact["dx"], exact["on_search_edge"])
    assert fast["on_search_edge"]


def test_texture_shift_rejects_window_larger_than_image():
    with pytest.raises(ValueError):
        align.texture_shift(np.ones((10, 10)), np.ones((10, 10)), search=6)


def test_shift_image_zeroes_wrapped_border():
    a = np.ones((5, 5), np.uint8)
    b = align.shift_image(a, 2, -1)
    assert b[:2].sum() == 0 and b[:, -1:].sum() == 0
    assert b[2:, :-1].all()
