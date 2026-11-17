"""Model-free alignment of a label grid to our rendered grid, from CT texture.

Published labels for most segments live on the 2.4 um render grid, while the pipeline renders at ~9.4 um.
Resizing by image shape gets the scale right, but the two grids differ by a constant offset of a few pixels.
We estimate that offset from papyrus texture alone (depth-mean of the published 2.4 um surface volume vs. our
render, both high-passed), so the ink model never influences how it is scored.
On w035 this gives (-8, -8); the offset that best matches native 9 um labels is (-8, -7), IoU 0.996.
"""
from __future__ import annotations

import numpy as np
from scipy import fft, ndimage

from vpipe.compare import resize_to

MIN_OVERLAP = 100  # shifts whose overlap has fewer pixels than this are not scored


def highpass(x: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x - ndimage.gaussian_filter(x, sigma)


def _prepare(ref: np.ndarray, ours: np.ndarray):
    R = resize_to(ref, ours.shape)
    O = np.asarray(ours, np.float32)
    return highpass(R), highpass(O), (O > 0) & (R > 0)


def _rank(scores: np.ndarray, search: int, valid: np.ndarray) -> dict:
    """Best and second-best shift, ordered exactly like sorting (r, dy, dx) tuples in reverse."""
    d = np.arange(-search, search + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    r, dy, dx = scores[valid], dy[valid], dx[valid]
    order = np.lexsort((dx, dy, r))[::-1]
    b, s = order[0], order[1]
    on_edge = abs(int(dy[b])) == search or abs(int(dx[b])) == search
    return {"dy": int(dy[b]), "dx": int(dx[b]), "r": float(r[b]), "r_second": float(r[s]),
            "on_search_edge": on_edge}


def masked_shift_correlation(X: np.ndarray, Y: np.ndarray, m: np.ndarray, search: int):
    """Pearson r between np.roll(X, (dy, dx)) and Y over m & np.roll(m, (dy, dx)), for every shift in
    [-search, search]^2 at once.

    Every masked sum that the correlation needs (count, sum, sum of squares, cross term) is a circular
    cross-correlation, computed by FFT on the full image. Circular is what np.roll does, so this is the
    same quantity the brute-force loop measures, with all overlap pixels used. Returns (r, n), both
    (2*search+1, 2*search+1) arrays indexed [dy + search, dx + search].
    """
    H, W = X.shape
    if 2 * search + 1 > min(H, W):
        raise ValueError(f"search window {search} too large for image {X.shape}")
    m = np.asarray(m, bool)
    # Pearson r does not change when a constant is added to either image; centring on the masked mean
    # keeps the variance terms (sum a^2 - (sum a)^2 / n) away from cancellation.
    x = np.where(m, X.astype(np.float64) - float(X[m].mean()), 0.0)
    y = np.where(m, Y.astype(np.float64) - float(Y[m].mean()), 0.0)
    idx = np.arange(-search, search + 1)
    win = np.ix_(idx % H, idx % W)

    def spec(a):
        return fft.rfft2(a, workers=-1)

    def xcorr(Fa, Fb):  # sum_p a(p) * b(p - s), for s in the window
        return fft.irfft2(Fa * np.conj(Fb), s=(H, W), workers=-1)[win]

    Fm, Fx, Fy = spec(m.astype(np.float64)), spec(x), spec(y)
    n = np.rint(xcorr(Fm, Fm))
    sa, sb, sab = xcorr(Fm, Fx), xcorr(Fy, Fm), xcorr(Fy, Fx)
    saa = xcorr(Fm, spec(x * x))
    sbb = xcorr(spec(y * y), Fm)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / n
        va = saa - sa * sa / n
        vb = sbb - sb * sb / n
        r = cov / np.sqrt(va * vb)
    r[~np.isfinite(r)] = -np.inf
    return r, n


def texture_shift(ref: np.ndarray, ours: np.ndarray, search: int = 15) -> dict:
    """Integer (dy, dx) such that np.roll(resize(ref), (dy, dx)) best matches `ours`.

    ref, ours: 2D texture images (e.g. depth means). ref is resized to ours.shape first.
    Pearson correlation of high-passed images over [-search, search]^2 inside the overlap of both images'
    nonzero support, evaluated for all shifts at once by FFT (see masked_shift_correlation). Same score and
    ranking as texture_shift_bruteforce(step=1); r and r_second use every overlap pixel.
    """
    X, Y, m = _prepare(ref, ours)
    r, n = masked_shift_correlation(X, Y, m, search)
    return _rank(r, search, n >= MIN_OVERLAP)


def texture_shift_bruteforce(ref: np.ndarray, ours: np.ndarray, search: int = 15, step: int = 5) -> dict:
    """Reference implementation of texture_shift: one np.roll + np.corrcoef per shift.

    `step` subsamples the overlap pixels for speed. Kept for tests and for cross-checking; it is about
    two orders of magnitude slower than texture_shift on full-size segments.
    """
    X, Y, m = _prepare(ref, ours)
    scores = []
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            xx = np.roll(X, (dy, dx), (0, 1))
            mm = m & np.roll(m, (dy, dx), (0, 1))
            a, b = xx[mm][::step], Y[mm][::step]
            if a.size < MIN_OVERLAP:
                continue
            scores.append((float(np.corrcoef(a, b)[0, 1]), dy, dx))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    on_edge = abs(best[1]) == search or abs(best[2]) == search
    return {"dy": best[1], "dx": best[2], "r": best[0], "r_second": second[0], "on_search_edge": on_edge}


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """np.roll with the wrapped border zeroed (no ink wraps around)."""
    out = np.roll(img, (dy, dx), (0, 1))
    if dy > 0:
        out[:dy] = 0
    elif dy < 0:
        out[dy:] = 0
    if dx > 0:
        out[:, :dx] = 0
    elif dx < 0:
        out[:, dx:] = 0
    return out
