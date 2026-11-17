import numpy as np
import pytest

from vpipe import consensus


def sheet(h, w, z, step=20.0, oy=0.0):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([c * step, oy + r * step, np.full_like(r, z)], -1), np.ones((h, w), bool)


def test_agreeing_runs_trust_each_other_and_jumps_are_not_trusted():
    ref = sheet(20, 20, 100.0)
    a = sheet(20, 20, 100.0)
    ax, av = a
    ax = ax.copy()
    ax[:5, :, 2] = 112.0              # top rows of A jumped to the next wrap (12 vx away)
    b = sheet(20, 20, 100.5)          # independent run on the right sheet
    t = consensus.agreement((ax, av), [b], tol=4)
    assert t[5:].all() and not t[:5].any()
    s = consensus.score_against_reference((ax, av), t, ref)
    assert s["precision_of_trusted"] == 1.0
    assert s["recall_of_on_sheet"] == pytest.approx(1.0)
    assert s["all_vertices_on_sheet"] == pytest.approx(0.75)


def test_no_overlap_means_no_trust():
    a = sheet(10, 10, 100.0)
    far = sheet(10, 10, 100.0, oy=10_000.0)
    assert not consensus.agreement(a, [far], tol=4).any()


def test_min_votes():
    a = sheet(10, 10, 100.0)
    b = sheet(10, 10, 101.0)
    c = sheet(10, 10, 140.0)
    assert consensus.agreement(a, [b, c], tol=4, min_votes=1).all()
    assert not consensus.agreement(a, [b, c], tol=4, min_votes=2).any()
