import numpy as np
import pytest

from vpipe import crop


def test_mask_bbox_same_grid_with_margin_and_clip():
    m = np.zeros((100, 80), np.uint8)
    m[20:30, 10:40] = 255
    assert crop.mask_bbox(m, (100, 80), margin=5) == (15, 5, 20, 40)
    assert crop.mask_bbox(m, (100, 80), margin=50) == (0, 0, 80, 80)


def test_mask_bbox_resizes_from_finer_grid():
    m = np.zeros((400, 320), np.uint8)
    m[80:120, 40:160] = 255                    # 4x finer than the render grid
    y0, x0, h, w = crop.mask_bbox(m, (100, 80), margin=0)
    assert abs(y0 - 20) <= 1 and abs(x0 - 10) <= 1 and abs(h - 10) <= 2 and abs(w - 30) <= 2


def test_mask_bbox_empty_raises():
    with pytest.raises(ValueError):
        crop.mask_bbox(np.zeros((10, 10)), (10, 10))


def test_paste_resizes_and_places():
    pred = np.full((5, 8), 200, np.uint8)      # model output on a coarser crop grid
    out = crop.paste(pred, (50, 60), 10, 20, 10, 16)
    assert out.shape == (50, 60) and out.dtype == np.uint8
    assert (out[10:20, 20:36] == 200).all()
    assert out[:10].sum() == 0 and out[:, :20].sum() == 0 and out[20:].sum() == 0
