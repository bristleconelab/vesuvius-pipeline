import numpy as np
import pytest

from vpipe import refine


def ct_with_sheet(z_sheet=40.0, z_next=56.0, shape=(80, 64, 64)):
    """Bright 6-voxel-thick sheet at z_sheet plus a fainter neighbouring sheet, mild noise."""
    z = np.arange(shape[0])[:, None, None]
    vol = 40 + 60 * np.exp(-((z - z_sheet) / 2.5) ** 2) + 45 * np.exp(-((z - z_next) / 2.5) ** 2)
    vol = np.broadcast_to(vol, shape).astype(np.float32).copy()
    vol += np.random.default_rng(0).normal(0, 2, shape).astype(np.float32)
    return vol


def flat_grid(z, h=8, w=8, step=6.0, o=10.0):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    xyz = np.stack([o + c * step, o + r * step, np.full_like(r, z)], -1)
    return xyz, np.ones((h, w), bool)


def test_refine_snaps_to_sheet_without_jumping_wraps():
    vol = ct_with_sheet()
    samp = refine.TiledSampler(vol)
    xyz, v = flat_grid(43.5)                       # 3.5 voxels off the sheet, toward the neighbour
    out, log = refine.refine(xyz, v, samp, radius=5, iterations=3)
    assert np.median(np.abs(out[v][:, 2] - 40.0)) < 0.6
    assert log[-1]["median_abs_shift"] < log[0]["median_abs_shift"]
    far, v2 = flat_grid(47.0)                      # beyond half the spacing: must not be pulled 7 voxels back
    out2, _ = refine.refine(far, v2, samp, radius=5, iterations=1)
    assert np.all(np.abs(out2[v2][:, 2] - 47.0) <= 5.0 + 1e-6)
    # a profile whose maximum sits on the window edge has no peak inside: the vertex must not move
    edge = refine.profile_offsets(samp, np.array([[30.0, 30.0, 47.0]]), np.array([[0.0, 0.0, 1.0]]), 3)
    assert np.isnan(edge[0])


def test_tiled_matches_untiled_on_small_grid():
    vol = ct_with_sheet()
    xyz, v = flat_grid(42.0)
    a, _ = refine.refine(xyz, v, refine.TiledSampler(vol), radius=5, iterations=2)
    b, _ = refine.refine_tiled(xyz, v, vol, tile=64, radius=5, iterations=2)
    assert np.allclose(a, b)


def test_nan_gaussian_ignores_nans():
    a = np.ones((5, 5))
    a[2, 2] = np.nan
    s = refine.nan_gaussian(a, 1.0)
    assert np.isnan(s[2, 2]) and np.allclose(s[np.isfinite(s)], 1.0)


def test_profile_offsets_flat_profile_is_nan():
    flat = np.full((20, 20, 20), 7.0, np.float32)
    samp = refine.TiledSampler(flat)
    off = refine.profile_offsets(samp, np.array([[10.0, 10.0, 10.0]]), np.array([[0.0, 0.0, 1.0]]), 4)
    assert np.isnan(off[0])


def test_mirror_array_assembles_chunks_and_caches(tmp_path):
    import json

    vol = (np.arange(20 * 18 * 16) % 251).astype(np.uint8).reshape(20, 18, 16)
    cs = (8, 8, 8)
    store, calls = {}, []
    store["v/0/.zarray"] = json.dumps({"shape": vol.shape, "chunks": cs, "dtype": "|u1", "fill_value": 0,
                                       "order": "C", "filters": None, "compressor": None,
                                       "dimension_separator": "/", "zarr_format": 2}).encode()
    for a in range(3):
        for b in range(3):
            for c in range(2):
                ch = np.zeros(cs, np.uint8)
                part = vol[a*8:(a+1)*8, b*8:(b+1)*8, c*8:(c+1)*8]
                ch[:part.shape[0], :part.shape[1], :part.shape[2]] = part
                if (a, b, c) != (2, 2, 1):            # one chunk missing on the server -> fill value
                    store[f"v/0/{a}/{b}/{c}"] = ch.tobytes()

    def fetch(url):
        calls.append(url)
        return store.get(url)

    arr = refine.MirrorArray("v", "0", tmp_path, fetch=fetch)
    got = arr[3:19, 2:17, 1:15]
    want = vol[3:19, 2:17, 1:15].copy()
    want[16 - 3:, 16 - 2:, 8 - 1:] = 0
    assert np.array_equal(got, want)
    n = len(calls)
    assert np.array_equal(arr[3:19, 2:17, 1:15], got) and len(calls) == n   # second read: no fetches


def test_mirror_array_reads_vc3d_cache_first(tmp_path):
    import json

    cs = (4, 4, 4)
    meta = json.dumps({"shape": [8, 8, 8], "chunks": cs, "dtype": "|u1", "fill_value": 0, "order": "C",
                       "filters": None, "compressor": None, "dimension_separator": "/", "zarr_format": 2}).encode()
    calls = []

    def fetch(url):
        calls.append(url)
        return meta if url.endswith(".zarray") else np.full(cs, 9, np.uint8).tobytes()

    ro = tmp_path / "vc3d"
    (ro / "level_0" / "0" / "0").mkdir(parents=True)
    (ro / "level_0" / "0" / "0" / "0.bin").write_bytes(np.full(cs, 5, np.uint8).tobytes())
    arr = refine.MirrorArray("v", "0", tmp_path / "m", fetch=fetch, read_only_dirs=[ro])
    a = arr[0:4, 0:4, 0:8]
    assert (a[:, :, :4] == 5).all() and (a[:, :, 4:] == 9).all()
    assert not any(u.endswith("/0/0/0") for u in calls)          # chunk (0,0,0) came from the VC3D cache


def test_mirror_array_after_prefetch_does_not_fetch_more(tmp_path):
    import json

    cs = (4, 4, 4)
    meta = json.dumps({"shape": [8, 8, 8], "chunks": cs, "dtype": "|u1", "fill_value": 0, "order": "C",
                       "filters": None, "compressor": None, "dimension_separator": "/", "zarr_format": 2}).encode()
    calls = []

    def fetch(url):
        calls.append(url)
        return meta if url.endswith(".zarray") else np.full(cs, 3, np.uint8).tobytes()

    arr = refine.MirrorArray("v", "0", tmp_path, fetch=fetch)
    assert arr.prefetch([(0, 0, 0)]) == 1
    n = len(calls)
    a = arr[0:8, 0:8, 0:8]
    assert len(calls) == n                       # nothing new fetched
    assert (a[:4, :4, :4] == 3).all() and (a[4:] == 0).all()
