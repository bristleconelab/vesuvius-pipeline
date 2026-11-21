
# ===================================================================================================== hx0358top
# The HerculeX layer stacks of the two largest PHerc0358 mosaics (m001, m002) from vesuvius-herculex-ink-0358 v1
# (stacks_top.tar, HerculeX 9f59e966, MIT, adambetka2), re-read with ink-sdk in BOTH depth directions.
# Stack layout (HerculeX ink/reading/piece_reading.py keep()/kept_face()): one file per piece,
#   <scroll>/<scan>/<sampling>/<unit>/piece-<id>.zst = struct 'III' (17, H, W) + zstd(uint8 C-order), in the piece image
#   frame (2 margin cells of `stride` px around the tifxyz grid). The 17 layers are window(cube) of a 31-layer stack
#   sampled along normal * turn (towards the scroll centre), reversed first when HerculeX chose the 'far' written face;
#   the window is centred on the intensity fall (layer 8 = the face). Layer order = the order HerculeX fed Reader v3.
# Here: ink-sdk 'forward' = that stored order (HerculeX's read), 'reverse' = the flipped stack. Pieces are packed into
# a few canvases (zarr v2, 17 layers, gaps of zeros) so each model loads once per canvas; triage windows must lie on one
# piece. Phase 1: Reader V3, Argus V2, TRACE v3B on every canvas. Phase 2: Studio + Reader V3 on crops around the top
# windows of both directions (Studio on whole canvases would take ~ 340 GPU-s/Mpx), then Studio x Reader V3 (ink-sdk
# combine). A-hn / v8-in need 24 layers and cannot run on 17-layer stacks.
import math, struct, tarfile

HX = dict(pack_mpx=110.0, gap=64, ct_layers=(5, 9), studio_top=10, crop_margin=128, win=384, step=192)
AN = T / 'an'


def find_input(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    return Path(hits[0]) if hits else None


def tar_members(path, want):
    """Yield (member name, bytes) of an uncompressed tar for names accepted by want(name), streaming once."""
    with tarfile.open(path) as tar:
        for m in tar:
            if m.isfile() and want(m.name):
                yield m.name, tar.extractfile(m).read()


def zst_stack(raw):
    import zstandard
    shape = struct.unpack_from('III', raw)
    return np.frombuffer(zstandard.ZstdDecompressor().decompress(memoryview(raw)[12:], max_output_size=int(np.prod(shape))),
                         dtype=np.uint8).reshape(shape)


def piece_key(name):
    p = Path(name)
    return p.parent.name, int(p.stem.split('-')[1])


def shelf_pack(sizes, width, gap):
    """Shelf packing of (h, w) boxes, tallest first. Returns {i: (y, x)} and the canvas (H, W)."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0])
    pos, y, x, shelf, W = {}, 0, 0, 0, 0
    for i in order:
        h, w = sizes[i]
        if x and x + w > width:
            y, x, shelf = y + shelf + gap, 0, 0
        pos[i] = (y, x)
        x += w + w % 2 + gap                      # even offsets keep the 2x grids aligned
        shelf = max(shelf, h + h % 2)
        W = max(W, x - gap)
    return pos, (y + shelf, W)


def stage_hx():
    src = find_input('stacks_top.tar')
    if src is None:
        sh('find /kaggle/input -maxdepth 4 | head -60', check=False)
        raise SystemExit('stacks_top.tar not found: vesuvius-herculex-ink-0358 must be its v1 output (v2 keeps no stacks)')
    sh(f'{sys.executable} -m pip install -q zstandard', check=False)
    level, faces_tar, mos_json = find_input('level.tar'), find_input('faces.tar'), find_input('mosaics.json')
    mosaic_of = {}
    if mos_json:
        for m in json.loads(mos_json.read_text()):
            for u, k in m['pieces']:
                mosaic_of[(u, int(k))] = m['rank']
    AN.mkdir(parents=True, exist_ok=True)
    # pass 1: shapes only (headers), to plan the canvases
    pieces = []
    for name, raw in tar_members(src, lambda n: n.endswith('.zst')):
        D, H, W = struct.unpack_from('III', raw)
        u, k = piece_key(name)
        pieces.append(dict(name=name, unit=u, piece=k, shape=(D, H, W), mosaic=mosaic_of.get((u, k))))
    if not pieces:
        raise SystemExit(f'no .zst stacks in {src}')
    note(stage='stacks', n=len(pieces), mpx=round(sum(p['shape'][1] * p['shape'][2] for p in pieces) / 1e6, 1),
         layers=sorted({p['shape'][0] for p in pieces}))
    # side records, sheet masks and HerculeX's own Reader v3 maps (faces.tar), tifxyz + layout (level.tar)
    keys = {(p['unit'], p['piece']) for p in pieces}
    side, sheet, taught, xyz, layout = {}, {}, {}, {}, {}
    from PIL import Image
    import io
    if faces_tar:
        for name, raw in tar_members(faces_tar, lambda n: '/faces/level-0/' in n):
            u = Path(name).parent.name
            stem = Path(name).name
            try:
                k = int(stem.split('-')[1].split('.')[0])
            except (IndexError, ValueError):
                continue
            if (u, k) not in keys:
                continue
            if stem.endswith('-taught-side.json'):
                side[(u, k)] = json.loads(raw)
            elif stem.endswith('-taught.png'):
                taught[(u, k)] = np.asarray(Image.open(io.BytesIO(raw)))
            elif stem == f'piece-{k:04d}.png':
                sheet[(u, k)] = np.asarray(Image.open(io.BytesIO(raw))) > 0
    if level:
        import tifffile
        def wanted(n):
            if n.endswith('sheet/layout.json'):
                return True
            q = Path(n).parts
            if len(q) < 4 or q[-4] != 'pieces' or q[-1] not in ('x.tif', 'y.tif', 'z.tif'):
                return False
            try:
                return (q[-3], int(q[-2].split('-')[1])) in keys
            except (IndexError, ValueError):
                return False
        for name, raw in tar_members(level, wanted):
            if name.endswith('sheet/layout.json'):
                for p in json.loads(raw)['pieces']:
                    layout[(p['unit'], int(p['piece']))] = p
                continue
            parts = Path(name).parts
            if len(parts) < 3 or parts[-1] not in ('x.tif', 'y.tif', 'z.tif'):
                continue
            try:
                key = (parts[-3], int(parts[-2].split('-')[1]))
            except (IndexError, ValueError):
                continue
            if key in keys:
                xyz.setdefault(key, {})[parts[-1][0]] = tifffile.imread(io.BytesIO(raw))
    note(stage='side-data', sides=len(side), sheets=len(sheet), taught=len(taught), xyz=len(xyz), layout=len(layout),
         written={w: sum(1 for s in side.values() if s.get('written') == w) for w in ('near', 'far')})
    # canvases
    total = sum(p['shape'][1] * p['shape'][2] for p in pieces) / 1e6
    n = max(2, int(math.ceil(total / HX['pack_mpx'])))
    bins = [[] for _ in range(n)]
    load = [0.0] * n
    for i in sorted(range(len(pieces)), key=lambda i: -pieces[i]['shape'][1] * pieces[i]['shape'][2]):
        b = int(np.argmin(load))
        bins[b].append(i)
        load[b] += pieces[i]['shape'][1] * pieces[i]['shape'][2]
    segs = {}
    for b, members in enumerate(bins):
        sizes = [pieces[i]['shape'][1:] for i in members]
        width = max(max(w for _, w in sizes), int(math.sqrt(sum(h * w for h, w in sizes) * 1.3)))
        pos, (Hc, Wc) = shelf_pack(sizes, width, HX['gap'])
        Hc, Wc = Hc + Hc % 2, Wc + Wc % 2
        for j, i in enumerate(members):
            pieces[i]['pack'], pieces[i]['yx'] = b, pos[j]
        segs[f'pack{b}'] = dict(sv=SV / f'pack{b}.zarr', valid_npy=None, shape=(17, Hc, Wc), surface=8, ref_dir=None,
                                members=members)
    import zarr
    arrays = {}
    for s, sd in segs.items():
        g = zarr.open_group(str(sd['sv']), mode='w', zarr_format=2)
        arrays[s] = g.create_array('0', shape=sd['shape'], dtype='uint8', chunks=(17, 128, 128))
        g.attrs['pixel_um'] = [CFG['voxel_um']] * 3
        H2, W2 = sd['shape'][1] // 2, sd['shape'][2] // 2
        sd.update(valid2=np.zeros((H2, W2), bool), owner2=np.full((H2, W2), -1, np.int32), ct2=np.zeros((H2, W2), np.float32),
                  taught2=np.full((H2, W2), np.nan, np.float32))
    by_name = {p['name']: i for i, p in enumerate(pieces)}
    for name, raw in tar_members(src, lambda n: n.endswith('.zst')):
        i = by_name[name]
        p = pieces[i]
        st = zst_stack(raw)
        if st.shape[0] != 17:
            raise SystemExit(f'{name}: {st.shape[0]} layers, expected 17')
        sd = segs[f"pack{p['pack']}"]
        y, x = p['yx']
        h, w = st.shape[1:]
        arrays[f"pack{p['pack']}"][:, y:y + h, x:x + w] = st
        key = (p['unit'], p['piece'])
        valid = sheet[key] if key in sheet and sheet[key].shape == (h, w) else st[8] > 0
        ctm = st[HX['ct_layers'][0]:HX['ct_layers'][1]].mean(0)
        y2, x2, h2, w2 = y // 2, x // 2, h // 2, w // 2
        v2 = pool_mask(valid, 2)[:h2, :w2]
        sd['valid2'][y2:y2 + h2, x2:x2 + w2] |= v2
        sd['owner2'][y2:y2 + h2, x2:x2 + w2][v2] = i
        sd['ct2'][y2:y2 + h2, x2:x2 + w2] = np.where(v2, pool(ctm, 2)[:h2, :w2], sd['ct2'][y2:y2 + h2, x2:x2 + w2])
        if key in taught and taught[key].shape == (h, w):
            t2 = pool(taught[key].astype(np.float32) / 255, 2)[:h2, :w2]
            sd['taught2'][y2:y2 + h2, x2:x2 + w2] = np.where(v2, t2, sd['taught2'][y2:y2 + h2, x2:x2 + w2])
        p['written'] = side.get(key, {}).get('written')
        lp = layout.get(key, {})
        p['stride'] = lp.get('stride')
        p['flip_rows'], p['flip_cols'] = bool(lp.get('flip_rows', False)), bool(lp.get('flip_cols', False))
        if key in xyz and len(xyz[key]) == 3:
            p['xyz'] = np.stack([xyz[key][a] for a in 'xyz'])
        del st
    for s, sd in segs.items():
        note(stage='pack', seg=s, shape=list(sd['shape']), pieces=len(sd['members']), valid_mpx=round(4 * sd['valid2'].sum() / 1e6, 1))
    return segs, pieces


def xyz_of(p, r, c):
    """Scan xyz (level-0 voxels) of full-resolution piece-frame pixel (r, c), after HerculeX's post xyz_at."""
    g, s = p.get('xyz'), p.get('stride')
    if g is None or not s:
        return None
    gi = int(np.clip(round((r - 2 * s) / s), 0, g.shape[1] - 1))
    gj = int(np.clip(round((c - 2 * s) / s), 0, g.shape[2] - 1))
    if g[0, gi, gj] < 0:
        ok = np.argwhere(g[0] >= 0)
        if not len(ok):
            return None
        gi, gj = ok[np.argmin(((ok - [gi, gj]) ** 2).sum(1))]
    return [round(float(v), 1) for v in g[:, gi, gj]]


def post_pack(seg, sd, status, final):
    """Phase 1: every map of the canvas pooled to 2x and kept (float16) for the global analysis."""
    try:
        d = AN / seg
        d.mkdir(parents=True, exist_ok=True)
        for (s, k), r in status.items():
            if s != seg or r.get('state') != 'done':
                continue
            for side, fname in SIDES.items():
                f = RUNS / seg / k / fname
                out = d / f'{k}_{side}.npy'
                if f.exists() and not out.exists():
                    m = np.load(f).astype(np.float32)
                    np.save(out, pool(m, 2)[:sd['valid2'].shape[0], :sd['valid2'].shape[1]].astype(np.float16))
        note(stage='post-pack', seg=seg)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        note(stage='post-failed', seg=seg, error=repr(exc))


def stroke_ratio(m, valid):
    """Stroke-scale over blob-scale structure of a 2x map (18.7 um px): std of a sigma 1.5 - 6 px band (strokes of
    ~0.1-0.4 mm) over std of the sigma 12 px low-pass (mm-scale blobs), inside valid."""
    from scipy import ndimage as ni
    if valid.sum() < 1000:
        return None
    f = np.where(valid, m, np.nanmedian(m[valid])).astype(np.float32)
    band = ni.gaussian_filter(f, 1.5) - ni.gaussian_filter(f, 6)
    low = ni.gaussian_filter(f, 12)
    inner = ni.binary_erosion(valid, iterations=12)
    if inner.sum() < 1000:
        return None
    return float(band[inner].std() / max(low[inner].std(), 1e-6))


def global_analysis(segs, pieces, keys):
    """z-mean per direction with statistics pooled over all canvases, stretches, triage, per-piece metrics, sheets."""
    from ink_sdk.ensemble import map_values
    maps = {s: {} for s in segs}
    for s in segs:
        for f in (AN / s).glob('*.npy'):
            maps[s][f.stem] = f
    members = [k for k in ('reader-v3', 'argus-v2', 'trace') if k in keys]
    stats = {}
    for k in members:
        for side in SIDES:
            vals = []
            for s, sd in segs.items():
                if f'{k}_{side}' in maps[s]:
                    m = map_values(np.load(maps[s][f'{k}_{side}']).astype(np.float32), 'ink9um')
                    vals.append(m[sd['valid2']][::7])
            if vals:
                v = np.concatenate(vals)
                stats[f'{k}_{side}'] = (float(v.mean()), float(v.std()))
    small = {s: {} for s in segs}
    for s, sd in segs.items():
        for side in SIDES:
            zs = []
            for k in members:
                name = f'{k}_{side}'
                if name not in maps[s]:
                    continue
                raw = np.load(maps[s][name]).astype(np.float32)
                small[s][name] = np.where(sd['valid2'], raw, np.nan)
                mu, sdv = stats[name]
                zs.append((map_values(raw, 'ink9um') - mu) / max(sdv, 1e-6))
            if len(zs) >= 2:
                small[s][f'zmean_{side}'] = np.where(sd['valid2'], np.mean(zs, 0), np.nan).astype(np.float32)
    names = sorted({n for s in small for n in small[s]})
    lohi = {}
    for n in names:
        v = np.concatenate([small[s][n][segs[s]['valid2']][::5] for s in segs if n in small[s]])
        lohi[n] = tuple(np.percentile(v, [1, 99.5]))
    normd = {s: {n: np.where(segs[s]['valid2'], np.clip((small[s][n] - lohi[n][0]) / max(lohi[n][1] - lohi[n][0], 1e-9), 0, 1),
                             np.nan).astype(np.float32) for n in small[s]} for s in segs}
    norm = lambda s, n: normd[s][n]  # noqa: E731
    summary = dict(target=TARGET, pieces=len(pieces), packs={s: list(sd['shape']) for s, sd in segs.items()}, members=members,
                   stretch={n: [float(a) for a in lohi[n]] for n in names}, maps={}, pieces_metrics=[], triage={})
    for s, sd in segs.items():
        for n in small[s]:
            summary['maps'].setdefault(n, []).append(dict(pack=s) | map_stats(norm(s, n), sd['valid2']))
    # per piece: stroke structure of each map and agreement with HerculeX's own Reader v3 map (forward should match it)
    for i, p in enumerate(pieces):
        sd = segs[f"pack{p['pack']}"]
        y2, x2 = p['yx'][0] // 2, p['yx'][1] // 2
        h2, w2 = p['shape'][1] // 2, p['shape'][2] // 2
        sl = (slice(y2, y2 + h2), slice(x2, x2 + w2))
        own = sd['owner2'][sl] == i
        rec = dict(unit=p['unit'], piece=p['piece'], mosaic=p['mosaic'], written=p.get('written'), pack=p['pack'],
                   shape=list(p['shape']), valid_mpx=round(4 * own.sum() / 1e6, 2), stroke_ratio={}, r_vs_herculex={})
        hxm = sd['taught2'][sl]
        for n in small[f"pack{p['pack']}"]:
            m = small[f"pack{p['pack']}"][n][sl]
            rec['stroke_ratio'][n] = stroke_ratio(m, own)
            rec['r_vs_herculex'][n] = pearson(m, hxm, own)
        summary['pieces_metrics'].append(rec)
    med = {}
    for n in names:
        vals = [r['stroke_ratio'].get(n) for r in summary['pieces_metrics'] if r['stroke_ratio'].get(n) is not None]
        rs = [r['r_vs_herculex'].get(n) for r in summary['pieces_metrics'] if np.isfinite(r['r_vs_herculex'].get(n, np.nan))]
        med[n] = dict(stroke_ratio_median=float(np.median(vals)) if vals else None,
                      r_vs_herculex_median=float(np.median(rs)) if rs else None, n=len(vals))
    summary['direction_medians'] = med
    # triage per canvas, windows on one piece only (>= 95 % of the window's valid pixels owned by one piece)
    W, S = HX['win'], HX['step']
    tops = {}
    for n in names:
        allw = []
        for s, sd in segs.items():
            if n not in small[s]:
                continue
            nm = norm(s, n)
            for w in triage(nm, sd['valid2'], win=W, step=S, top=400):
                o = sd['owner2'][w['y']:w['y'] + w['h'], w['x']:w['x'] + w['w']]
                o = o[o >= 0]
                if not o.size:
                    continue
                ids, cnt = np.unique(o, return_counts=True)
                if cnt.max() < 0.95 * o.size:
                    continue
                i = int(ids[cnt.argmax()])
                p = pieces[i]
                r_full = 2 * (w['y'] + w['h'] // 2) - p['yx'][0]
                c_full = 2 * (w['x'] + w['w'] // 2) - p['yx'][1]
                other = n[:-3] + ('rev' if n.endswith('fwd') else 'fwd')
                osc = None
                if other in small[s]:
                    ov = sd['valid2'][w['y']:w['y'] + w['h'], w['x']:w['x'] + w['w']]
                    osc = window_score((norm(s, other)[w['y']:w['y'] + w['h'], w['x']:w['x'] + w['w']] > 0.6), ov)
                allw.append(w | dict(pack=s, piece_index=i, unit=p['unit'], piece=p['piece'], mosaic=p['mosaic'],
                                     written=p.get('written'), centre_piece_px_full=[int(r_full), int(c_full)],
                                     scan_xyz=xyz_of(p, r_full, c_full), score_other_direction=osc))
        allw.sort(key=lambda w: -w['score'])
        tops[n] = allw[:40]
        for k, w in enumerate(tops[n], 1):
            w['rank'] = k
        (OUT / f'triage_{n}.json').write_text(json.dumps(dict(map=n, grid='2x of the canvas', window=W, step=S, threshold=0.6,
                                                              windows=tops[n]), indent=1, default=float))
        summary['triage'][n] = dict(n=len(tops[n]), top_score=tops[n][0]['score'] if tops[n] else None,
                                    median_top20=float(np.median([w['score'] for w in tops[n][:20]])) if tops[n] else None)
    # contact sheets: CT | this direction | the other direction, for the ensembles and each model
    for n in names:
        if not tops.get(n):
            continue
        other = n[:-3] + ('rev' if n.endswith('fwd') else 'fwd')
        panels = [('CT', None), (n, n), (other, other)]
        hx_sheet(tops[n], segs, small, norm, pieces, panels, OUT / f'triage_{n}.jpg', f'{n} top {len(tops[n])}')
    # overview PNGs at 4x
    for s, sd in segs.items():
        png(to8(stretch(pool(np.where(sd['valid2'], sd['ct2'], np.nan), 2), pool_mask(sd['valid2'], 2))), OUT / f'{s}_ct_4x.png')
        for n in small[s]:
            png(to8(pool(norm(s, n), 2)), OUT / f'{s}_{n}_4x.png')
    return summary, tops, small, norm


def oriented(a, p):
    a = a[::-1] if p.get('flip_rows') else a
    return a[:, ::-1] if p.get('flip_cols') ^ (p.get('written') == 'far') else a


def hx_sheet(wins, segs, small, norm, pieces, panels, path, title, tile=256, cols=2):
    from PIL import Image, ImageDraw
    k = len(panels)
    pw, ph = k * (tile + 4) + 12, tile + 22
    rows = (len(wins) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * pw, rows * ph + 24), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), title + '  panels: ' + ' | '.join(n for n, _ in panels) + '  (written face up, as HerculeX shows it; 384 px at 2x = 7.2 mm)',
           fill=(230, 230, 230))
    for j, w in enumerate(wins):
        s, p = w['pack'], pieces[w['piece_index']]
        sd = segs[s]
        sl = (slice(w['y'], w['y'] + w['h']), slice(w['x'], w['x'] + w['w']))
        v = sd['valid2'][sl]
        ox, oy = (j % cols) * pw, 24 + (j // cols) * ph
        for q, (label, n) in enumerate(panels):
            if n is None:
                a = stretch(sd['ct2'][sl], v)
            elif callable(n):
                a = n(w)
            elif n in small[s]:
                a = norm(s, n)[sl]
            else:
                continue
            img = Image.fromarray(to8(oriented(a, p))).resize((tile, tile))
            sheet.paste(img, (ox + q * (tile + 4), oy + 18))
        xyz = w.get('scan_xyz')
        where = f"xyz {xyz[0]:.0f},{xyz[1]:.0f},{xyz[2]:.0f}" if xyz else 'xyz ?'
        osc = w.get('score_other_direction')
        d.text((ox + 2, oy + 2), f"#{w.get('rank', j + 1)} m{w.get('mosaic')} {w.get('written') or '?'} s={w['score']:.3f}"
               f" other={osc if osc is None else round(osc, 3)} {where}", fill=(255, 220, 120))
    sheet.save(path, quality=90)


def studio_phase(segs, pieces, tops, small, norm, summary):
    """Studio + Reader V3 on crops around the top windows of both z-mean directions, then Studio x Reader V3."""
    import zarr
    chosen = []
    for n in ('zmean_fwd', 'zmean_rev'):
        for w in tops.get(n, [])[:HX['studio_top']]:
            if all(not (w['pack'] == c['pack'] and abs(w['y'] - c['y']) < w['h'] / 2 and abs(w['x'] - c['x']) < w['w'] / 2)
                   for c in chosen):
                chosen.append(dict(w, from_map=n))
    if not chosen:
        return
    m = HX['crop_margin']
    side_px = 2 * HX['win'] + 2 * m
    half = (len(chosen) + 1) // 2
    groups = [chosen[:half], chosen[half:]] if len(chosen) > 1 else [chosen]
    ssegs = {}
    for g, crops in enumerate(groups):
        if not crops:
            continue
        cols = min(4, len(crops))
        rows = (len(crops) + cols - 1) // cols
        Hc, Wc = rows * (side_px + HX['gap']), cols * (side_px + HX['gap'])
        path = SV / f'studio{g}.zarr'
        grp = zarr.open_group(str(path), mode='w', zarr_format=2)
        arr = grp.create_array('0', shape=(17, Hc, Wc), dtype='uint8', chunks=(17, 128, 128))
        grp.attrs['pixel_um'] = [CFG['voxel_um']] * 3
        for j, c in enumerate(crops):
            src = open_render(segs[c['pack']]['sv'])
            y0, x0 = 2 * c['y'] - m, 2 * c['x'] - m
            Hs, Ws = src.shape[1:]
            a0, b0, a1, b1 = max(y0, 0), max(x0, 0), min(y0 + side_px, Hs), min(x0 + side_px, Ws)
            cy, cx = (j // cols) * (side_px + HX['gap']), (j % cols) * (side_px + HX['gap'])
            block = np.asarray(src[:, a0:a1, b0:b1])
            # keep only the window's own piece in the crop
            own = segs[c['pack']]['owner2'][a0 // 2:(a1 + 1) // 2, b0 // 2:(b1 + 1) // 2] == c['piece_index']
            own = np.repeat(np.repeat(own, 2, 0), 2, 1)[:a1 - a0, :b1 - b0]
            block = np.where(own[None], block, 0).astype(np.uint8)
            arr[:, cy + a0 - y0:cy + a1 - y0, cx + b0 - x0:cx + b1 - x0] = block
            c['studio_yx'] = (cy, cx)
        ssegs[f'studio{g}'] = dict(sv=path, valid_npy=None, shape=(17, Hc, Wc), surface=8, ref_dir=None, crops=crops)
    status = run_all(ssegs, [k for k in ('studio', 'reader-v3') if k in summary['keys']], post_fn=lambda *a: None)
    lohi, res = {}, []
    loaded = {}
    for g, sd in ssegs.items():
        runs = {k: RUNS / g / k for k in ('studio', 'reader-v3') if status.get((g, k), {}).get('state') == 'done'}
        if len(runs) == 2:
            for side in ('forward', 'reverse'):
                combine([runs['studio'], runs['reader-v3']], 'product', side, RUNS / g / 'prodSR')
            runs['prodSR'] = RUNS / g / 'prodSR'
        for k, d in runs.items():
            for side, fname in SIDES.items():
                if (d / fname).exists():
                    loaded[(g, f'{k}_{side}')] = pool(np.load(d / fname).astype(np.float32), 2)
    names = sorted({n for _, n in loaded})
    for n in names:
        v = np.concatenate([a[np.isfinite(a)][::5] for (g, nn), a in loaded.items() if nn == n])
        lohi[n] = tuple(np.percentile(v, [1, 99.5]))
    W = HX['win']
    for g, sd in ssegs.items():
        for c in sd['crops']:
            cy, cx = c['studio_yx']
            y2, x2 = (cy + m) // 2, (cx + m) // 2
            v = segs[c['pack']]['valid2'][c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']]
            c['studio'] = {}
            for n in names:
                if (g, n) not in loaded:
                    continue
                a = loaded[(g, n)][y2:y2 + c['h'], x2:x2 + c['w']]
                nm = np.clip((a - lohi[n][0]) / max(lohi[n][1] - lohi[n][0], 1e-9), 0, 1)
                nm = np.where(v[:a.shape[0], :a.shape[1]], nm, np.nan)
                c['studio'][n] = window_score(np.nan_to_num(nm) > 0.6, v[:a.shape[0], :a.shape[1]])
                c.setdefault('_crops', {})[n] = nm
            res.append(c)
    res.sort(key=lambda c: -max([s for s in c['studio'].values() if s is not None] or [0]))
    panels = [('CT', None)] + [(n, (lambda n: lambda w: w['_crops'].get(n, np.full((w['h'], w['w']), np.nan)))(n))
                               for n in ('prodSR_fwd', 'prodSR_rev', 'studio_fwd', 'studio_rev')]
    for k, c in enumerate(res, 1):
        c['rank'] = k
    hx_sheet(res, segs, small, norm, pieces, panels, OUT / 'studio_windows.jpg', 'Studio x Reader V3 on the top z-mean windows', cols=1)
    for c in res:
        c.pop('_crops', None)
    (OUT / 'studio_windows.json').write_text(json.dumps(dict(stretch={n: [float(a) for a in lohi[n]] for n in names}, windows=res),
                                                        indent=1, default=float))
    summary['studio'] = {f'{g}/{k}': {kk: vv for kk, vv in r.items() if kk != 'tail'} for (g, k), r in status.items()}


def main_hx():
    setup()
    keys = fetch(CFG['profiles'])
    segs, pieces = stage_hx()
    phase1 = [k for k in ('reader-v3', 'argus-v2', 'trace') if k in keys]
    status = run_all(segs, phase1, post_fn=post_pack)
    summary, tops, small, norm = global_analysis(segs, pieces, phase1)
    summary['keys'] = keys
    summary['jobs'] = {f'{s}/{k}': {kk: vv for kk, vv in r.items() if kk != 'tail'} for (s, k), r in status.items()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    if 'studio' in keys and hours() < CFG['no_start_after_h'] - 1.5:
        try:
            studio_phase(segs, pieces, tops, small, norm, summary)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            note(stage='studio-failed', error=repr(exc))
    summary['hours'] = round(hours(), 2)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    sh(f'du -sh {OUT}; ls -la {OUT}', check=False)
    print(json.dumps(dict(direction_medians=summary.get('direction_medians'), triage=summary.get('triage')), indent=1, default=float))


if __name__ == '__main__':
    main_hx()
