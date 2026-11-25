"""Write kaggle/hecate_heldout.ipynb: score the Hecate 9.6 um ink model on held-out PHerc0841 segments on a Kaggle GPU.

Everything is fetched inside the notebook from public sources (anonymous S3 open-data bucket, the scrollprize/hecate
Hugging Face repo, the MIT-licensed PHerc 343 First Letters repo, this repo); every downloaded file with a published
hash is checked. Results are printed and written to /kaggle/working/hecate_results.json.

    python kaggle/make_hecate_notebook.py
"""
import json
from pathlib import Path

CELLS = [
    ("markdown", """# Hecate on held-out PHerc0841 segments (vesuvius-pipeline bake-off)

Scores the public **Hecate 9.6 um** ink model (scrollprize/hecate) on three labelled PHerc0841 segments that are not
in the public 9 um models' training set, with the same metric and label alignment as
[vesuvius-pipeline](https://github.com/bristleconelab/vesuvius-pipeline) `scripts/bakeoff_hecate.sh`. Only the
supervision-mask bounding box (+128 px) is run. Needs: GPU accelerator, internet on."""),
    ("code", """import subprocess, sys, os, hashlib, json, glob, time, shutil, urllib.request
def sh(cmd):
    print('$', cmd); r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:]); r.check_returncode(); return r.stdout
sh('pip install -q "zarr>=3" tifffile imagecodecs awscli')
if not os.path.isdir('/kaggle/working/vp'):
    sh('git clone -q --depth 1 https://github.com/bristleconelab/vesuvius-pipeline /kaggle/working/vp')
sys.path.insert(0, '/kaggle/working/vp/src')
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"""),
    ("code", """W = '/kaggle/working/hec'; os.makedirs(W, exist_ok=True)
FILES = {  # file: (url, sha256)
 'hecate.py': ('https://huggingface.co/scrollprize/hecate/resolve/main/hecate.py',
               'c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59'),
 'hecate_9.6um.pth': ('https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth',
               '809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d'),
 'hecate_prep.py': ('https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/e8370f1/ink/scripts/hecate_prep.py',
               '8249994e6da8d29a1c4005c518450456f570642da03531f10a874ee97f2e4a5a'),
}
for f, (u, want) in FILES.items():
    p = f'{W}/{f}'
    if not os.path.exists(p): urllib.request.urlretrieve(u, p)
    got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert got == want, (f, got)
    print('sha256 OK', f)"""),
    ("code", """B = 's3://vesuvius-challenge-open-data/PHerc0841/segments'
SV9 = '9.366um-1.2m-113keV-volume-20250821151531.zarr'
SV24 = '2.403um-0.22m-77keV-volume-20260319124803.zarr'
SEGS = {  # name: segment id (labels on the 2.403 um volume grid)
 '0841-w00': '20260220213127-w00',
 '0841-ag1': '20260220214732-auto_grown_20260220144552896',
 '0841-ag2': '20260221022814-auto_grown_20260220174252405',
}
def s3(src, dst): sh(f'aws s3 sync --no-sign-request --only-show-errors {src} {dst}')
def s3cp(src, dst): subprocess.run(f'aws s3 cp --no-sign-request --only-show-errors {src} {dst}', shell=True)"""),
    ("code", """import numpy as np, tifffile, zarr
from vpipe import crop, evaluate
RES = {}
for name, seg in SEGS.items():
    D = f'/kaggle/working/data/{name}'
    s3(f'{B}/{seg}/surface-volumes/{SV9}/0', f'{D}/sv9.zarr/0')
    for m in ('.zattrs', '.zgroup', 'zarr.json'): s3cp(f'{B}/{seg}/surface-volumes/{SV9}/{m}', f'{D}/sv9.zarr/{m}')
    s3(f'{B}/{seg}/surface-volumes/{SV24}/2', f'{D}/sv24.zarr/2')
    for m in ('.zattrs', '.zgroup', 'zarr.json'): s3cp(f'{B}/{seg}/surface-volumes/{SV24}/{m}', f'{D}/sv24.zarr/{m}')
    s3(f'{B}/{seg}/ink-labels/2.403um-volume-20260319124803', f'{D}/labels')
    lab = sorted(glob.glob(f'{D}/labels/*/inklabels.zarr'))[0].rsplit('/', 1)[0]
    shape = tuple(zarr.open(f'{D}/sv9.zarr', mode='r')['0'].shape[1:])
    y0, x0, h, w = crop.mask_bbox(np.asarray(zarr.open(f'{lab}/supervision.zarr', mode='r')['0'][...]), shape, 128)
    print(name, 'render', shape, 'crop', y0, x0, h, w)
    for direction in ('reverse', 'forward'):
        T = f'{D}/{direction}'; os.makedirs(T, exist_ok=True)
        t = time.time()
        sh(f"cd {T} && python {W}/hecate_prep.py --src {D}/sv9.zarr --out {T}/prep.zarr --native-um 9.366 --native-um-z 9.366 "
           f"--target-um 9.6 --z-center 13.5 --nz 16 --crop {y0} {x0} {h} {w}")
        sh(f"cd {T} && python {W}/hecate.py --checkpoint {W}/hecate_9.6um.pth --input {T}/prep.zarr --spacing-um 9.6 "
           f"--output {T}/out.png --device cuda --batch-size 16" + (' --reverse' if direction == 'reverse' else ''))
        dt = time.time() - t
        from PIL import Image
        tifffile.imwrite(f'{T}/ink.tif', crop.paste(np.asarray(Image.open(f'{T}/out.png')), shape, y0, x0, h, w))
        r = evaluate.eval_ink(f'{T}/ink.tif', f'{lab}/inklabels.zarr', None, None, mask=f'{lab}/supervision.zarr',
                              align_ref=f'{D}/sv24.zarr', align_ref_level=2, align_ours=f'{D}/sv9.zarr')
        m = r['labels']['metrics_in_mask']
        RES[f'{name} {direction}'] = {'roc_auc': m['roc_auc'], 'average_precision': m['average_precision'],
                                      'best_f05': m['best_f05'], 'seconds': round(dt), 'crop': [y0, x0, h, w],
                                      'alignment': r['labels']['texture_alignment']}
        print('RESULT', name, direction, 'AUC %.4f AP %.4f F05 %.4f  %ds' % (m['roc_auc'], m['average_precision'], m['best_f05'], dt))
        json.dump(RES, open('/kaggle/working/hecate_results.json', 'w'), indent=1)
    shutil.rmtree(D)
print(json.dumps(RES, indent=1))"""),
]


def main() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = Path(__file__).with_name("hecate_heldout.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
