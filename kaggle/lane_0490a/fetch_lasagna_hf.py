"""fetch_lasagna_hf.py SCAN Z0 Z1 DST : level-2 nx/ny/grad_mag chunks of SharpMeow/lasagna-fields for z in [Z0, Z1] (+-1 chunk)."""
import sys, os, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
S, Z0, Z1, DST = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
H = {"PHerc0490A": "20250521151210-lasagna-de45f011"}
B = f"https://huggingface.co/buckets/SharpMeow/lasagna-fields/resolve/{S}/representations/predictions/lasagna/{H[S]}"
def get(url, path, ok404=False):
    if os.path.exists(path): return 1
    for t in range(12):
        try:
            d = urllib.request.urlopen(url, timeout=60).read()
            os.makedirs(os.path.dirname(path), exist_ok=True); open(path + ".part", "wb").write(d); os.replace(path + ".part", path); return 1
        except urllib.error.HTTPError as e:
            if e.code == 404 and ok404: return 0
            time.sleep(min(60, 2 ** t))
        except Exception: time.sleep(min(60, 2 ** t))
    raise RuntimeError("failed " + url)
n = 0
for ch in ("nx", "ny", "grad_mag"):
    src = f"{B}/{S}_{ch}.ome.zarr"; dst = f"{DST}/las_008_{ch}.ome.zarr"
    for f in (".zattrs", ".zgroup", "2/.zarray"): get(f"{src}/{f}", f"{dst}/{f}", ok404=(f == ".zgroup"))
    a = json.load(open(f"{dst}/2/.zarray")); cz, cy, cx = a["chunks"]; _, Y, X = a["shape"]
    zs = range(max(0, Z0 // 4 // cz - 1), Z1 // 4 // cz + 2)
    keys = [(z, y, x) for z in zs for y in range(-(-Y // cy)) for x in range(-(-X // cx))]
    with ThreadPoolExecutor(8) as ex:
        got = sum(ex.map(lambda k: get(f"{src}/2/{k[0]}/{k[1]}/{k[2]}", f"{dst}/2/{k[0]}/{k[1]}/{k[2]}", ok404=True), keys))
    n += got; print(ch, "chunks", got, "of", len(keys), flush=True)
print("lasagna chunks", n)
