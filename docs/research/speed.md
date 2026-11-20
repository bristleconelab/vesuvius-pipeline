# Speed and cost of the CPU chain: what helps, with evidence

Researched 2026-11-16. Pins: villa `2d3fb38d` (2026-11-14), VC3D AppImage `f7756c7` (2026-11-03, release `stable`).
Read-only: GitHub PRs/issues via `gh`, villa source at both commits (blobless clone), anonymous S3 metadata
listings. Nothing was run on the 16-vCPU test VM for this note. **F** = fact with a source, **I** = my inference.

## 0. Where the time goes (our measurements)

| Run | Render | Inference |
|---|---|---|
| w035 cold, container (`results/w035-e2e-1/timings.json`) | 688 s | 470 s |
| PHerc0139 w030, w045 (shared chunk cache) | 1574-1581 s | 242-654 s (range over all segments) |
| PHerc0841 segments (shared chunk cache) | 167-520 s | same range |

- **F** The 9.362 µm volume `20250728140407-9.362um-1.2m-113keV-masked.zarr/0/.zarray`: `chunks [128,128,128]`,
  `dtype |u1`, **`compressor: null`**. Every chunk is 2,097,152 bytes on the wire. The bucket is in `us-east-1`
  (`x-amz-bucket-region`).
- **F** w035 cold render: `remote_cache_bytes 10884384882` (`results/w035-e2e-1/streamed.txt`). The VC3D persistent cache
  stores raw chunks by default (`kRemoteCacheDelta3dDefault = false`, `RemoteCacheSettings.hpp` at f7756c7), so this is
  roughly the downloaded bytes: 10.88 GB / 2 MiB ≈ **5,190 chunks**, i.e. **15.8 MB/s and about 7.5 chunks/s** over 688 s.
  (The earlier "2.2-2.4 chunks/s" figure does not match this. Either it counted something else or the cache holds
  more than one fetch; worth rechecking.)
- **F** The w035 mesh `meta.json` gives `area_vx2 25082013` → 21.98 cm² at 9.362 µm. So we read **0.50 GB per cm²**.
  #2534 reports the same figure for its own minimal-read Python chain ("0.5 GB of reads per cm²"). **I:** we already
  fetch only the chunks the renderer needs (#1905 and #1657 are merged in f7756c7; checked with `git merge-base`).
  A smarter chunk *selection* will not save much. Fewer bytes come only from skipping the render or reading a
  different product.
- **F** the test VM measured 13.7 MB/s (Hetzner Ashburn) and 24.8 MB/s (OVH) earlier. 15.8 MB/s is inside that
  range. **I:** the render is link-bound. Render CPU work matters only on warm-cache reruns.
- **Conclusion:** render is 2.5-6x the inference time on the new segments. **Download bytes and download overlap are the
  main levers.** Inference tuning comes second.

## 1. Ranked recommendations (gain / effort)

| # | Change | Stage | Expected gain | Effort | Output changes? |
|---|---|---|---|---|---|
| 1 | Use the published 9 µm surface volume when one exists (screening mode) | render | w035: 10.9 GB → 0.745 GB (**14.6x fewer bytes**). Est. ~50 s instead of 688 s | script flag | no (render vs published already 99.73 % identical) |
| 2 | `--prefetch-remote` on `vc_render_tifxyz` | render | unknown. Bounded by link capacity (see test 2a) | 1 flag | no (same chunks, same sampling) |
| 3 | Overlap render of segment N+1 with inference of segment N | batch | up to (R+I)/max(R,I): **≈1.15-1.4x** throughput on PHerc0139-size segments | small script change | no |
| 4 | glibc heap env vars (same effect as #2452) + `--no-compile` A/B + `--loader-threads` | inference | #2452: **1.20-1.32x** (2-core Zen 3). `--no-compile`: unquantified | env vars + flags | heap: byte-identical (per #2452). no-compile: possibly last-bit changes |
| 5 | OpenVINO forward (#2453, unmerged) | inference | **1.13-1.29x** over the best torch setting | cherry-pick PR + 59 MB wheel | yes, ≤1 uint8 level |
| 6 | `--overlap 0.25` instead of 0.5 | inference | ~2.2x fewer patches (arithmetic) | 1 flag | yes, must re-score AUC |

Not recommended or not applicable: #2536, #2411, #2521, group-idx 1, rendering 17 instead of 28 slices,
`--pyramid false`, Delta3D cache. Reasons in §3.

## 2. Details per candidate

### 1. Skip the render when a published 9 µm surface volume exists

- **F** Published `surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr` level 0 is
  `shape [28, 5820, 5240]`, `chunks [28,128,128]`, uncompressed u8. Its sizes (S3 listing, level 0 only):
  - w035: 1,624 objects, **0.745 GB**
  - w045: 2,741 objects, **1.257 GB**
  - w030 (`20250108000005-w030_2025010818`): 2,851 objects, **1.307 GB**

  PHerc0841 w00 also has `9.366um-1.2m-113keV-volume-20250821151531.zarr`.
- **F** The model card (huggingface.co/scrollprize/ink_9um README) runs inference directly on that published
  zarr by URL: `infer https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/9.362um-...zarr <ckpt> w043.tif`.
- **F** Our render of w035 matches the published one: 99.73 % voxels identical, 100 % within 1 grey level
  (`results/w035_render.json`).
- **I** gain: at the measured 15.8 MB/s, 0.745 GB is ~47 s instead of 688 s for w035, and ~80 s for w045 instead of
  ~1,575 s.
- **When it is legitimate:**
  - Fine for **ink screening of published segments** (M5 held-out checks, M8 screening). There the question is
    "what does the checkpoint see on this segment", not "can we produce the surface volume".
  - Not legitimate for M2/M3/M6 (the reproducible "tifxyz → render" claim and the Grand-Prize-style automation story).
  - Not possible for segments with no published surface volume: new meshes we trace (M7), and the 22 raw
    PHerc1203 `auto_grown_*` meshes in #2534 ("no surface volume and no ink prediction for them").
  - Each output must record which input it used (rendered vs published) so screening results are not passed off as
    render reproductions.
- **Risk:**
  - Inference reads pyramid **level 3** for the occupancy scan (`DEFAULT_OCCUPANCY_SCAN_LEVEL = "3"`,
    `infer.py:56`, used at `infer.py:425-430`). Sync the whole zarr, or stream it by URL, not only `/0`. If level 3
    is missing it falls back to the nearest level with a warning (`infer.py:380-394`), and the patch set may differ.
  - The published surface volume and our render differ in 0.27 % of voxels by 1 level, so ink maps will not be
    byte-identical to a rendered run.
- **Command** (in `run_segment.sh`, behind e.g. `INPUT=published`). `$SV9` is already synced by the script:
  ```bash
  run python -m vesuvius.ink_detection.inference.infer "$RAW/surface-volumes/$SV9" \
    "$ROOT/checkpoints/ink_9um/$CKPT" "$RUN/ink.tif" --overlap 0.5 --blend-mode hann --batch-size 8 --device cpu
  ```
  The script currently runs `s3 "surface-volumes/$SV9"` anyway for evaluation, so the published input is already on
  disk. Only the render call needs to be skipped.

### 2. `vc_render_tifxyz --prefetch-remote` (in f7756c7, default off)

- **F** Help text: "With a remote --volume, fetch the chunks this render reads into the shared remote cache before
  rendering instead of streaming them during the render" (`vc_render_tifxyz.cpp:1183` at f7756c7).
- **F** With the flag, the tool collects every chunk key the render will sample (`collectPrefetchKeysForRows`) and
  queues them all at once. It logs `Prefetch: N chunk(s)` and waits (`vc_render_tifxyz.cpp:1928-1966`;
  `prefetchChunks(keys, true)` at :500).
- **F** Without it, the zarr-output path renders tile row by tile row: `#pragma omp parallel for` over the tiles of
  one row, each thread sampling its own tile, with an implicit barrier per row (`vc_render_tifxyz.cpp:757-772`).
- **F** The process-wide fetch scheduler has up to 64 workers with **adaptive** admission between 2 and 64
  (`defaultProcessChunkCacheServiceOptions`, `ChunkCache.cpp:758-763`; adaptive bounds :691-697). No CLI or INI key
  sets it (`RemoteCacheSettings.hpp` only has the cache dir, Delta3D and disk-budget keys).
- **I:** streaming mode can only have the chunk requests of one tile row (at most 16 threads) in flight, and pays a
  row barrier. One full queue lets the 64-worker scheduler fill the link. On a high-RTT path (a far client → us-east-1)
  this is the likeliest single render win. Not measured; the gain is capped by what the link can do (2a).
- **Bonus:** the `Prefetch: N chunk(s)` line is an exact per-segment download count (the estimate #2478 asks for).
- **Risk:** low. The same chunks are fetched and the sampling is unchanged. The output should be byte-identical:
  check it with `vpipe.evaluate render` against the previous render.
- **Command:** add `--prefetch-remote` to the render call. Time it cold with `VC3D_HOME=$RUN/home`.
- **2a. Link-capacity test first (2 min):** fetch ~200 known chunk keys with 1 vs 32 parallel connections, e.g.
  `aria2c -j32 -x1 -i urls.txt` (aria2 is installed on the test VM). Interpretation:
  - If 32 streams give much more than 15.8 MB/s, the render is under-using the link and #2 should help.
  - If they don't, the link is saturated: only #1 and #3 help, plus a host closer to us-east-1.

### 3. Pipeline overlap: render N+1 while inferring N

- **F** Render streams over the network; inference is CPU-bound (1161 % CPU, 2.1 GB RAM). Peak render RSS was 5.9 GB.
  Both fit in 62 GB.
- **I:** for PHerc0139 segments (render ~1,575 s, inference 242-654 s), a two-stage pipeline hides almost all of the
  inference. Throughput gain ≈ (R+I)/max(R,I) = 1.15-1.4x, plus #2 or #1 on top.
- **Open question before adopting:** how much CPU does the render use? "~75 % busy" is ambiguous. `/usr/bin/time -v`
  "Percent of CPU" 75 % means 0.75 core, so overlap is nearly free. 75 % of 16 cores would mean real contention.
  Read the render's `Percent of CPU` and `User time` from `run.log`.
- **Risk:** timings in a run are no longer clean per stage. Keep isolated cold runs for the published table.
- **Command:**
  - Split `run_segment.sh` into `render` and `infer` phases.
  - A driver runs `render(seg[i+1]) &` then `infer(seg[i])`, then `wait`.
  - Or, cheapest: `vc_render_tifxyz --prefetch-remote` for the next segment in the background. The fetch is
    network-only, then the CPU part is short.

### 4. Inference flags and environment (no code change)

- **F #2452 (open, unmerged, 2026-11-05):**
  - What it does: `mallopt(M_MMAP_MAX=0)` + 2 GiB trim threshold for CLI CPU runs.
  - Speed-up: "1.20x faster than the best setting without this PR, with byte-identical output". On the w035 crop at
    batch 4: 44.7 → 33.8 s (1.32x). 3-model ensemble: 1.22x.
  - Cost: "+0.3 GB peak RSS". Measured on 2-core and 4-core Zen 3 only.
  - Code: `ctypes.CDLL(None).mallopt(-4, 0)` and `mallopt(-1, 2**31-1)`.
- **I:** glibc reads the same two settings from the environment (`mallopt(3)`: `MALLOC_MMAP_MAX_`,
  `MALLOC_TRIM_THRESHOLD_`), so we can get the effect at our pin with no patch:
  `-e MALLOC_MMAP_MAX_=0 -e MALLOC_TRIM_THRESHOLD_=2147483647` on the inference `podman run`. Gain on 16 vCPU is
  unmeasured. Verify the TIFF is byte-identical to the previous run.
- **F `--no-compile`:** `compile_model` defaults to True on every device. CPU models do go through `torch.compile`
  with `mode="reduce-overhead"` (`inference_runtime.py:206-222`, `infer.py:1720-1722`). Our runs never passed
  `--no-compile`, so they ran compiled. #2453: "`torch.compile` is slower than eager on CPU" and "Tried and dropped:
  … `torch.compile` (slower than eager)". It gives no numbers. #2452's own command uses `--no-compile`.
  **I:** A/B it. It also removes the compile warm-up. The output may change in the last bits, so re-hash and re-score.
- **F `--amp-dtype`:** autocast is applied only when `device.type == "cuda"` (`infer.py:945-949`). This is why bf16 did
  nothing for us. No CPU AMP exists at this pin.
- **F `--loader-threads` (#2003, merged, in our pin):**
  - Replaces 4 spawned DataLoader processes with an ordered thread pool. Same batches, same order.
  - "Read the full 11,006-patch schedule 53.2 s (4 spawn workers, 37.5 s of it startup) → 13.5 s (`--loader-threads 4` + chunk cache)".
  - At our pin `--chunk-cache-size` defaults to 0 (`infer.py:1677-1683`). Pass `--chunk-cache-size 256` explicitly
    (~71 MB per worker per #2003).
  - **I:** saves the per-worker spawn and import (tens of seconds per run). Small but free.
- **Threads:** the CLI has no thread flag. **I:** 1161 % on 16 vCPUs may mean torch picked fewer intra-op threads than
  vCPUs, for example 8 physical cores with SMT. Check `lscpu` and
  `python -c 'import torch;print(torch.get_num_threads())'` in the container. Try `OMP_NUM_THREADS=16` vs default,
  and two inference processes with 8 threads each on two segments. Unmeasured.
- **Suggested inference line** (A/B each change against the current one, hash and score the TIFF):
  ```bash
  run env MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=2147483647 \
    python -m vesuvius.ink_detection.inference.infer "$RUN/render.zarr" "$ROOT/checkpoints/ink_9um/$CKPT" "$RUN/ink.tif" \
    --overlap 0.5 --blend-mode hann --batch-size 8 --device cpu \
    --loader-threads 4 --chunk-cache-size 256 --no-compile
  ```
  (`podman run -e VAR=…` works too. `env` inside keeps it on one line.)

### 5. OpenVINO backend (#2453, open, unmerged)

- **F:**
  - "`--backend openvino` makes CPU flat ink inference 1.27x faster than the fastest torch setting (1.53x faster than
    the CLI defaults), with maps within one uint8 level of torch."
  - Table, 2-core Zen 3: seed42 PHerc0343 114.4 → 90.0 s (1.27x); 3-model ensemble 330.6 → 255.8 s (1.29x);
    **w035 crop 33.8 → 30.0 s (1.13x)**; 4-core: 1.19x / 1.18x.
  - Parity: "max |Δ| 1 uint8 level, r ≥ 0.999997"; AUC 0.9448982 → 0.9448995.
  - Limits: fp32 only, Zen 3 Linux only. It "refuses to start under a tcmalloc `LD_PRELOAD`". Int8 broke parity;
    ONNX Runtime was slower than OpenVINO.
  - Origin: the PHerc0800 screen "took 3.47 VM-hours, almost all of it in the model forward" (#2452/#2453).
  - #2567 checked OpenVINO + masking on Windows CPU: both engines differ by ≤1 level.
- **I:**
  - For a single-checkpoint run on w035-like data, expect ~1.1-1.2x. It is worth more for ensembles.
  - It needs a patched villa (cherry-pick #2453 onto `2d3fb38d`) and the `cpu-inference` extra, which breaks
    "pinned upstream commit" purity.
  - Do it only for M8 screening with ensembles, after #4.
- **Command (after the patch):** `--backend openvino --loader-threads 4 --batch-size 4` (the PR's command).

### 6. Fewer patches: `--overlap 0.25`

- **F** Patch 128, default overlap 0.5 → stride 64 (`infer.py:183-202`, `DEFAULT_OVERLAP = 0.5`). The 2D U-Net was
  trained on 128×128 patches (ink_9um README). Sliding positions per axis: `range(0, L-128+1, stride)` + last
  (`infer.py:267-272`).
- **I** w035 (5820×5240): stride 64 → 90×82 = 7,380 positions; stride 96 → 61×55 = 3,355 (**2.2x fewer**); stride 128
  → 46×41 = 1,886 (3.9x fewer, visible seams likely).
- **Risk:** changes the image and drifts from Tutorial 5 settings. Accept only if AUC/AP on w035 and the held-out w030/w045
  stay within noise. Our harness (`vpipe.evaluate ink`) can decide this in one run per segment.
- **Command:** `--overlap 0.25` (keep `--blend-mode hann`).

## 3. Checked and set aside

| Item | Status at our pins | Why not for us |
|---|---|---|
| #2536 vc_reader render 2.3x less CPU, identical pixels (open) | Not merged. Targets the vc_reader renderer from #2440 (merged 2026-11-04), which is **not in VC3D f7756c7** (`git merge-base`: #2440 is an ancestor of `2d3fb38d`, not of `f7756c7`) | Our render is link-bound. The gain is CPU seconds on cached data ("Look at CPU time … wall time moves"). It needs a self-built VC3D from main + PR. Revisit only for warm-cache batch re-renders |
| #2411 default `--cache-gb` = min(16 GB, half RAM) (open) | Not merged | Prevents OOM kills under 8 GB limits. We pass `--cache-gb 16` on 62 GB, so there is no speed effect. Keep 16 (peak 5.9 GB) |
| #2521 / #2522 uint8 histogram robust normalization, 7.5x faster per patch, byte-identical (open) | Not merged | Flat inference does use it: `normalize_image` → `vc_reader.transforms.Normalize('robust')` for `robust_mad` (`normalization.py:29-37`). **I:** 59.8 ms vs 8.0 ms per 17×384×384 patch scales to ~6.6 ms per 17×128×128 patch. Over ~7,400 patches that is ≈ 50 CPU-s of ~5,000 CPU-s per run (~1 %). It matters on GPU, not here |
| `--group-idx 1` (18.7 µm) | available | ~8x fewer bytes (**I**), but ink_9um expects "~9 µm isotropic surface volumes" (README). Rendering at 18.7 µm changes the model input: not legitimate |
| Render 17 slices instead of 28 | available | The model center-crops 17 of 28 layers (`center_crop_layer_indices`, `infer.py:205-219`: layers 6..22). 17-slice offsets (−8..8) are not the same samples as 28-slice ones (−13.5..13.5), so the result is no longer comparable with the published surface volume. **I:** chunk saving is only ~5-10 % for a sheet crossing 128³ chunks |
| `--pyramid false` | available | Inference reads level 3 for occupancy (`infer.py:56,425`). Dropping the pyramid changes patch scheduling |
| Delta3D remote cache (`perf/remote_cache_delta3d`) | available, off | Saves disk, not network |
| Parallel downloader (s5cmd/aria2) + local render | — | Same bytes. It helps only if #2/2a shows VC3D under-uses the link. `--prefetch-remote` is the in-tool version |
| Disk budget | `remote_cache_min_free_gib` default 20 | **F** default min-free 20 GiB, max unlimited (`RemoteCacheSettings.hpp`). If `/data` free space drops below 20 GiB, the shared cache evicts and re-downloads silently. Monitor free space for M8 |
| Run near the data | bucket in `us-east-1` | **I:** a us-east-1 CPU instance would make the 0.5 GB/cm² read nearly free. Outside the "own VM, no spend" scope; noted as the structural fix |

## 4. Community CPU-only chains

- **#2453/#2452 (divyadityabhardwaj):** CPU-only ink screen of PHerc0800's six segments (~9.4 cm²) with a 3-model
  ensemble, 3.47 VM-hours, "almost all of it in inference". It starts from published surface volumes, with no render
  step. Their fixes are §2.4 and §2.5.
- **#2534 (wil-pe):** MacBook, CPU, <2 GB RAM: m7 prediction → tifxyz → surface volume → ink_9um. "About 3 minutes of
  CPU and 0.5 GB of reads per cm²". Their renderer correlates 0.97 with a published patch (not bit-exact). Our
  render is bit-exact for 99.73 % of voxels at the same 0.5 GB/cm². Their only edge is RAM, which we don't need.
- **vesuvius-automesh** (docs/prior-work.md): CPU-only Mac, ~30 GB chunk cache, anonymous S3, geometry only.
- **#2478:** asks for a pre-render download estimate. Its rough count was 1.2-8.8 GB per segment for seven 9 µm segments.
  `--prefetch-remote`'s `Prefetch: N chunk(s)` line already gives the exact count.
- **I:** nobody public overlaps render and inference across segments or reuses published surface volumes for
  screening as a deliberate mode. Those two plus `--prefetch-remote` are where our chain can beat the reports above.

## 5. Untested (and how to test)

- Nothing here was run on the test VM.
- Checked 2026-11-20: `--prefetch-remote` exists in `vc_render_tifxyz.cpp` at f7756c7, and `--no-compile`,
  `--loader-threads`, `--chunk-cache-size` exist in `infer.py` at `2d3fb38d`; `MALLOC_MMAP_MAX_` /
  `MALLOC_TRIM_THRESHOLD_` are glibc environment variables, not flags. None is a default in `scripts/` until a test
  below measures it as a win with unchanged output.
- Gains for #2, #3, the heap env vars, `--no-compile`, `OMP_NUM_THREADS`, `--overlap 0.25` and OpenVINO on our 16-vCPU
  box are **unmeasured**. The cited numbers come from 2-4-core Zen 3 VMs (#2452/#2453) and a Xeon 4114 (#2536).
- Test order:
  1. 2a link test.
  2. w035 cold render with `--prefetch-remote` (`VC3D_HOME=$RUN/home`). Compare wall time and output against
     `results/w035-e2e-1`.
  3. Inference A/B on w035: heap env, `--no-compile`, `--loader-threads 4 --chunk-cache-size 256`, one change at a
     time. Record wall time and TIFF sha256 for each.
  4. w035 inference on the published surface volume, scored with `vpipe.evaluate ink`.
  5. `--overlap 0.25` scored on w035, w030 and w045.
