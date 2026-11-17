# Vesuvius Challenge open data inventory (as of 2026-11-16)

Method note: "MEASURED" = sizes I summed from a live S3 `ListObjectsV2` listing or read from a dl.ash2txt.org nginx directory listing on 2026-11-15/16 (UTC). "DOC" = quoted from scrollprize.org or a README. "RAW" = uncompressed size computed as z*y*x bytes from the `shape` field in the bucket's `metadata.json` (uint8); stored size is lower because empty chunks are not written. "INFERRED" = my arithmetic. Nothing large was downloaded; no form was filled in or submitted.

## 1. Hosting, mirrors, streaming and access requirements

### Takeaway
The primary home is now the public AWS Open Data bucket `s3://vesuvius-challenge-open-data/` (us-east-1, anonymous reads, OME-Zarr, organised by sample ID). The old `https://dl.ash2txt.org/` server is still online and openly browsable without a password, but it now holds legacy `.volpkg` data and curated bundles. Curated ML datasets (ink labels, surfaces, spiral) and model weights are on Hugging Face. HTTP range requests and chunk-level streaming work on all three hosts, so small crops can be read without downloading whole volumes.

### Cited Findings
- The data page says: "We host the dataset in the open data bucket: `s3://vesuvius-challenge-open-data/`, usable with any S3-compatible client (e.g., AWS CLI, boto3, s3fs, etc.)". It also says the bucket is "browsable directly" at https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html. No dl.ash2txt.org link and no registration step appear on that page. — [scrollprize.org/data](https://scrollprize.org/data)
- AWS registry entry "Vesuvius Challenge - CT Scans of Herculaneum Papyri": ARN `arn:aws:s3:::vesuvius-challenge-open-data`, region us-east-1, access command `aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/`, licence CC BY-NC 4.0, updated "As additional papyri are scanned and reconstructed", managed by Vesuvius Challenge, docs at github.com/ScrollPrize/open-data. No size is given. — [registry.opendata.aws](https://registry.opendata.aws/vesuvius-challenge-herculaneum-scrolls/)
- MEASURED: an anonymous S3 listing of the bucket root (2026-11-16) returned 46 sample prefixes plus `_thumbnails/`, and the root files `LICENSE.txt` (19,346 B, the full CC BY-NC 4.0 legal text), `index.html`, `metadata.json` (17.7 MB uncompressed, 1.41 MB gzip), `metadata.min.json` and `metadata.schema.json`. — [S3 root listing](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/?list-type=2&delimiter=/)
- Sample prefixes present on S3 (MEASURED): PHerc0009B, PHerc0125, PHerc0139, PHerc0172, PHerc0175A, PHerc0175B, PHerc0191, PHerc0211, PHerc0257, PHerc0268, PHerc0306B, PHerc0332, PHerc0343, PHerc0343P, PHerc0358, PHerc0483A, PHerc0483B, PHerc0490A, PHerc0490B, PHerc0500P2, PHerc0800, PHerc0813, PHerc0814, PHerc0826, PHerc0841, PHerc0846A, PHerc0846B, PHerc1203, PHerc1218, PHerc1299, PHerc1447, PHerc1451, PHerc1545, PHerc1667, PHerc1667Cr1Fr3, PHerc51Cr4Fr8, PHercMAN5, PHercMANB, PHercMANBp, PHercParis1Fr34, PHercParis1Fr39, PHercParis2Fr143, PHercParis2Fr47, PHercParis3, PHercParis4. — same listing
- On-disk layout per sample: `{SAMPLE_ID}/volumes/` (OME-Zarr, sometimes TIFF), `segments/` (meshes, surface volumes, optional ink results), `representations/` (predictions). PHercParis4 also has `photos/` and `transforms/` (MEASURED). — [scrollprize.org/data](https://scrollprize.org/data)
- "OME-Zarr is the primary distribution format because it is cloud-optimized (chunked, multi-resolution) and supports streaming / partial reads—so you don't need to download entire terabyte-scale volumes". — [scrollprize.org/data](https://scrollprize.org/data)
- MEASURED: the Scroll 1 2.4 µm volume `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/` is Zarr v2 with levels `0`–`5`, chunks 128³, dtype `|u1`, `"compressor": null`, `dimension_separator "/"`. Level 0 shape is [75784, 32693, 32693] and level 5 is [2369, 1022, 1022]. A sibling `...-fast-q1-c64-v0.3.2.vcd3d.zarr/` holds a compressed Zarr v3 copy with levels 0–7. — S3 `.zarray`/`zarr.json` reads
- The ink tutorial states: "The public `vesuvius-challenge-open-data` S3 bucket is read anonymously — no AWS account needed." It also says you can download "just the chunks your segments touch, with `vesuvius.ink_detection.preprocessing.download_required_zarr_chunks`". `vc_render_tifxyz --volume s3://...` streams chunks into `~/.VC3D/remote_cache` "so only the parts of the scroll your segment touches ever get downloaded". — [scrollprize.org/tutorial5](https://scrollprize.org/tutorial5) (via [llms-full.txt](https://scrollprize.org/llms-full.txt))
- The `vesuvius` library README says "Inference can be performed on remote data (http, s3) stored as Zarr arrays". It lists an entrypoint `vesuvius.accept_terms` that "accepts the data terms and conditions; pass `--yes` to accept non-interactively". — [villa/vesuvius README](https://github.com/ScrollPrize/villa/blob/main/vesuvius/README.md)
- MEASURED: dl.ash2txt.org (nginx/1.26.0) answers HTTP 200 with an open directory index and no auth prompt. Top level (2026-11-15): `community-uploads/` (24 user dirs, latest 2026-Nov-12), `datasets/`, `fragments/`, `full-scrolls/`, `ml-models/`, `other/`, `LICENSE.txt` (2.4 KiB). The server sends `accept-ranges: bytes`. A `Range: 0-99` request on `other/campfire.zip` returned HTTP 206 with 100 bytes. CORS allows `Range`. — [dl.ash2txt.org](https://dl.ash2txt.org/)
- MEASURED: a second HTTPS host, `https://data.aws.ash2txt.org/`, is an open nginx index. It serves the legacy EduceLab-licensed Zarr volumes under `samples/<ID>/volumes/...`. For example, `samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr/0/.zarray` is shape [14258, 7888, 8096], 128³ chunks, uncompressed. `metadata.json` lists this host as the `access_root` for those volumes. — [data.aws.ash2txt.org](https://data.aws.ash2txt.org/)
- MEASURED: the Hugging Face bucket `scrollprize/datasets` is public. Its API reports `size` 3,681,271,204,074 B (about 3.68 TB) across 11,657,660 files, updated 2026-10-20. The top-level directories are `ink`, `ink_9um`, `ink_paris4_3d_4.8um_gambiarra`, `spiral` and `surfaces`. `ink/` contains `0009b, 0139, 0500p2, 1667, 814, 841, man5, phercparis4, unused`. — [HF API](https://huggingface.co/api/buckets/scrollprize/datasets)
- The tutorial says that if you hit HF rate limits you should "create a free account, generate a read token". — [tutorial5](https://scrollprize.org/tutorial5)
- Model checkpoints are hosted at https://huggingface.co/scrollprize, for example `scrollprize/ink_9um`, `surface_recto_3dunet` and `surface_m7_nnunet`. — [2026_open_problems](https://scrollprize.org/2026_open_problems)

### Inferences
- Registration is no longer a technical gate. Every host I probed served data anonymously. The Data Agreement form still exists, and the legacy prize rules made it a condition of prize eligibility (see section 2).
- For a 14–25 MB/s link (about 50–90 GB/h), always stream S3 Zarr chunks for scroll volumes. Download whole objects only for segments and their surface volumes.

### Gaps
- I did not check whether `index.html` on S3 is a static browser or a generated listing. It does not matter for access.
- I did not find an official torrent or a non-AWS mirror. The Hugging Face bucket mirrors the curated datasets only, not the raw volumes.

## 2. Licence and data agreement

### Takeaway
There are two regimes. (a) New scans released directly by Vesuvius Challenge (2025–2026 ESRF scans, most of the S3 bucket) are CC BY-NC 4.0. (b) The legacy EduceLab-Scrolls data (Scrolls 1–4 and Fragments 1–6 scanned at Diamond Light Source before 2025) is under the "EduceLab Data License", i.e. the dl.ash2txt.org LICENSE.txt / Data Agreement text. That text forbids redistribution without written approval, forbids publishing hidden-text revelations without consent, and requires EduceLab citation. The Data Agreement is still a live Google Form asking for name, email and a typed signature. I did not fill it in.

### Cited Findings
- Data page: "CC-BY-NC 4.0 (unless otherwise noted for specific assets)". It also says Scrolls 1–4 and Fragments 1–6 scanned at DLS before 2025 "are from the EduceLab-Scrolls Dataset, copyrighted by EduceLab/The University of Kentucky. Permission to use the data linked herein according to the terms outlined above is granted to Vesuvius Challenge". — [scrollprize.org/data](https://scrollprize.org/data)
- MEASURED per-volume licence in `metadata.json`: new volumes carry `"license": {"name": "CC BY-NC 4.0"}`. The DLS-era volumes (for example PHerc1667 `20231107190228-3.240um-88keV`, PHerc0332 3.24/7.91 µm, PHercParis4 `20230205180739`/`20230206171837` 7.91 µm) carry `"license": {"name": "EduceLab Data License", "url": "https://dl.ash2txt.org/LICENSE.txt"}`. PHerc0172 (Scroll 5, DLS 2024) is marked CC BY-NC 4.0. — [metadata.json](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
- Verbatim from https://dl.ash2txt.org/LICENSE.txt (MEASURED, 2.4 KiB, dated 2025-May-12):
  - "By registering for downloads from the Vesuvius Challenge Data Server and Vesuvius Challenge Discord Server, I agree to license the data from Vesuvius Challenge under the following terms:"
  - "I agree not to disclose, publish, or otherwise make public - outside of the official Vesuvius Challenge Discord - any revelation of hidden text or related code without the express written consent of the Vesuvius Challenge and its Papyrology Team. This restriction applies to full or partial transcriptions, translations, interpretations, ink label images, and any discussions or commentary related to textual revelations derived from data provided by the Vesuvius Challenge."
  - "I will not redistribute the data without the written approval of Vesuvius Challenge."
  - "Vesuvius Challenge reserves the right to use in any way, including in an academic or other publication, all submissions or results produced from this dataset."
  - "I understand that all submissions will be reviewed by the Vesuvius Challenge Review Team, and that prizes will be awarded as the sole discretion of Vesuvius Challenge."
  - "When I post or upload data in Discord (e.g. text, pictures, code), I agree to license it to other participants under these same terms."
  - "If I am working in a team, every member will agree to these terms individually."
  - EduceLab add-on: cite "EduceLab-Scrolls" in any abstract; reference Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023), *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT*, arXiv, https://doi.org/10.48550/arXiv.2304.02084; and include "Data used in the preparation of this article were obtained from the EduceLab-Scrolls dataset [above citation]." in the methods section. — [dl.ash2txt.org/LICENSE.txt](https://dl.ash2txt.org/LICENSE.txt)
- MEASURED: `full-scrolls/Scroll5/LICENSE.txt` and the `fragments/PHerc0009B|0343P|0500P2/LICENSE.txt` files (18.9 KiB each) are the CC BY-NC 4.0 legal text. — [dl.ash2txt.org/full-scrolls/Scroll5/](https://dl.ash2txt.org/full-scrolls/Scroll5/)
- FAQ: "Can I publish my results? Generally yes, with the conditions that are specified in the [Data Agreement](https://forms.gle/HV1J6dJbmCB2z5QL8)". The conditions: publications must cite the dataset(s) used, and "You won't publish any revelation of hidden text (or associated code) without the written approval of Vesuvius Challenge." — [scrollprize.org/faq](https://scrollprize.org/faq)
- FAQ on social media: if you are competing for a prize that involves reading new text (2027 Grand Prize, First Letters, PHerc. Paris. 4 Title), "do not make your discovery public — e.g., entire words or passages — until the prize is officially announced". — [scrollprize.org/faq](https://scrollprize.org/faq)
- The Data Agreement form (viewed, not submitted) is titled "Vesuvius Challenge Data Agreement". It repeats the LICENSE.txt terms. Fields: Email (required), Full name (required), Team leader name (optional), Team leader email (optional), Substack mailing-list checkbox, a required checkbox "I agree to the data licensing terms as stated above.", and a required "digital signature" (retype full name). It appeared to be accepting responses. — [forms.gle/HV1J6dJbmCB2z5QL8](https://forms.gle/HV1J6dJbmCB2z5QL8)
- The archived 2023 Grand Prize rule read: "To qualify for the Grand Prize, you must have registered at the time you downloaded the data." — [scrollprize.org/grand_prize (archived)](https://scrollprize.org/grand_prize)
- Citation for the new scans: Angelotti, Parsons, Johnson, Dal Prà, Rudolph, Tafforeau, Mirone, Henderson, Schilling, McDonald, Josey, Nader, Parker, Seales, *Vesuvius Challenge - CT Scans of Herculaneum Papyri*, Vesuvius Challenge. The technical preprint for the 2026 scans is arXiv 2606.29085. — [scrollprize.org/data](https://scrollprize.org/data)

### Inferences
- What a participant must personally do: (1) for prize eligibility, submit the Data Agreement form before downloading (name, email, typed signature; every team member individually); (2) for EduceLab-era data, follow the no-redistribution and citation rules; (3) for CC BY-NC data, give attribution and make no commercial use. Redistributing CC BY-NC data non-commercially is allowed by the licence itself. The stricter "no redistribution" and "no hidden-text publication" clauses come from the agreement and the EduceLab terms.
- `vesuvius.accept_terms --yes` suggests the Python library still shows the terms once before use. I did not run it.

### Gaps
- No current prize page I read says whether the Data Agreement is required for the 2026–2027 prizes. I only confirmed the archived 2023 rule. Check /prizes before submitting a prize entry.
- I could not fetch `accept_terms.py` (404 at the path I guessed), so its exact prompt text is not confirmed.

## 3. Per-scroll inventory (scan parameters, volumes, sizes)

### Takeaway
Every sample now has an OME-Zarr multiscale volume on S3 named `<scanID>-<pixel>um-<det.dist>m-<keV>keV-masked.zarr`. New 2025–2026 ESRF BM18 scans at 2.4 µm and 1.129 µm are tens of TB each (RAW). The DLS 7.91 µm volumes are about 0.1–1.3 TB RAW, and the 9.36/8.64 µm ESRF survey scans about 0.6–2.4 TB RAW. Every volume has 128³-chunk pyramid levels. Levels 3–5 are tens of GB down to about 1 GB, which is how a 1 TB disk works with them. Ink has been read in PHerc. Paris. 4, PHerc. 1667 (read in full), PHerc. 172, PHerc. 139 (title and author), and First Letters were awarded for PHerc. 306B and PHerc. 343 (Oct 2026).

### Cited Findings
Volume table (MEASURED from `metadata.json`; RAW = uncompressed uint8 z*y*x; S3 path prefix `s3://vesuvius-challenge-open-data/<sample>/volumes/`; "EDU" = EduceLab licence, served from `https://data.aws.ash2txt.org/samples/<sample>/volumes/`). Source: [metadata.json](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)

| Sample (community name) | Volume long_id | Shape z,y,x | RAW | Licence |
|---|---|---|---|---|
| PHercParis4 (Scroll 1) | 20230205180739-7.910um-54keV-masked.zarr | 14369,7888,8096 | 918 GB | EDU |
| PHercParis4 | 20230206171837-7.910um-54keV-masked.zarr | 10529,7812,8316 | 684 GB | EDU |
| PHercParis4 | 20260310170716-45.532um-11.0m-74keV-masked.zarr | 4071,2264,2264 | 21 GB | CC |
| PHercParis4 | 20260310173927-45.532um-11.0m-110keV-masked.zarr | 4066,2264,2264 | 21 GB | CC |
| PHercParis4 | 20260323153942-2.400um-0.2m-137keV-masked.zarr | 6625,8431,8431 | 471 GB | CC |
| PHercParis4 | 20260411134726-2.400um-0.2m-78keV-masked.zarr (main current volume; has ink-3D, surface predictions, spiral dataset) | 75784,32693,32693 | 81.0 TB | CC |
| PHercParis4 | 20260608103018-1.129um-0.2m-78keV-masked.zarr | 59969,36006,32354 | 69.9 TB | CC |
| PHercParis3 (Scroll 2) | 20260427095331-2.400um-0.2m-78keV-masked.zarr | 68417,42403,42403 | 123.0 TB | CC |
| PHerc0332 (Scroll 3) | 20231027191953-3.240um-53keV-masked.zarr | 22929,9414,9414 | 2.03 TB | EDU |
| PHerc0332 | 20231201141544-3.240um-70keV-masked.zarr | 22929,9414,9414 | 2.03 TB | EDU |
| PHerc0332 | 20231117143551-7.910um-53keV-masked.zarr | 9777,3550,3400 | 118 GB | EDU |
| PHerc0332 | 20251211183505-2.399um-0.2m-78keV-masked.zarr | 33592,15761,15761 | 8.3 TB | CC |
| PHerc1667 (Scroll 4) | 20231107190228-3.240um-88keV-masked.zarr | 26132,7960,8120 | 1.69 TB | EDU |
| PHerc1667 | 20231117161658-7.910um-53keV-masked.zarr (the volume all 20 published segments are on) | 11173,3340,3440 | 128 GB | EDU |
| PHerc1667 | 20251217075048-2.399um-0.2m-78keV-masked.zarr | 37076,15229,15229 | 8.6 TB | CC |
| PHerc1667 | 20260323082859-1.129um-0.2m-59keV-masked.zarr | 42209,22122,20276 | 18.9 TB | CC |
| PHerc0172 (Scroll 5) | 20241024131838-7.910um-53keV-masked.zarr (segments' base volume) | 21000,6700,9100 | 1.28 TB | CC |
| PHerc0172 | 20241024131839-7.910um-53keV-masked.zarr | 20820,6700,9100 | 1.27 TB | CC |
| PHerc0139 | 20250728140407-9.362um-1.2m-113keV-masked.zarr | 20974,6621,6621 | 919 GB | CC |
| PHerc0139 | 20260102150214 / 20260930100252 -2.399um-0.2m-78keV | 76953,26511,26511 | 54.1 TB each | CC |
| PHerc0139 | 20260413113053-1.129um-0.2m-59keV | 19297,35971,32318 | 22.4 TB | CC |
| PHerc0139 | four 2.403um-77keV scans + two 9.362um pag0/pag50 variants | — | 10.3–13.2 TB each; 919 GB each | CC |
| PHerc0814 | 9.362um 113keV / 2.399um 78keV / 1.129um 59keV | 19390,5784,5784 / 74568,23891,23891 / 46721,22078,20284 | 649 GB / 42.6 TB / 20.9 TB | CC |
| PHerc0841 | 9.366um 113keV / 2.403um 77keV | — | 1.15 TB / 14.0 TB | CC |
| Other 2025 ESRF survey scrolls at 8.64 µm 116 keV or 9.362 µm 113 keV (single volume each): PHerc0125, 0175A, 0175B, 0191, 0211, 0257, 0268, 0306B, 0343, 0358, 0483A, 0483B, 0490A, 0490B, 0800, 0813, 0826, 0846A/B, 1203, 1218, 1447, 1451, 1545 | — | — | 0.6–2.4 TB each | CC |
| PHerc1299, PHerc1451, PHercMAN5, PHercMANB (2.399 µm 78 keV) | — | — | 20.9 / 26.0 / 9.7 / 51.9 TB | CC |

- MEASURED stored sizes (S3 listing sums) for the Scroll 1 2.4 µm volume `20260411134726`, which has uncompressed chunks: level 5 = 1.16 GB (556 objects); level 4 = 6.98 GB (3,329 objects); level 3 = 47.4 GB (22,625 objects). The compressed `...-fast-q1-c64-v0.3.2.vcd3d.zarr` level 5 = 0.40 GB. The whole-scroll 45.532 µm volume `20260310170716` (all levels) = 5.41 GB. — S3 listing
- Scan sources: the 2026 scans were taken on the BM18 beamline at ESRF with "high-resolution phase-contrast X-ray microtomography". The DLS scans form the EduceLab legacy set. — [scrollprize.org/firstscroll](https://scrollprize.org/firstscroll); [scrollprize.org/data](https://scrollprize.org/data)
- Legacy dl.ash2txt.org copies (MEASURED listings): `full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230205180739/` has 14,376 TIFFs (00000–14375) of 121.8 MiB each. `volumes_zarr_standardized/` holds `54keV_7.91um_Scroll1A.zarr/`, `54keV_7.91um_Scroll1B.zarr/`, and archives `54keV_7.91um_Scroll1A.7z` (362.6 GiB) and `Scroll1B.7z` (231.9 GiB). Scroll 5 `PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.7z` is 579.2 GiB. Scroll 4 `PHerc1667.volpkg/volumes_small/20231107190228_small.tif` is 3.2 GiB. Scroll2 `PHercParis3.volpkg` has volumes 20230206082907, 20230210143520, 20230212125146 and `volumes_zarr_standardized/54keV_7.91um_Scroll2A.zarr/`. Scroll3 `PHerc332.volpkg` has `volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/`. — [dl.ash2txt.org/full-scrolls/](https://dl.ash2txt.org/full-scrolls/)
- "standardized" Zarr = "normalization of intensity values to the range of air - papyrus, removing bits without information (keeping 5 most significant bits), adjusting dataformat (uint8), compression with Blosc + zstd". — [Scroll5 volumes_zarr_standardized/README.txt](https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/README.txt)
- Ink status: "PHerc. 1667 — read in full … the lower parts of some twenty-two columns"; PHerc. Paris. 4 ink "directly visible inside the scroll itself, in the three-dimensional X-ray data" at higher resolution, matching the 2023 Grand Prize reading; PHerc. 139 title and author recovered (Philodemus, *On Gods*, Book 8). — [scrollprize.org/firstscroll](https://scrollprize.org/firstscroll). First Letters prizes for PHerc. 306B and PHerc. 343 were awarded in October 2026. — [scrollprize.org/winners](https://scrollprize.org/winners). The home summary says "by 2025 ~70% of scroll PHerc. 172 had been digitally unwrapped". — [llms-full.txt](https://scrollprize.org/llms-full.txt)
- MEASURED: segments carrying ink predictions in `metadata.json`: PHercParis4 has 80 of 82, PHerc0172 53, PHerc1667 19 of 20, PHerc0139 39 segments (75 ink-detection objects), PHerc0814 19, PHerc0841 3, and fragments PHerc0009B, PHerc0343P and PHerc0500P2. PHerc0332 (Scroll 3) has 2 segments without ink-detection outputs. PHercParis3 (Scroll 2) has 0 segments. — [metadata.json](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)

### Inferences
- INFERRED: level 2 of the Scroll 1 2.4 µm volume is about 8 × 47 GB ≈ 380 GB stored, level 1 about 3 TB, and level 0 tens of TB. On a 1 TB disk, the only full volumes you can hold are levels 3–5 of a 2.4 µm volume, the 45 µm overview, or the DLS 7.91 µm volumes at a reduced level. The 7.91 µm PHerc1667 volume is 128 GB RAW and is the smallest legacy full-resolution volume that has published segments.
- Download time at 14–25 MB/s: 47 GB ≈ 0.5–1 h; 128 GB ≈ 1.5–2.5 h; 380 GB ≈ 4–7.5 h.

### Gaps
- I did not measure the stored size of the PHerc1667 7.91 µm volume or of any 7.91 µm volume's levels. For uncompressed chunks it is ≤ RAW. Not confirmed.
- The PHerc0343P 8.64 µm volume and three of PHerc0500P2's four volumes have `shape: null` in the metadata. Size not confirmed.
- The per-sample S3 listing showed no Scroll 2 segments and no Scroll 2 ink. I found no scrollprize.org statement about Scroll 2 ink. Not confirmed either way.

## 4. Segments: what they are, sizes, reproduction targets

### Takeaway
A segment is a traced papyrus sheet surface. The current format is `tifxyz` (x.tif/y.tif/z.tif + meta.json) plus OBJ meshes, one transformed tifxyz per registered volume, and "surface volumes" (`layers-zarr`) rendered at each volume's resolution. Optional extras are `ink-detection/` TIFF predictions (+ ds8 JPG), `ink-labels/` (inklabels + supervision zarr) and 3D ink-prediction zarrs. The legacy format (dl.ash2txt.org `paths/<id>/`) is `.obj` + `.ppm` + `layers/00–64.tif` (65 TIFFs) + mask/composite. One segment's surface volume ranges from about 0.04 GB (45 µm) through about 1 GB (9.36 µm) and about 5–6 GB (legacy 7.91 µm, 65 layers) to about 57 GB (2.4 µm).

### Cited Findings
- Formats: segment surface volumes are "OME-Zarr and/or TIFF stacks (`00.tif`, `01.tif`, …)"; geometry is "OBJ meshes, plus TIFXYZ (x/y/z TIFF triplet + metadata)". — [scrollprize.org/data](https://scrollprize.org/data)
- MEASURED legacy segment `full-scrolls/Scroll1/PHercParis4.volpkg/paths/20230827161847/`: `20230827161847.obj` 19.2 MiB, `.ppm` 2.1 GiB, `.tif` (composite) 83.8 MiB, `_cellmap.tif` 62.2 MiB, `_mask.png` 100.4 KiB, `_points.obj` 129.3 MiB, `pointset.vcps` 131.9 MiB, `_webknossos.zip` 2.9 MiB, `area_cm2.txt`, `author.txt`, `meta.json`, and `layers/` with 65 TIFFs (00–64) of 88.2 MiB each. INFERRED total ≈ 5.6 GiB for layers + 2.5 GiB for the rest ≈ 8.1 GiB. `paths/` holds 284 entries for Scroll 1. — [dl.ash2txt.org paths/20230827161847](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20230827161847/)
- MEASURED: on S3, 20230827161847 does not appear among the 82 PHercParis4 segments in `metadata.json`. The 2023-era S3 segments are 20230702185753, 20230929220926, 20231005123336, 20231007101619, 20231012184424, 20231016151002, 20231022170901, 20231031143852, 20231106155351, 20231210121321 and 20231221180251, plus 2026 re-traces on the 2.4 µm volume (for example `20260602230115-20230702185753_v14`, `20260603222816-20231005123336_v2_flatboi`, `20260604223808-20231210121321_v8`, `20260623*-w0xx-0yy`). — [metadata.json](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
- MEASURED: PHercParis4 segment `20231016151002` (15,960 × 10,080 px). The S3 objects are `mesh/intermediate/{tifxyz_original,tifxyz_normalized,*_original.obj,*_flattened.obj,*_normalized.obj}`, `mesh/20231016151002-on-{20260310170716-45.532um,20230205180739-7.91um,20260411134726-2.4um}.tifxyz/`, `surface-volumes/45.532um-11.0m-74keV-volume-20260310170716.zarr/` (0.036 GB measured), `surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/` (size not measured), `ink-detection/` (0.169 GB measured, including `PHercParis4-20231016151002-2.4um-...-new_canon_autoresearch_recipe-tile256-stride128.tif` and a ds8 JPG), `ink-labels/2.4um-volume-20260411134726/20260918/` (0.7 MB measured), a 3D ink prediction zarr `...ink3d-20260428123845-v3-78k-fullsup-L2-3d-ink-max.zarr/`, and `surface-renders/` alpha overlays. — S3 listing + metadata.json
- MEASURED: PHerc0139 segment `20260317000000-w035_2026031718`, total 90.6 GB: `surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr` 1.02 GB; `2.399um-...-20260102150214.zarr` 56.9 GB; `1.129um-...-20260413113053-L1.zarr` 32.4 GB; `ink-detection/` 72.7 MB (+0.8 MB downsampled); `ink-labels/` 0.4 MB; tifxyz per volume 0.9–62.7 MB; `mesh/intermediate` 31.5 MB. — S3 listing
- MEASURED PHerc1667 segments (all on the 7.91 µm volume 20231117161658, each with layers-zarr + ink-detection): w011–w041 "flatboi" segments of about 9–10k × 10–12k px; larger ones `20240304144031-w018` (30,040 × 12,960), `20240304141531-w013` (24,460 × 13,000), `20240304161941-w023` (31,340 × 12,800); ink labels on w013, w018, w023, w028, w029, w031; plus a merged `20260612121456-w011_..._merged_v4_flatboi_straightened_v4` (184,099 × 14,179) with no layers listed. — [metadata.json](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
- Legacy PHerc1667 `paths/` on dl.ash2txt.org: 20231111135340, 20231122192640, 20231210132040, 20240304141530/31, 20240304144030/31, 20240304161940/41, and a merged 20240304161941-20240304144031-20240304141531-… — [dl.ash2txt.org Scroll4 paths](https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/)
- Grand Prize banner region (archived dataset, MEASURED listing): `datasets/grand-prize-banner-region/gp_meshes.7z` 275.1 MiB (DOC: 288 MB); `volumes/gp_volume.zarr/` (DOC: 77 GB); `volumes/gp_tifstack.7z` 363.2 GiB (DOC: 389.9 GB); predictions `gp_legendary-medial-cubes.7z` (DOC 5.8 GB) and `...-softmax.7z` (DOC 146.8 GB). The README says the tifstack is "original uint16 TIFF stacks … non GP region is masked" and the meshes are "all `.obj` triangular meshes for the GP region segments". — [grand-prize-banner-region README](https://dl.ash2txt.org/datasets/grand-prize-banner-region/README.txt); [data_datasets](https://scrollprize.org/data_datasets)
- Four combined Grand Prize composites in `other/`: `combined-composite-16_36-segment_21_22.png` (659.7 MiB), `..._51_52.png` (680.5 MiB), `combined-composite-18_45-segment_21_22.png` (658.2 MiB), `..._51_52.png` (677.5 MiB) (MEASURED). — [dl.ash2txt.org/other/](https://dl.ash2txt.org/other/)

### Inferences
- The best "official result" reproduction targets on S3 today are PHercParis4 `20231016151002` (has an official 2.4 µm ink TIFF, ink labels, and a 3D ink prediction) and any PHerc1667 w0xx flatboi segment on the 128 GB-RAW 7.91 µm volume (official ink-detection, several with labels).

### Gaps
- MEASURED (lower bound): the S3 2.4 µm surface volume `PHercParis4/segments/20231016151002/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/` is more than 142.9 GB. My listing stopped at 80,000 objects, so the true size is larger. Excluding that volume, the rest of the segment is 0.45 GB: mesh/intermediate 313 MB, 2.4 µm tifxyz 49.8 MB, 3D ink zarr 32.5 MB, 45 µm surface volume 35.7 MB, ink labels 0.7 MB. The HF tutorial copy of the same segment is DOC "about 25 GB", so the HF version is much smaller (fewer layers or compressed) and is the one to download.
- I did not measure the size of a PHerc1667 flatboi segment's layers-zarr (the listing did not finish). Not confirmed.
- I could not tell from the listings exactly which segment IDs make up the 2023 Grand Prize banner. Not confirmed.

## 5. Fragments with infrared ground truth and the 2023 Kaggle ink dataset

### Takeaway
Fragments 1–6 (Paris collection, DLS, EduceLab licence) are still under `dl.ash2txt.org/fragments/FragN/`, with IR photos and hand-made `inklabels.png` aligned to a 65-layer surface volume (about 6.4 GiB for Frag1). New ESRF fragments PHerc0009B, PHerc0343P and PHerc0500P2 have multispectral folders and S3 segments. The 2023 Kaggle ink-detection dataset's size could not be confirmed (Kaggle requires login).

### Cited Findings
- MEASURED `fragments/` listing: Frag1–Frag6, PHerc0009B, PHerc0343P, PHerc0500P2. Frag1 = `PHercParis2Fr47.volpkg/` (S3 sample PHercParis2Fr47). The new fragments have `cases/ multispectral/ paths/ volumes/` and a CC BY-NC LICENSE.txt. — [dl.ash2txt.org/fragments/](https://dl.ash2txt.org/fragments/)
- MEASURED `fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/`: `ir.png` 7.6 MiB, `inklabels.png` 86.8 KiB, `mask.png` 60.2 KiB, `result.obj` 7.3 MiB, `result.ppm` 2.3 GiB, `surface_volume.ppm` 2.3 GiB, `alignment.psd` 217.5 MiB, `faux_rgb.png` 14.7 MiB, `00_canny_raw.ply` 2.4 GiB, and `surface_volume/` with 65 TIFFs of 98.8 MiB each (INFERRED ≈ 6.3 GiB). There are also `Frag1_54keV_3.24um.mp4` and `Frag1_88keV_3.24um.mp4`. — [Frag1 working dir](https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/)
- "The first ink labels came from detached fragments, where the exposed writing can be photographed in infrared and aligned with the surface volume." "No infrared ground truth exists in scrolls". — [tutorial5](https://scrollprize.org/tutorial5); [data_datasets](https://scrollprize.org/data_datasets)
- The 2023 Kaggle competition was a $100,000 Ink Detection Progress Prize on fragments: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/. — [llms-full.txt](https://scrollprize.org/llms-full.txt)
- MEASURED: S3 sample PHerc0009B has 19 segments, 18 ink-detection, 1 ink-labels set. PHerc0500P2 has 46 segments, 48 ink-detection, 1 ink-labels set. HF `ink/0009b` and `ink/0500p2` exist. — metadata.json; HF API

### Inferences
- For a first local ink model trained against true IR ground truth, Frag1 `surface_volume/` (about 6.3 GiB) + `inklabels.png` + `mask.png` + `ir.png` is the smallest complete set.

### Gaps
- Kaggle 2023 dataset size and file list are not confirmed: the Kaggle data tab returned only the title, and the Kaggle API needs authentication. Its licence is also not confirmed.
- Sizes of Frag2–Frag6 were not listed.

## 6. Kaggle 2025–2026 Surface Detection competition dataset

### Takeaway
The competition ran with a $200,000 pool (announced November 2025) and has finished, with winners listed. Dataset size, contents and licence could not be confirmed without a Kaggle login.

### Cited Findings
- "Surface prediction was the subject of a dedicated Kaggle competition, 'Vesuvius Challenge – Surface Detection,' with a $200,000 prize pool in November 2025. The competition's scoring metric was deliberately topology-aware". — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Winners: $60k 1st Vesuvius Team, $40k 2nd "risk of overfitting", $30k 3rd W & A, … 10th "Vibes & Scrolls Trade-off". — [scrollprize.org/winners](https://scrollprize.org/winners); [Kaggle overview](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/overview)
- The 1st-place nnU-Net component card cites "Dataset100_VesuviusSurface" with 786 training volumes, single-channel CT read from .tif, labels background/surface/ignore, and Apache-2.0 for the weights. — [huggingface.co/scrollprize/surface_m7_nnunet](https://huggingface.co/scrollprize/surface_m7_nnunet) (via search snippet; not opened in full)
- An open alternative: the curated `surface-labels` dataset (voxelized recto surfaces paired with scroll sub-volumes) at HF `buckets/scrollprize/datasets/tree/surfaces`. — [data_datasets](https://scrollprize.org/data_datasets)

### Gaps
- Kaggle dataset GB size, file layout and licence are not confirmed (WebFetch returned only the page title; Kaggle API returned 401 Unauthenticated).

## 7. Smallest sensible end-to-end test set (CPU-only, 1 TB, 14–25 MB/s)

### Takeaway
The cheapest complete chain with an official comparison target is PHerc0139 segment w035 at 9.362 µm. The official tutorial uses it for exactly this. You fetch the tifxyz (about 1 MB), stream only the touched chunks of the 9.362 µm volume to render a 28-layer surface volume (DOC about 900 MB; the published copy is 1.02 GB MEASURED), run the public `scrollprize/ink_9um` checkpoint, and compare against the published `ink-detection/` TIFF (72.7 MB MEASURED). The total is about 2–3 GB. For a Scroll 1 / Scroll 4 target, use PHercParis4 `20231016151002` or a PHerc1667 flatboi segment.

### Cited Findings
- Tutorial steps (DOC): `aws s3 sync --no-sign-request s3://vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz/ ink-dataset/pherc0139/w035/w035.tifxyz`; then `vc_render_tifxyz --volume s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/ … --num-slices 28 --voxel-size 9.362 --flip-normals`: "For w035 this takes about five minutes and writes a ~900 MB Zarr". "For w035 the rendered surface volume is also already published, under the segment's `surface-volumes/` folder". Then `python -m vesuvius.ink_detection.inference.infer ink-dataset/pherc0139/w035/w035_9um.zarr checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth predictions/w035_9um.tif`. The checkpoint comes via `uvx --from huggingface_hub hf download scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth`. — [tutorial5](https://scrollprize.org/tutorial5)
- The ink tutorial "assumes an NVIDIA GPU with a working CUDA installation". The 2.5D training example takes "about an hour and a half on a single H100", and full-segment inference "on the order of an hour on a single GPU". `--mask-path region.tif` limits inference to a region. — [tutorial5](https://scrollprize.org/tutorial5)
- Tutorial training set (DOC): HF `ink/phercparis4/w00_20231016151002`, "about 25 GB", synced via `uvx --from huggingface_hub hf buckets sync hf://buckets/scrollprize/datasets/ink/phercparis4/w00_20231016151002 ./ink-dataset/phercparis4/w00_20231016151002`. It contains x/y/z.tif, meta.json, `<seg>.zarr`, `<seg>_inklabels.zarr/.tif`, `<seg>_supervision_mask.zarr/.tif`. MEASURED on HF: it also has `preds/tile256_stride128_layers1_63_hann_fwd.tif` (395.8 MB), an official prediction to compare against. — [tutorial5](https://scrollprize.org/tutorial5); HF API
- The `ink_9um` labels set holds "24 segments annotated on pooled 2.4 µm renders … and 5 on native 9.362 µm renders". — [tutorial5](https://scrollprize.org/tutorial5)

### Inferences
- On CPU only, run inference with `--mask-path` on a small crop of w035 and compare it pixel-wise to the published ink TIFF. Full-segment CPU inference timing is not documented and is likely many hours.
- To start upstream at "CT sub-volume → surface", stream a small crop of the 9.362 µm PHerc0139 volume around the w035 bounding box and run the surface model (`surface_m7` / `surface_recto_3dunet`). Then compare your surface against the published tifxyz. I did not test this.

### Gaps
- I did not test CPU-only runtimes or whether `vc_render_tifxyz` (VC3D) builds on macOS ARM.
- The exact chunk count or bytes streamed for w035 was not measured. DOC says only the touched chunks are fetched.

## 8. Recommended download tools and recipes

### Takeaway
Use AWS CLI `--no-sign-request` (or s3fs/zarr streaming) for S3. Use `hf buckets sync` for Hugging Face. Use `rclone copy :http:` with `--http-url` for dl.ash2txt.org. All three resume interrupted transfers.

### Cited Findings
- S3: `aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/` — [AWS registry](https://registry.opendata.aws/vesuvius-challenge-herculaneum-scrolls/); `aws s3 sync --no-sign-request s3://…/mesh/…tifxyz/ <dst>` — [tutorial5](https://scrollprize.org/tutorial5)
- HF: `uvx --from huggingface_hub hf buckets sync hf://buckets/scrollprize/datasets/ink/phercparis4 ./ink-dataset/phercparis4`. "`hf buckets sync` works like `rsync`: re-running it resumes interrupted downloads". — [tutorial5](https://scrollprize.org/tutorial5)
- dl.ash2txt.org (archived tutorial): `rclone copy :http:/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230205180739/ . --http-url https://dl.ash2txt.org/ --progress --size-only`. For a 1 cm central slab: `for i in `seq 6000 7250`; do wget http://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230205180739/0$i.tif; done`. INFERRED: 1,251 slices × 121.8 MiB ≈ 149 GiB. — [tutorial3 (archived)](https://scrollprize.org/tutorial3)
- Spiral dataset: `rclone copy :http: ./spiral_datasets/phercparis4 --http-url https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ --transfers 32 -P`. The tutorial says "~90 GB", while the data_datasets page says "~49.6 GB total". The two documentation figures conflict. MEASURED files in it include `fibers.afv` 10.4 GiB and `fiber_directions.npz` 2.6 GiB. — [tutorial_spiral](https://scrollprize.org/tutorial_spiral); [data_datasets](https://scrollprize.org/data_datasets)
- Community helper: "Efficient Data Downloader … scripts to efficiently download data with rclone, by James Darby" — https://github.com/JamesDarby345/VesuviusDataDownload. — [llms-full.txt](https://scrollprize.org/llms-full.txt)
- A quick-start notebook: https://github.com/ScrollPrize/open-data/blob/main/examples/get-to-know-a-dataset.ipynb. — [scrollprize.org/data](https://scrollprize.org/data)
- I found no aria2 recipe in the official docs (grep of llms-full.txt). — [llms-full.txt](https://scrollprize.org/llms-full.txt)

### Inferences
- Suggested download order on a 1 TB disk:
  1. `metadata.json` (17.7 MB).
  2. The w035 tifxyz, published 9.362 µm surface volume and ink TIFF (about 1.1 GB).
  3. The `ink_9um` checkpoint.
  4. Frag1 surface_volume + IR labels (about 6.5 GiB).
  5. HF `w00_20231016151002` (about 25 GB DOC).
  6. The Scroll 1 2.4 µm volume level 4 (7 GB) or level 3 (47 GB) for context.
  7. The PHerc1667 7.91 µm volume (≤128 GB) only if you need full-resolution scroll-space work.

### Gaps
- `rclone` against S3 with anonymous config was not tested. The documented path is AWS CLI.
