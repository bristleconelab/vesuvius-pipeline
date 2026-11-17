# Vesuvius Challenge: open problems and organisers' wish list (as of 2026-11-16)

Method note: official pages were read through a summarising fetcher, so quotes are short fragments it returned verbatim; GitHub facts come from direct `gh` API queries on 2026-11-16 (issue bodies read verbatim). Discord was not accessed. "Inference" sections are mine, not sourced facts.

## 1. What the latest official posts say are the hard parts (June to November 2026)

### Takeaway
The organisers say supervised surface models have saturated on what human annotators can label, densely packed / twisted regions still defeat automation, and ink does not yet generalise across scrolls at 8-9 µm. Since July the stated goal is full automation (Grand Prize: at most 8 human hours); the public posts from August on are mostly prize announcements, not problem statements.

### Cited Findings
Post-by-post (all dates from the Substack archive, [archive](https://scrollprize.substack.com/archive)):

- **2026-06-05, "The hard parts are getting clearer" (May update, Vesuvius Challenge and Giorgio Angelotti).** Key quotes: "We are close to saturating the performance of models trained on the approximate surface labels our annotators can produce."; densely packed regions "are not rare"; "These regions still defeat fully (and partial) automated approaches."; "manual annotation does not scale."; Neural Tracing "soon drifts off the intended surface" and inference is "very slow"; spiral fit "still does not conform well where annotations—whether automatic or manual—are missing." Proposed direction: self-supervised 3D DINO. Semi-automated fiber tracing at about 2 cm/minute where the latent representation holds. May prizes: $10,000 Ben Kyles (Scroll Fiesta automatic mesher), $2,000 Paulo Sergio Camillo (3D CT augmentations). No per-scroll numbers and nothing on ink in this post. — [Substack](https://scrollprize.substack.com/p/the-hard-parts-are-getting-clearer)
- **2026-06-25, first complete scroll announced.** PHerc. 1667 (Scroll 4) read end to end: about 1.4 m of papyrus, around twenty-two columns (lower parts; roughly 8 cm of an original 19-24 cm roll height survives). Preprint arXiv 2606.29085. The page does not state human hours or compute. — [scrollprize.org/firstscroll](https://scrollprize.org/firstscroll)
- **2026-07-15, "A New $1M Grand Prize for 2027".** Besides Scroll 4: bottom 70% of PHerc. 172 (Scroll 5) unwrapped, PHerc. 139 identified as Philodemus "On Gods, Book 8", about 140 columns unwrapped from PHerc. Paris 4 (Scroll 1) rescanned at 2.4 µm. Quotes: methods "still require humans in the loop."; "Full automation is required to read the collection within a reasonable timeframe, and legibility could improve."; Grand Prize for a team that will "fully unroll and make readable another complete scroll from the eligible set" with "no more than eight hours of human input."; "a guaranteed $20,000 will be awarded every month to the best open-source submission". June prizes: $2,000 Camillo, $1,000 Joseph Balmaceda (fiber format converter). — [Substack](https://scrollprize.substack.com/p/a-new-1m-grand-prize-for-2027)
- **2026-08-06, "$33.5K awarded in July".** $33,500 to 13 teams/individuals from 58 submissions. Community focus was "alternative virtual-unwrapping pipelines and integrating them with VC3D."; "Other valuable contributions included bug fixes, consistency improvements, and metadata checks." — [Substack](https://scrollprize.substack.com/p/335k-awarded-in-july)
- **2026-08-18, "From CT Scan to Ancient Text: A First Letters Prize Workflow" (Paul Henderson).** Reference workflow: VC3D -> spiral fitter (umbilicus, tracks, normals) -> optional Lasagna flatten (GPU) -> `vc_render_tifxyz` to Zarr -> ink inference with pretrained 9 µm checkpoint (`vesuvius`, CUDA) -> label/retrain loop. Admitted pain points: spiral fit "may fail in a difficult region"; "Ink cannot be read in the original CT volume"; "Ink models recover local signal; they should not invent plausible writing."; "there are still many scrolls in which nobody has seen a single letter." RAM/disk needs not specified. — [Substack](https://scrollprize.substack.com/p/from-ct-scan-to-ancient-text-a-first)
- **2026-10-08, "$50K First Letters Prize awarded for PHerc. 343".** Erwin Nieuwlaar; 8.64 µm scan; spiral-fitter + own surface-snapping + averaging four ink models none trained on PHerc. 343; second-hand RTX 3090 plus "about 300 dollars spent on rent uptil now". Scrolls with ink listed at that time: PHerc. 1667, PHerc. 1447, PHerc. 343. — [Substack](https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for)
- **2026-11-02, "Writing seen in PHerc. 306B" (Henderson, Nicolardi).** Liam Warner and Nicola Bodill; GrowPatch on m7 surface predictions; product of two ink models (public TRACE v3B and a fine-tune). Stated next goal: "reliable ink detection on 8–9um resolution scans." Several other submissions reached 7-9 characters and missed the threshold; submissions exist for 13 other scrolls without meeting the criteria. Bodill reports "spending hundreds of hours manually creating ink labels" since 2023. — [Substack](https://scrollprize.substack.com/p/writing-seen-in-pherc-306b-for-the)
- **2026-11-15, "First letters emerge from PHerc. 211".** Priyank Aranke; full-height spiral fit, stitched windings, DinoVol-guided growth, residual 3D U-Net on 16 planes of native ~9 µm CT; training scrolls chosen by 3D-DINO similarity; 2 µm ink predictions transferred down to 9 µm as labels; trained on 12 other scrolls/fragments. Manual: surface "refined ... by hand" in a VC3D fork; labels "checked and corrected by hand." 37 submissions from 15 contestants/teams on this scroll. Seven First Letters prizes remain. — [Substack](https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211)
- There is no separate Substack post for the August, September or October progress prizes; those appear only on the winners page. — [archive](https://scrollprize.substack.com/archive), [winners](https://scrollprize.org/winners)
- Background (older): 2026-03-19 "We are cooking": Kaggle surface detection ended; "Almost all of the top 10 submissions used large ensembles of our baseline, a ResEncUNet trained with the nn-UNet framework."; recipes such as "training for more than 4,000 epochs"; Neural Tracer "still does not perform on par with the state of the art."; "We currently have 4 agents working 24/7." — [Substack](https://scrollprize.substack.com/p/we-are-cooking)

### Inferences
- The organisers' own framing moved from "better models" (June) to "full automation + reproducibility" (July onward). The hard-parts post itself contains no infrastructure ask; the infrastructure asks live on the open-problems page and in the prize criteria (sections 2 and 5).
- First Letters wins in Oct-Nov all used public tooling (spiral fitter / GrowPatch + public 9 µm ink checkpoints) plus careful manual refinement, which suggests the bottleneck for new scrolls is surface quality and false-positive control more than model architecture.

### Gaps
- No public post gives human hours or compute for the PHerc. 1667 unwrapping; the Nature paper / arXiv 2606.29085 was not read here and may contain it.
- Discord announcements (where most day-to-day asks are made) were not read by design.

## 2. Official open problems / wish list / prize pages

### Takeaway
The open-problems page (last updated 2026-10-14) names six help areas, almost all modelling/geometry; the one explicitly infrastructural ask is "Reproducible streaming pipelines and cheaper compute/storage paths", plus evaluation suites, diagnostics and scan-quality metrics. The GitHub "help wanted" wishlist the prizes page points to is currently empty.

### Cited Findings
- Open-problems page, last updated October 14, 2026. Three flagged problems: (1) degraded scan regions / "scanning whole scrolls at sub-micron resolution is not yet practical"; (2) "No method yet traces a complete, correct surface through a scroll automatically"; (3) ink recovery beyond PHerc. 1667: "But it is not guaranteed to work everywhere." — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Bottlenecks table, "help" column: compressed regions -> "Scan-quality metrics, and scroll-specific acquisition recipes."; no connectivity -> "Better geometry priors, fiber tracing, and topology-aware tools."; approximate labels -> "Label snapping, active learning, self-supervised approaches."; sheet switches (currently "VC3D inspection and manual correction.") -> "Smarter surface tracing algorithms that avoid introducing sheet switches."; ink depth ambiguity -> "Direct 3D ink segmentation where possible."; cross-scroll generalisation -> "Multi-scroll training, better labels, stronger diagnostics."; **data scale -> "Reproducible streaming pipelines and cheaper compute/storage paths."** — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Data-scale section: "A model that requires copying tens of terabytes locally shuts most contributors out before they start."; contributions should "read chunks directly from OME-Zarr;" "run tiled inference;" "avoid unnecessary full-volume copies;". — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Evaluation asks: "Devise better evaluation suites and loss functions to fit the spiral."; "relative winding number annotations seem to have a great impact on the spiral fit"; "we do not always know which part of the pipeline is limiting us"; "The pipeline works, but not without a person checking its output at almost every stage"; goal to "make the whole workflow reproducible enough for collection-scale reading". — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Appendix: neural tracer / autoregressive mesh model: "Inference is slow at the 2.4 µm resolution the pipeline runs at"; "Neither problem is solved yet." — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- The page does not mention CPU, Docker or CI. — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Prizes page: open pool $2,140,000. Grand Prize $800,000 / $100,000 / $50,000 / $50,000, deadline 2027-06-25, 13 eligible volumes. First Letters $50,000 per scroll, up to 8 more, 10 letters in 4 cm², 20 eligible volumes. Title prize for PHerc. Paris. 4 $50,000. Progress prizes monthly; next deadline 23:59 Pacific 2026-11-30. — [prizes](https://scrollprize.org/prizes)
- Progress prize levels: "Best Submission of the Month: $20,000, guaranteed every month."; others typically $20,000 / $10,000 / $5,000 / $2,500 / $1,000 / $500 / $250. Judges favour submissions that are "released or open-sourced early.", "Actually get used", "Improve results quantitatively and/or qualitatively on real data.", "Resolve outstanding bugs in tools that people are using", "Are well documented." Core requirements: problem identification and solution, documentation, technical integration (standard community formats). — [prizes](https://scrollprize.org/prizes)
- Wishlist: "We maintain a public wishlist of ideas that would make excellent progress prize submissions." linking to villa issues labelled `help wanted`, `VC3D`, `good first issue`. — [prizes](https://scrollprize.org/prizes)
- Queried 2026-11-16: zero open issues carry `help wanted`, `good first issue` or `VC3D`; one carries `open-data` (#1835). The `help wanted` label description is "Good candidate for a Progress Prize". — [help wanted query](https://github.com/ScrollPrize/villa/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22), [labels](https://github.com/ScrollPrize/villa/labels)
- Eligible Grand Prize volumes (13): PHerc0125, 0191, 0211, 0257, 0268, 0358, 0800, 0813, 0826, 1203, 1218, 1447, 1545. First Letters list (20) adds 0175A/B, 0483A/B, 0490A/B, 0846A/B and omits 1447. — [prizes](https://scrollprize.org/prizes)
- llms-full.txt: "Get Started" has no body; older tutorials are archived; current tutorial is "Virtual Unwrapping with VC3D" (updated July 2026) and Tutorial 5 (ink). Tutorial 5 "assumes an NVIDIA GPU with a working CUDA installation."; 2.5D training "takes about an hour and a half on a single H100."; native 3D "about ten hours on an H100."; inference "on the order of an hour on a single GPU for a full segment."; no CPU-only path and no Docker instructions described. — [llms-full.txt](https://scrollprize.org/llms-full.txt)
- CONTRIBUTING.md (villa): PRs must come from a human running the tool "on real scroll data"; "Synthetic or toy examples are not accepted"; bugfix PRs need before/after screenshots; "We may close PRs which appear to be simple \"fishing expeditions\" for llms". — [CONTRIBUTING.md](https://github.com/ScrollPrize/villa/blob/main/CONTRIBUTING.md)

### Inferences
- The advertised wishlist being empty means the real wish list is (a) the open-problems page, (b) the unlabelled issue tracker, (c) Discord. A contributor cannot pick a ticket; they must find and evidence a problem by actually running the pipeline on real data.
- The only wishlist item that maps directly onto DevOps skills is "Reproducible streaming pipelines and cheaper compute/storage paths"; evaluation suites/diagnostics are the second-best fit.

### Gaps
- The prizes page may not yet reflect the 2026-11-15 PHerc. 211 award (0211 still appears in the First Letters list as fetched). The 306B post says "20 scrolls remain eligible" while the prizes page says up to 8 more prizes; these are different counts (eligible volumes vs prizes left), not verified further.
- Full llms-full.txt (570k chars) was read only up to 200k.

## 3. Ink: why only some scrolls, per-scroll status, approaches

### Takeaway
Ink reading is confirmed on PHerc. 1667 (complete), Paris 4, 172, 139, and since autumn 2026 first letters on 1447, 343, 306B and 211; most of the ~20 eligible 9 µm scrolls still show nothing convincing. The organisers say they often cannot tell which pipeline stage is to blame.

### Cited Findings
- "In some scrolls, predictions improve and then plateau. In others, current models show little or no convincing ink." Possible causes listed: weak scan signal, poor surface placement, label mismatch, differing ink chemistry. Requested: a way to tell "no ink" from "no ink recovered yet". "Ink not recovered from a scroll doesn't necessarily mean the signal isn't there." — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Resolution matters: "We have evidence that working on 1.1 µm data yields cleaner results than 2.4 µm data."; full-scroll regime about 2.4 µm (ESRF BM18, ~78 keV, 22 cm propagation); prize-eligible scans are around 9 µm (e.g. 9.362 µm, 8.64 µm). — [2026_open_problems](https://scrollprize.org/2026_open_problems), [PHerc. 343 post](https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for)
- Status: PHerc. 1667 fully read; PHerc. Paris. 4 ink "directly visible inside the scroll itself" in higher-resolution data, matching the 2023 reading "one-to-one"; PHerc. 139 title recovered. — [firstscroll](https://scrollprize.org/firstscroll). PHerc. 172 bottom 70% unwrapped — [Grand Prize post](https://scrollprize.substack.com/p/a-new-1m-grand-prize-for-2027). First letters: 343 (2026-10-08), 306B (2026-11-02), 211 (2026-11-15), and ink on 1447 — [winners](https://scrollprize.org/winners), [PHerc. 211 post](https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211)
- Approaches in use: iterative pseudo-labelling (PHerc. 1667 iteration-0..5 checkpoints); direct 3D ink segmentation on Paris 4 (`ink_3d_dino_guided`); autonomous agent swarm that nearly doubled validation Dice on 1667 training only on 139; self-supervised 3D DINO (`dinovol`); distilling 2 µm predictions onto 9 µm volumes (Domenico Russo reader-v2/v3); ensembling/multiplying models to suppress false positives. — [2026_open_problems](https://scrollprize.org/2026_open_problems), [winners](https://scrollprize.org/winners), [306B post](https://scrollprize.substack.com/p/writing-seen-in-pherc-306b-for-the)
- Topography-based: arXiv 2603.27698 "Ink Detection from Surface Topography of the Herculaneum Papyri" (Angelotti, Nicolardi, Henderson, Seales; 2026-03-29) uses optical profilometry on opened fragments PHerc. 248, 250, 500P2. — [arXiv](https://arxiv.org/html/2603.27698v1)
- Validation weaknesses found by the community: Tutorial 5 "trains with no validation set" (issue #1231, 2026-07-25, open); all three shipped `ink_9um` validation masks split within connected regions, adjacency "worth an extra +0.14 and +0.07 F1" (#1638, 2026-08-29, closed); a data-leakage bug in 3D ink training was fixed (Miika Kestilä, Sept, $250). — [#1231](https://github.com/ScrollPrize/villa/issues/1231), [#1638](https://github.com/ScrollPrize/villa/issues/1638), [winners](https://scrollprize.org/winners)
- Render path affects ink score: rendering from the coarser published mesh "costs 0.04 AUC" (0.857 vs 0.897) on a PHerc0139 segment; replicated on a second machine to the fourth decimal. — [#1845](https://github.com/ScrollPrize/villa/issues/1845) (2026-09-20, open)
- CPU-only ink screening is being done: PR #2453 (2026-11-05, open) reports screening PHerc0800's six segments with a 3-model ensemble "on a CPU-only VM took 3.47 VM-hours" and adds an OpenVINO backend 1.27x faster; issue #2534 (2026-11-13) reports a CPU-only MacBook chain "under 2 GB of RAM" from m7 surface prediction to `ink_9um`. — [#2453](https://github.com/ScrollPrize/villa/pull/2453), [#2534](https://github.com/ScrollPrize/villa/issues/2534)

### Inferences
- Ink inference on rendered segments is feasible without a GPU (hours per scroll on a few cores); training is not practical on CPU (H100-hours quoted).
- A systematic, reproducible "ink screening of every public segment on every eligible scroll with every public checkpoint, with negative results published" is exactly the diagnostic the open-problems page asks for ("no ink" vs "not recovered yet"), and several individuals are doing fragments of it ad hoc (#2534, #2453, #2049 commenters).

### Gaps
- No official per-scroll ink status table was found; status of Scroll 2 and Scroll 3 is not stated in any 2026 source read.
- Who found ink in PHerc. 1447 and when was not confirmed from a primary post.
- New scan plans (energies, higher resolution for eligible scrolls) are not public beyond the open-problems page statements.

## 4. Automated segmentation: coverage, error modes, metrics, what Kaggle changed

### Takeaway
Only PHerc. 1667 is fully unwrapped; elsewhere coverage is partial and public numbers are sparse. Sheet switches in compressed regions remain the main failure, quality is still checked by eye in VC3D, and the Kaggle competition mostly yielded better training recipes for the same nnU-Net baseline rather than a new method.

### Cited Findings
- Coverage numbers available: PHerc. 1667 complete (~1.4 m); bottom 70% of PHerc. 172; about 140 columns of PHerc. Paris 4. — [firstscroll](https://scrollprize.org/firstscroll), [Grand Prize post](https://scrollprize.substack.com/p/a-new-1m-grand-prize-for-2027). William Stevens' patch pipeline: ~365 cm² on PHerc. 1667 (Aug, $20,000). — [winners](https://scrollprize.org/winners)
- Many eligible scrolls have no public segments: PHerc1447 "has no public surfaces" as of 2026-07-13 (only surface predictions) — [llms-full.txt](https://scrollprize.org/llms-full.txt); data browser shows "Segments: 0" for PHerc. 1203, which has 22 unreviewed raw auto-grown meshes (~130 cm²) — [#2534](https://github.com/ScrollPrize/villa/issues/2534); PHerc0813 "has no published segments" — [#1588](https://github.com/ScrollPrize/villa/issues/1588)
- Error modes: "Meshes can jump from one wrap to another."; predictions that "disappear and barely make sense" or "form blobs"; in PHerc. 1203 a published m7 prediction is "one fused mass (99 percent of the voxels in a 3.7 mm region)". — [2026_open_problems](https://scrollprize.org/2026_open_problems), [hard parts post](https://scrollprize.substack.com/p/the-hard-parts-are-getting-clearer), [#2534](https://github.com/ScrollPrize/villa/issues/2534)
- Quality measurement: Kaggle metric "was deliberately topology-aware rather than voxel-accuracy-based" and "rewarded connected, gap-free predictions"; no numeric metric is defined for full-scroll unwrapping; sheet switches are handled by "VC3D inspection and manual correction." — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Kaggle Surface Detection ($200,000, ended March 2026): top 10 were mostly "large ensembles of our baseline"; hole post-processing; team planned to retrain on 2.4 µm data. — [We are cooking](https://scrollprize.substack.com/p/we-are-cooking), [winners](https://scrollprize.org/winners)
- Community QA tools already rewarded: windcheck (self-intersections in tifxyz), TIFXYZ Doctor, scroll-data-audit (July, $1,000 each); windaudit (Sept, $500); GrowPatch comparison of three 9 µm surface models (Yupeng Tang, Oct, $500). — [winners](https://scrollprize.org/winners)
- Unmet evaluation needs in issues: fiber evaluation reference "98% ... drawn by the beam tracer" (#2264, 2026-10-23); "tifxyz meta.json has no isometry numbers, so segment grids can't be compared or checked for stretch" (#2279, 2026-10-24); held-out bands of a sheet tracer "aren't recorded" (#2444, 2026-11-04); duplicate / nested windings in PHerc0139 (#1547, #2167). — [#2264](https://github.com/ScrollPrize/villa/issues/2264), [#2279](https://github.com/ScrollPrize/villa/issues/2279), [#2444](https://github.com/ScrollPrize/villa/issues/2444), [#2167](https://github.com/ScrollPrize/villa/issues/2167)
- Still manual per the August workflow: choosing Z-range, checking winding direction, inspecting meshes for sheet jumps, judging fiber continuity, labelling ink. — [workflow post](https://scrollprize.substack.com/p/from-ct-scan-to-ancient-text-a-first)

### Inferences
- There is no public, standard, automatically computed "unwrapping quality scorecard" per scroll (area, self-intersections, sheet-switch count, stretch, winding consistency). Pieces exist as separate prize-winning tools; nobody appears to run them together across the open-data bucket on a schedule. This is CPU-only work.

### Gaps
- Per-scroll segmented area in cm² / percent for all scrolls: not found in any page read (the data browser was not scraped).
- Kaggle top-solution write-ups were not read.

## 5. 2026 progress prizes for tooling / infrastructure / validation

### Takeaway
The $20,000 monthly prize has gone to unwrapping methods every month (Kyles, Stevens, Lerognon, Lomeli-Rodriguez), but a long tail of $250-$1,000 awards goes to infrastructure, platform fixes, data audits and validation harnesses; several of those were CPU-only work. Judges' comments are one-liners only.

### Cited Findings
- $20,000 winners: July Ben Kyles (ScrollFiesta, "directly ingests our zarr volumes and outputs tifxyz patches"); Aug William Stevens; Sept Samir Lerognon (fiber models + spiral fitter integration); Oct Jaime Lomeli-Rodriguez (HercUNet / HercuViT). — [winners](https://scrollprize.org/winners), [July post](https://scrollprize.substack.com/p/335k-awarded-in-july)
- Infrastructure/validation awards, $1,000 tier: Hyojun Kwon ink validation harness with held-out splits; Josep Carreras windcheck; Aviad Cohen TIFXYZ Doctor; Matteo Bulloni scroll-data-audit (catalog metadata checks); Prasad Khake multiscale zarr 3 read fixes; TAUIL Abd Elilah reproducibility fixes + ROI inference + negative-result analysis; Erwin Nieuwlaar tifxyz metadata fixes (all July); Domenico Russo benchmarks of 14 released 9 µm ink checkpoints; Lutfiya Miller and Chris Müller documented First Letters run on PHerc. 0826 (Aug); Forrest McDonald volcomp chunked compressor (Sept). — [winners](https://scrollprize.org/winners)
- $250-$500 tier examples: "Ink tutorial downloads 2.4 GB instead of 92 GB" (Marcel Gwerder, Sept, $500); renderer loads only needed chunks (Scheirer, Sept, $500); ~10% lossless CT compression improvement (Hendrick, Oct, $500); Discord chat search service (Wainer, Oct, $500); found zarr downsampling bug (Nathaniel, Oct, $250); many Windows / Apple Silicon / AMD compatibility fixes. — [winners](https://scrollprize.org/winners)
- Award counts on the winners page: July 13 awards (58 submissions), Aug 15, Sept 27, Oct 33. — [winners](https://scrollprize.org/winners), [July post](https://scrollprize.substack.com/p/335k-awarded-in-july)
- Explicit requests found for: reproducible streaming pipelines and cheaper compute/storage (open-problems page); Docker image for Grand Prize reproduction ("Please create a Docker image that we can easily run to reproduce your work, and please include system requirements."); "For fully automated software, consider a Docker image" for First Letters; held-out validation with k-fold; documentation ("Hands-on guides for every stage of the pipeline"). — [2026_open_problems](https://scrollprize.org/2026_open_problems), [prizes](https://scrollprize.org/prizes)
- No explicit organiser request was found for: CI builds, CPU-only inference, or a benchmark harness as such; these exist only as community PRs/issues. — [2026_open_problems](https://scrollprize.org/2026_open_problems), [prizes](https://scrollprize.org/prizes)
- Existing CI in villa (20 workflows): VC3D CI, CodeQL, Linux AppImage, macOS app, Windows installer, publish builder images to ghcr, PyPI publishing for `vesuvius`, test-vesuvius, GPU-manager tests, PR time limits, large-PR review gate, progress-prize automation. — [villa actions](https://github.com/ScrollPrize/villa/actions)

### Inferences
- Infrastructure work is rewarded but capped: every infra/validation award so far is $250-$1,000; no tooling-only contribution has taken the $20,000 slot. A realistic expectation for a strong infra contribution is $500-$2,500 per month unless it visibly unlocks results on a new scroll.
- The pattern in rewarded infra work: a measured before/after on real scroll data, small scope, merged into villa or usable with VC3D formats (tifxyz, OME-Zarr).

### Gaps
- No judges' narrative for Aug-Oct awards exists publicly beyond the winners-page one-liners.

## 6. Open GitHub issues describing unmet infrastructure needs (ScrollPrize/villa)

### Takeaway
The tracker (58 open issues, 124 open PRs on 2026-11-16) is dominated by community-filed data-integrity and reproducibility findings with few maintainer replies; the clearest infra gaps are release/version reproducibility, stale container images, headless Linux use, PyPI lag, data-bucket integrity and resource-bounded rendering.

### Cited Findings
Repo facts (queried 2026-11-16): 383 stars, last push 2026-11-15, 58 open issues, 124 open PRs; 1,252 PRs created 2026-07-01..2026-11-16 (688 merged, 440 closed unmerged); 217 issues created in that window; 102 distinct PR authors in October 2026; GitHub Discussions has 1 thread (2024). — [villa](https://github.com/ScrollPrize/villa), [pulls](https://github.com/ScrollPrize/villa/pulls)

Reproducibility / release / packaging:
- #2049 (2026-10-12, open, 7 comments): "VC3D releases are deleted on promotion, so published results cannot be reproduced"; the PHerc 343 result pins a VC3D build that "is not downloadable". A commenter proposed "create-only, commit-addressed VC3D releases" and asked for maintainer confirmation. Releases now include `vc3d-archive-latest-*` tags (first seen dated 2026-11-08 to 11-13), so this may be partly addressed; issue still open. — [#2049](https://github.com/ScrollPrize/villa/issues/2049), [releases](https://github.com/ScrollPrize/villa/releases)
- #1588 (2026-08-24, closed): "container images are 3+ months stale: :edge AND :main both ship the 2026-05-13 build"; PR #1982 (2026-10-05) changed docs to point to the AppImage because the tutorial's `:stable` Docker tag "does not exist". — [#1588](https://github.com/ScrollPrize/villa/issues/1588), [#1982](https://github.com/ScrollPrize/villa/pull/1982)
- #2153 (2026-10-19, open, 0 comments): "Linux AppImage: CLI tools fail on headless servers (unbundled GL/X11 libs; direct binary invocation crashes)". — [#2153](https://github.com/ScrollPrize/villa/issues/2153)
- #1300 (2026-08-02, open, 0 comments): `pip install vesuvius` on Python 3.14 silently installs 0.1.10 from Dec 2024; PyPI latest 0.2.4 excludes 3.14 while `main` requires it. — [#1300](https://github.com/ScrollPrize/villa/issues/1300)
- #2425 (2026-11-02, open, filed by organiser giorgioangel): "repository is 1GB" (630 MB website, 130 MB deprecated). — [#2425](https://github.com/ScrollPrize/villa/issues/2425)
- #1296 (2026-08-01, open, 61 bot comments): "[automation] Progress Prize rollover needs attention" - a scheduled workflow keeps failing. — [#1296](https://github.com/ScrollPrize/villa/issues/1296)
- #2562 (2026-11-15): `vc_render_tifxyz` never exits on Windows. — [#2562](https://github.com/ScrollPrize/villa/issues/2562)

Resource / streaming / cost:
- #2294 (2026-10-24, open): `render_ink.py` full-scroll render "holds about 12 bytes per output pixel in RAM", caches cannot be limited; "can run past 64 GB of RAM and fill a few hundred GB of disk"; a fix branch exists. — [#2294](https://github.com/ScrollPrize/villa/issues/2294)
- #2478 (2026-11-07, open, 0 comments): proposal for a download estimate before `vc_render_tifxyz`; rough count 1.2-8.8 GB per 9 µm segment. — [#2478](https://github.com/ScrollPrize/villa/issues/2478)
- #2522 (2026-11-13): normalisation "took 93% of the CPU time per flat ink patch, so the GPU waited for the loader". — [#2522](https://github.com/ScrollPrize/villa/issues/2522)
- Open PRs: #2411 size render chunk cache to the machine; #2536 CPU render 2.3x faster; #2453 OpenVINO CPU ink backend; #2552 headless spiral fit on public inputs reading `s3://` without AWS keys; #2401 `gp_submission` (cut Grand Prize column meshes and check the folder). — [pulls](https://github.com/ScrollPrize/villa/pulls)

Open-data integrity (found by community audits):
- #1757 (2026-09-10, 0 comments): Scroll 5 volume levels 1-5 stored uncompressed (389 GB) plus 8 community pyramids; found with `zarr-pyramid-audit`. — [#1757](https://github.com/ScrollPrize/villa/issues/1757)
- #1965 (2026-10-04): PHerc0343P level 0 has 555 of 8,543 chunks at 8x/64x declared size; #1963 PHerc0814 surface volume has headers but zero chunks; #2107 (2026-10-15) 18 of 33 ink surface volumes have truncated pyramid level 2; #2092 missing last z chunk; #1964 broken debug mesh published among real segments. — [#1965](https://github.com/ScrollPrize/villa/issues/1965), [#1963](https://github.com/ScrollPrize/villa/issues/1963), [#2107](https://github.com/ScrollPrize/villa/issues/2107)
- #1835 (2026-09-19, assigned jrudolph, label open-data): "Four published volume transforms do not fit their own published landmarks"; #1843 same for PHerc1667 1.129 µm. — [#1835](https://github.com/ScrollPrize/villa/issues/1835), [#1843](https://github.com/ScrollPrize/villa/issues/1843)
- #2370 (2026-10-30, 7 comments) ink canvases misaligned with tifxyz by up to 286 px; #2442 (2026-11-04) hecate ink map offset 19 px; #2369 label zarrs 59 px smaller; #1654 mixed uint8/uint16 in `instance-labels-harmonized`. — [#2370](https://github.com/ScrollPrize/villa/issues/2370), [#2442](https://github.com/ScrollPrize/villa/issues/2442)

Validation / documentation:
- #1231 (2026-07-25, assigned) Tutorial 5 has no validation set; follow-up: "once you have a prediction, there is no supported way to score it." PR #2210 (2026-10-21, open) adds an `evaluate` that scores validation and training regions separately. — [#1231](https://github.com/ScrollPrize/villa/issues/1231), [#2210](https://github.com/ScrollPrize/villa/pull/2210)
- #1691 (2026-09-03, 0 comments) checkpoint/config for a public ink prediction not available; #2346 rendering parameters for published 65-slice volumes not documented; #2105 README example contradicts catalog winding direction. — [#1691](https://github.com/ScrollPrize/villa/issues/1691), [#2346](https://github.com/ScrollPrize/villa/issues/2346)

### Inferences
- Many issues are filed in a uniform "In one sentence / I was trying to / Using" template with before/after numbers; this is the house style, and several look AI-assisted (PR #1982 says so explicitly). The bar for a new audit issue is therefore measured evidence on real bucket data, and "one more zarr audit" has become a crowded genre since September.
- Maintainer bandwidth is the real constraint: 124 open PRs, automated PR time limits, and zero-comment infra issues weeks old (#2153, #1300, #1757, #2478). Contributions that reduce maintainer load (CI that catches regressions on real data, pinned artefacts) are more valuable than more findings.

### Gaps
- Sibling repos under the ScrollPrize org and community repos (ScrollFiesta, spiral fitter forks, khj1222's scorecard, zarr-pyramid-audit) were not inspected.
- Maintainer positions on #2049, #2153, #1300 are unknown (no replies visible in the first comments read).

## 7. How the "fully automated, at most 8 human hours" requirement is demonstrated; who is close

### Takeaway
The rule is documentation plus independent reproduction: a Docker image with system requirements, tifxyz meshes, programmatically generated images, held-out validation and a written account of human input. No one has published a pipeline shown to read a complete eligible scroll within 8 human hours.

### Cited Findings
- "The unrolling pipeline should be fully automated; up to 8 documented hours of human annotation / input are tolerated." Code open source under a permissive licence on GitHub; pipeline should integrate into VC3D; 100% of recto unrolled (tolerance: disconnected outer patches under 10%); at least 70% of each counted column's characters legible. — [prizes](https://scrollprize.org/prizes)
- Submission package: tifxyz meshes, programmatically generated images, "Please create a Docker image that we can easily run to reproduce your work, and please include system requirements.", held-out validation ("Run your method on the public input renders/volumes with known ground truth", k-fold if trained on them), false-positive mitigation. Unlimited manual effort may go into generic ML training datasets if published CC-BY-NC 4.0. — [prizes](https://scrollprize.org/prizes)
- The page does not specify how hours are audited beyond documentation and the technical team reproducing the pipeline. — [prizes](https://scrollprize.org/prizes)
- Candidates: ScrollFiesta (Kyles et al., $20,000 July, $5,000 Sept, "producing clean surfaces over large areas"); Will Stevens' patch pipeline with annealing-based consistent-connectivity (~365 cm² on 1667); UFSM "fast end-to-end unwrapping" (Forrest McDonald, Oct, $1,000); fiber-based route (Lerognon); HercUNet/HercuViT (Lomeli-Rodriguez). None is described as completing an eligible scroll. — [winners](https://scrollprize.org/winners)
- Organisers' own statement of the gap: "The pipeline works, but not without a person checking its output at almost every stage". — [2026_open_problems](https://scrollprize.org/2026_open_problems)
- Tooling toward submission packaging exists as an open PR: #2401 "vesuvius: add gp_submission, cut Grand Prize column meshes and check the folder" (2026-10-31). — [#2401](https://github.com/ScrollPrize/villa/pull/2401)
- Even first-letters wins needed heavy manual work (hand-refined surface, hand-corrected labels; "hundreds of hours" of labels). — [PHerc. 211 post](https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211), [306B post](https://scrollprize.substack.com/p/writing-seen-in-pherc-306b-for-the)

### Inferences
- Every Grand Prize team will need the same plumbing: a containerised, headless, resumable driver that chains spiral fit / patch growth -> flatten -> render -> ink inference from the public S3 bucket, logs wall-clock and human-interaction time, and emits the submission folder. The pieces are headless-hostile today (#2153, #2552, #2294, stale Docker images). A neutral, method-agnostic orchestration + "human-hours ledger" + submission validator is plausibly useful to all teams and to the judges, and does not need a GPU to build (GPU needed only to run the heavy stages at full scale).
- The scientific bottleneck (correct surfaces in compressed regions) is not something infrastructure alone fixes; an orchestration layer would not by itself win the Grand Prize.

### Gaps
- No public statement on how organisers will verify the 8-hour claim in practice (screen recordings? logs?). Likely discussed on Discord, not read.
- Performance numbers (scroll area per GPU-hour) for ScrollFiesta / UFSM / Stevens pipelines were not collected.

## 8. Crowded vs neglected; candidate gaps for a no-GPU DevOps contributor

### Takeaway
Crowded: surface models, ink models, platform-compatibility fixes, one-off data audits. Comparatively neglected: release/artefact reproducibility, headless containerised end-to-end runs, continuous data-integrity and regression checks against real data, a unified evaluation scorecard, and collection-wide negative-result ink screening. All assessments below are inference from the evidence in sections 1-7.

### Cited Findings
- Volume evidence of crowding: 102 distinct PR authors in October 2026; 1,252 PRs since July with 440 closed unmerged; 33 progress awards in October, 14 of them $250 platform/compat fixes. — [pulls](https://github.com/ScrollPrize/villa/pulls), [winners](https://scrollprize.org/winners)
- Evidence of neglect: zero-comment infra issues (#2153 headless AppImage 2026-10-19; #1300 PyPI lag 2026-08-02; #1757 uncompressed pyramids 2026-09-10; #2478 download estimate 2026-11-07); #2049 reproducibility open since 2026-10-12; `help wanted` list empty. — [#2153](https://github.com/ScrollPrize/villa/issues/2153), [#1300](https://github.com/ScrollPrize/villa/issues/1300), [#1757](https://github.com/ScrollPrize/villa/issues/1757), [#2049](https://github.com/ScrollPrize/villa/issues/2049)

### Inferences
Candidate gaps, each with evidence pointer, who is on it, and what a winning contribution would look like (all "winning" descriptions are my inference from the judging criteria in section 2):

1. **Reproducible, headless, containerised end-to-end run (CT in bucket -> ink PNG) on a public segment.** Evidence: stale ghcr images (#1588), nonexistent `:stable` tag (#1982), headless AppImage failure (#2153), spiral fit needing AWS keys on a public bucket (#2552), Docker required for Grand Prize. On it: scattered individuals (Bullo27 docs, tguguen #2552, rodriguescarson's clean-box reproduction of 343). Winning shape: a pinned image (digest + VC3D commit + checkpoint hashes) that reproduces a published First Letters image (343 or 211, both have public MIT code) bit-for-bit or within stated tolerance on a CPU box, run nightly in CI against `main`, with a report of drift. Neglected-to-moderate. Fits 16 cores / 62 GB (rendering and ink inference on one segment are CPU-feasible per #2453/#2534; full spiral fit needs a GPU).
2. **Immutable, commit-addressed release artefacts and provenance manifest.** Evidence #2049. On it: saschaivanov-lgtm offered a prototype pending maintainer approval; archive tags appeared in November, so check current state before starting. Winning shape: small workflow change + manifest (commit, SHA-256, platform) + doc. Small prize tier; partly done.
3. **Continuous open-data integrity monitor.** Evidence: #1757, #1963, #1965, #2107, #2092, #1964, #1835 all found by one-off audits; Bulloni's scroll-data-audit and sgsllc-jr's zarr-pyramid-audit exist. Winning shape: not another audit but a scheduled job that runs the existing checks over the whole bucket (header-only, cheap), publishes a public status page / machine-readable defect list, and opens deduplicated issues. Crowded as one-offs, neglected as a maintained service. Pure CPU + network.
4. **Unified unwrapping-quality scorecard / regression benchmark.** Evidence: "Devise better evaluation suites"; no numeric metric for unwrapping; separate tools windcheck, TIFXYZ Doctor, windaudit; #2279 (no isometry numbers), #2264 (fiber eval). Winning shape: one CLI + CI job that scores any tifxyz set (area, self-intersections, stretch/isometry, winding consistency, overlap with known-good PHerc. 1667 surfaces) and a public leaderboard of the competing pipelines' published meshes. Moderately neglected; needs geometry understanding, CPU-only.
5. **Collection-wide ink screening with published negatives.** Evidence: request to tell "no ink" from "no ink recovered yet"; #2534 offer for PHerc. 1203; Yupeng Tang ($500) and Russo ($1,000) did checkpoint comparisons. Winning shape: every public segment x every public 9 µm checkpoint, CPU inference (OpenVINO PR #2453), standard render parameters (#1845 lesson: mesh/frame choice moves AUC by 0.04), browsable results. Getting crowded quickly (Oct-Nov), and a GPU owner does it faster; could incidentally surface a First Letters candidate.
6. **Resource-bounded rendering / download estimation / cost accounting.** Evidence #2294, #2478, #2411, #2522; "cheaper compute/storage paths". On it: ChaseHendrick, stantheman0128, protonjls, LynnHenry. Crowded at the PR level.
7. **Packaging hygiene: PyPI/`main` Python-range mismatch (#1300), repo size (#2425, filed by an organiser), failing prize-rollover automation (#1296).** Low competition, low prize ceiling ($250-$500), but direct maintainer pain; #1296 involves the organisers' private Google resources and is probably not fixable from outside.

Crowded (avoid as primary bet): new surface-segmentation models, new ink architectures (GPU-bound, saturated by teams with clusters and by the organisers' own agent swarm), Windows/Mac/AMD compatibility fixes (dozens of $250 awards), single-defect zarr audits.

Ranking for the stated profile (inference): gaps 1 + 3 + 4 combined as "reproducibility and continuous verification for the public pipeline and data" are the best fit; they match the only infra item on the official bottleneck table and the Grand Prize Docker requirement, and are bounded to a few weeks. Expected prize size from precedent is $500-$2,500 per month, not $20,000. Constraint to respect: CONTRIBUTING.md requires evidence from real scroll data and human use; purely synthetic CI will be rejected.

### Gaps
- Could not verify on Discord whether an organiser-run nightly reproduction or data-status page already exists privately.
- Did not test any of the CPU-feasibility claims locally; they rest on community reports (#2453, #2534), not on measurement here.
- Third-party write-ups (HN, Reddit, X) for June-November 2026 were not searched within the call budget.
