# Plan: upgrade to FLUX 3 Image Klein and rebrand to Darkroom

Drafted 2026-10-21, research pass the same day. To be implemented in a later
session, one phase per topic branch, each green on the four gates before merge
(`--no-ff`, Conventional Commits with an evidence body — see CLAUDE.md).

## At a glance

Read this first. Everything below is detail.

```text
Phase 1a  hygiene + baseline      cache guard, .gitignore, example licenses, eval set, FLUX.2 golden run (old lock)
   │       gate: four gates, CI
Phase 1b  toolchain               mflux 0.23 → everything latest (pillow<13), Streamlit 1.66 re-verify
   │       gate: four gates, smoke, drift check vs FLUX.2 golden, CI
Phase 2   FLUX 3 swap (feat!)     one model instance, pinned FLUX + VLM, BFL upsampling, license
   │       gate: peak ≤ 26 GB AND time ≤ 60 s AND quality ≥ FLUX.2  ──► none passes: ABORT
   │                                                                     (ship 1a/1b/3 as 0.9.1)
Phase 3   Darkroom rebrand        name, light theme, screenshots
   │       gate: four gates, CI
Release   0.10.0                  drift check, disk check, bump, release-notes lead, repo details
```

| Phase | Session | Branch | Commits | Blocks on |
|---|---|---|---|---|
| 1a | 1 | `chore/hygiene-and-eval` | 4 | — |
| 1b | 1 | `chore/mflux-0.23` | 2 | 1a merged + CI |
| 2 | 1 (largest) | `feat/flux3-klein` | as needed, `feat!:` | 1b merged + CI; **exit criteria** |
| 3 | 1 | `feat/darkroom-rebrand` | as needed | 2 merged + CI (or the abort path) |
| Release | short | `main` | 1 bump | 3 merged; smoke + drift + disk |
| 4–6 | — | — | — | not scheduled |

- **Never:** `hf cache prune` or `uv cache clean` (Disk plan); hand-edit `uv.lock`; load a model by bare repo id (Weight pinning); push or edit public pages without the owner's OK.
- **The owner decides:** which unrelated cached models to free (asked once, before Phase 2's first download); Release's public edits; anything on the abort path.

## Decisions (agreed with the owner)

| Topic | Decision |
|---|---|
| Model | FLUX 3 Image Klein, distilled, **Q8** MLX build `mflux-community/flux-3-image-klein-mflux-q8` (9.9 GB). Q6 (`-q6`, 7.6 GB) if the memory budget below is not met |
| Weights source | `mflux-community` (converted with mflux 0.23.0, not gated). `mlx-community` has no FLUX 3 build yet |
| Dependencies | **Upgrade every locked package to its latest release** (`uv lock --upgrade`) in Phase 1b, except Pillow, held below 13 by a commented constraint until upstream fixes its import (Facts). Lower bounds in `pyproject.toml` move only where the code starts relying on a newer feature |
| License | **Code stays MIT** (reconfirmed 2026-10-21 over Apache-2.0 and dual MIT/Apache: compatible with every dependency and the vendored Apache-2.0 prompts, already locked by `test_license.py`). Accept the FLUX Kommunity License v1.0 for the weights, which are downloaded at runtime and never bundled. **Disclose it in Phase 2, in the same commit as the model swap**, so no pushed state claims Apache-2.0 for Kommunity weights. Show it **in the README and in the sidebar's model caption**, each locked by a test. Bundled example images: **verify their source license, else replace them** (Phase 1a, before the golden run that uses them) |
| Display name | **Darkroom**, with the model named descriptively in a subtitle ("local image generation and editing with FLUX 3 Image Klein"). Repo and package stay `flux-image` |
| Theme | **A light "Darkroom" that mirrors the dark one** (Phase 3; designed 2026-10-21). This replaces "light mode stays stock": same neutral surround, the dark pair's page and text swapped, one shared ember primary, and each accent at the same hue on the opposite side of its surface. Colors only, so the fold budget is unaffected |
| Upsampling | **On by default**, BFL's external-VLM system prompts, greedy decoding, APC; an off switch remains for users who paste full captions |
| Upsampling VLM | **`mlx-community/Qwen3-VL-8B-Instruct-4bit` preferred** (decided 2026-10-21 on the 2B-vs-8B measurements in Facts), subject to the memory budget. Try combinations in order and take the first that passes: (1) 8B-4bit + FLUX Q8, (2) 8B-4bit + FLUX Q6, (3) `mlx-community/Qwen3-VL-4B-Instruct-8bit` (~5 GB download) + Q8, (4) the current 2B + Q8 as a last resort |
| Safety classifier | Not added (BFL's reference CLI loads one; the license does not require it). README points at the license's Section 4 restrictions instead |
| Release | **0.10.0** (`uv version --bump minor`) after Phase 3 and a green smoke run |
| Scope | One model, as today. No FLUX.2 fallback. Bounding-box layout is a possible Phase 4, generation history (SQLite) a possible Phase 5, and a measured, display-adaptive canvas a possible Phase 6, none of them part of this release |

## Target machine

The owner's Mac: **Apple M2 Max, 32 GB** unified memory. Disk: 460 GB, **41 GB
free (91% used)** on 2026-10-21. Every memory and disk decision below is
measured against this machine.

## Facts established in the planning session

All verified 2026-10-21 by reading source or running code in a scratch venv
(`mflux==0.23.0`, `pillow==12.3.0`), unless marked otherwise.

### mflux 0.23.0 (PyPI 2026-10-16; pulls mlx 0.32.4)

- **Package layout.** `mflux.models.flux3.variants` exports `Flux3Klein` and `Flux3KleinEdit`. `ModelConfig.flux3_image_klein()` / `flux3_image_klein_base()`.
- **Both classes are the same model.** They subclass `_Flux3KleinCore`, whose `__init__` and `Flux3Initializer.init` have no class-specific logic.
  - `Flux3Klein.generate_image` is a 10-line wrapper over `self._generate(...)`.
  - `Flux3KleinEdit.generate_image` opens the references, then calls `self.encode_references(...)` and the same `_generate`.
- **Signatures.** `__init__(quantize, model_path, model_config, compute_precision, lora_paths, lora_scales, bake_lora)`.
  - `Flux3Klein.generate_image(seed, prompt, num_inference_steps=None, height=1024, width=1024, guidance=None)`.
  - `Flux3KleinEdit.generate_image(seed, prompt, image_paths, num_inference_steps=None, height=None, width=None, guidance=None)`.
- **Guidance is ignored on the distilled checkpoint.** `_resolve_sampling` returns `guidance=None` whenever `model_config.supports_guidance` is false (it is, for `flux3_image_klein`). The predictor then skips the unconditional pass, so nothing runs CFG. Passing `guidance=1.0` is harmless but meaningless.
- **Steps default to 16** for distilled, 50 for base (`_resolve_sampling`).
- **`Flux3KleinEdit` opens every `image_paths` entry with `PIL.Image.open(path)`.** Probed: `PIL.Image.open(<PIL.Image>)` raises `AttributeError: 'Image' object has no attribute 'read'`.
  - **The app's current "PIL Images go straight through" contract breaks on FLUX 3**: the first edit would fail.
  - A `BytesIO` does open, but is outside the `list[Path | str]` annotation, which mflux 0.23 types precisely enough for ty to flag.
- **Reference limits** (`flux3_reference_images.py`, mirroring BFL's pipeline):
  - 1–10 references; `[]` raises `ValueError("At least one reference image is required")`.
  - A single reference is fitted to ≤ 4 MP and padded to multiples of 16. Each of several references is capped at 1 MP.
  - When `width`/`height` are `None`, the output takes the reference canvas size and is cropped back to the fitted box.
- **Output size.** `Config` rounds width and height **down to multiples of 16** (with a warning). No upper bound for text-to-image. The app's 32-multiple, 256–1024 sizes are valid as-is.
- **`call_in_loop` contract unchanged.** The keywords are `t, seed, prompt, latents, config, time_steps`; `denoised` is opt-in by declaring it. The app's `_ProgressReporter` should work unmodified.
- **The model keeps every encoded prompt.** `model.prompt_cache = {}` is set once in `Flux3Initializer.init` and never pruned, keyed by prompt text. On an `@st.cache_resource` model, every distinct (upsampled, so nearly always distinct) prompt stays resident for the whole server lifetime.
- **mflux's `MemorySaver` / `--low-ram` is not usable here.** It deletes the text encoder or transformer from the model (`model.text_encoder = None`), which is fine for a one-shot CLI and fatal for a cached model.

### Model card (black-forest-labs/flux-3-image-klein; card public, weights gated)

- The model is a 5.6B transformer with a **Qwen3-VL-4B-Instruct text encoder**. Up to 10 references.
- Distilled: 16 steps, no guidance. Base: 50 steps, guidance 3.5.
- "Do not use the model without a prompt upsampler"; a short prompt as typed gives generic output.
- Peak memory 31 GB BF16 on a B200 at 1024² without the upsampler.
- License: FLUX Kommunity License v1.0 covers non-commercial use, plus commercial output generation for Qualifying Users under US$5M gross annualized revenue.
  - Section 4 restrictions include using outputs to improve a competing model.
  - FLUX/BFL trademarks need permission except for attribution or "necessary descriptive references".

### BFL prompt upsampling (github.com/black-forest-labs/flux-3-image-klein, Apache-2.0, `main` @ `bfc8889a4497f846416f253a3972a1941ac698b9`, 2026-10-17)

- **Use `src/flux3/pup_prompts_external_vlm.json`, not the upsampler's `system_prompts.json`.** The README says an external VLM should be given that file. Both have keys `t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`, and their contents differ.
  - **The external `t2i` prompt is ~7,410 words (~10k tokens)**; `i2i` is ~1,002.
  - The `t2i` length dominates enhancement latency on a 2B model and must be measured (see Phase 2).
- **Decoding** (`src/flux3/pup.py`): `SAMPLING = {"temperature": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 3584}`.
  - At temperature 0 this is greedy; `top_p`/`top_k` are inert. That resolves the model card's apparent contradiction.
  - `PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}`. Stop tokens `<|im_end|>`, `<|endoftext|>`. `enable_thinking=False`.
- **Edit user message:** the reference images first, then exactly
  `Reference images follow in order (the first image, the second image, ...).\nUser request: {instruction}\nOutput the caption.`
  Text-to-image sends the bare request as the user message.
- **Reference preprocessing for the upsampler** (`policy_reference`): EXIF-transpose, alpha composited over white, RGB, sizes rounded to multiples of 32. Pixel budget by reference count: 1 MP for one, 724² for two, 512² for three or more. Then a JPEG q92 round trip.
- **mlx-vlm 0.7.2** (repo's locked version) supports `presence_penalty` in `generate()` (`mlx_vlm/generate/types.py`, `generate/ar.py`), so BFL's settings map 1:1: `temperature=0.0`, `presence_penalty=1.5|0.0`.
  - Phase 1b moves to **0.7.10**. Before Phase 2 builds on this fact or on APC, re-run the appendix script once on 0.7.10, and confirm that `presence_penalty` and `mlx_vlm.apc.APCManager` behave as measured below.

### Upsampling measured on this machine (2026-10-21)

Qwen3-VL-2B-Instruct-8bit with mlx-vlm 0.7.2, on the M2 Max. The setup matched the app (768px images, the same chat template), with BFL's external-VLM prompts and decoding. The script is in the appendix.

| Case | Prompt tokens | Prefill | Total | Output | Peak mem |
|---|---|---|---|---|---|
| t2i, capybara, cold | 9,920 | 7.8 s | 10.5 s | 243 tok / 188 words | 4.4 GB |
| t2i, kawaii sticker, cold | 9,912 | 6.8 s | 9.0 s | 198 tok / 159 words | 5.6 GB |
| t2i, kawaii sticker, **APC hit** | 9,912 | **0.2 s** | **2.5 s** | 201 tok | 5.4 GB |
| i2i, 3 references, cold | 2,573 | 3.2 s | 4.7 s | 190 tok / 164 words | 4.9 GB |
| i2i, "make it snowy", 1 ref, cold | 1,706 | 1.2 s | 2.1 s | 112 tok / 92 words | 5.3 GB |

- **Prefill dominates text-to-image.** The ~10k-token system prompt prefills at ~1,450 tok/s, so ~7 s of a ~9–10 s enhancement. Generation runs at ~91 tok/s (t2i) and ~130 tok/s (i2i).
- **mlx-vlm's automatic prefix cache (APC) removes it.** With one `mlx_vlm.apc.APCManager()` passed as `apc_manager=` to `generate()`, a *different* t2i request reusing the same system prompt prefilled in 0.2 s: 9.0 s → 2.5 s. It hit 5 of 8 times. Two oddities for the implementing session:
  - The very first APC call stores, which is expected. But the immediate repeat of that same capybara request also missed (6.8 s).
  - i2i calls gained little or nothing, likely because image tokens break the prefix. i2i is only 2–5 s cold, so it doesn't matter.
- **`VLM_MAX_TOKENS = 256` is too low.** BFL-style captions ran 112–243 tokens, and one came within 13 tokens of the cap. Set it to ~1024 (BFL's ceiling is 3584).
- **The VLM's peak is ~5.6 GB, not its 2.7 GB of weights.** The 10k-token context's KV cache accounts for the difference. Use the peak in the memory budget.
- **i2i output is supposed to be an instruction.** BFL's *external* `i2i` prompt asks for "ONE detailed, self-contained instruction":
  - "an imperative editing instruction" for a single reference that keeps its scene;
  - "Create a photograph of …" for compositions;
  - "Describe the CHANGE, not the whole scene."

  Both models complied. The model card's "complete caption of the reference with the edit folded in" describes BFL's own fine-tuned upsampler and its different prompt set.

#### Qwen3-VL-8B-Instruct-4bit, same script, same machine

Already in the owner's HF cache (5.78 GB), with no `refs/main`, so load it by its
local snapshot path when offline.

| Case | Prefill | Total | Output | Peak mem |
|---|---|---|---|---|
| t2i, capybara, cold | 28.4 s (~350 tok/s) | 36.0 s | 299 tok / 231 words | 8.0 GB |
| t2i, kawaii sticker, cold | 28.1 s | 32.0 s | 154 tok / 124 words | 9.5 GB |
| t2i, kawaii sticker, **APC hit** | 0.3 s | **4.0 s** | 143 tok | 9.1 GB |
| i2i, 3 references, cold | 7.6 s | 10.5 s | 158 tok / 131 words | 8.5 GB |
| i2i, "make it snowy", cold | 4.6 s | 5.7 s | 59 tok / 45 words | 8.7 GB |

#### 2B vs 8B-4bit (n = 4, read, not scored)

| | 2B-8bit | 8B-4bit |
|---|---|---|
| Medium matches request | **No**: the kawaii die-cut sticker became "Full-color outdoor photograph…" | **Yes**: "Full-color die-cut sticker illustration, kawaii anime style…" |
| Visual grounding | **Wrong**: called the cat photo's dark green shutters "blue" | Correct: "dark green shutters" |
| "Describe the change, not the scene" | Re-describes the scene (92 words for "make it snowy") | Follows it (45 words) |
| i2i voice | Correct | Correct |
| t2i with APC hit / cold | 2.5 s / 9–10.5 s | 4.0 s / 32–36 s |
| i2i (no APC benefit) | 2–5 s | 6–10.5 s |
| Peak memory | 4.4–5.6 GB | 8.0–9.5 GB |

#### APC behavior (both models)

- **The first request of a session never hits.** In both runs, the first request and its two immediate repeats all prefilled cold, while the *next, different* request hit. That's systematic, not noise.
  - Mitigation: warm APC when the VLM loads by running one throwaway t2i enhancement inside `_get_vlm()`'s load spinner. Every user request is then a "next" request.
  - With 8B, that warm-up costs ~30 s once per server start.
- **The default APC pool evicts under 8B.** It logged `memory_evictions=161` on 8B and 0 on 2B. If i2i calls evict the t2i system-prompt blocks, the next t2i request is cold again: ~28 s on 8B.
  - Size `APCManager(num_blocks=...)` to hold the t2i and i2i prefixes, and measure the pool's memory.
  - Then test the interleaved sequence t2i → i2i → t2i.

### Weight pinning

- **Revisions** (Hub, 2026-10-21): `mflux-community/flux-3-image-klein-mflux-q8` @ `7e46237dea3999a9792959f5e000aeb1b9f9e7df`, `-q6` @ `ace05578b60f0185b0c1212d23fb13616d88d6c7`, both last modified 2026-10-19.
- **mflux pins revisions only for repos in its own table.** `mflux.models.common.resolution.model_revisions.ModelRevisions.PINNED` maps repo ids to SHAs. `path_resolution.py` passes `revision=ModelRevisions.get(path)` to `snapshot_download`.
  - The table pins `black-forest-labs/flux-3-image-klein` (`8454017…`) but **not** the `mflux-community` builds, which therefore load `main`.
  - The table's own comment: "A local path, a mflux save or a repo that is not listed loads as before."
- **Pin through a local path:** `model_path=snapshot_download(MODEL_REPO, revision=MODEL_REVISION)` (huggingface_hub, public API) inside the getter. Don't patch `ModelRevisions.PINNED`.
  - `_reload_app()` must also patch that `snapshot_download`, or the mocked suite hits the network in CI.
  - Keep the repo id for the sidebar link.
  - A snapshot fetched by full SHA gets no ref, so `hf cache prune` would treat it as detached and delete it (Disk plan).
- **Pin the VLM the same way. Today it floats.** `_get_vlm()` loads `VLM_MODEL_ID` by repo id, so it resolves `main` at load time. That causes three problems:
  - **It confounds the Evaluation drift check.** Upsampling is greedy, so a changed enhanced prompt should mean *our* change. An upstream re-upload of the VLM would look like one.
  - **It breaks offline loading.** The cached `Qwen3-VL-8B-Instruct-4bit` has no `refs/main` (Facts above), so loading it by repo id fails with `HF_HUB_OFFLINE=1`. A `snapshot_download(..., revision=SHA)` resolves from the cache.
  - **It is inconsistent** with the reasoning for pinning FLUX 3.
  
  Revisions as of 2026-10-21 (Hub `main`; each matches the snapshot already cached where one is):
  - `mlx-community/Qwen3-VL-8B-Instruct-4bit` @ `defcdea7cc7a4b0858fea563cbbce171d328e457` (cached);
  - `mlx-community/Qwen3-VL-4B-Instruct-8bit` @ `0943db6e15185b86be368d3cf0704aec740b142b` (not cached);
  - `mlx-community/Qwen3-VL-2B-Instruct-8bit` @ `b0338e0e843d8e1befe873d144b81fefdc47efa6` (cached; today's VLM).
  
  **Phase 2:**
  - add `VLM_REVISION` beside `VLM_MODEL_ID`, for whichever rung wins;
  - load through `vlm_path = snapshot_download(VLM_MODEL_ID, revision=VLM_REVISION)`, then `load(vlm_path)` and `load_config(vlm_path)`;
  - patch that call in `_reload_app()`;
  - lock the pair with a constants test, as `test_model_repo_is_the_8bit_build` does;
  - keep the repo id for any sidebar link;
  - add a Do-not-touch row ("Don't load either model by bare repo id").
  
  The `hf cache prune` hazard applies to this snapshot too.

### Other

- **Pillow 13.0.0 is broken; stay on 12.x.** Its `PIL/_typing.py` does `from numbers import _IntegralLike`, a private name absent from uv's CPython 3.12.15, 3.13.12 *and* 3.14.6 alike (`ImportError: cannot import name '_IntegralLike' from 'numbers'`).
  - `import PIL.Image` alone succeeds, because it does not reach `_typing`. **mflux's model modules do reach it, and fail.** That holds for both `mflux.models.flux3.variants` and the app's current `mflux.models.flux2.variants` import (verified).
  - The app imports mflux at module level and every test imports the app, so the mocked suite fails at collection. **CI's `pytest` gate and the local Stop hook both catch it.** A one-line check like `python -c "import PIL"` does not.
  - `pillow==12.3.0` (the repo's locked version) works on all three.
- **uv pin:** bumped to `==0.13.4` (merged 2026-10-21, ahead of Phase 1). Verified on 0.13.4: re-locking gave no `uv.lock` diff, `uv version --bump` still re-locks, and `0.9.0-rc1` is still stored as `0.9.0rc1`. `uv self update --dry-run` reports 0.13.4 as the latest release, so the full upgrade needs no pin bump.

### Full dependency upgrade, rehearsed (2026-10-21)

`uv lock --upgrade` with `pillow<13`, in a scratch `git archive HEAD` copy of the repo. The real lock was not touched.

| Package | Locked | Latest | Note |
|---|---|---|---|
| mflux | 0.20.0 | 0.23.0 | |
| mlx / mlx-metal | 0.32.2 | 0.32.4 | macOS only; on Linux the lock forks and keeps 0.32.2 (its CUDA build stops there) |
| mlx-vlm | 0.7.2 | 0.7.10 | Linux fork stays at 0.7.6. 0.7.10 newly depends on `datasets` 5.1, which pulls ~12 packages (aiohttp, multiprocess, xxhash, …) and **downgrades fsspec** 2026.9.0 → 2026.7.0 |
| streamlit | 1.64.0 | **1.66.1** | breaks 9 tests, see below |
| transformers | 5.17.0 | 5.20.2 | |
| pillow | 12.3.0 | 13.0.0 | held at 12.3.0 |
| ruff | 0.16.8 | 0.17.1 | `format --check` and `check` both clean |
| ty | 0.0.83 | 0.0.90 | clean |
| Majors, transitive | | | websockets 16 → 17, python-dateutil 2.9 → 3.0, pyarrow 25 → 26 |

**Gates:** ruff format, ruff check and ty pass. **pytest: 9 failed, 228 passed**, and all 9 failures share one cause:

- **Streamlit 1.66's `AppTest` models form gating.** A `set_value()` on a form widget followed by a bare `.run()` is now discarded, as in production. The prompt arrives empty and the Run hits the empty-request guard ("Enter a prompt or add an input image.").
  - Every failing test does `at.text_input(key="prompt_input").set_value(...).run(...)` and then clicks Run. That pattern occurs 10 times in `tests/test_streamlit_app.py`.
  - **The fix works on both versions:** `set_value(...)` with no `.run()`, then `click().run()` on Run, submits the form in a single run. Verified on 1.64.0 and 1.66.1.
  - It **invalidates CLAUDE.md's Testing convention "`AppTest` does not model form gating"**. Rewrite it, since the old advice to route non-submit prompt changes through session state no longer applies.

Not yet checked: the smoke suite, and everything CLAUDE.md states as measured on Streamlit 1.64 (Phase 1b step 3).

## Memory budget (blocks merging Phase 2)

Today the app holds **two copies of the image model** (`_get_model()` and
`_get_edit_model()` are separate cache entries) plus the VLM. For FLUX 3 Q8 that
is 2 × 9.9 GB of FLUX weights plus the VLM, whose measured peak with BFL's
10k-token prompt is ~5.6 GB on 2B and **~9.5 GB on the preferred 8B-4bit** (see
Facts): ≈ 25–29 GB before any FLUX activations. Even with a single FLUX instance,
8B-4bit + Q8 is ~19.4 GB before activations, which is why the VLM choice is
gated on this budget. CLAUDE.md measured
24 GB real footprint from 8.6 GB of weights on today's model. On 32 GB that
plan swaps.

**Load one FLUX 3 model for both modes.** Since both classes are `_Flux3KleinCore`, a single
app-side subclass covers both through public methods only:

```python
class _Flux3KleinAnyMode(Flux3KleinEdit):
    # Flux3Klein.generate_image only calls self._generate, which the shared
    # core provides, so the edit model can also serve text-to-image.
    generate_text_image = Flux3Klein.generate_image
```

Then one `_get_model()` getter (dropping `_get_edit_model()`), wrapped in
`_materialized`, and `infer()` picks `generate_text_image` or `generate_image`
by whether `image_list` is set. Verify under ty, and with the smoke tests that
both modes produce images from the one instance. Fallback if it misbehaves:
keep two classes but load only one at a time (evict the other cache entry and
call `mx.clear_cache()` on a mode switch).

**Bound the prompt cache.** Clear `model.prompt_cache` in `infer()`'s existing
`finally` block, next to the callback removal. Nothing in the app re-uses an
encoding across runs except an identical re-Run, and that costs one text-encoder
pass.

**Serialize Runs: one at a time per server.** The app has no inference lock
today (verified 2026-10-21: nothing in `streamlit_app.py` takes one). With one
shared FLUX instance, two browser tabs or a double submit would run
`generate_image` concurrently on the same object, with three effects:
- both Runs' `_ProgressReporter`s are registered on the same `model.callbacks`, so each progress bar also counts the other Run's steps;
- one Run's `finally` clears `model.prompt_cache` under the other;
- two sets of activations break this budget.

The fix:
- Hold a process-wide `threading.Lock` from an `@st.cache_resource` getter. Take it around `infer()`, and separately around the VLM's `generate()`, since APC's pool is shared too.
- A Run that has to wait sets its status to "Waiting for the other Run to finish…", with `expanded=True` per the status rule.
- Test it with two threads: the second `infer()` starts only after the first's `finally`, and no reporter sees the other Run's steps.
- Add a Do-not-touch row ("Don't remove the inference lock"), naming that test.

**This budget assumes serialized Runs.**

**Budget:** measured on the Streamlit server process, with the VLM loaded and
enhancement on, across this sequence: a 1024² text-to-image Run, an edit with 3
references, and a text-to-image Run again.
- **The gate is the process's peak `phys_footprint` ≤ 26 GB**, which is 32 GB minus the 6 GB headroom the owner keeps for other apps; adjust if they say otherwise.
- The peak matters, not the after-Run value: swap starts at the peak.
- **How to measure:** sample `/usr/bin/footprint -p <pid>` (or `vmmap --summary <pid>`) about once a second during the sequence, and keep the maximum. Note the tool and its exact field in the record. MLX's `mx.get_peak_memory()` counts only MLX buffers and can sit inside that figure; record it too, but don't gate on it.
- **The worst case is measured once too:** an edit with **10 references**, the upload cap. That is 10 images of VLM image tokens plus 10 × 1 MP in FLUX 3. If it breaks the peak, lower the upload cap to the largest count that passes, and say so in the existing sidebar warning.
- If Q8 fails, try `mx.set_cache_limit`/`mx.clear_cache()` after each Run, then Q6.
- Record the numbers in CLAUDE.md's General gotchas, replacing the FLUX.2 figure.

## Disk plan

Free space was 41 GB when the plan was drafted and **40 GB** when this section
was revised (both 2026-10-21; `df -h ~`, 460 GB disk, 91% used). Every store
below shares that one disk, so the budget is their sum, not the model downloads
alone.

| Store | Size (2026-10-21) | What grows it |
|---|---|---|
| HF cache `~/.cache/huggingface/hub` | 66.1 GB across 23 repos (`hf cache ls`), mostly the owner's unrelated models | model downloads, revision bumps |
| uv cache `~/.cache/uv` | 11 GB | Phase 1b's upgrade adds the new wheels beside the old ones |
| `.venv` | 1.6 GB | Phase 1b (mlx-vlm 0.7.10 adds `datasets` and ~12 small packages) |
| App outputs | 0 | nothing, by design (see Rules) |

**The FLUX.2 weights are not in the HF cache**, so Phase 1a's golden run downloads
them. Already cached and reusable: `mlx-community/Qwen3-VL-2B-Instruct-8bit`
(2.7 GB) and `mlx-community/Qwen3-VL-8B-Instruct-4bit` (5.8 GB).

| Step | Change | Free after (approx.) |
|---|---|---|
| Start | — | 40 GB |
| Phase 1a golden run | + `mlx-community/flux2-klein-4b-8bit` 8.6 GB, + eval outputs (~10 PNGs per run) | ~31 GB |
| Phase 1b upgrade | + new wheels in the uv cache, + venv growth (measure with `du -sh ~/.cache/uv .venv` before and after) | ~30 GB |
| Phase 1b merged | − `uv cache prune` (see Rules) | measure |
| Phase 2 | + FLUX 3 Q8 9.9 GB | 20 GB |
| Phase 2, if needed | + Q6 7.6 GB and/or Qwen3-VL-4B-8bit ~5 GB | 8–15 GB, **below the floor**: needs the owner's answer on what to free (Rules) |
| After Phase 2 merges | − FLUX.2 8.6 GB, − whichever quant and VLM lost | back above 30 GB |

### How the HF cache stores files (verified 2026-10-21, huggingface_hub 1.32.0)

- **One content-addressed pool.** File bodies live in a shared `hub/blobs/<xx>/<sha256>` (61 GB). A repo's own `blobs/` holds only symlinks into it, so `du` on a `models--…` folder reports megabytes (4.4 MB for the 5.8 GB Qwen3-VL-8B). **Size repos with `hf cache ls`, never `du`.**
- **Identical files are stored once across repos.** The Q8 and Q6 builds may share unquantized files, so a fallback's real cost is the `df` difference after the download, not its repo size. Record the measured delta.
- **Never `rm -rf` a repo folder.** The weights are in the pool, so this frees almost nothing and orphans the blobs. Delete with `hf cache rm` after listing.
- **Never run `hf cache prune`.** Its dry run on 2026-10-21 would have deleted **56.8 GB: 16 entire repos**, including `Qwen3-VL-8B-Instruct-4bit` and nearly all the owner's models.
  - It removes "detached" revisions, meaning no ref points at them. Every repo here downloaded without a branch name has none.
  - huggingface_hub writes a ref only when `revision` is not the full commit hash. The app's FLUX 3 snapshot, pinned by full SHA (Weight pinning), will therefore be detached too, and a prune would delete the app's own weights. Confirm with `hf cache ls` after the first download (`refs` column empty).
  - Add this as a CLAUDE.md gotcha in Phase 2. From Phase 1a step 2 on, `guard-cache.sh` denies it, along with `uv cache clean` and recursive `rm` of either cache. That hook is a backstop, not a substitute for the rule.
- `hf cache ls` prints 2 warnings, for `hub/version.txt` and an empty `hub/cohr-converted/`. Both are harmless and not this project's to clean.

### Rules

- **Download headroom.** The 15 GB floor applies *after* the download. The hub writes partial `.incomplete` files at full size while downloading, so check that `df -h ~` free space ≥ download size + 15 GB. **If it isn't, stop and ask the owner**; macOS degrades when the disk is nearly full.
  - An interrupted download leaves `.incomplete` files behind. List them with `find ~/.cache/huggingface/hub -name '*.incomplete'` (there were 0 on 2026-10-21) and delete them with the owner's OK. Don't use `hf cache prune`, even though it also clears them.
- **The owner's models: ask once, up front.** Before Phase 2's first FLUX 3 download, show the owner `hf cache ls` and ask which models, if any, they would free if a fallback needs room. Record the answer here.
  - The largest unrelated entries: `Qwen-Image-2.1-mflux-q4` 9.6 GB, `granite-vision-4.1-4b` 8.0 GB, `c4ai-command-r7b-12-2024-4bit` 4.5 GB, `cohere-transcribe-03-2026-mlx-8bit` 4.1 GB, `tiny-aya-global-8bit-mlx` 3.6 GB, `North-Micro-Vision-Instruct-8bit` 3.2 GB.
  - Without an answer, a fallback that would cross the floor waits.
- **Deletions** go through `hf cache rm` (list first, confirm the exact repo id) and **only with the owner's go-ahead**.
- **uv cache.** After Phase 1b merges, run `uv cache prune` with the owner's OK. It removes only dangling entries and cached environments, all of them re-downloadable, and has no dry-run flag in 0.13.4. Record `du -sh ~/.cache/uv` before and after. **Never `uv cache clean`**: it empties the cache for all of the owner's projects.
- **Revision bumps.** A new `MODEL_REVISION` downloads a new snapshot beside the old one. Unchanged files are shared through the pool and changed ones are stored again. Once the new revision passes smoke, remove the superseded one with `hf cache rm` (check its help for revision-level removal), with the owner's OK. The old snapshot is detached, so nothing else will ever remove it.
- **The app writes no outputs to disk.** Results live in `st.session_state` and vanish with the session, and saving goes through the browser. Phase 2's edit-input PNGs live in a `TemporaryDirectory` for one `infer()` call. Any future feature that persists images (history, gallery) needs its own disk budget and a cap; see Possible Phase 5. State this in CLAUDE.md's General gotchas in Phase 2.
- **Release check** (see Release): confirm the cleanup row above actually happened.

## Evaluation

Agreed 2026-10-21. Before this section, the plan evaluated code (the four gates,
smoke) and resources (memory, disk) with thresholds, but **output quality only
by reading 4 upsampled captions unscored, and image quality not at all**. Yet
quality is what drives the costliest choices: the 8B VLM, 16 steps, and Q8 over
Q6. This section adds a time budget, a scored evaluation set, and drift
detection. It gates Phase 2's merge alongside the Memory budget.

| Layer | Method | Gate |
|---|---|---|
| Code | Four gates + Stop hook + CI per phase | All green (existing) |
| Real surface | `pytest -m smoke` | Green before each phase merge (existing) |
| Memory / disk | `phys_footprint`, `df` | ≥ 6 GB / ≥ 15 GB free (existing) |
| **Time** | Latency script, below | **≤ 60 s** warm text-to-image Run (new) |
| **Quality** | Scored evaluation set, below | FLUX 3 ≥ FLUX.2, no zero scores (new) |
| **Drift** | Golden runs + pixel/text diff | Flags for review; never auto-fails (new) |

### Latency budget (blocks merging Phase 2)

- **Budget: ≤ 60 s for one 1024² text-to-image Run, warm, upsampling on.** Warm means the models are loaded and the prompt cache has been primed by the load-time warm-up. The budget is the owner's (2026-10-21).
- **What is timed:** wall clock from the Run submit to the result drawn, which is `_resolve_prompt` plus `infer()`. Take the median of 3 Runs after one discarded warm Run, at fixed seeds.
- **Recorded, not gated:** an edit with 3 references; upsampling off; the cold first Run of a server (model load plus the warm-up, ~30 s of it on 8B). If an edit exceeds 60 s, bring the numbers to the owner rather than assuming the budget covers edits.
- **Baseline:** Phase 1b records FLUX.2's timings as shipped (4 steps, upsampling off by default) on the upgraded toolchain, so the Phase 2 numbers can be read against what users have today.
- **If it fails, the levers in order:**
  1. `DEFAULT_STEPS` 16 → 12;
  2. a 768² default size;
  3. the Q6 build;
  4. a smaller VLM rung.
  
  **Each lever must be re-scored on the evaluation set before it's adopted**: a faster Run that fails the quality gate is not a pass.
- **A script, not a test.** The threshold describes the owner's M2 Max, so a pytest assertion would be flaky on any other machine. Record the numbers in CLAUDE.md with date and versions, beside the memory figures.

### Evaluation set

- **10 fixed cases, all seeded.** Seed `20261021 + case index`; text-to-image at 1024², edits at the size the references give.
  - **Text-to-image (6):**
    1. the vase with exact hex colors (color adherence);
    2. the Berlin TV Tower infographic (dense typography and layout);
    3. the wet capybara (photorealism);
    4. the kawaii die-cut sticker (medium);
    5. "a red bicycle", a bare prompt that tests what upsampling adds;
    6. "three red apples to the left of a blue mug on a wooden table" (counting and placement).
    
    Cases 1–4 are the app's `EXAMPLE_PROMPTS`, verbatim.
  - **Edits (4):**
    7. the three-reference composition, `EDIT_EXAMPLES[0]`;
    8. "make it snowy" on `cat_window` (a change that preserves the scene);
    9. "turn this into a watercolor painting" on `woman1` (style, identity kept);
    10. "replace the bird with a hummingbird" on `bird` (object swap).
    
    The edit cases use whichever images Phase 1a's license check leaves in `examples/`. That check runs **before** the FLUX.2 golden run, so both versions are scored on the same inputs.
- **Run as shipped,** with each version's shipped defaults: steps, default size and upsampling. For FLUX 3, upsampling is on; for the FLUX.2 baseline it is off, its default. The "1024²" above is today's default. If Phase 2 adopts the 768² time lever, FLUX 3 is evaluated at 768², because that is what users would get. Record each enhanced prompt. Upsampling is greedy, so the enhanced prompt is deterministic per version and is part of the output.
- **Where things live:**
  - Tracked: `eval/cases.json` (prompts, references, seeds, sizes) and `eval/run_eval.py`. The script imports `streamlit_app`'s `infer`/`_resolve_prompt`, so it measures the app's real path, and also prints the latency figures above. It needs real weights, so it is not part of `pytest`. ruff and ty cover it as they do every tracked `.py`. Add `eval/` to CLAUDE.md's Architecture table.
  - **Outputs outside the repo** in `~/Library/Application Support/flux-image/eval/<label>/`, where the label is a version plus the date. Each run holds one PNG per case plus a `manifest.json`: case, prompt, enhanced prompt, seed, steps, size, model repo + revision, mflux/mlx/mlx-vlm versions, timings, scores. Record the revision **as resolved**: the snapshot SHA actually loaded, read from the cache, even for a model the app doesn't pin. FLUX.2 floats to `main`, so an upstream re-upload between Phase 1a and 1b would otherwise look like toolchain drift. The `.gitignore` safety net and the "no outputs in the repo" rule hold. Add the directory's size to the Disk plan; it's about 10 PNGs per run.

### Scoring rubric (the owner scores; 0–2 per criterion)

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| **Adherence**: subjects, attributes, counts, text, colors | misses the request | partly | all of it |
| **Preservation** (edits only): what should stay unchanged | identity or scene lost | drifted | intact |
| **Medium / style**: photo, sticker, watercolor, infographic | wrong medium | mixed | right |
| **Artifacts**: hands, garbled text, seams, warping | distracting | minor | none visible |
| **Upsampled prompt**: faithful, no invented subjects, describes the change for edits | contradicts the request | padded or partly off | faithful |

- **Blind:** `run_eval.py` also writes a contact sheet (an HTML file next to the outputs, never published). It shows each case's two versions side by side, in random left/right order and unlabeled; the key is kept in the manifest. That keeps "the new one" from scoring itself up.
- **Phase 2 quality gate:**
  - FLUX 3's **image total** ≥ FLUX.2's over the 10 cases. The image total sums only the four image rows: adherence, preservation (edits only), medium and artifacts. The "upsampled prompt" row is left out because the FLUX.2 baseline ships with upsampling off, so it has no upsampled prompt to score. Including it would compare totals with different maximums (10 points per edit case for FLUX 3 against 8 for FLUX.2);
  - **no case scores 0** on adherence, or on preservation for an edit;
  - upsampling has its own gate, on FLUX 3 only: no case scores 0 on the "upsampled prompt" row.
  
  Record per-case scores in the manifest and the totals in the Phase 2 commit body.
- **The VLM fallback ladder is judged by quality too, not memory alone.** Score each rung that passes the memory budget on the upsampled-prompt row, the 2B last resort included. Replace the 4-case read in Facts with these scores.

### Drift check (golden runs)

- **Golden runs:**
  - `golden-flux2-0.9.0`, captured in Phase 1a **before** the lock changes;
  - `golden-flux3-<rev>`, captured when Phase 2 merges.
  
  A golden run is an evaluation-set run marked as the reference.
- **Re-run the set with the same seeds** after Phase 1b's upgrade, any later mflux/mlx/mlx-vlm bump, a `MODEL_REVISION` bump, and before the release. Compare against the golden run:
  - images: mean absolute difference on a 256px grayscale downscale, with numpy and Pillow only (already installed). **Flag** any case above 2% of the range for a human look;
  - enhanced prompts: an exact text comparison. Greedy decoding should reproduce them on unchanged versions, so any change is a flag.
- **Flags never fail anything by themselves.** Version bumps legitimately move pixels. A flagged case is re-scored against the rubric, and only a score drop blocks.
- The threshold is a starting point. Record the actual differences from Phase 1a's repeat run (same versions) and Phase 1b's comparison (new versions), and tune it from those.

## Phase 1a: hygiene and evaluation baseline (`chore/hygiene-and-eval`)

Goal: put the guards and the measuring stick in place **before anything moves**.
That means the cache guard hook, the `.gitignore` safety net, and the evaluation
set with its FLUX.2 golden run, captured on the **current, unchanged lock**. It
is the only chance to record the pre-upgrade baseline. Four commits: the cache
guard, `.gitignore`, the example images' license, then the evaluation set. The
two hygiene commits land before anything downloads, and the images are final
before the golden run uses them. One session; merge and pass CI
before Phase 1b.

1. ~~Bump the uv pin to 0.13.4~~: done, and it is the latest uv (see Facts).
2. **Cache guard hook, first and in its own commit**, before this phase downloads
   anything or anyone runs the post-merge `uv cache prune`. The Disk plan's two
   "never" rules (`hf cache prune`, `uv cache clean`) are prose today. This hook
   enforces them. It is repo-shared, not local: the prune hazard comes from the
   app's own SHA pinning (Weight pinning), so it affects anyone who clones the repo.
   - **`.claude/hooks/guard-cache.sh`, a `PreToolUse` hook on matcher `Bash`**, wired in `.claude/settings.json` as its own matcher group beside `guard-paths.sh`'s, with the `${CLAUDE_PROJECT_DIR}` prefix. Track it executable (`100755`). It reads `.tool_input.command` and denies with the same `permissionDecision: "deny"` JSON as `guard-paths.sh`, with a reason that names the Disk plan rule.
   - **Deny:**
     - `hf cache prune` without `--dry-run`, also through `uv run`, `uvx` or a leading `VAR=value`;
     - `uv cache clean`, in every form (a bare one empties the cache for all of the owner's projects);
     - a recursive `rm` whose target lies under `~/.cache/huggingface`, `$HOME/.cache/huggingface`, `$HF_HOME`, `~/.cache/uv`, a `hub/blobs` path or a `models--*` folder (the latter orphans pool blobs, see the Disk plan). Match paths case-insensitively, as `guard-paths.sh` does, because macOS is case-insensitive.
   - **Allow:** `hf cache ls`, `hf cache rm`, `hf cache prune --dry-run`, `uv cache prune`, `uv cache dir`, `du`, `df` and `find` without `-delete`. These are the commands the Disk plan prescribes.
   - **Match commands, not text.** A pattern counts only at the start of a command segment: the start of the string, or after `;`, `&&`, `||`, `|`, `(` or a newline, past any `VAR=value`, `sudo`, `uv run` or `uvx` prefix. Otherwise a `git commit -m` that merely *mentions* `hf cache prune` would be denied, and this plan's own commit messages do.
   - **Missing `jq`:** fall back to matching the raw payload anywhere, which over-denies (fail-closed, as `guard-paths.sh` does for a path it can't read). Name the fix in the reason: install `jq`.
   - **Tests in `tests/test_hooks.py`.**
     - Add a `_run_bash_hook(script, command, ...)` helper; `_run_hook` hard-codes an Edit payload.
     - Add a `TestGuardCache` class covering each deny and each allow above, the `uv run`/env-prefix forms, the `git commit -m "... hf cache prune ..."` allow, the case-insensitive path match, and the jq-less fail-closed path.
     - **Update `test_scripts_are_bound_to_the_correct_events`** to `PreToolUse == {"guard-paths.sh", "guard-cache.sh"}`.
     - **Narrow `test_tool_matchers_cover_edit_write_multiedit`.** It asserts that *every* `PreToolUse` group covers `Edit|Write|MultiEdit`, so the new `Bash` group fails it. Make it select the groups wired to the path and format hooks, and don't loosen it. Add a separate assertion that the cache guard's group matches `Bash`.
   - **Docs:**
     - In CLAUDE.md's hooks section, add a `guard-cache.sh` bullet with the same "backstop, not a security boundary" scope note as `guard-paths.sh`: an alias, a script or Python's `shutil.rmtree` all bypass it.
     - Add a Do-not-touch row ("Don't remove `guard-cache.sh` or narrow its prune/clean patterns", guarded by `TestGuardCache`) and a row on matching at command-segment starts.
     - Re-sync README's Hooks section.
   - Hooks are snapshotted at session start, so run `/hooks` to activate it, then confirm in-session that `uv run hf cache prune --dry-run` is allowed. Never test the deny by running the real command.
   - Commit `harden(hooks): deny cache-wiping commands the disk plan forbids`, with the dry-run's 56.8 GB figure in the body.
3. **`.gitignore` safety net, its own commit**, also before any download. Keep the
   hand-curated file. Don't adopt GitHub's stock `Python.gitignore`: here, being
   ignored also means an edit no longer arms the Stop hook's suite
   (`mark-tests-pending.sh` arms on "not gitignored"). The secrets block also
   mirrors `guard-paths.sh`/`test_secrets.py`, with `!.env.example`-style
   exceptions the template lacks. Append one block:

   ```gitignore
   # Model weights and local data. Phase 2 downloads into the HF cache and
   # Phase 5 keeps history outside the repo; these catch a misconfigured path.
   *.safetensors
   *.gguf
   *.db
   *.sqlite
   *.sqlite3
   *.db-journal
   *.db-wal
   *.db-shm
   ```

   - **Weights rules:** a `local_dir` or `HF_HOME` pointed into the repo would otherwise stage 9.9 GB. GitHub rejects the push, but only after the commit exists.
   - **Database rules:** a stray history database would commit prompts in plain text.
   - **Never add `*.png` or `*.webp`.** `docs/screenshot-dark.png` and `examples/*.webp` must stay tracked (`test_readme.py`). Nor any rule that could catch Phase 2's vendored JSON or its `LICENSE`/`NOTICE` sidecar.
   - **A two-way test in `tests/test_secrets.py`**, which already owns "what must never be committed". Using `git check-ignore`:
     - assert a sample path for each new pattern is ignored (`model.safetensors`, `history.db`, `history.db-wal`, …);
     - assert the assets the plan relies on are **not** ignored: the hero, each example image, `uv.lock`, `.python-version`, `.claude/hooks/guard-cache.sh`, plus the vendored JSON and sidecar paths once Phase 2 fixes them;
     - mirror `test_secrets.py`'s existing pairing of the `.env` deny with its template allows.
   - **Measurement and screenshot artifacts are not ignored; they live in scratch.** That covers the appendix script, any BFL JSON copy it reads, and headless Chrome's `--user-data-dir`. Run them from Claude's scratch directory or a temp dir, never the repo root, so no rule is needed and nothing is staged by accident.
   - Commit `chore(git): ignore model weights and local databases`.
4. **Settle the example images' license first** (moved here from Phase 2). The
   edit cases are built on `examples/*.webp`, and the golden run below is the
   only pre-upgrade baseline, so the images must be final before it.
   - Find the FLUX.2 [klein] Space's license and the images' origin, and record both in the README.
   - If either can't be confirmed, replace them now with BFL's Apache-2.0 `examples/inputs/` or images the owner made. Update `EDIT_EXAMPLES` and its provenance comment, and re-run the tests that pin the example dimensions (e.g. woman1's 512×768 → 672×1024).
   - Commit `docs(examples): record the example images' license` or `chore(examples): replace unlicensed example images`.
   
   This makes Phase 1a four commits.
5. **Evaluation set and FLUX.2 golden run.** Download the FLUX.2 weights (Disk
   plan: check `df` first). Add `eval/cases.json` and `eval/run_eval.py`
   (Evaluation) and commit `feat(eval): add the seeded evaluation set and runner`.
   Run the set **on the current lock** and keep the result as
   `golden-flux2-0.9.0`, latencies included. Then run it once more, unchanged,
   and compare it with the golden run. Bit-identical output answers open
   question 10's determinism half, and the measured differences are the first
   data for the drift threshold.

## Phase 1b: toolchain (`chore/mflux-0.23`)

Goal: upgrade **every dependency to its latest release** (Decisions) **on the
current FLUX.2 model**, so any breakage is attributable to the toolchain alone.
Two commits, mflux then everything else, so a smoke regression can be bisected
to one side or the other. Starts from Phase 1a merged; one session.

1. **mflux alone.** `uv lock --upgrade-package mflux`. Confirm in the diff that mflux becomes 0.23.0
   and mlx 0.32.4, and that **Pillow does not move**. Gates, then
   **`uv run pytest -m smoke`** on real FLUX.2 weights. Watch most
   closely the tests behind the Do-not-touch rules that depend on mflux internals:
   - `test_cached_models_generate_on_a_new_thread` (the `_materialized` stream bug: does it still reproduce on mlx 0.32.4 without `mx.eval`? keep the guard either way unless proven dead);
   - `test_progress_callback_fires_and_is_deregistered` (contract read above, but run it);
   - `test_edit_accepts_the_pil_images_the_ui_builds` (FLUX.2's edit class, still PIL-tolerant).

   Commit `chore(deps): upgrade mflux to 0.23`.
2. **Everything else.** Change `"pillow"` to `"pillow<13"` in `pyproject.toml`, with a
   comment citing the `_IntegralLike` import error, which mflux's model modules
   trigger and `import PIL.Image` alone does not (Facts). Then `uv lock --upgrade`. Check the diff against the
   rehearsal table in Facts. If a package moved further since 2026-10-21, note it
   and re-run what that package's row would have affected.
   - Try `pillow` unconstrained first: a 13.0.x fix may have shipped. Test the fix with `uv run python -I -c "import mflux.models.flux3.variants"` or the suite, not `import PIL`. Drop the cap only if that passes on 3.12.
   - Re-run the dependency license scan on the upgraded `.venv`: every installed distribution's `License-Expression` or classifiers, flagging anything copyleft or unknown. On 2026-10-21 (old lock) all 92 were permissive, apart from MPL-2.0 file-level copyleft in `certifi`/`tqdm` (no obligation unless their files are modified) and `hf_transfer` with blank metadata (confirm from its repo). The upgrade adds `datasets`, `aiohttp` and others, so check those. Record the result in the commit body.
   - Fix the 10 `set_value(...).run(...)`-then-Run sites in `tests/test_streamlit_app.py` with the same-run submit pattern (Facts), and rewrite CLAUDE.md's form-gating Testing convention to match.
   - Run the gates, then **`uv run pytest -m smoke` again**. mlx-vlm 0.7.2 → 0.7.10 is a jump the mocked suite cannot judge. `test_tokenizer_decode_drops_the_grounding_markers` matters most: if mlx-vlm now skips Qwen3-VL's grounding markers itself, record it, but keep the `token_ids` decode.
   - **Drift check against `golden-flux2-0.9.0`** (Evaluation): re-run the set on the upgraded lock and compare. Re-score flagged cases, and record the measured differences, which tune the 2% threshold. Record FLUX.2's latency on the new toolchain as the Phase 2 baseline.
3. **Re-verify what CLAUDE.md pins to Streamlit 1.64**, using the `developing-with-streamlit` skill for 1.66 docs. Each item is either re-measured or confirmed on 1.66.1, and its version note updated:
   - the canvas fold measurements (576² at 1440×880, 480² at 1280×720, 660² from 1920×968, the collapsed-sidebar misses, the 640px column stacking) and `TestCanvasGeometry`'s offsets (canvas top 152, caption 30, padding 160), re-measured with the screenshot recipe;
   - the theme frontend facts `TestThemeConfig`'s docstring lists: the minified `xn`/`yn` mode builders, `secondary` set to `primary` in any custom theme, the −15-point hover lightness, and the `textColor` console warning. Check **light mode** too, since Phase 3's light theme relies on the same behaviors there: the white primary label, the sidebar falling back to `secondaryBackgroundColor`, the 0.6 caption opacity, and light alerts drawn as the hue at 10% alpha (`f(hue, .9)` in 1.64's `St`). Also confirm the stock light alert palette recorded under Phase 3;
   - `st.status` closing on a label change without `expanded=True`;
   - `st.button`'s single-line default in columns (the `wrap=True` row), and an int `initial_sidebar_state` still auto-collapsing;
   - the `st.image` hover toolbar, and whether a per-element toolbar switch now exists.
   The hero screenshot is re-shot in Phase 3 anyway. Until then, leave its "on Streamlit 1.64.0" note alone.
4. Update version-specific prose in CLAUDE.md ("verified on mlx 0.32.2 / mflux
   0.20.0", "since mflux 0.20 …", "new in mlx-vlm 0.7.1", the Streamlit
   1.64 notes above) to what was re-verified. Also note that the lock now
   carries `datasets`, which mlx-vlm 0.7.10 requires and the app never imports.
5. Commit `chore(deps): upgrade all dependencies to latest`, with the version table and test fixes in the body.

## Phase 2: model swap (`feat/flux3-klein`, commit as `feat!:`)

### Exit criteria: ship or abort (decided 2026-10-21, before starting)

Phase 2 merges only if **one combination** of FLUX build (Q8, Q6) and VLM rung
(8B-4bit, 4B-8bit, 2B-8bit) passes **all three** gates together, measured on the
target machine:

1. **Memory:** peak `phys_footprint` ≤ 26 GB over the budget's sequence, with Runs serialized (Memory budget).
2. **Time:** ≤ 60 s warm text-to-image (Latency budget).
3. **Quality:** FLUX 3's image total ≥ FLUX.2's, no zero scores, and no zero on the upsampled-prompt row (Evaluation).

The time levers (12 steps, a 768² default) count as part of a combination, each
re-scored. Try the combinations in the Decisions order and **take the first
that passes all three**, not the best on any one gate.

**Abort path: no combination passes.**
- **Stop. Don't add rungs**: no 3-bit build, no third-party quant, no CPU offload. Those are new decisions for the owner, not fallbacks.
- **Don't merge `feat/flux3-klein`.** Push the branch and keep it as a record, with the measurements in its last commit body.
- **Ship what is independent of FLUX 3** as **0.9.1** (`uv version --bump patch`), or 0.10.0 if the light theme lands, since that's a feature: Phase 1a/1b, plus Phase 3's light theme and the parts of Phase 3 that don't name FLUX 3. The Darkroom name can ship on FLUX.2. The FLUX 3 license disclosure and the GitHub topic swap **do not** ship. Re-check the README and repo details still say FLUX.2.
- **Record it:** in this plan, a dated "FLUX 3 shelved" note with each combination's three numbers. Add a memory entry so a later session doesn't redo it blind.
- **Revisit when** any of these changes: an official `mlx-community` FLUX 3 build, an mflux release with memory work (e.g. text-encoder offload between Runs), or different hardware. Then re-run the same three gates.

### Model loading and inference (`streamlit_app.py` ~lines 8–9, 72–87, 125–141, 380–420)

- Import `Flux3Klein, Flux3KleinEdit` from `mflux.models.flux3.variants`. Add the
  single-instance subclass from the Memory budget section, with one getter:
  `model_path=MODEL_REPO`, `model_config=ModelConfig.flux3_image_klein()`, wrapped in `_materialized`.
- `MODEL_REPO = "mflux-community/flux-3-image-klein-mflux-q8"` plus
  `MODEL_REVISION = "7e46237dea3999a9792959f5e000aeb1b9f9e7df"`, loaded through
  the local-path pin described under Weight pinning. Rewrite the comment block:
  size, community conversion, why the pin, and where quantization is recorded
  (check the safetensors metadata as CLAUDE.md describes for today's build).
  Add a Do-not-touch row: don't pass the bare repo id to `model_path`, because
  mflux-community repos are not in mflux's revision table and would float to `main`.
- `DEFAULT_STEPS = 16` (mflux's own default and the model card's).
- **Drop `GUIDANCE`** and stop passing `guidance`: mflux nulls it for the
  distilled checkpoint (Facts). Replace the "pinned to 1.0" gotcha and its
  Do-not-touch row with that fact. Update the tests that assert `guidance=GUIDANCE`.
- **Normalize each input image once, where it is opened.** That covers uploads and examples alike, before both FLUX and the VLM see it. Apply `ImageOps.exif_transpose`, so a sideways phone photo is described *and* edited upright; BFL's `policy_reference` does the same for the upsampler. Then composite any alpha over white and convert to RGB, as `policy_reference` does. Both paths, `_vlm_images()` and the edit branch, then work from the same upright pixels. Test it with a mocked upload carrying an EXIF orientation tag.
- **Fix the edit input.** `image_paths` must be real paths. In `infer()`'s edit branch, write each normalized image to a `tempfile.TemporaryDirectory()` as PNG and pass the paths; the directory lives for the call.
  - First **downscale a copy to mflux's own budget**: ≤ 4 MP for a single reference, ≤ 1 MP each for several (Facts). mflux would shrink it anyway, and PNG-encoding a full 24 MP phone photo costs seconds for nothing.
  - Use copies, as `_vlm_images()` does, since `thumbnail()` resizes in place.
  
  Then:
  - Replace the Do-not-touch row "Don't fix `image_paths` to accept only `str` paths" and the matching Gotcha, which are both false on FLUX 3.
  - Rewrite `tests/test_smoke.py::test_edit_accepts_the_pil_images_the_ui_builds` to drive the real UI-shaped input through `infer()`.
  - Add a mocked test asserting `generate_image` receives `str`/`Path` entries.
- Clear `model.prompt_cache` in `infer()`'s `finally` (Memory budget).
- **Add the inference lock** (Memory budget, "Serialize Runs"), with its two-thread test and Do-not-touch row. It lands in the same commit as the single shared instance, because that is what makes concurrent Runs collide.
- Sizes: keep `MAX_IMAGE_SIZE` 1024 and the 32-rounding in `_dimensions_from_images`
  (valid for mflux's 16-rounding). Larger sizes cost memory this machine lacks.
- Uploads: cap at 10 references (mflux raises beyond that). Surface it as a
  sidebar warning, not an exception. **The cap may end lower:** the Memory
  budget measures the 10-reference worst case, and if that breaks the peak, the
  cap becomes the largest count that passes. Keep the number in one constant
  that the warning and its test read.

### Prompt upsampling (~lines 56–69, 144–265, 591–596)

- Vendor `pup_prompts_external_vlm.json` (`t2i` and `i2i` only; leave the `_bbox`
  keys for Phase 4) from BFL @ `bfc8889a`, under Apache-2.0, with an attribution
  comment and the SHA. A JSON file next to the app, loaded at import, is better
  than 8k words of string constants. Replace `UPSAMPLE_PROMPT_TEXT_ONLY`/`_WITH_IMAGES`.
  The "don't reword without re-measuring" rule becomes "these are BFL's prompts; don't edit them".
- Edit messages use BFL's three-line scaffold (Facts). Text-to-image sends the bare request.
- Decoding: `temperature=0.0`, `presence_penalty=1.5` for text-to-image and `0.0` for
  edits. Drop the old `top_p`, `repetition_penalty` and `repetition_context_size`
  and their justifying comments. ty does not check `generate()`'s kwargs keys, so add a
  mocked test asserting the exact kwargs passed.
- `VLM_MAX_TOKENS`: raise from 256 to ~1024. Measured captions ran 112–243 tokens
  (Facts), and BFL allows 3584. Keep the truncation guard.
- Image preprocessing for the VLM: consider matching BFL's `policy_reference` budget
  (1 MP / 724² / 512² by count) instead of the flat 768px `VLM_MAX_IMAGE_SIZE`. Measure
  prefill cost against caption quality before changing it.
- **Add APC for the fixed system prompt.** Measured 9.0 s → 2.5 s per t2i
  enhancement (Facts). Hold one `APCManager()` alongside the VLM: return it from
  `_get_vlm()` as a fourth element, or give it its own `@st.cache_resource`, so it
  lives as long as the model. Pass it as `apc_manager=` to every `generate()`.
  - Warm it at load, and size its pool (see APC behavior in Facts). `APC_*` env settings and `apc.stats` are the levers.
  - Check APC is safe across Streamlit's per-rerun threads. It holds an `RLock`, but its stored KV arrays are MLX arrays, so this is the same per-thread stream question as `_materialized`. Smoke-test enhancement on two successive reruns.
  - Measure the APC pool's memory against the budget.
  - Don't trim BFL's prompt by hand.
- Toggle: keep the key `auto_enhance_toggle` and the label "Prompt upsampling", flip the
  default to `True`, and say in its caption that FLUX 3 expects full scene captions,
  so switch it off only when pasting one. About ten tests set it explicitly; also
  update any that assume the default is off.
- **Switch `VLM_MODEL_ID` to `mlx-community/Qwen3-VL-8B-Instruct-4bit`**, pinned with `VLM_REVISION` (Weight pinning; Decisions)
  and walk the fallback order against the memory budget. Measure each combination
  with both models resident: a t2i Run, then an i2i Run with 3 references, then a
  t2i Run again, which also exercises APC eviction.
  - If falling back to 4B-8bit, re-check medium and grounding on the four cases in Facts with the appendix script (pass the model id as the third argument).
  - Record the chosen pair, its footprint and its latencies in CLAUDE.md, replacing the 2B-era Qwen3-VL gotchas that no longer apply (sampling parameters, the 120-word cap, the "prompt wording does not fix it" note).
  - The 768px `VLM_MAX_IMAGE_SIZE` measurements were taken on 2B. Re-check the prefill cost on 8B, since 8B prefills ~4× slower.

### Copy, docs, tests

- Sidebar model caption (~line 666): FLUX 3 Image Klein Q8, link to `MODEL_REPO`,
  **plus the license name linked to the FLUX Kommunity License** (Decisions), so
  anyone running the app sees which terms its outputs fall under.
  - Lock it with an `AppTest` assertion that the sidebar caption names and links the license.
  - The caption sits in the sidebar, so the canvas fold budget is unaffected unless the example labels re-wrap. Check that at the measured viewports.
  - Re-check the link's contrast against `TestThemeConfig`'s caption-opacity pair. It uses the same `linkColor`, so it should already pass.

### License disclosure (moved from Phase 3; same commit as the model swap)

From the commit that loads FLUX 3, the README's current License section ("all
permissive", FLUX.2 "Apache-2.0") is false. The fix ships in that commit:

- **README License section:**
  - Add the FLUX Kommunity License: link, the non-commercial and US$5M summary, and the Section 4 restrictions, including no use of outputs to train a competing model.
  - State that the weights come from `mflux-community`'s unofficial conversion, and that the license binds whoever uses them.
  - Add that FLUX and Black Forest Labs are trademarks of their owner and the project is not affiliated.
  - Credit BFL's Apache-2.0 prompts and Qwen3-VL's own license (read the chosen VLM's model card; don't assume).
  - Drop "all permissive".
  - Keep `[MIT License](LICENSE)` intact (`test_readme_declares_project_license`).
- **Tests in `tests/test_license.py`:** assert the README names and links the model's license, and no longer claims "all permissive".
- **Vendored prompts' notice.** JSON cannot hold a comment, so the plan's "attribution comment" cannot live in the file itself.
  - Ship a sidecar: Apache-2.0's text plus a short notice (source repo, path, SHA `bfc8889a`, "subset: `t2i` and `i2i` keys") next to the JSON, as `LICENSE`/`NOTICE` in its directory.
  - Copy BFL's own `NOTICE` too, if their repo has one.
  - Add a test that the sidecar exists, is git-tracked, and names the SHA the loader's comment cites.
- **Bundled example images:** their license was settled in Phase 1a (before the golden run). Here, only fold that record into the rewritten License section. If they were replaced with BFL's `examples/inputs/`, attribute them alongside the prompts.
- **CLAUDE.md:** a Do-not-touch row ("Don't remove the model-license line from the README or the sidebar caption"), naming both tests. Also update the `LICENSE` row in the Architecture table, which says the three MIT declarations are the whole story.
- README Setup: download sizes (9.9 GB plus the VLM). Rewrite CLAUDE.md's Overview
  and the whole "mflux / FLUX.2 Klein" gotchas section from the Facts above and the
  smoke results, not carried over. Drop what no longer holds (no pre-quantized base
  build, VAE quantization counts, two separate weight copies).
- Run the full smoke suite with the new classes, adding a both-modes-from-one-instance test.
- **First-run download status.** On a fresh machine, the first Run triggers ~10 GB of FLUX 3 plus ~5.8 GB of VLM downloads. Today that happens inside a generic spinner, and huggingface_hub's progress bars go only to the terminal, so the app looks hung for many minutes.
  - Before calling each getter, check whether its pinned snapshot is already local: `snapshot_download(repo, revision=REV, local_files_only=True)` raises `LocalEntryNotFoundError` when it is not.
  - If it isn't local, set the Run's `st.status` label to "Downloading FLUX 3 Image Klein, 9.9 GB, first run only…" (or the VLM's name and size, in `vlm_load_slot`) with `expanded=True`, per the existing status rule. Otherwise keep today's labels.
  - Mocked tests cover both branches, cached and not cached, asserting the label text and that the check uses the pinned revision.
  - Keep the sizes in one place, beside `MODEL_REPO`/`VLM_MODEL_ID`, so the README and the label can't disagree.
- **README "Upgrading to 0.10" section**, short:
  - new downloads (sizes from those constants);
  - upsampling now **on** by default, and how to turn it off;
  - slower Runs, with the measured median from the Latency budget;
  - the license change;
  - **removing the old FLUX.2 weights safely with `hf cache rm mlx-community/flux2-klein-4b-8bit`, never `hf cache prune`**, with one line on why (Disk plan).
  
  Add a `tests/test_readme.py` assertion that the section exists and names `hf cache rm`.
- **Evaluation gates** (Evaluation section), alongside the Memory budget, before merge:
  - the latency budget, ≤ 60 s warm text-to-image;
  - the quality gate on the 10-case set: FLUX 3's image total ≥ FLUX.2's (the four image rows only), no zero scores, scored blind from the contact sheet;
  - every VLM rung considered is scored too.
  
  Save the merged state's run as `golden-flux3-<rev>`. Put the totals, per-mode latencies and the chosen levers, if any, in the `feat!:` commit body.
- **Keep `eval/run_eval.py` working across the swap.** Phase 1a wrote it against FLUX.2's `infer()`, and this phase drops `guidance`, routes both modes through one instance, and changes the defaults.
  - The runner should call only the app's public path (`_resolve_prompt` and `infer()`, with the shipped defaults read from the app's constants). It should never call the getters or pass model-specific keywords, so that the swap needs no runner change beyond what `infer()`'s signature forces.
  - Re-run it on the Phase 1b lock (FLUX.2) once after any runner change, to confirm it still reproduces `golden-flux2-0.9.0` before scoring FLUX 3 against that baseline.
- `EDIT_EXAMPLES` comment says "from the FLUX.2 [klein] space". Check they still give
  sensible edits; consider BFL's own examples (`examples/inputs/`, Apache-2.0) if not.
  Their license was settled in Phase 1a. Replacing them now, for quality
  rather than license, invalidates the edit cases' FLUX.2 baseline. If that
  happens, re-capture those 4 cases on FLUX.2 *before* merging Phase 2, while
  the Phase 1b lock still runs it, and record why.
- **Disk:** follow the Disk plan section.

## Phase 3: rebrand to Darkroom (`feat/darkroom-rebrand`)

- `APP_TITLE = "Darkroom"`, plus a subtitle naming the model descriptively. Check
  `page_title`, `st.title` and `st.logo` usage, and `tests/test_streamlit_app.py:192`.
  A subtitle line may shift the measured fold budget. Re-measure with the screenshot
  recipe if the sidebar header grows.
- README: title, intro, hero alt text. CLAUDE.md: Overview's first line. The palette
  is already called "Darkroom".
- README License section: the license content landed in Phase 2. Here, only re-read
  it for the new name: "Darkroom" must not suggest a BFL product, so keep the
  non-affiliation line next to the title's first mention of FLUX.
- `pyproject.toml` `description`: set it to the exact description agreed under
  Release step 4, which the GitHub repo description will copy verbatim. It
  currently says "FLUX.2 Klein", so Phase 2 may already have touched it; Phase 3
  sets the final text.
- Re-shoot `docs/screenshot-dark.png` with the recipe in CLAUDE.md (new generation with
  FLUX 3, update the "last re-shot" note), and add its light counterpart (Light
  theme, Screenshots, below).

### Light theme: "Darkroom" for daylight (designed 2026-10-21)

Today light mode is stock. Its blue-grey `#F0F2F6`/`#31333F` and red `#FF4B4B`
share nothing with the dark theme: a white Run label at 3.30:1 (fails AA), a red
progress bar at 2.95:1, and a navy cast around the image. Replace it with a
light palette derived from the dark one, so switching modes changes lightness,
not identity.

**Rules that make the two complementary** (each becomes a test, below):

1. **Neutrals swap.** Light page and text are dark text and page (`#FAFAFA` ↔ `#121212`), all R=G=B. Neither mode tints the image being judged. Light uses `#FAFAFA`, not pure white, to cut glare around a picture.
2. **One brand color.** `primaryColor` is `#C24E0A` in both modes. Its white Run label is 4.79:1 in both.
3. **Accents mirror across their surface.** Same OKLCH hue, lightness flipped. The sidebar ember sits *lighter* than the brand color in dark mode (L 0.671) and *deeper* in light mode (L 0.538). Gold stays at hue 86.0° and slate at 252.5°.
4. **Alert fills are shared.** The same gold and slate tints at the same alpha, so a warning looks like a warning in both modes.
5. **Panel steps are tuned per mode, not mirrored.** Mirroring dark's page-to-panel contrast ratio (1.25) gave `#E2E2E2`, too heavy in light mode, and it pushed the caption link to near-black. Light uses `#F0F0F0`: stock light's panel lightness without its blue tint.

**Proposed `.streamlit/config.toml` addition** (the dark tables are unchanged):

```toml
[theme.light]
primaryColor = "#C24E0A"              # shared ember: white label 4.79:1, 4.59:1 on the page
backgroundColor = "#FAFAFA"           # dark's textColor
secondaryBackgroundColor = "#F0F0F0"  # neutral stock-light panel tone
textColor = "#121212"                 # dark's backgroundColor; pinned (console warning)
linkColor = "#280900"                 # renders #786560 through the 0.6 caption: 4.81:1
yellowBackgroundColor = "#E9AB2B29"   # dark's gold fill
yellowTextColor = "#7D5E01"           # gold, hue 86.0: 5.20 page / 4.83 sidebar
blueBackgroundColor = "#5B8FD624"     # dark's slate fill
blueTextColor = "#36669B"             # slate, hue 252.5: 4.96:1
redTextColor = "#B9333A"              # stock red90 deepened: 4.38 -> 4.85:1 on #FAFAFA

[theme.light.sidebar]
primaryColor = "#B54602"              # deeper ember for slider values: 4.80:1
```

**Audit** (scratch script, same model as `TestThemeConfig`: WCAG 2.x, translucent fills composited, caption at 0.6, hover at −15 HSL lightness):

| Pair | Dark (shipped) | Light (proposed) | Stock light |
|---|---|---|---|
| Text on page / panel | 17.95 / 14.50 | 17.95 / 16.44 | 12.53 / 11.18 |
| White Run label on primary | 4.79 | 4.79 | **3.30** |
| Primary on page / panel (≥ 3) | 3.91 / 3.16 | 4.59 / 4.20 | |
| Hover `#793106` on page (≥ 3) | **2.01** (accepted cost) | 8.92 | |
| Sidebar accent (slider values) | 4.77 | 4.80 | |
| Caption link, rendered | 5.05 (`#A89087`) | 4.81 (`#786560`) | |
| Warning, page / sidebar | 11.04 / 8.73 | 5.20 / 4.83 | |
| Banner (blue) | 9.41 | 4.96 | |
| Error (red) | 5.07 (stock) | 4.85 (override) | 4.38 on `#FAFAFA` |

- Stock red fails on any off-white page: its `#BD4043` is tuned for pure white. Hence the light-only `redTextColor`. It sets the text key only, never `redColor`, which would re-derive the unset keys; that is the existing alert test's rule.
- Stock light alert palette (1.64 `St`, for `_STOCK_LIGHT_ALERTS`): red `#FF2B2B1A` / `#BD4043`, yellow `#FFFF121A` / `#926C05`, blue `#1C83FF1A` / `#0054A3`. Fills are the hue at 10%, against dark's 20%. Stock light surfaces: `#FFFFFF`, `#F0F2F6`, text `#31333F`, primary `#FF4B4B`.
- Two of CLAUDE.md's three accepted costs go away in light mode. The progress bar turns ember at 4.20:1 on its track, against stock's red 2.95. The Run label reaches AA. Dark's hover cost (2) remains, and is dark-only now.

**Tests (`TestThemeConfig`):**
- `test_only_the_dark_tables_are_declared` → `test_only_the_mode_tables_are_declared`: `set(theme) == {"light", "dark"}`, each with at most a `sidebar` subtable. A top-level `[theme]` key or a `[theme.sidebar]` table still fails: it would leak into *both* modes now.
- `_load_dark_theme()` becomes `_load_mode_theme(mode)`. Parametrize every contrast test over `("dark", "light")`, with `_STOCK_LIGHT_ALERTS` beside `_STOCK_DARK_ALERTS`. Keep the color-keys-only shape test, applied to all four tables.
- The hover check is asserted for light only, with a comment pointing at dark's accepted cost. Don't weaken it for dark silently.
- **A new `test_modes_are_complementary`** locks rules 1–4:
  - `primaryColor` is equal across modes;
  - every surface and text color is R=G=B;
  - light `backgroundColor`/`textColor` equal dark `textColor`/`backgroundColor`;
  - each accent pair (sidebar primary, link, yellow and blue text) is within ±2° of OKLCH hue across modes, and on opposite sides of its surface's lightness;
  - the yellow and blue fills are identical across modes.
  - It needs a small sRGB→OKLCH helper next to `_relative_luminance`.
- Check each test against its mutant, as the dark suite was: stock red primary in light, a blue-grey panel, a hue drift > 2°, and a `[theme.sidebar]` table.

**Docs:**
- Rewrite `config.toml`'s header and CLAUDE.md's theme bullet: "the dark theme is this repo's; the light theme is Streamlit's" becomes one palette in two modes, with the rules above.
- Update the Do-not-touch row to "Don't put theme keys outside the four mode tables, or non-color keys inside them", and add a row "Don't break the cross-mode invariants", guarded by `test_modes_are_complementary`.
- Remove accepted costs (1) and (3) and keep (2) as dark-only.
- README: the hero caption's "Streamlit's stock light theme ships too" goes.

**Screenshots:**
- Re-shoot dark (below) and add **`docs/screenshot-light.png`**, using the same recipe with CDP `prefers-color-scheme: light`, from the same seeded result so the two compare like for like.
- Show both in the README with a `<picture>` element keyed on `prefers-color-scheme`, so GitHub serves the shot that matches the reader's mode. `test_readme.py` already requires local images to exist and be tracked, so the new file is covered once referenced.
- Verify on the asset: page `#FAFAFA`, panel `#F0F0F0`, Run `#C24E0A`.

**Visual checks AppTest can't make:**
- the translucent `_blank_canvas` swatch must still read as a frame on `#FAFAFA`;
- the ember focus ring on the prompt field;
- the light progress bar.

Check all three at the screenshot viewports.

## Release

After Phase 3 and a green `uv run pytest -m smoke`:

1. **Drift check** against `golden-flux3-<rev>`. Phase 3 changed only UI and theme, so no image or enhanced prompt should move; any flag means the inference path changed by accident. Investigate it before bumping.
2. **Disk check.** Run `df -h ~` and `hf cache ls`. Confirm that FLUX.2 and the unused fallback quant and VLM are gone (removed with the owner's OK), that free space is back above 30 GB, and that the app's FLUX 3 snapshot and VLM are still present. Record the numbers in the Disk plan.
3. `uv version --bump minor` (**0.10.0**), commit, push to `main`. The `release` job publishes `v0.10.0`.
   **Then write the notes' opening paragraph by hand.** `--generate-notes` lists PRs by label, and nothing in that list says "breaking". With the owner's OK, since it edits a public page, prepend a short lead with `gh release edit v0.10.0 --notes-file <file>`. Keep the generated list below it, and keep the file in scratch, not the repo. The lead covers: FLUX 3 Image Klein replacing FLUX.2; the download sizes; upsampling on by default; slower Runs, with the measured median; the FLUX Kommunity License; and a link to the README's "Upgrading to 0.10". Verify with `gh release view v0.10.0`.
4. **GitHub repo details**, once `v0.10.0` is published and `main` runs FLUX 3. Changing them earlier would describe an app `main` doesn't run yet. This is outward-facing, so ask the owner before running it. Nothing in the repo or CI can catch these details drifting, which is why it is a written step.
   - **Description**: the same text as `pyproject.toml`'s `description`, which Phase 3 sets, so the two stay identical as they are today:
     > Darkroom: local image generation and editing on Apple Silicon with FLUX 3 Image Klein, built on MLX and Streamlit, with vision-aware prompt upsampling by Qwen3-VL.

     Name FLUX descriptively; don't phrase it as "Black Forest Labs FLUX 3 …", which reads as a BFL product (trademark note under Decisions → License).
   - **Topics** (19 of GitHub's 20; agreed 2026-10-21):
     - **remove** `flux2`: the app no longer runs it, and there's no fallback;
     - **add** `flux3`, `generative-ai`, `diffusion-models`, `local-ai`, `macos`, `prompt-upsampling`;
     - **keep** `apple-silicon`, `mlx`, `mflux`, `mlx-vlm`, `flux`, `black-forest-labs`, `text-to-image`, `image-to-image`, `image-editing`, `image-generation`, `qwen3-vl`, `streamlit`, `vlm`;
     - **don't add** `darkroom`, which is an established topic for photo-editing apps such as darktable.
   - Command, from the repo root:
     ```bash
     gh repo edit \
       --description "Darkroom: local image generation and editing on Apple Silicon with FLUX 3 Image Klein, built on MLX and Streamlit, with vision-aware prompt upsampling by Qwen3-VL." \
       --remove-topic flux2 \
       --add-topic flux3,generative-ai,diffusion-models,local-ai,macos,prompt-upsampling
     ```
     Verify with `gh repo view --json description,repositoryTopics`: the description equals pyproject's, and there are 19 topics.
   - **Homepage**: leave it empty. There is no hosted demo, since the app needs Apple Silicon and local weights.
   - If the VLM fallback lands on something other than Qwen3-VL (Phase 2), update `qwen3-vl` and the description's last clause first.

## Possible Phase 4: bounding-box layout (not scheduled)

FLUX 3 places elements from boxes written into the caption, on a 0–1000 grid, and
supports box-based edits. BFL's `t2i_bbox`/`i2i_bbox` external prompts and
`src/flux3/bbox.py` define the format. A UI for drawing boxes on the canvas would
be its own design effort.

## Possible Phase 5: generation history with SQLite (not scheduled)

Today the app persists nothing (Disk plan, Rules), so it has no database and
needs none. This phase adds one only if the owner wants past generations to
survive a session. It is a separate release after 0.10.0, on its own branch
(`feat/generation-history`).

**Decide with the owner first:**
- whether a history is wanted at all, or whether option A below is enough;
- the cap, by count or GB, and where it sits in the Disk plan;
- the storage location;
- whether enhanced prompts are kept too, since prompts are stored in plain text on disk.

### Option A, the lighter one: settings inside the PNG, no database

When an image is saved, write prompt, enhanced prompt, seed, size, steps and
model revision into the PNG's text chunks (Pillow's `PngInfo`). The file
carries its own settings, and nothing accumulates on disk. Take this option if
the need is only "see how this image was made".

### Option B: SQLite via the standard library

- **Engine: `sqlite3` from the standard library.** This adds no dependency and changes nothing in `uv.lock`.
  - Not `st.connection("sql")`, which pulls in SQLAlchemy for a single local file.
  - Not Postgres or MySQL: a server process for one local user.
  - Not DuckDB, which is analytical and would add a dependency.
  - No vector database, since no feature retrieves by similarity.
- **Images as PNG files, the database holds paths.** No image BLOBs: they grow the database file without bound and make eviction awkward.
- **Location: outside the repo**, e.g. `~/Library/Application Support/flux-image/` (the database file plus an `images/` directory). Nothing lands in the working tree, so `test_secrets.py`'s `git ls-files` scan and the hooks are unaffected. If a repo-local path is chosen instead, Phase 1a's `*.db`/`*-wal`/`*-shm` rules already ignore the database. Ignore its `images/` directory explicitly, *not* via `*.png`, and extend the two-way test in `test_secrets.py`.
- **Schema, one table:** id, created_at, prompt, enhanced_prompt, seed, width, height, steps, model_repo, model_revision, mode (t2i or edit), reference count, enhancement time, generation time, image path. Write the row only after a successful Run, so a stored row always has its image.
- **Threads.** Streamlit runs each rerun on a new thread, the same hazard behind `_materialized`, and `sqlite3` connections refuse cross-thread use by default (`check_same_thread=True`). Pick one approach and pin it with a test:
  - open a short-lived connection per Run (simplest, preferred);
  - or cache one connection with `check_same_thread=False`, a lock and `PRAGMA journal_mode=WAL`.
- **Cap and eviction.** Enforce the owner's cap after each insert, deleting the oldest rows *and* their files in one step. On startup, reconcile orphans in both directions: files with no row, and rows whose file is missing.
- **UI:** a sidebar history list, newest first, in a keyed expander. Clicking an entry loads its image onto the canvas and its settings into the widgets, through `on_click` callbacks, because of the widget-key rules in CLAUDE.md. Check the canvas fold budget if the sidebar grows, and re-shoot the hero if the sidebar's look changes.
- **Tests:** mocked, using pytest's `tmp_path` for the database and image directory, with the path injected so no test touches the real location. Cover:
  - insert only on success;
  - eviction removes files;
  - orphan reconciliation;
  - generation on a fresh thread after the store was opened on another;
  - the history click restoring the widget values.
- **Docs:** replace the "app writes no outputs to disk" Rule and the matching CLAUDE.md gotcha with the cap. Add Do-not-touch rows for the cap and for the threading choice, naming their tests. Mirror the README (Usage, and where history lives on disk).

## Possible Phase 6: a measured canvas that adapts to the display (not scheduled)

Agreed 2026-10-21. It is a separate release after 0.10.0, on its own branch
(`feat/measured-canvas`), because it rewrites the canvas's measured invariants
and their Do-not-touch rows.

**Why the canvas can't adapt today.** Streamlit gives Python no viewport size
(`st.context` has none in 1.64.0 or 1.66.1, verified), and `st.image` has no
`height` in either version. So `_canvas` bounds height through width, using
fixed constants: a 0.6 column share and a 660px cap. Those constants are one
compromise across every display, which is where CLAUDE.md's measured misses
come from:

| Display | Today | With a measured canvas (1.64 offsets: top 152, caption 30; re-measure) |
|---|---|---|
| 1280×720, sidebar collapsed | 660², clips 92px | ≈ 538² (720 − 182), fits |
| 1366×768 / 1440×790, collapsed | 660², clips 44 / 22px | ≈ 586² / 608², fits |
| 640×700 (columns stack) | 608², clips 60px; jumps from 365² at 641px | ≈ 518², fits, no jump |
| Tablet portrait 820×1180 | 0.6 × width, half the height unused | width-bound, full main width |
| 2560×1440 | 660², ~500px of height unused | 1024² (the image's own width) |
| Phone 390×844 | 358², fits | unchanged (width-bound) |

**Design:**

- **A viewport probe: an inline CCv2 component, zero height, drawing nothing.**
  - Register it with `st.components.v2.component("viewport_probe", js=…)` and mount it in main directly above `result_slot`, with `height=0`.
  - CCv2 renders in a shadow root in the page, not an iframe, so its JS reads the real `window.innerWidth/innerHeight`. Its own `parentElement.getBoundingClientRect()` gives the canvas top and the main area's width.
  - It reports `{vw, vh, top, main_w}` through `setStateValue`, rounded to 8px buckets.
  - A `ResizeObserver` on its parent catches window resizes and sidebar collapse/expand, debounced at about 200ms.
  - It emits only on a bucket change, and ignores height-only changes under about 10%, which is a phone's address bar showing and hiding.
  - About 25 lines of JS, inline in `streamlit_app.py`, so it stays one file and makes no network fetch.
- **Sizing:** `_canvas_display_width(w, h, viewport)` becomes:
  `min(main_w, (vh − top − CAPTION − BOTTOM_GAP) × aspect, image native width)`, floored at 256px so a tiny window scrolls rather than collapsing.
  - `CAPTION` and `BOTTOM_GAP` are measured constants, re-measured with the screenshot recipe and pinned.
  - Never upscale past the image's own width.
  - Draw with `st.image(width=…)` inside `st.container(horizontal_alignment="center")` instead of the `[m, share, m]` columns. That removes the ≤640px stacking jump.
- **Fallback = today, exactly.** With no measurement, today's geometry applies unchanged: first paint before the JS reports, `AppTest`, JS disabled, or a probe error. That means `CANVAS_HEIGHT_RATIO`, `CANVAS_MAX_HEIGHT` and the column share. All existing canvas tests keep passing as fallback tests. First paint may therefore reflow once to the measured size; it is the idle swatch, not a result.
- **Pause during a Run.** A CCv2 state change triggers a rerun, and a rerun mid-Run would interrupt the generation.
  - Mount the probe with `data={"paused": True}` on the Run branch. Its JS holds emissions while paused, then re-measures and emits once the next idle render unpauses it.
  - **Verify in a browser** that a resize during a Run neither interrupts nor queues a stale size.
- **Unchanged:** the sidebar (`SIDEBAR_WIDTH` 320, still auto-collapsing), the prompt bar in main, the notices/status order, and the swatch at the same geometry as the result. The `st.image` hover toolbar's overlap with the prompt bar stays a known miss.
- On Streamlit 1.66, also pass `st.image(alt=…)`: the prompt, truncated, for a result, and "Empty canvas, W × H" for the swatch.

**Verified 2026-10-21 (1.64.0, scratch probe):** `AppTest` runs an app with an inline CCv2 component without error. The unmeasured state is the `default`. A test can inject a measurement by seeding `st.session_state["<key>"] = {"size": {...}}` before `run()`, and the script reads it back.

**Tests:**
- **The display matrix.** Inject viewports for each row of the table above, sidebar open and collapsed, plus 1920×1080 and 1440×880. Assert the picture and its caption end within `vh`, and the picture within `main_w`.
- This replaces `TestCanvasGeometry`'s modeled fold for the measured path. Keep the model for the fallback path.
- A fallback test: no measurement gives exactly today's widths.
- A pause test: the Run branch mounts the probe paused.
- A floor test: a 300px-tall window draws 256, not less.
- A no-upscale test: a 512² result on 2560×1440 draws at 512.
- Mutants to check against: drop the pause, drop the floor, use `vh` without `top`, re-add the columns.

**Docs:**
- Rewrite CLAUDE.md's canvas bullet: two bounds when measured, today's bounds as the fallback.
- Update the known misses (most go away) and the Do-not-touch rows for `CANVAS_MAX_HEIGHT` and the trio (now fallback-only), and add rows for the pause, the floor and the probe.
- Note in Architecture that this is the repo's first JavaScript. It styles nothing and only measures, so the "no custom CSS/HTML" rule stands.
- README: a sentence under Usage.
- Re-shoot both hero screenshots if the 1440×880 geometry moved.

**Costs accepted:**
- one rerun per resize bucket (cheap while idle, since the models are cached);
- one reflow on first paint;
- a dependency on CCv2's inline API, so re-check it on each Streamlit upgrade, as the theme's frontend facts are.

## Open questions for the implementing session

1. Does the `_materialized` per-thread stream bug still reproduce on mlx 0.32.4? (Phase 1b smoke)
2. Does the single-instance subclass type-check under ty and produce both modes? (Phase 2 smoke)
3. Q8 footprint against the budget above. (Phase 2 measurement)
4. ~~Prefill latency~~: measured (Facts). Still open: whether a load-time warm-up fixes the first-request miss, and pool sizing against eviction. Concurrent use is settled by the inference lock (Memory budget). Still open is whether APC's stored MLX arrays work on a *later* rerun's thread, which is the `_materialized` question.
5. ~~Can mflux pin a revision?~~: only for repos in its table. Pin through `snapshot_download(..., revision=)` to a local path.
6. Which rung of the VLM fallback order fits the memory budget, and does it pass the quality gate? (Phase 2 measurement + Evaluation)
7. Does Streamlit 1.66.1 keep 1.64's canvas geometry and theme behavior, or do the fold constants need retuning? (Phase 1b step 3)
8. Has Pillow fixed 13.0's `_IntegralLike` import, so the `<13` cap can go? (Phase 1b step 2; re-check before release)
9. Does FLUX 3 at 16 steps fit 60 s on the M2 Max, and if not, which lever passes both the time and the quality gate? (Phase 2, Evaluation)
10. Is a 2% mean-absolute-difference flag the right sensitivity, and is MLX output bit-identical on unchanged versions? (Phase 1a repeat run; Phase 1b drift check)

## Phase exit checklist

Before merging each phase branch:

- [ ] `uv run ruff format --check . && uv run ruff check . && uv run ty check . && uv run pytest` green
- [ ] `uv run pytest -m smoke` green (Phases 1b and 2; Phase 3 if the UI changed what smoke covers). Phase 1a changes no model code, so it owes its golden run and repeat run instead
- [ ] Measurements this phase owes are recorded **in CLAUDE.md**, with date and versions, not only in this plan
- [ ] Every Do-not-touch row and Gotcha the phase invalidated is rewritten or removed, and any new invariant has a row naming the test that guards it
- [ ] README re-synced with CLAUDE.md where mirrored content moved
- [ ] This plan updated: tick the phase, strike resolved open questions, note deviations
- [ ] Evaluation done where the phase owes it: Phase 1a's golden run and repeat run before the lock moves; Phase 1b's drift check; Phase 2's latency and quality gates and its golden run; flagged cases re-scored. Manifests are saved outside the repo, and the results are in the commit body
- [ ] Disk checked against the Disk plan: `df -h ~`, `hf cache ls`, `du -sh ~/.cache/uv .venv` recorded, and no `.incomplete` files left behind
- [ ] Commit body states the evidence; merge `--no-ff`; ask before pushing
- [ ] **CI green for the phase before the next one starts.** With the owner's OK, push the phase branch and open a PR. The `ci` job runs all four gates on a clean macOS runner, including the `uv sync --locked` check the local `.venv` and uv cache can mask, and the `release` job cannot fire on a PR. CI tests only the pushed tip, so each commit's own greenness still rests on the local Stop hook

## Handoff: starting the implementing session

Paste this as the first message of the new session:

> Read `docs/plans/flux3-upgrade.md` (start with "At a glance") and CLAUDE.md,
> then implement Phase 1a on `chore/hygiene-and-eval`. Follow the plan's facts
> rather than re-deriving them, run the phase exit checklist, and stop to check
> with me before Phase 1b. If anything in the plan proves wrong, say so and
> update the plan before continuing.

**One phase per session**: 1a, 1b, 2, 3, then Release. For each later phase,
use the same prompt with the phase and its branch changed: 1b
`chore/mflux-0.23`, 2 `feat/flux3-klein`, 3 `feat/darkroom-rebrand`. Phase 2 is
the largest and needs a fresh context. **Its prompt adds:** "apply the Phase 2
exit criteria. If no combination passes, stop and take the abort path; don't
add rungs." Each phase starts only once the previous one is merged and green in
CI (checklist).

## Appendix: upsampling measurement script

Used for the 2026-10-21 measurements. Run with the repo's venv from a directory
holding BFL's `pup_prompts_external_vlm.json`:

`HF_HUB_OFFLINE=1 .venv/bin/python -I measure_pup.py pup_prompts_external_vlm.json .`

```python
"""Measure BFL's external-VLM upsampling prompts on Qwen3-VL-2B (mlx-vlm)."""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.apc import APCManager
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

MODEL = "mlx-community/Qwen3-VL-2B-Instruct-8bit"
prompts = json.loads(Path(sys.argv[1]).read_text())
examples = Path(sys.argv[2]) / "examples"

I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)

CASES = [
    (
        "t2i",
        "Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo",
        [],
    ),
    ("t2i", "A kawaii die-cut sticker of a chubby orange cat waving", []),
    (
        "i2i",
        "The person from image 1 is petting the cat from image 2, the bird from image 3 is next to them",
        ["woman1.webp", "cat_window.webp", "bird.webp"],
    ),
    ("i2i", "make it snowy", ["cat_window.webp"]),
]

model, processor = load(MODEL)
config = load_config(MODEL)


def images_for(names):
    out = []
    for n in names:
        im = Image.open(examples / n).convert("RGB")
        im.thumbnail((768, 768))
        out.append(im)
    return out


def run(task, request, names, apc):
    imgs = images_for(names)
    user = I2I_SCAFFOLD.format(instruction=request) if imgs else request
    messages = [
        {"role": "system", "content": prompts[task]},
        {"role": "user", "content": user},
    ]
    formatted = apply_chat_template(processor, config, messages, num_images=len(imgs))
    kwargs = dict(
        image=imgs or None,
        max_tokens=3584,
        temperature=0.0,
        presence_penalty=1.5 if task == "t2i" else 0.0,
    )
    if apc is not None:
        kwargs["apc_manager"] = apc
    mx.reset_peak_memory()
    t = time.perf_counter()
    r = generate(model, processor, formatted, **kwargs)
    text = processor.tokenizer.decode(r.token_ids, skip_special_tokens=True).strip()
    return time.perf_counter() - t, r, text


apc = APCManager()
for task, request, names in CASES:
    for label, mgr in (("cold", None), ("apc-1", apc), ("apc-2", apc)):
        wall, r, text = run(task, request, names, mgr)
        print(
            task,
            label,
            f"{wall:.1f}s",
            r.prompt_tokens,
            r.generation_tokens,
            r.peak_memory,
        )
print(apc.stats)
```
