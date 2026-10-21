# Plan: upgrade to FLUX 3 Image Klein and rebrand to Darkroom

Drafted 2026-10-21, research pass the same day. To be implemented in a later
session, one phase per topic branch, each green on the four gates before merge
(`--no-ff`, Conventional Commits with an evidence body — see CLAUDE.md).

## Decisions (agreed with the owner)

| Topic | Decision |
|---|---|
| Model | FLUX 3 Image Klein, distilled, **Q8** MLX build `mflux-community/flux-3-image-klein-mflux-q8` (9.9 GB). Q6 (`-q6`, 7.6 GB) if the memory budget below is not met |
| Weights source | `mflux-community` (converted with mflux 0.23.0, not gated). `mlx-community` has no FLUX 3 build yet |
| License | Accept the FLUX Kommunity License v1.0 for the weights; code stays MIT. Disclose it in README and lock it with a test |
| Display name | **Darkroom**, with the model named descriptively in a subtitle ("local image generation and editing with FLUX 3 Image Klein"). Repo and package stay `flux-image` |
| Upsampling | **On by default**, BFL's external-VLM system prompts, greedy decoding, APC; an off switch remains for users who paste full captions |
| Upsampling VLM | **`mlx-community/Qwen3-VL-8B-Instruct-4bit` preferred** (decided 2026-10-21 on the 2B-vs-8B measurements in Facts), subject to the memory budget. Try combinations in order and take the first that passes: (1) 8B-4bit + FLUX Q8, (2) 8B-4bit + FLUX Q6, (3) `mlx-community/Qwen3-VL-4B-Instruct-8bit` (~5 GB download) + Q8, (4) the current 2B + Q8 as a last resort |
| Safety classifier | Not added (BFL's reference CLI loads one; the license does not require it). README points at the license's Section 4 restrictions instead |
| Release | **0.10.0** (`uv version --bump minor`) after Phase 3 and a green smoke run |
| Scope | One model, as today. No FLUX.2 fallback. Bounding-box layout is a possible Phase 4, not part of this release |

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

### Other

- **Pillow hazard:** the newest Pillow from PyPI failed to import (`cannot import name '_IntegralLike' from 'numbers'`) on uv's CPython 3.12.15 and 3.13.12. `pillow==12.3.0` (the repo's locked version) works. Root cause not investigated.
- **uv pin:** bumped to `==0.13.4` (merged 2026-10-21, ahead of Phase 1). Verified on 0.13.4: re-locking gave no `uv.lock` diff, `uv version --bump` still re-locks, and `0.9.0-rc1` is still stored as `0.9.0rc1`.

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

**Budget:** after a 1024² text-to-image run, then an edit run with 3 references,
with the VLM loaded and enhancement on, measured `phys_footprint` must leave
**≥ 6 GB** headroom on 32 GB (the owner keeps other apps open; adjust if they
say otherwise). If Q8 fails it, try `mx.set_cache_limit`/`mx.clear_cache()`
after each Run, then Q6. Record the numbers in CLAUDE.md's General gotchas,
replacing the FLUX.2 figure.

## Disk plan

Free space was 41 GB on 2026-10-21. **The FLUX.2 weights are not in the HF
cache**, so Phase 1's smoke run downloads them. Already cached and reusable:
`mlx-community/Qwen3-VL-2B-Instruct-8bit` (2.66 GB) and
`mlx-community/Qwen3-VL-8B-Instruct-4bit` (5.78 GB).

| Step | Change | Free after (approx.) |
|---|---|---|
| Start | — | 41 GB |
| Phase 1 smoke | + `mlx-community/flux2-klein-4b-8bit` 8.6 GB | 32 GB |
| Phase 2 | + FLUX 3 Q8 9.9 GB | 22 GB |
| Phase 2, if needed | + Q6 7.6 GB and/or Qwen3-VL-4B-8bit ~5 GB | 10–17 GB |
| After Phase 2 merges | − FLUX.2 8.6 GB, − whichever quant and VLM lost | back above 30 GB |

- Check `df -h ~` before each download. **Stop and ask the owner if free space would
  drop below 15 GB**; macOS degrades when the disk is nearly full.
- Deletions go through `hf cache` (list first, confirm the exact repo id) and **only
  with the owner's go-ahead**. Their cache holds unrelated models (e.g.
  `mlx-community/Qwen-Image-2.1-mflux-q4`, 9.6 GB), and those are theirs to decide on.

## Phase 1: toolchain (`chore/mflux-0.23`)

Goal: move to mflux 0.23 / mlx 0.32.4 **on the current FLUX.2 model**, so any
breakage is attributable to the toolchain alone.

1. ~~Bump the uv pin to 0.13.4~~: done (see Facts).
2. `uv lock --upgrade-package mflux`. Confirm in the diff that mflux becomes 0.23.0
   and mlx 0.32.4, and that **Pillow does not move**. If it does, constrain it in
   `pyproject.toml` with a comment citing the import error above.
3. Gates, then **`uv run pytest -m smoke`** on real FLUX.2 weights. Watch most
   closely the tests behind the Do-not-touch rules that depend on mflux internals:
   - `test_cached_models_generate_on_a_new_thread` (the `_materialized` stream bug: does it still reproduce on mlx 0.32.4 without `mx.eval`? keep the guard either way unless proven dead);
   - `test_progress_callback_fires_and_is_deregistered` (contract read above, but run it);
   - `test_edit_accepts_the_pil_images_the_ui_builds` (FLUX.2's edit class, still PIL-tolerant).
4. Update version-specific prose in CLAUDE.md ("verified on mlx 0.32.2 / mflux
   0.20.0", "since mflux 0.20 …") to what was re-verified.
5. Commit `chore(deps): upgrade mflux to 0.23`.

## Phase 2: model swap (`feat/flux3-klein`, commit as `feat!:`)

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
- **Fix the edit input.** `image_paths` must be real paths. In `infer()`'s edit branch,
  write each PIL image to a `tempfile.TemporaryDirectory()` as PNG after
  `ImageOps.exif_transpose` (a re-encode drops EXIF, so apply orientation first),
  and pass the paths. The directory lives for the call. Then:
  - Replace the Do-not-touch row "Don't fix `image_paths` to accept only `str` paths" and the matching Gotcha, which are both false on FLUX 3.
  - Rewrite `tests/test_smoke.py::test_edit_accepts_the_pil_images_the_ui_builds` to drive the real UI-shaped input through `infer()`.
  - Add a mocked test asserting `generate_image` receives `str`/`Path` entries.
- Clear `model.prompt_cache` in `infer()`'s `finally` (Memory budget).
- Sizes: keep `MAX_IMAGE_SIZE` 1024 and the 32-rounding in `_dimensions_from_images`
  (valid for mflux's 16-rounding). Larger sizes cost memory this machine lacks.
- Uploads: cap at 10 references (mflux raises beyond that). Surface it as a
  sidebar warning, not an exception.

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
- **Switch `VLM_MODEL_ID` to `mlx-community/Qwen3-VL-8B-Instruct-4bit`** (Decisions)
  and walk the fallback order against the memory budget. Measure each combination
  with both models resident: a t2i Run, then an i2i Run with 3 references, then a
  t2i Run again, which also exercises APC eviction.
  - If falling back to 4B-8bit, re-check medium and grounding on the four cases in Facts with the appendix script (pass the model id as the third argument).
  - Record the chosen pair, its footprint and its latencies in CLAUDE.md, replacing the 2B-era Qwen3-VL gotchas that no longer apply (sampling parameters, the 120-word cap, the "prompt wording does not fix it" note).
  - The 768px `VLM_MAX_IMAGE_SIZE` measurements were taken on 2B. Re-check the prefill cost on 8B, since 8B prefills ~4× slower.

### Copy, docs, tests

- Sidebar model caption (~line 666): FLUX 3 Image Klein Q8, link to `MODEL_REPO`.
- README Setup: download sizes (9.9 GB plus the VLM). Rewrite CLAUDE.md's Overview
  and the whole "mflux / FLUX.2 Klein" gotchas section from the Facts above and the
  smoke results, not carried over. Drop what no longer holds (no pre-quantized base
  build, VAE quantization counts, two separate weight copies).
- Run the full smoke suite with the new classes, adding a both-modes-from-one-instance test.
- `EDIT_EXAMPLES` comment says "from the FLUX.2 [klein] space". Check they still give
  sensible edits; consider BFL's own examples (`examples/inputs/`, Apache-2.0) if not.
- **Disk:** follow the Disk plan section.

## Phase 3: rebrand to Darkroom (`feat/darkroom-rebrand`)

- `APP_TITLE = "Darkroom"`, plus a subtitle naming the model descriptively. Check
  `page_title`, `st.title` and `st.logo` usage, and `tests/test_streamlit_app.py:192`.
  A subtitle line may shift the measured fold budget. Re-measure with the screenshot
  recipe if the sidebar header grows.
- README: title, intro, hero alt text. CLAUDE.md: Overview's first line. The palette
  is already called "Darkroom".
- README License section ("all permissive" becomes false):
  - Add the FLUX Kommunity License: link, the non-commercial and US$5M summary, and the Section 4 restrictions, including no use of outputs to train a competing model.
  - Add that FLUX and Black Forest Labs are trademarks of their owner and the project is not affiliated.
  - Credit BFL's Apache-2.0 prompts.
  - Keep `[MIT License](LICENSE)` intact (`test_readme_declares_project_license`).
  - Add a test in `tests/test_license.py` asserting the README names and links the model's license.
- `pyproject.toml` `description`, if it names the model or the old title.
- Re-shoot `docs/screenshot-dark.png` with the recipe in CLAUDE.md (new generation with
  FLUX 3, update the "last re-shot" note).

## Release

After Phase 3 and a green `uv run pytest -m smoke`: `uv version --bump minor`
(**0.10.0**), commit, push to `main`. The `release` job publishes `v0.10.0`.

## Possible Phase 4: bounding-box layout (not scheduled)

FLUX 3 places elements from boxes written into the caption, on a 0–1000 grid, and
supports box-based edits. BFL's `t2i_bbox`/`i2i_bbox` external prompts and
`src/flux3/bbox.py` define the format. A UI for drawing boxes on the canvas would
be its own design effort.

## Open questions for the implementing session

1. Does the `_materialized` per-thread stream bug still reproduce on mlx 0.32.4? (Phase 1 smoke)
2. Does the single-instance subclass type-check under ty and produce both modes? (Phase 2 smoke)
3. Q8 footprint against the budget above. (Phase 2 measurement)
4. ~~Prefill latency~~: measured (Facts). Still open: whether a load-time warm-up fixes the first-request miss, pool sizing against eviction, and whether APC is thread-safe across reruns.
5. ~~Can mflux pin a revision?~~: only for repos in its table. Pin through `snapshot_download(..., revision=)` to a local path.
6. Which rung of the VLM fallback order fits the memory budget? (Phase 2 measurement)

## Phase exit checklist

Before merging each phase branch:

- [ ] `uv run ruff format --check . && uv run ruff check . && uv run ty check . && uv run pytest` green
- [ ] `uv run pytest -m smoke` green (Phases 1 and 2; Phase 3 if the UI changed what smoke covers)
- [ ] Measurements this phase owes are recorded **in CLAUDE.md**, with date and versions, not only in this plan
- [ ] Every Do-not-touch row and Gotcha the phase invalidated is rewritten or removed, and any new invariant has a row naming the test that guards it
- [ ] README re-synced with CLAUDE.md where mirrored content moved
- [ ] This plan updated: tick the phase, strike resolved open questions, note deviations
- [ ] Disk checked against the Disk plan
- [ ] Commit body states the evidence; merge `--no-ff`; ask before pushing

## Handoff: starting the implementing session

Paste this as the first message of the new session:

> Read `docs/plans/flux3-upgrade.md` and CLAUDE.md, then implement Phase 1 on
> `chore/mflux-0.23`. Follow the plan's facts rather than re-deriving them, run
> the phase exit checklist, and stop to check with me before Phase 2. If anything
> in the plan proves wrong, say so and update the plan before continuing.

One phase per session is also fine. Phase 2 is the largest and benefits from a
fresh context. The per-phase prompt is the same with the phase number changed.

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
