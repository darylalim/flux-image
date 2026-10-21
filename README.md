# FLUX.2 Klein Studio

[![CI](https://github.com/darylalim/flux-image/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/flux-image/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/darylalim/flux-image)](https://github.com/darylalim/flux-image/releases)
[![License](https://img.shields.io/github/license/darylalim/flux-image)](LICENSE)

Streamlit application for generating and editing images using Black Forest Labs [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Apple Silicon with MLX, running the [8-bit quantized distilled 4B](https://huggingface.co/mlx-community/flux2-klein-4b-8bit) weights.

<p align="center">
  <img src="docs/screenshot-dark.png" width="100%" alt="FLUX.2 Klein Studio in its graphite dark theme — settings sidebar, prompt bar and canvas">
</p>
<p align="center"><sub>Settings and examples in the sidebar, a real 4-step generation on the canvas — the app's graphite dark theme (Streamlit's stock light theme ships too)</sub></p>

## Features

- Unified generation and editing — text-to-image by default; uploading one or more images switches to editing automatically
- 8-bit quantized weights — an 8.6GB download instead of 16GB, with no local quantization pass at load (see Requirements for the memory picture, which is a separate question from download size)
- Native Apple Silicon performance via MLX — inference runs on MLX, not PyTorch
- Sidebar + canvas layout: input images, settings and examples live in the sidebar; the main area is a full-width prompt bar over a canvas sized to keep the whole image and its seed on screen
- Enter to run — the prompt row is a borderless form, so pressing Enter submits the run
- Multi-image upload for editing and compositing workflows
- Auto-dimension: width/height sliders adjust to match the first input image's aspect ratio
- Optional vision-aware prompt upsampling via Qwen3-VL-2B-Instruct — a toggle in the sidebar's Advanced settings; the VLM can see uploaded images when enhancing edit prompts (loaded on first use)
- Clickable examples — text-to-image prompts plus an editing example with bundled input images (loading one replaces any manual upload)
- Per-step progress shown on the canvas, right where the image will land, with a labeled status while the prompt is enhanced and spinners for first-time model loads
- Configurable seed, dimensions, and inference steps in the sidebar's Advanced settings
- A graphite dark theme built for judging images — a neutral surround with no color cast and one ember accent, at WCAG AA contrast — beside Streamlit's stock light theme (except that, as under any custom theme, light mode's Run progress bar turns red). Colors only, set in `.streamlit/config.toml` (no CSS), so the in-app appearance switcher still picks the mode
- Graceful failure handling — empty runs are blocked; unreadable uploads and generation errors surface inline instead of crashing the app

## Requirements

- Apple Silicon Mac (M1+) on macOS 14+
- Python 3.12+
- **16GB unified memory minimum; 24GB+ comfortable.** Weight size understates what a run actually needs: with only the text-to-image model resident, a 1024×1024 generation measured a 24GB process footprint against 8.6GB of weights. Most of the excess is MLX's Metal buffer cache — retained for reuse and reclaimable under pressure rather than leaked — but it is real memory while the session is warm. Generating *and* editing in one session keeps two independent weight copies resident, since they are separate cached models.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Run the application: `uv run streamlit run streamlit_app.py`

Models download automatically on first use and are cached locally for reuse: **~8.6GB** for the 8-bit FLUX.2 Klein weights, plus ~2.7GB for Qwen3-VL if you turn prompt upsampling on. Text-to-image and editing share the same download.

The app loads [`mlx-community/flux2-klein-4b-8bit`](https://huggingface.co/mlx-community/flux2-klein-4b-8bit) — the distilled FLUX.2 Klein 4B pre-quantized to 8-bit by mflux, which is roughly half the 16GB the bf16 original would pull. The 50-step base variant has no pre-quantized build under `mlx-community`, so the app does not offer it.

## Usage

The studio opens with input images, settings and examples in the sidebar, and a prompt bar over the canvas. Until the first run, a blank canvas previews the output size. On narrow windows the sidebar collapses behind the » toggle, but the prompt bar and the canvas stay put.

**Text-to-image** — type a prompt, then press **Enter** or click **Run**. The generated image appears on the canvas, labeled with the seed used.

**Editing and compositing** — expand **Input image(s)** in the sidebar and upload one or more files (JPG, PNG, or WebP). The app switches to editing automatically — there's no generate/edit switch to flip. The width and height sliders snap to the first image's aspect ratio; describe the change and Run. With multiple images you can composite across them — the bundled editing example turns these three inputs into a single scene:

<p align="center">
  <img src="examples/woman1.webp" height="260" alt="Editing input: person">
  <img src="examples/cat_window.webp" height="260" alt="Editing input: cat by a window">
  <img src="examples/bird.webp" height="260" alt="Editing input: bird">
</p>

**Advanced settings** (in the sidebar, collapsed by default):

- **Prompt upsampling** — a vision-language model (Qwen3-VL-2B) rewrites your prompt into a more descriptive one; when editing, it can see your uploaded images. Off by default.
- **Seed** — *Randomize* is **on** by default, so each Run varies. Turn it off and set a seed for reproducible results.
- **Width / Height / Number of inference steps** — fine-tune output size and sampling. The distilled model is tuned for 4 steps (the default), but the steps slider stays open if you want to push further. There is no guidance control: the model is guidance-free by design, so guidance stays fixed at 1.0. Higher values would switch on classifier-free guidance, roughly doubling the cost of each step and pulling the output away from what the distilled model was tuned for.

**Examples** — in the sidebar, click a prompt example to fill the box, or an editing example to load its prompt together with its bundled input images.

## Development

Lint, format, type-check, and run the unit tests (no GPU or model download required):

```bash
uv run ruff check .   # Lint
uv run ruff format .  # Format
uv run ty check .     # Type check
uv run pytest         # Unit tests
```

The unit tests mock the model classes, so they never download weights — which
also means they cannot catch a breaking change in mflux or mlx-vlm. A separate
opt-in suite exercises the real stack end to end and is worth running before
tagging a release (Apple Silicon and ~8.6GB of weights required):

```bash
uv run pytest -m smoke   # Real model load + generation
```

### Continuous integration

[GitHub Actions](.github/workflows/ci.yml) runs the same four checks on every push to `main` and every pull request, on a single `macos-latest` (Apple Silicon) runner — `mlx` only ships a CUDA build for Linux, so the suite can't import on a GPU-less Linux runner, and macOS is the app's target platform anyway. No model weights are downloaded (the tests mock the model loaders), so CI stays fast and needs no Hugging Face token. A secret-leak guard (`tests/test_secrets.py`) also fails the build if any tracked file contains a recognizable secret — an HF or GitHub token, a private key, or an AWS access key — or if `.env`/`secrets.toml` is ever committed. A license-consistency check (`tests/test_license.py`) likewise keeps the `LICENSE` file, the `pyproject.toml` metadata, and the README License section in agreement, so relicensing one without the others fails CI. A README-asset guard (`tests/test_readme.py`) fails the build if the README embeds a local image — a `docs/` screenshot or an `examples/` input — that is missing from the repo or not git-tracked (a broken image on GitHub). `.python-version` pins the interpreter to 3.12 so local `uv` and CI resolve the same runtime, and `[tool.uv] required-version` in `pyproject.toml` pins uv itself — `setup-uv` reads that key, so local and CI share one uv and the lockfile can't churn between them.

### Releases

Bump `version` in `pyproject.toml` and push to `main`. That is the whole ritual — `uv version --bump patch` does the bump and re-locks in one step, and [CI](.github/workflows/ci.yml)'s `release` job takes it from there: it runs only after all four gates pass, and publishes a [GitHub release](https://github.com/darylalim/flux-image/releases) with auto-generated notes, tagging the exact commit that passed. It fires only when the declared version has no matching tag yet, so re-runs and unrelated pushes are no-ops, and a pull request never publishes.

Two details are load-bearing. Because the job declares `needs: ci`, a commit that fails any gate cannot ship — the tag-triggered workflow this replaced had no way to know, and two of the first three releases published before their own CI run had finished. And the tag and the release are created by a single `gh release create --target` call rather than pushed separately: a ref pushed with the default `GITHUB_TOKEN` starts no workflow run, so splitting them would produce a tag, no release, and a green build. `tests/test_ci.py` locks both, and a cross-workflow guard (`tests/test_workflows.py`) fails the build if any workflow's `run:` script interpolates a `${{ }}` expression directly — the GitHub Actions command-injection vector — or pushes a git ref.

Before bumping, run `uv run pytest -m smoke`. It loads real weights and is the only test that exercises the live mflux/mlx-vlm surface; no GitHub runner can run it, so it is a manual step by necessity.

### Claude Code hooks

This repo ships opt-in [Claude Code](https://claude.com/claude-code) hooks (in `.claude/`) that run the checks above automatically as you edit. They are split by cost: **`ruff format` runs per edit** on the changed `.py`, `.pyi`, `.md` or `.ipynb` (the same set CI checks, so nothing is fixed only in CI), while every whole-project gate — `ruff check --fix`, `ty`, and the `pytest` suite — runs **once at the end of a turn**, and only if an edit touched something the suite asserts on (`ruff format` also runs tree-wide every turn, since it cannot lose work). A separate guard blocks `Edit`/`Write` to `.env`, `secrets.toml` and `uv.lock` (it does not intercept `Bash` writes; without `jq` it keeps working on a bash-only path rather than blocking everything). They require [`jq`](https://jqlang.github.io/jq/) and activate on session start (run `/hooks` to review). `tests/test_hooks.py` covers their behavior; `.claude/settings.local.json` (personal overrides) is gitignored.

## License

This project is released under the [MIT License](LICENSE).

It builds on components with their own licenses, all permissive:

- **FLUX.2 Klein 4B** (distilled) and **Qwen3-VL-2B-Instruct** — [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (open weights, commercial use permitted). The 8-bit weights this app loads are an Apache-2.0 quantized redistribution of the same model. The 9B FLUX.2 Klein variants carry a non-commercial license and are **not** used here.
- **mflux** and **mlx-vlm** — MIT
- **Streamlit** — Apache-2.0

Model weights are downloaded at runtime from Hugging Face rather than bundled in this repository.
