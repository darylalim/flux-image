import contextlib
import json
import math
import random
import tempfile
import threading
from pathlib import Path
from typing import cast

import mlx.core as mx
import streamlit as st
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mflux.models.common.config import ModelConfig
from mflux.models.flux3.variants import Flux3Klein, Flux3KleinEdit
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as load_vlm
from mlx_vlm.apc import APCManager
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image, ImageOps
from streamlit.runtime.media_file_storage import MediaFileStorageError

# Interim: Phase 3 of docs/plans/flux3-upgrade.md renames the app to Darkroom.
APP_TITLE = "FLUX 3 Klein Studio"

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1024

# Start width of the sidebar in px (set_page_config accepts 200-600; Streamlit's
# default is 300). An int keeps "auto" behavior -- expanded on desktop,
# collapsed on narrow viewports. 320 is the narrowest that keeps every example
# label to two lines (at 300 two of them wrap to three, growing the sidebar
# from 931 to 998px and pushing the model caption past an 880px fold) and the
# Width | Height pair side by side inside Advanced settings; every px it gives
# back widens the canvas.
SIDEBAR_WIDTH = 320

# The canvas. st.image has no height parameter and Streamlit exposes no
# viewport size to Python, so a picture's drawn height is bounded through its
# width, two ways at once:
#  - CANVAS_HEIGHT_RATIO: the picture's column spans min(1, aspect x ratio) of
#    the main area, so any aspect is drawn at most ratio x main-area-width tall.
#    It tracks the window: with the sidebar open, 0.6 keeps picture and seed
#    caption on screen in laptop-shaped windows down to 1280x720 and 1440x790.
#    At viewports of 640px or less Streamlit stacks the columns and the
#    picture spans the full width, so only the cap below bounds it there.
#  - CANVAS_MAX_HEIGHT: _canvas's inner frame, which st.image stretches to
#    fill, gets width = max height x aspect, and Streamlit caps an int
#    container width to its column. It holds where the main area widens
#    without the window growing taller -- a collapsed sidebar, or a 16:9
#    monitor, where 0.6 x width alone would overshoot the fold -- in windows
#    at least 842px tall; shorter ones with the sidebar collapsed clip.
# Retune all three constants together, re-measuring against
# TestCanvasGeometry._FOLD_VIEWPORTS and the canvas bullet in CLAUDE.md.
CANVAS_HEIGHT_RATIO = 0.6
CANVAS_MAX_HEIGHT = 660

# Fill of the blank canvas that stands in for the picture before it exists.
# Mid-gray at low alpha reads as a faint panel on both the stock light theme
# and the graphite dark one, so no theme color is hard-coded.
CANVAS_FILL = (128, 128, 128, 28)

# The prompt upsampler, pinned like the image model (see MODEL_REVISION):
# upsampling decodes greedily, so a changed enhanced prompt should mean a change
# here, never an upstream re-upload. The cached snapshot of this repo also has
# no refs/main, so loading it by repo id would fail offline.
VLM_MODEL_ID = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
VLM_REVISION = "defcdea7cc7a4b0858fea563cbbce171d328e457"
VLM_LABEL = "Qwen3-VL 8B"
VLM_DOWNLOAD_GB = 5.8

# Longest edge handed to the VLM. Qwen3-VL's processor does not cap input
# resolution (preprocessor_config has max_pixels=None and
# size.longest_edge=16777216), unlike SmolVLM's 512px tiling. Measured: a
# 4032x3024 phone photo costs 11987 prompt tokens and 50s of prefill
# against 575 tokens and 0.7s at 768px, with identical subject coverage
# across 768/1024/full-res on the 3-image example bundled at the time.
VLM_MAX_IMAGE_SIZE = 768

# Generation cap. Named so the guard below cannot drift from the kwarg: a
# run that stops because it hit the cap is a mid-sentence fragment, and
# that is a worse FLUX prompt than the user's own words. BFL-style captions
# measured 112-243 tokens; BFL's own ceiling is 3584.
VLM_MAX_TOKENS = 1024

# BFL's prompt upsampling system prompts for an external VLM, vendored
# unmodified (the "t2i" and "i2i" keys) from github.com/black-forest-labs/
# flux-3-image-klein src/flux3/pup_prompts_external_vlm.json at bfc8889a,
# Apache-2.0 -- see prompts/LICENSE and prompts/NOTICE. These are BFL's
# prompts: don't edit them. The t2i prompt is ~10k tokens, which is why the
# VLM runs with a prefix cache (_get_apc).
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
UPSAMPLE_PROMPTS = json.loads(
    (PROMPTS_DIR / "pup_prompts_external_vlm.json").read_text(encoding="utf-8")
)

# BFL's user message for an edit (src/flux3/pup.py, I2I_SCAFFOLD[False]);
# text-to-image sends the bare request.
I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)

# BFL's decoding (src/flux3/pup.py): greedy, so the enhanced prompt is
# deterministic per version, with a presence penalty on text-to-image only.
UPSAMPLE_PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}

# The app ships exactly one model: FLUX 3 Image Klein (distilled), converted
# to MLX and quantized to 8 bits by the mflux-community organization with mflux
# 0.23.0 -- an unofficial conversion, not gated (mlx-community has no FLUX 3
# build). 9.9 GB for the whole repo. mflux detects the quantization from the
# safetensors' own quantization_level metadata, so loading passes no quantize.
#
# Loaded by local snapshot path, never by repo id: mflux pins revisions only for
# the repos in its own table (ModelRevisions.PINNED), which lists BFL's
# originals but not mflux-community's builds, so a bare id would float to main.
# A snapshot fetched by full SHA gets no ref in the HF cache, so `hf cache
# prune` would delete it as detached -- never run it (CLAUDE.md).
MODEL_REPO = "mflux-community/flux-3-image-klein-mflux-q8"
MODEL_REVISION = "7e46237dea3999a9792959f5e000aeb1b9f9e7df"
MODEL_LABEL = "FLUX 3 Image Klein"
MODEL_QUANT_LABEL = "Q8"
MODEL_DOWNLOAD_GB = 9.9
MODEL_LICENSE_NAME = "FLUX Kommunity License"
MODEL_LICENSE_URL = (
    "https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md"
)

# The distilled checkpoint's step count (mflux's default and the model
# card's); this seeds the steps slider. There is no guidance: mflux nulls it
# for the distilled checkpoint, so nothing runs classifier-free guidance.
DEFAULT_STEPS = 16

# mflux accepts 1-10 references and raises beyond that. The uploader keeps the
# first MAX_REFERENCES and warns.
MAX_REFERENCES = 10

# mflux's own per-reference pixel budgets (flux3_reference_images.py): one
# reference is fitted to 4 MP, each of several to 1 MP. infer() downscales a
# copy to these before writing it out, so a 24 MP phone photo isn't PNG-encoded
# for nothing.
SINGLE_REFERENCE_MAX_PIXELS = 4 * 1024 * 1024
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024

# Shipped defaults the UI seeds its widgets with. Module constants rather than
# literals in the UI block, so eval/run_eval.py measures what users get.
# Upsampling is on: the model card says not to use the model without one, and
# a short prompt as typed gives generic output.
DEFAULT_SIZE = 1024
DEFAULT_AUTO_ENHANCE = True

EXAMPLE_PROMPTS = [
    "Create a vase on a table in living room, the color of the vase is a gradient of color, starting with #02eb3c color and finishing with #edfa3c. The flowers inside the vase have the color #ff0088",
    "Photorealistic infographic showing the complete Berlin TV Tower (Fernsehturm) from ground base to antenna tip, full vertical view with entire structure visible including concrete shaft, metallic sphere, and antenna spire. Slight upward perspective angle looking up toward the iconic sphere, perfectly centered on clean white background. Left side labels with thin horizontal connector lines: the text '368m' in extra large bold dark grey numerals (#2D3748) positioned at exactly the antenna tip with 'TOTAL HEIGHT' in small caps below. The text '207m' in extra large bold with 'TELECAFÉ' in small caps below, with connector line touching the sphere precisely at the window level. Right side label with horizontal connector line touching the sphere's equator: the text '32m' in extra large bold dark grey numerals with 'SPHERE DIAMETER' in small caps below. Bottom section arranged in three balanced columns: Left - Large text '986' in extra bold dark grey with 'STEPS' in caps below. Center - 'BERLIN TV TOWER' in bold caps with 'FERNSEHTURM' in lighter weight below. Right - 'INAUGURATED' in bold caps with 'OCTOBER 3, 1969' below. All typography in modern sans-serif font (such as Inter or Helvetica), color #2D3748, clean minimal technical diagram style. Horizontal connector lines are thin, precise, and clearly visible, touching the tower structure at exact corresponding measurement points. Professional architectural elevation drawing aesthetic with dynamic low angle perspective creating sense of height and grandeur, poster-ready infographic design with perfect visual hierarchy.",
    "Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo",
    "A kawaii die-cut sticker of a chubby orange cat, featuring big sparkly eyes and a happy smile with paws raised in greeting and a heart-shaped pink nose. The design should have smooth rounded lines with black outlines and soft gradient shading with pink cheeks.",
]

_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# Editing examples as (prompt, [image paths]). The images are BFL's own
# generations from github.com/black-forest-labs/flux-3-image-klein (Apache-2.0,
# examples/inputs/2.png and 3.png at bfc8889). mug.webp is 2.png cropped to 3:4
# so the example also exercises portrait sizing; retriever.webp is 3.png as is.
# The prompt is the two-reference edit from that repo's examples/README.md.
EDIT_EXAMPLES = [
    (
        "Print the dog from image 2 onto the mug in image 1",
        [
            str(_EXAMPLES_DIR / "mug.webp"),
            str(_EXAMPLES_DIR / "retriever.webp"),
        ],
    ),
]


def _materialized(model):
    """Evaluate a freshly built mflux model's weights before it is cached.

    Streamlit runs each rerun on a new thread, and MLX's default streams are
    per-thread. mflux hands back lazily loaded weights bound to the loading
    thread's stream, so a cached model would work only on the rerun that built
    it: every later Run fails with "There is no Stream(cpu, 0) in current
    thread". mlx-vlm's load() already evaluates eagerly, so _get_vlm is fine.
    """
    mx.eval(model.parameters())
    return model


class _Flux3KleinAnyMode(Flux3KleinEdit):
    """One FLUX 3 instance serving both modes.

    Flux3Klein and Flux3KleinEdit are the same model: both subclass mflux's
    _Flux3KleinCore and add only a generate_image wrapper over the shared
    _generate. Holding one instance instead of two saves a full 9.9 GB copy
    of the weights. generate_image is the edit; generate_text_image is
    text-to-image, borrowed from Flux3Klein, whose body only calls the
    _generate this class also has. (Inheriting from both classes instead fails
    ty: their generate_image signatures conflict.)
    """

    generate_text_image = Flux3Klein.generate_image


def _local_snapshot(repo_id, revision):
    """The cached snapshot path of repo_id@revision, or None if not downloaded."""
    try:
        return snapshot_download(repo_id, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def _pinned_snapshot(repo_id, revision):
    """Local path of repo_id at exactly `revision`, downloading it if needed.

    The local check first: it needs no network, so a cached model loads
    offline.
    """
    return _local_snapshot(repo_id, revision) or snapshot_download(
        repo_id, revision=revision
    )


# model_path picks the weights; model_config supplies the architecture (its
# transformer/text-encoder overrides), which mflux does not read from the repo.
# It is mflux's default today, stated so the pairing is visible at the call
# site, and load-bearing if MODEL_REPO ever points at the Base checkpoint.
@st.cache_resource(show_spinner=f"Loading {MODEL_LABEL} ({MODEL_QUANT_LABEL})…")
def _get_model() -> _Flux3KleinAnyMode:
    return _materialized(
        _Flux3KleinAnyMode(
            model_path=_pinned_snapshot(MODEL_REPO, MODEL_REVISION),
            model_config=ModelConfig.flux3_image_klein(),
        )
    )


@st.cache_resource(show_spinner="Loading Qwen3-VL prompt enhancer…")
def _get_vlm():
    path = _pinned_snapshot(VLM_MODEL_ID, VLM_REVISION)
    model, processor = load_vlm(path)
    config = load_config(path)
    return model, processor, config


@st.cache_resource(show_spinner=False)
def _get_apc():
    """The VLM's automatic prefix cache, as long-lived as the VLM itself.

    BFL's t2i system prompt is ~10k tokens; a prefix hit skips its prefill.
    Built directly rather than via mlx_vlm.apc.from_env, which by default
    adds a 20 GB on-disk tier under ~/.cache/mlx-vlm: the app writes nothing
    to disk, so this pool is memory-only (disk=None).
    """
    return APCManager()


@st.cache_resource(show_spinner=False)
def _inference_lock():
    """One process-wide lock: a single Run at a time per server.

    The FLUX instance is shared by every session, so two concurrent Runs would
    count each other's steps in their progress bars (one callbacks registry),
    clear each other's prompt cache, and hold two sets of activations, which
    the memory budget does not allow. The VLM's generate() takes it too, so
    its APC pool and its peak never overlap a diffusion pass.
    """
    return threading.Lock()


@contextlib.contextmanager
def _serialized(on_wait=None):
    """Hold the inference lock for the block.

    If another Run has it, on_wait(True) is called before blocking and
    on_wait(False) once the lock is ours, so a status can say "waiting" and
    then go back to what it was doing.
    """
    lock = _inference_lock()
    if not lock.acquire(blocking=False):
        if on_wait is not None:
            on_wait(True)
        lock.acquire()
        if on_wait is not None:
            on_wait(False)
    try:
        yield
    finally:
        lock.release()


def _normalized(image):
    """An input image as both FLUX and the VLM should see it.

    Upright (EXIF orientation applied, so a sideways phone photo is described
    and edited upright), any transparency composited over white, then RGB --
    the same steps as BFL's policy_reference. Applied once, where an image is
    opened, so every later consumer works from the same pixels.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    return image.convert("RGB")


def _open_image(source):
    """Open an upload, a path or a file object as a normalized RGB image."""
    return _normalized(Image.open(source))


def _vlm_images(image_list):
    """Downscaled copies of the inputs for the VLM.

    Copies, not the originals: `infer()` still passes the full-resolution
    images to mflux, and PIL's thumbnail() resizes in place.
    """
    if not image_list:
        return None
    resized = []
    for image in image_list:
        downscaled = image.copy()
        downscaled.thumbnail((VLM_MAX_IMAGE_SIZE, VLM_MAX_IMAGE_SIZE))
        resized.append(downscaled)
    return resized


def upsample_prompt(prompt, image_list: list | None = None, on_wait=None):
    try:
        model, processor, config = _get_vlm()
        task = "i2i" if image_list else "t2i"
        user = I2I_SCAFFOLD.format(instruction=prompt) if image_list else prompt
        messages = [
            {"role": "system", "content": UPSAMPLE_PROMPTS[task]},
            {"role": "user", "content": user},
        ]
        # apply_chat_template returns a str at runtime for Qwen3-VL, though the
        # stub types it as a broader union.
        formatted_prompt = cast(
            str,
            apply_chat_template(
                processor,
                config,
                messages,
                num_images=len(image_list) if image_list else 0,
            ),
        )
        apc = _get_apc()
        with _serialized(on_wait):
            result = vlm_generate(
                model,
                processor,
                formatted_prompt,
                # mlx_vlm types `image` as str | list[str] | None; it also
                # accepts PIL Images at runtime. ty cannot see the mismatch:
                # _vlm_images has no return annotation, so its result is
                # Unknown. ty doesn't check the keyword names below either, so
                # a typo would be dropped silently -- a mocked test pins them.
                image=_vlm_images(image_list),
                max_tokens=VLM_MAX_TOKENS,
                # BFL's decoding: temperature 0 is greedy (its top_p/top_k are
                # inert there, so they are not passed).
                temperature=0.0,
                presence_penalty=UPSAMPLE_PRESENCE_PENALTY[task],
                apc_manager=apc,
            )
        # mlx-vlm reports "length" when the cap cut generation off rather
        # than the model stopping on its own; that text is a fragment.
        if result.finish_reason == "length":
            return prompt
        # Decode the ids rather than read result.text. Qwen3-VL is
        # grounding-trained, and mlx-vlm's skip_special_tokens skips only
        # tokenizer.all_special_ids. Before transformers 5.18 that was just
        # <|im_end|> and <|endoftext|> here, so <|box_start|>-style markers
        # reached FLUX verbatim; nothing floors transformers, so keep it.
        # The tokenizer's own skip covers all 14 of its special tokens.
        enhanced = processor.tokenizer.decode(
            result.token_ids, skip_special_tokens=True
        ).strip()
        return enhanced or prompt
    except Exception:
        st.warning(
            "Prompt enhancement failed. Using original prompt.",
            icon=":material/warning:",
        )
        return prompt


def _resolve_prompt(prompt, image_list, auto_enhance, on_wait=None):
    """Resolve the final prompt, optionally auto-enhancing via the VLM.

    Returns was_enhanced=False when the VLM call fails or produces output
    identical to the input, so the caller can avoid showing a misleading
    "Enhanced prompt" banner.
    """
    if not auto_enhance:
        return prompt, False
    enhanced = upsample_prompt(prompt, image_list=image_list, on_wait=on_wait)
    if enhanced == prompt:
        return prompt, False
    return enhanced, True


def _set_example_prompt(example):
    """Fill the prompt box from a text-to-image example (clears any example images)."""
    st.session_state.prompt_input = example
    st.session_state.pop("example_images", None)


def _load_edit_example(prompt, images):
    """Load an editing example: its prompt plus its bundled input images."""
    st.session_state.prompt_input = prompt
    st.session_state.example_images = list(images)
    # Cycle the uploader key so a fresh, empty uploader renders — otherwise a
    # stale manual upload survives the rerun and silently overrides the example.
    st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1
    # Open the input panel via the keyed expander's state; later user toggles
    # are respected because the widget syncs this key on every change.
    st.session_state.input_expander = True


def _clear_example_images():
    st.session_state.pop("example_images", None)


def _capped_references(sources):
    """The first MAX_REFERENCES input images, with a warning if there were more.

    mflux raises beyond its limit; a warning under the uploader explains the
    cut instead. UI-only: it draws a warning.
    """
    if len(sources) <= MAX_REFERENCES:
        return sources
    st.warning(
        f"{MODEL_LABEL} edits at most {MAX_REFERENCES} images; using the first "
        f"{MAX_REFERENCES}.",
        icon=":material/warning:",
    )
    return sources[:MAX_REFERENCES]


def _truncate(text, length=70):
    return text if len(text) <= length else text[:length].rstrip() + "…"


class _ProgressReporter:
    def __init__(self, callback):
        self._callback = callback

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        self._callback(t + 1, config.num_inference_steps)


def _dimensions_from_images(image_list):
    """Calculate output dimensions matching the aspect ratio of the first input image."""
    w, h = image_list[0].size
    if w == 0 or h == 0:
        return 1024, 1024
    aspect = w / h
    if aspect >= 1:
        new_w = 1024
        new_h = round(1024 / aspect / 32) * 32
    else:
        new_h = 1024
        new_w = round(1024 * aspect / 32) * 32
    return max(256, min(MAX_IMAGE_SIZE, new_w)), max(256, min(MAX_IMAGE_SIZE, new_h))


def _reference_paths(image_list, directory):
    """Write the references into `directory` as PNGs and return their paths.

    Each is a copy downscaled to mflux's own budget for its count (mflux would
    shrink it there anyway), so a full-resolution phone photo isn't encoded for
    nothing. PNG keeps the pixels exact; compress_level 1 keeps it fast.
    """
    budget = (
        SINGLE_REFERENCE_MAX_PIXELS
        if len(image_list) == 1
        else MULTI_REFERENCE_MAX_PIXELS
    )
    paths = []
    for index, image in enumerate(image_list):
        w, h = image.size
        if w * h > budget:
            scale = math.sqrt(budget / (w * h))
            size = (max(1, math.floor(w * scale)), max(1, math.floor(h * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        path = directory / f"reference-{index}.png"
        image.save(path, format="PNG", compress_level=1)
        paths.append(str(path))
    return paths


def _canvas_spec(width, height):
    """Column spec centering a width x height picture on the canvas, or None.

    The middle share is what bounds the drawn height to CANVAS_HEIGHT_RATIO x
    the main area's width; None means the picture is wide enough to span it.
    """
    if width <= 0 or height <= 0:
        return None
    share = min(1.0, width / height * CANVAS_HEIGHT_RATIO)
    if share >= 1.0:
        return None
    margin = (1.0 - share) / 2
    return [margin, share, margin]


def _canvas_display_width(width, height):
    """Pixel width at which a width x height picture is CANVAS_MAX_HEIGHT tall."""
    if width <= 0 or height <= 0:
        return CANVAS_MAX_HEIGHT
    return max(1, round(CANVAS_MAX_HEIGHT * width / height))


def _blank_canvas(width, height):
    """The empty canvas: a CANVAS_FILL swatch with the output's aspect ratio.

    Tiny on purpose -- st.image scales it, and a flat fill has no detail to
    lose. Slider sizes are multiples of 32, so dividing keeps the ratio exact.
    """
    return Image.new("RGBA", (max(1, width // 32), max(1, height // 32)), CANVAS_FILL)


def _canvas(width, height):
    """The container a width x height picture is drawn in, centered in main.

    Its width is the picture's drawn width -- the lesser of the column share
    (the ratio bound) and _canvas_display_width (the pixel cap; Streamlit caps
    an int container width to its parent) -- so a width="stretch" picture
    fills it exactly and its caption and the run status line up with the
    picture's edges. Calls st.columns, so only the UI block may call it.
    """
    spec = _canvas_spec(width, height)
    column = st.columns(spec, gap=None)[1] if spec else st.container()
    return column.container(horizontal_alignment="center").container(
        width=_canvas_display_width(width, height), gap="xsmall"
    )


def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    num_inference_steps=None,
    image_list=None,
    progress_callback=None,
    on_wait=None,
):
    if num_inference_steps is None:
        num_inference_steps = DEFAULT_STEPS

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    model = _get_model()

    with _serialized(on_wait):
        reporter = None
        if progress_callback is not None:
            reporter = _ProgressReporter(progress_callback)
            model.callbacks.register(reporter)

        try:
            if image_list:
                # mflux opens every image_paths entry with PIL.Image.open, which
                # rejects a PIL Image, so the references go through real files
                # that live for this call only.
                with tempfile.TemporaryDirectory(prefix="flux-image-") as tmp:
                    image = model.generate_image(
                        seed=seed,
                        prompt=prompt,
                        image_paths=_reference_paths(image_list, Path(tmp)),
                        num_inference_steps=num_inference_steps,
                        width=width,
                        height=height,
                    )
            else:
                image = model.generate_text_image(
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                )
        finally:
            if reporter is not None:
                model.callbacks.in_loop.remove(reporter)
            # mflux keeps every encoded prompt for the model's lifetime, and an
            # upsampled prompt is nearly always new, so on a cached model this
            # grows without bound. Only an identical re-Run would reuse one.
            model.prompt_cache.clear()

    return image.image, seed


# The UI lives under this guard as a testability seam, not ceremony. Plain
# `import streamlit_app` (the unit tests) leaves __name__ == "streamlit_app", so
# the UI is skipped and the helpers above import without a Streamlit runtime;
# AppTest.from_file() and `streamlit run` exec the module as "__main__", so the
# UI runs. Dropping the guard would fire the whole UI on every plain import.
if __name__ == "__main__":
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/auto_awesome:",
        layout="wide",
        initial_sidebar_state=SIDEBAR_WIDTH,
    )
    # Fills the sidebar's header row, and stays in the app header when the
    # sidebar (and the title inside it) is collapsed.
    st.logo(":material/auto_awesome:", size="large")

    # Main area = prompt bar + canvas; every other control lives in the
    # sidebar. The prompt stays out of the sidebar on purpose: "auto" collapses
    # it on narrow viewports, and the primary action must never sit behind
    # that toggle.
    #
    # A borderless form makes Enter in the prompt box submit the run; the
    # horizontal container lets the input stretch while the button hugs its
    # icon+label content.
    with (
        st.form("prompt_form", border=False),
        st.container(horizontal=True, vertical_alignment="bottom"),
    ):
        prompt = st.text_input(
            "Prompt",
            placeholder="Enter your prompt",
            key="prompt_input",
            label_visibility="collapsed",
            width="stretch",
        )
        run_clicked = st.form_submit_button(
            "Run",
            type="primary",
            icon=":material/play_arrow:",
        )

    # Run notices (empty-run guard, enhancement failure, generation error)
    # render between the prompt bar and the canvas: in main,
    # so a collapsed sidebar cannot hide them, and outside result_slot, which
    # the bottom render block overwrites on every run. Empty, it takes no space.
    notices = st.container()

    # The canvas. The blank canvas, the run's status and the final image share
    # this one slot at one geometry, so each state draws where the image lands.
    result_slot = st.empty()

    # The enhanced prompt the model actually received, under the canvas: in
    # main, and below the picture so its height never pushes the picture down.
    enhanced_slot = st.empty()

    # Sidebar: the title, input images, settings and examples. What matters to
    # Streamlit is script order, not where a block draws: the image loading
    # below writes the width/height slider keys, so it -- and the setdefault
    # seeding -- must execute before the Advanced settings expander creates
    # those sliders. Keep this block's internal order.
    with st.sidebar:
        st.title(APP_TITLE, anchor=False)

        # Optional input images — uploading any switches to editing
        # automatically. The keyed expander mirrors its open/closed state in
        # session state, so loading an example opens it programmatically (see
        # _load_edit_example) while the user's own toggling is respected.
        _has_example_images = bool(st.session_state.get("example_images"))
        with st.expander(
            "Input image(s) (optional)",
            icon=":material/image:",
            key="input_expander",
            on_change="rerun",
        ):
            uploaded_files = st.file_uploader(
                "Input images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.get('uploader_nonce', 0)}",
            )
            if not uploaded_files and _has_example_images:
                st.caption("Loaded example images:")
                # Unreadable paths would crash this preview before the load
                # guard below runs; that guard warns and clears them. A file
                # that isn't an image raises UnidentifiedImageError (an
                # OSError); one that can't be opened at all raises
                # MediaFileStorageError from Streamlit's media storage, a plain
                # Exception subclass.
                # 72px keeps the bundled three on one row in the 320px sidebar.
                with contextlib.suppress(OSError, MediaFileStorageError):
                    st.image(st.session_state.example_images, width=72)
                st.button("Clear example images", on_click=_clear_example_images)

        # Load warnings render here, under the uploader or example that caused
        # them.
        image_list = None
        if uploaded_files:
            # A manual upload overrides any loaded example images
            st.session_state.pop("example_images", None)
            uploaded_files = _capped_references(uploaded_files)
            try:
                image_list = [_open_image(f) for f in uploaded_files]
                _image_key = tuple((f.name, f.file_id) for f in uploaded_files)
            except OSError:
                st.warning(
                    "Could not load one or more uploaded images.",
                    icon=":material/warning:",
                )
                _image_key = ()
        elif st.session_state.get("example_images"):
            example_images = _capped_references(st.session_state.example_images)
            try:
                image_list = [_open_image(p) for p in example_images]
                _image_key = tuple(example_images)
            except OSError:
                st.warning(
                    "Could not load the example images.",
                    icon=":material/warning:",
                )
                st.session_state.pop("example_images", None)
                _image_key = ()
        else:
            _image_key = ()
        if _image_key != st.session_state.get("prev_images", ()):
            st.session_state.prev_images = _image_key
            # An enhanced prompt is tied to its image set; drop it when that
            # changes.
            st.session_state.pop("auto_enhanced_prompt", None)
            # Match the sliders to a new input image, but leave a manual size
            # untouched when the image set is cleared. Writes the width/height
            # slider keys, so it must execute before the Advanced settings
            # expander below instantiates those sliders.
            if image_list:
                _w, _h = _dimensions_from_images(image_list)
                st.session_state.width_slider = _w
                st.session_state.height_slider = _h

        if "last_prompt" not in st.session_state:
            st.session_state.last_prompt = ""

        # The prompt lives in a form, so typed edits only arrive here on
        # submit (where the Run branch drops the banner anyway); this check
        # catches prompt changes pushed via session state by example clicks.
        if prompt != st.session_state.last_prompt:
            st.session_state.last_prompt = prompt
            st.session_state.pop("auto_enhanced_prompt", None)

        final_prompt = prompt

        st.session_state.setdefault("width_slider", DEFAULT_SIZE)
        st.session_state.setdefault("height_slider", DEFAULT_SIZE)
        st.session_state.setdefault("steps_slider", DEFAULT_STEPS)

        # The three sliders below are instantiated on every run — two
        # invariants to preserve:
        #  1. Keys seeded above (width/height on image change, plus the steps
        #     default) are written before this line in *script* order, so they
        #     land before the widgets exist. Keep those blocks above this
        #     expander; the sidebar they draw in does not change that.
        #  2. Do NOT gate this body with on_change="rerun" + `.open` to skip it
        #     when collapsed: the slider return values feed infer() below, so
        #     they must be assigned every run — a collapsed, un-run body leaves
        #     them unset (NameError at generate time).
        with st.expander("Advanced settings", icon=":material/tune:", expanded=False):
            auto_enhance = st.toggle(
                "Prompt upsampling",
                value=DEFAULT_AUTO_ENHANCE,
                key="auto_enhance_toggle",
            )
            st.caption(
                f"{MODEL_LABEL} expects a full scene caption, so a VLM expands "
                "your prompt into one. Switch off only when pasting a caption."
            )

            randomize_seed = st.toggle("Randomize seed", value=True)
            seed_val = st.number_input(
                "Seed",
                min_value=0,
                max_value=MAX_SEED,
                value=0,
                step=1,
                disabled=randomize_seed,
            )

            col_w, col_h = st.columns(2)
            with col_w:
                width = st.slider(
                    "Width",
                    min_value=256,
                    max_value=MAX_IMAGE_SIZE,
                    step=32,
                    key="width_slider",
                )
            with col_h:
                height = st.slider(
                    "Height",
                    min_value=256,
                    max_value=MAX_IMAGE_SIZE,
                    step=32,
                    key="height_slider",
                )

            num_inference_steps = st.slider(
                "Number of inference steps",
                min_value=1,
                max_value=100,
                step=1,
                key="steps_slider",
            )

        # One full-width button per example: the sidebar is one column wide.
        st.markdown("**Examples**")
        for _i, _example in enumerate(EXAMPLE_PROMPTS):
            st.button(
                _truncate(_example),
                key=f"example_{_i}",
                on_click=_set_example_prompt,
                args=(_example,),
                width="stretch",
                help=_example,
                # Explicit so the label wraps wherever the button lands: since
                # Streamlit 1.63 a button placed directly in a column or a
                # horizontal container defaults to one ellipsized line.
                wrap=True,
            )

        st.markdown("**Editing examples**")
        for _i, (_ex_prompt, _ex_imgs) in enumerate(EDIT_EXAMPLES):
            st.button(
                _truncate(_ex_prompt),
                key=f"edit_example_{_i}",
                on_click=_load_edit_example,
                args=(_ex_prompt, _ex_imgs),
                width="stretch",
                help=_ex_prompt,
                wrap=True,
            )
            st.image(_ex_imgs, width=56)

        # App metadata closes the sidebar (Streamlit's layout guidance: settings
        # and small app info there, content in main).
        # The license line is load-bearing: anyone running the app sees which
        # terms its outputs fall under (test_sidebar_caption_names_the_license).
        st.caption(
            f"[{MODEL_LABEL}](https://huggingface.co/{MODEL_REPO}) "
            f"{MODEL_QUANT_LABEL} under the [{MODEL_LICENSE_NAME}]"
            f"({MODEL_LICENSE_URL}) · prompt upsampling by "
            f"[{VLM_LABEL}](https://huggingface.co/{VLM_MODEL_ID}) · runs on MLX"
        )

    if run_clicked and not final_prompt.strip() and not image_list:
        # The Run button is always enabled, so guard the empty request here
        # rather than run the VLM and a full diffusion pass on nothing.
        notices.warning(
            "Enter a prompt or add an input image.",
            icon=":material/warning:",
        )
    elif run_clicked:
        st.session_state.pop("auto_enhanced_prompt", None)

        cache = st.session_state.setdefault("_enhance_cache", {})
        cache_key = (final_prompt, _image_key) if auto_enhance else None
        needs_vlm = auto_enhance and cache_key not in cache

        # The run's one status sits under the blank canvas, where the seed
        # caption will go: the canvas holds still from idle to run to result,
        # and the fold budget that keeps the caption on screen keeps the
        # status label there too. It opens before the VLM call so an enhanced
        # Run is never silent, and carries the whole run in its label. It stays
        # closed while enhancing, when its body would be empty.
        with result_slot.container(), _canvas(width, height):
            st.image(_blank_canvas(width, height), width="stretch")
            status = st.status(
                "Enhancing prompt…" if needs_vlm else "Generating image…",
                expanded=not needs_vlm,
            )
            # Room for the first-use VLM load spinner, under the status. A
            # cached-function spinner is a transient that overflows an empty
            # container, so in notices it would overlap the canvas and leave
            # it shifted for the rest of the run.
            vlm_load_slot = st.container()

        def _waiting_then(label, expanded):
            """An on_wait for _serialized: say "waiting" while another
            session's Run holds the inference lock, then restore `label`."""

            def on_wait(waiting):
                if waiting:
                    status.update(
                        label="Waiting for the other Run to finish…", expanded=True
                    )
                else:
                    status.update(label=label, expanded=expanded)

            return on_wait

        if needs_vlm:
            # On a fresh machine the first Run downloads the VLM; huggingface_hub
            # reports progress only to the terminal, so say so here.
            if _local_snapshot(VLM_MODEL_ID, VLM_REVISION) is None:
                status.update(
                    label=f"Downloading {VLM_LABEL}, {VLM_DOWNLOAD_GB} GB, "
                    "first run only…",
                    expanded=True,
                )
            # Warm the VLM here so the call in notices below is a cache hit. A
            # failed load is retried there, where upsample_prompt reports it.
            with vlm_load_slot, contextlib.suppress(Exception):
                _get_vlm()
            status.update(label="Enhancing prompt…", expanded=False)

        if cache_key is not None and cache_key in cache:
            run_prompt, was_auto_enhanced = cache[cache_key], True
        else:
            # In notices, not the status: upsample_prompt's failure warning
            # must outlive result_slot, which the bottom block overwrites.
            with notices:
                run_prompt, was_auto_enhanced = _resolve_prompt(
                    final_prompt,
                    image_list,
                    auto_enhance,
                    on_wait=_waiting_then("Enhancing prompt…", False),
                )
            if was_auto_enhanced and cache_key is not None:
                cache[cache_key] = run_prompt
                # Bound the per-session cache so a long editing session
                # can't accumulate enhanced prompts without limit.
                if len(cache) > 32:
                    cache.pop(next(iter(cache)))

        generation_error = None
        with status:
            # expanded=True on every mid-run update: in Streamlit 1.64 a label
            # change without it resets the status to closed, hiding the bar.
            if _local_snapshot(MODEL_REPO, MODEL_REVISION) is None:
                status.update(
                    label=f"Downloading {MODEL_LABEL}, {MODEL_DOWNLOAD_GB} GB, "
                    "first run only…",
                    expanded=True,
                )
            else:
                status.update(label="Generating image…", expanded=True)
            progress_bar = st.progress(0, text="Starting…")

            def _update_progress(step, total):
                progress_bar.progress(step / total, text=f"Step {step}/{total}")
                # The label repeats the count, so progress reads from the
                # header alone when the body sits below the fold.
                status.update(
                    label=f"Generating image… step {step}/{total}", expanded=True
                )

            try:
                image, used_seed = infer(
                    run_prompt,
                    seed_val,
                    randomize_seed,
                    width,
                    height,
                    num_inference_steps,
                    image_list=image_list,
                    progress_callback=_update_progress,
                    on_wait=_waiting_then("Generating image…", True),
                )
            except Exception as exc:
                status.update(label="Generation failed", state="error")
                generation_error = str(exc)
            else:
                status.update(label="Image generated", state="complete")
                st.session_state.result_image = image
                st.session_state.result_seed = used_seed
                # Only on success: the banner sits under the image it made,
                # never under an older one after a failed run. A retry still
                # reuses the enhancement from _enhance_cache.
                if was_auto_enhanced:
                    st.session_state.auto_enhanced_prompt = run_prompt

        # Surface a failure in notices, where it survives the bottom block
        # re-rendering result_slot (which overwrites the status).
        if generation_error is not None:
            notices.error(
                f"Image generation failed: {generation_error}",
                icon=":material/error:",
            )

    if auto_enhance and "auto_enhanced_prompt" in st.session_state:
        enhanced_slot.info(
            f"Enhanced prompt: {st.session_state.auto_enhanced_prompt}",
            icon=":material/auto_awesome:",
        )

    # The canvas: the result image, or the blank canvas at the requested size.
    # Both share one geometry, so the image lands exactly where the blank
    # canvas stood; the placeholder text is a theme-aware :gray[] line.
    with result_slot.container():
        if "result_image" in st.session_state:
            _result = st.session_state.result_image
            with _canvas(*_result.size):
                st.image(_result, width="stretch")
                if st.session_state.result_seed is not None:
                    st.caption(f"Seed: {st.session_state.result_seed}")
        else:
            with _canvas(width, height):
                st.image(_blank_canvas(width, height), width="stretch")
                if image_list:
                    _n = len(image_list)
                    _hint = (
                        f"Editing {_n} input image{'s' if _n > 1 else ''}: "
                        "describe the change, then Run."
                    )
                else:
                    _hint = (
                        "Describe an image and Run, or start from an example "
                        "in the sidebar."
                    )
                # One element, two lines: the placeholder, then a small hint --
                # the only pointer to the examples when the sidebar is collapsed.
                # Non-breaking spaces keep "W × H" whole on a narrow canvas.
                st.markdown(
                    f":gray[:material/image: Your image will appear here · "
                    f"{width} × {height}]  \n:gray[:small[{_hint}]]"
                )
