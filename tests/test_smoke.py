"""Opt-in smoke tests: real weights, real generation, no mocks.

Every other test in this suite patches mflux's FLUX 3 classes with stubs, so an
mflux API break sails through the whole green suite. These are the only tests
that exercise the real stack end to end, which makes them the pre-release gate
— and also why they are deselected by default (see
``[tool.pytest.ini_options]`` in ``pyproject.toml``): they need ~15.7GB of
weights on disk (9.9GB FLUX 3 Image Klein Q8 + 5.8GB Qwen3-VL 8B) and Apple
Silicon, neither of which CI has.

    uv run pytest -m smoke            # all of them
    uv run pytest -m smoke -k text    # just text-to-image

Weights come from the HF cache at the app's pinned revisions; the first run
downloads them.
"""

import re
from unittest.mock import Mock

import pytest
from PIL import Image

pytestmark = pytest.mark.smoke

# Smallest the app allows, so a smoke run stays cheap. Both are multiples of 32.
_W = _H = 256


@pytest.fixture(scope="module")
def app():
    """The real module, unmocked. Module-scoped so weights load once."""
    import streamlit_app

    return streamlit_app


def _is_real_image(image, size):
    """A PIL image of the requested size that is not a uniform blank canvas."""
    assert isinstance(image, Image.Image), f"not a PIL Image: {type(image)}"
    assert image.size == size, f"{image.size} != {size}"
    low, high = image.convert("L").getextrema()
    assert low != high, "image is a single flat colour — generation produced nothing"


def test_text_to_image_generates_a_real_image(app):
    image, seed = app.infer(
        "a red cube on a white table, studio lighting",
        seed=1234,
        width=_W,
        height=_H,
    )
    _is_real_image(image, (_W, _H))
    assert seed == 1234  # not randomized unless asked


def test_edit_runs_from_the_images_the_ui_builds(app):
    """The editing path's real contract, which the mocks cannot check.

    Both UI branches build ``image_list`` with ``_open_image(...)`` -- PIL
    Images, not paths -- while mflux's FLUX 3 edit opens every
    ``image_paths`` entry with ``PIL.Image.open``, which raises on a PIL Image.
    So ``infer()`` writes the references to temporary files. Drive it exactly
    the way the UI does rather than passing paths.
    """
    image_list = [
        app._open_image(app._EXAMPLES_DIR / name)
        for name in ("mug.webp", "retriever.webp")
    ]
    image, _ = app.infer(
        "print the dog from image 2 onto the mug in image 1",
        seed=7,
        width=_W,
        height=_H,
        image_list=image_list,
    )
    _is_real_image(image, (_W, _H))


def test_both_modes_run_on_one_instance(app):
    """The single-instance subclass: text-to-image borrows Flux3Klein's
    generate_image onto the edit class, so prove both produce images from the
    one cached model, in either order."""
    model = app._get_model()
    reference = [app._open_image(app._EXAMPLES_DIR / "retriever.webp")]
    for image_list in (None, reference, None):
        image, _ = app.infer(
            "a dog on a beach", seed=11, width=_W, height=_H, image_list=image_list
        )
        _is_real_image(image, (_W, _H))
    assert app._get_model() is model
    assert model.prompt_cache == {}, "infer() left encoded prompts behind"


def test_progress_callback_fires_and_is_deregistered(app):
    """mflux's CallbackRegistry has no unregister(), so infer() cleans up itself.

    The models are ``@st.cache_resource``-cached, so a reporter left behind
    fires on every later Run. Assert both halves against the real registry: the
    callback is actually invoked, and ``in_loop`` is empty afterwards.
    """
    seen = []
    image, _ = app.infer(
        "a blue sphere on grey concrete",
        seed=3,
        width=_W,
        height=_H,
        progress_callback=lambda step, total: seen.append((step, total)),
    )
    _is_real_image(image, (_W, _H))
    assert seen, "progress_callback never fired"
    assert seen[-1][0] <= seen[-1][1], f"step exceeded total: {seen[-1]}"
    assert seen[-1][1] == app.DEFAULT_STEPS
    assert app._get_model().callbacks.in_loop == [], "reporter left registered"


def test_cached_models_generate_on_a_new_thread(app):
    """Streamlit runs every rerun on a fresh thread, and MLX's default streams
    are per-thread. A model whose lazily loaded weights were never evaluated
    works only on the thread that built it: in the app, every Run after the
    first failed with "There is no Stream(cpu, 0) in current thread". The
    mocked suite and the single-threaded tests above cannot see this, so load
    here and generate on another thread, the way two reruns would.
    """
    import threading

    app._get_model()
    image_list = [app._open_image(app._EXAMPLES_DIR / "retriever.webp")]
    results = {}

    def generate(name, **kwargs):
        try:
            results[name] = app.infer("a dog", seed=5, width=_W, height=_H, **kwargs)
        except Exception as exc:  # surfaced below with the pipeline's name
            results[name] = exc

    for name, kwargs in (("txt2img", {}), ("edit", {"image_list": image_list})):
        worker = threading.Thread(target=generate, args=(name,), kwargs=kwargs)
        worker.start()
        worker.join()
        assert not isinstance(results[name], Exception), f"{name}: {results[name]!r}"
        _is_real_image(results[name][0], (_W, _H))


def _upsample_without_fallback(app, monkeypatch, prompt, image_list=None):
    """Call upsample_prompt and fail if it quietly fell back to ``prompt``.

    It returns the caller's prompt on any exception (after an st.warning, a
    no-op outside a Streamlit runtime), on a length cap-hit and on an empty
    decode -- so without these checks a broken real stack still yields a
    non-empty string and passes. A cap-hit failing here is signal too: BFL's
    captions measured 112-243 tokens against the 1024-token cap.
    """
    warning = Mock()
    monkeypatch.setattr(app.st, "warning", warning)
    enhanced = app.upsample_prompt(prompt, image_list=image_list)
    warning.assert_not_called()
    assert isinstance(enhanced, str)
    assert enhanced.strip()
    assert enhanced != prompt, "upsample_prompt fell back to the original"
    return enhanced


def test_prompt_upsampling_returns_usable_text(app, monkeypatch):
    """Qwen3-VL is the third real dependency and equally mocked elsewhere."""
    enhanced = _upsample_without_fallback(app, monkeypatch, "a cat")
    # token_ids ends with <|im_end|> on a natural stop, and Qwen3-VL's
    # grounding markers (<|box_start|>, <|object_ref_start|>, ...) are missed
    # by mlx-vlm's own skip set -- hence upsample_prompt decoding token_ids
    # with the tokenizer's full skip. The markers only appear if the model
    # happens to emit one; the contract tests below don't depend on that.
    assert not re.search(r"<\|[a-z_]+\|>", enhanced), enhanced


def test_generate_token_ids_hold_only_the_completion(app):
    """upsample_prompt decodes ``result.token_ids``, so it rests on that list
    being the completion alone. Prompt ids in it would hand the whole chat
    transcript, system prompt included, to FLUX; ``None`` would make every
    enhancement fall back. The mocked suite builds token_ids itself and
    cannot see either.
    """
    model, processor, config = app._get_vlm()
    prompt = app.apply_chat_template(
        processor, config, [{"role": "user", "content": "a cat"}], num_images=0
    )
    result = app.vlm_generate(model, processor, prompt, max_tokens=32)
    assert result.token_ids is not None
    assert len(result.token_ids) == result.generation_tokens
    if result.finish_reason == "stop":
        assert result.token_ids[-1] in processor.tokenizer.all_special_ids


def test_tokenizer_decode_drops_the_grounding_markers(app):
    """The contract upsample_prompt's marker stripping rests on, checked on
    the real tokenizer: ``decode(skip_special_tokens=True)`` drops the
    grounding markers and ``<|im_end|>``. If it stops holding, markers reach
    FLUX.

    mlx-vlm's ``result.text`` skips only ``all_special_ids``. Up to
    transformers 5.17 that held 2 of Qwen3-VL's 14 special tokens, so the
    markers leaked; from 5.18.0 it holds all 14 (bisected 2026-11-12), so
    on the current lock ``result.text`` is clean too. The ``token_ids``
    decode is kept anyway: nothing floors transformers at 5.18, and it
    costs nothing. So this test no longer asserts the leak itself.
    """
    _, processor, _ = app._get_vlm()
    tok = processor.tokenizer
    markers = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"]
    marker_ids = tok.convert_tokens_to_ids(markers)
    ids = (
        tok.encode("A", add_special_tokens=False)
        + marker_ids[:1]
        + tok.encode(" cat", add_special_tokens=False)
        + marker_ids[1:]
        + tok.convert_tokens_to_ids(["<|im_end|>"])
    )
    assert tok.decode(ids, skip_special_tokens=True) == "A cat"


def test_prompt_upsampling_sees_multiple_images(app, monkeypatch):
    """The multi-image VLM path, unmocked.

    Every other VLM test patches ``vlm_generate``, so nothing else proves that
    Qwen3-VL's processor accepts the list of PIL Images the UI actually builds
    (mlx-vlm annotates ``image`` as ``str | list[str] | None``), or that
    multi-image prefill succeeds at all. Same reasoning as
    ``test_edit_accepts_the_pil_images_the_ui_builds`` on the mflux side.
    """
    request, paths = app.EDIT_EXAMPLES[0]
    images = [app._open_image(p) for p in paths]
    sizes = [i.size for i in images]

    enhanced = _upsample_without_fallback(app, monkeypatch, request, images)

    # Grounding markers must not reach the FLUX prompt.
    assert not re.search(r"<\|[a-z_]+\|>", enhanced), enhanced
    # _vlm_images downscales copies; infer() still needs the originals.
    assert [i.size for i in images] == sizes


def test_prefix_cache_works_across_rerun_threads(app, monkeypatch):
    """APC stores MLX arrays, and MLX streams are per-thread -- the same trap
    _materialized guards for mflux. Streamlit runs every rerun on a new
    thread, so enhance on two successive threads, sharing the cached APC
    pool, the way two Runs would."""
    import threading

    results = {}

    def enhance(name, prompt):
        try:
            results[name] = _upsample_without_fallback(app, monkeypatch, prompt)
        except BaseException as exc:  # surfaced below with the thread's name
            results[name] = exc

    for name, prompt in (("first", "a red bicycle"), ("second", "a blue kite")):
        worker = threading.Thread(target=enhance, args=(name, prompt))
        worker.start()
        worker.join()
        assert not isinstance(results[name], BaseException), (
            f"{name}: {results[name]!r}"
        )
    assert app._get_apc().disk is None
