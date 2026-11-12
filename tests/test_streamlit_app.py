import contextlib
import importlib
import io
import math
import re
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


class _MockGeneratedImage:
    """Mock mflux GeneratedImage with a .image attribute."""

    def __init__(self, image=None):
        self.image = image or Image.new("RGB", (64, 64))


def _make_mock_model():
    """Create a mock mflux model that returns a dummy GeneratedImage.

    `callbacks.in_loop` is a real list and `callbacks.register` appends to it,
    mirroring mflux's CallbackRegistry so the reporter-cleanup path in
    `infer()` is actually exercised.
    """
    model = MagicMock()
    model.generate_image.return_value = _MockGeneratedImage()
    callbacks = MagicMock()
    callbacks.in_loop = []
    callbacks.register.side_effect = callbacks.in_loop.append
    model.callbacks = callbacks
    return model


class _MockGenerationResult:
    """Mock mlx-vlm GenerationResult with .text, .finish_reason and .token_ids.

    finish_reason is real: mlx-vlm sets "stop" on a natural end and "length"
    when max_tokens cut generation off, and upsample_prompt branches on it.
    token_ids holds _FakeTokenizer pieces rather than ints; by default the
    whole text is one piece, followed on a natural stop by the <|im_end|>
    id that the real list also ends with.
    """

    def __init__(self, text="enhanced prompt", finish_reason="stop", token_ids=None):
        self.text = text
        self.finish_reason = finish_reason
        if token_ids is None:
            token_ids = [text, "<|im_end|>"] if finish_reason == "stop" else [text]
        self.token_ids = token_ids


class _FakeTokenizer:
    """Stands in for Qwen3-VL's HF tokenizer, where each "id" is a text piece.

    decode(skip_special_tokens=True) drops every added special token, which
    is the real tokenizer's contract (tests/test_smoke.py checks it against
    the real one) and the reason upsample_prompt decodes ids itself.
    """

    SPECIAL = frozenset(
        {"<|im_end|>", "<|endoftext|>", "<|box_start|>", "<|box_end|>"}
        | {"<|object_ref_start|>", "<|object_ref_end|>"}
    )

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(
            t for t in token_ids if not (skip_special_tokens and t in self.SPECIAL)
        )


def _passthrough_cache_resource(func=None, **_kwargs):
    """Stand-in for st.cache_resource supporting bare and parameterized forms.

    The app uses ``@st.cache_resource(show_spinner="…")``, which calls the
    decorator factory with kwargs first; a plain ``lambda f: f`` would choke
    on that form.
    """
    if func is None:
        return lambda f: f
    return func


def _make_mock_vlm():
    """Create a mock VLM (model, processor, config) triple."""
    mock_processor = MagicMock()
    mock_processor.tokenizer = _FakeTokenizer()
    mock_model = MagicMock()
    mock_config = MagicMock()
    return mock_model, mock_processor, mock_config


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / ".streamlit" / "config.toml"
# AppTest resolves a relative script path against the *calling test file*
# (it used to resolve against the working directory), so hand it an
# absolute path and stay independent of both.
_APP_PATH = _REPO_ROOT / "streamlit_app.py"

# WCAG 2.x AA minimums: 4.5:1 for normal-size text, 3:1 for the visual boundary
# of a UI component (success criterion 1.4.11).
_WCAG_AA_TEXT = 4.5
_WCAG_AA_NON_TEXT = 3.0
# Streamlit (1.64 through 1.67, measured) draws a whole st.caption block at
# this opacity, so a caption link's rendered color is a blend with the surface
# under it.
_CAPTION_OPACITY = 0.6
# Streamlit 1.67's stock dark (fill, text) per alert hue -- `Ot` in
# static/js/utils.*.js. An alert key [theme.dark] leaves unset falls back to
# these. 1.67's redesigned palette made the fills opaque; 1.64's were the hue
# at 20% alpha (red #FF6C6C33 under #FF6C6C).
_STOCK_DARK_ALERTS = {
    "red": ("#412023", "#F2919A"),
    "yellow": ("#302917", "#FBD54F"),
    "blue": ("#1E2A3D", "#85B8F8"),
}
# The alerts the app draws, and where: notices.error (red), notices.warning and
# upsample_prompt's warning (yellow) and the enhanced-prompt banner (blue) in
# main; image-load warnings (yellow) in the sidebar.
_APP_ALERTS = {"red": ("page",), "yellow": ("page", "sidebar"), "blue": ("page",)}


def _load_dark_theme():
    """Return the ([theme.dark], [theme.dark.sidebar]) tables of config.toml."""
    with _CONFIG_PATH.open("rb") as fh:
        dark = tomllib.load(fh)["theme"]["dark"]
    return dark, dark.get("sidebar", {})


def _rgba(color):
    """(r, g, b, alpha) of a #rrggbb or #rrggbbaa color; alpha in 0..1."""
    h = color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return r, g, b, int(h[6:8], 16) / 255 if len(h) == 8 else 1.0


def _over(fg, bg, opacity=1.0):
    """#rrggbb[aa] fg composited over opaque bg, returned as #rrggbb."""
    *rgb, alpha = _rgba(fg)
    a = alpha * opacity
    mixed = (f * a + b * (1 - a) for f, b in zip(rgb, _rgba(bg)[:3], strict=True))
    return "#" + "".join(f"{round(c):02x}" for c in mixed)


def _relative_luminance(color):
    """WCAG relative luminance of an opaque #rrggbb color."""
    *rgb, alpha = _rgba(color)
    assert alpha == 1.0, f"{color} is translucent: composite it with _over first"
    srgb = [c / 255 for c in rgb]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    r, g, b = linear
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(fg, bg):
    """WCAG contrast ratio between two opaque #rrggbb colors (>= 1.0)."""
    lo, hi = sorted((_relative_luminance(fg), _relative_luminance(bg)))
    return (hi + 0.05) / (lo + 0.05)


def _reload_app(mock_model, *, mock_edit_model=None, mock_vlm=None):
    """Reload app module with mocked heavy dependencies and passthrough cache."""
    with (
        patch(
            "mflux.models.flux2.variants.Flux2Klein", return_value=mock_model
        ) as mock_cls,
        patch(
            "mflux.models.flux2.variants.Flux2KleinEdit",
            return_value=mock_edit_model or mock_model,
        ) as mock_edit_cls,
        patch("mflux.models.common.config.ModelConfig") as _mock_model_config,
        patch("mlx_vlm.load") as mock_load,
        patch("mlx_vlm.generate") as _mock_generate,
        patch("mlx_vlm.prompt_utils.apply_chat_template") as _mock_chat,
        patch("mlx_vlm.utils.load_config") as mock_load_config,
        patch("streamlit.cache_resource", _passthrough_cache_resource),
    ):
        if mock_vlm is not None:
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_load_config.return_value = mock_vlm_config

        import streamlit_app

        importlib.reload(streamlit_app)
        return streamlit_app, mock_cls, mock_edit_cls


class TestConstants:
    def test_app_title(self):
        import streamlit_app

        assert streamlit_app.APP_TITLE == "FLUX.2 Klein Studio"

    def test_max_seed(self):
        import streamlit_app

        assert streamlit_app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self):
        import streamlit_app

        assert streamlit_app.MAX_IMAGE_SIZE == 1024

    def test_vlm_model_id(self):
        import streamlit_app

        assert streamlit_app.VLM_MODEL_ID == "mlx-community/Qwen3-VL-2B-Instruct-8bit"

    def test_model_repo_is_the_8bit_build(self):
        """The app is pinned to the pre-quantized 8-bit distilled repo.

        Dropping the "-8bit" suffix would silently fall back to a 16GB bf16
        download and double resident memory, with no other visible symptom.
        """
        import streamlit_app

        assert streamlit_app.MODEL_REPO == "mlx-community/flux2-klein-4b-8bit"

    def test_generation_defaults(self):
        import streamlit_app

        assert streamlit_app.DEFAULT_STEPS == 4
        # Above 1.0 mflux runs CFG against a blank negative -- a second
        # transformer pass per step the distilled model doesn't need.
        assert streamlit_app.GUIDANCE == 1.0

    def test_no_mode_tables_remain(self):
        """The base variant was dropped; no mode plumbing should linger."""
        import streamlit_app

        for removed in (
            "MODE_DEFAULTS",
            "MODE_LABELS",
            "LABEL_TO_MODE",
            "MODE_LABEL_LIST",
            "MODELS",
            "EDIT_MODELS",
            "_get_model_base",
            "_get_edit_model_base",
        ):
            assert not hasattr(streamlit_app, removed), removed


class TestThemeConfig:
    """.streamlit/config.toml: a dark-only color theme over stock light.

    Two contracts. Shape: every key lives in [theme.dark] or
    [theme.dark.sidebar] and is a color. The frontend builds each mode as the
    top-level [theme] keys plus that mode's table, so anything outside the dark
    tables leaks into light mode (a lone top-level `font` would populate
    neither table and cost the Light/Dark/System switcher outright). Contrast:
    the palette is this repo's own now, so WCAG AA is too.

    The contrast tests model these Streamlit frontend behaviors, last
    re-checked in the browser on 1.67.0 (re-check them on an upgrade): the
    sidebar inherits every dark key it does not override, its background
    falling back to secondaryBackgroundColor; the primary button's label is
    white; the prompt field and the progress track are
    secondaryBackgroundColor, the progress fill primaryColor; links are
    linkColor, else blueTextColor; st.caption draws at 0.6 opacity; and
    st.error/warning/info draw red/yellow/blue *BackgroundColor (translucent
    in this config, opaque in 1.67's stock) under *TextColor, falling back to
    _STOCK_DARK_ALERTS. Scope: this reads the committed file, so it cannot see
    a theme injected at runtime (STREAMLIT_THEME_*, ~/.streamlit), nor the one
    light-mode cost a custom theme carries (the Run progress bar turns the
    stock light primary instead of stock's blue; see CLAUDE.md).
    """

    def test_only_the_dark_tables_are_declared(self):
        with _CONFIG_PATH.open("rb") as fh:
            theme = tomllib.load(fh)["theme"]
        assert set(theme) == {"dark"}, (
            f"[theme] keys outside [theme.dark] leak into light mode: "
            f"{sorted(set(theme) - {'dark'})}"
        )
        tables = {k for k, v in theme["dark"].items() if isinstance(v, dict)}
        assert tables <= {"sidebar"}, f"unexpected tables: {sorted(tables)}"

    def test_only_color_keys_are_set(self):
        # The canvas fold budget is measured against stock font, size and
        # radius metrics, and a mode switch should change nothing but color.
        # Values must be #rrggbb, the format the contrast helpers read; only an
        # alert fill, which they composite, may carry an alpha byte. (The app
        # draws no charts, so the chart*Colors lists have no place here.)
        dark, sidebar = _load_dark_theme()
        tables = {"theme.dark": dark, "theme.dark.sidebar": sidebar}
        alert_fill = re.compile(r"(red|orange|yellow|green|blue|violet)BackgroundColor")
        opaque, translucent = r"#[0-9A-Fa-f]{6}", r"#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?"
        for name, table in tables.items():
            for key, value in table.items():
                if isinstance(value, dict):
                    continue
                assert key.endswith("Color"), f"[{name}] {key}"
                pattern = translucent if alert_fill.fullmatch(key) else opaque
                assert re.fullmatch(pattern, value), (
                    f"[{name}] {key} = {value!r}: not a #rrggbb color"
                )

    def test_body_text_contrast_meets_wcag_aa(self):
        dark, sidebar = _load_dark_theme()
        text = dark["textColor"]
        pairs = {
            "page": (text, dark["backgroundColor"]),
            "prompt field": (text, dark["secondaryBackgroundColor"]),
            "sidebar": (
                sidebar.get("textColor", text),
                sidebar.get("backgroundColor", dark["secondaryBackgroundColor"]),
            ),
        }
        for name, (fg, bg) in pairs.items():
            ratio = _contrast_ratio(fg, bg)
            assert ratio >= _WCAG_AA_TEXT, f"text on {name}: {ratio:.2f}:1"

    def test_run_button_meets_wcag_aa(self):
        # Streamlit draws the primary button's label white on primaryColor --
        # 1.64's stock #FF4B4B gave 3.30:1. The button must also stand out from the
        # page, and as the prompt field's focus border and the progress fill
        # it sits on secondaryBackgroundColor.
        dark, _ = _load_dark_theme()
        primary = dark["primaryColor"]
        label = _contrast_ratio("#ffffff", primary)
        assert label >= _WCAG_AA_TEXT, f"Run label: {label:.2f}:1"
        for surface in ("backgroundColor", "secondaryBackgroundColor"):
            ratio = _contrast_ratio(primary, dark[surface])
            assert ratio >= _WCAG_AA_NON_TEXT, f"primary on {surface}: {ratio:.2f}:1"

    def test_sidebar_accent_reads_as_text(self):
        # The slider values are text in the sidebar's primaryColor. A primary
        # dark enough for a white label cannot also pass on a non-black panel,
        # so the sidebar carries its own, lighter one.
        dark, sidebar = _load_dark_theme()
        accent = sidebar.get("primaryColor", dark["primaryColor"])
        panel = sidebar.get("backgroundColor", dark["secondaryBackgroundColor"])
        ratio = _contrast_ratio(accent, panel)
        assert ratio >= _WCAG_AA_TEXT, f"slider values: {ratio:.2f}:1"

    def test_caption_links_meet_wcag_aa_through_the_caption_opacity(self):
        # The app's only links are in the sidebar's model caption. Unset, a
        # link is the theme's blueTextColor, which measured 2.77:1 through the
        # caption's opacity on 1.64's stock dark.
        dark, sidebar = _load_dark_theme()
        link = sidebar.get("linkColor", dark.get("linkColor"))
        assert link is not None, "linkColor unset: links fall back to blueTextColor"
        panel = sidebar.get("backgroundColor", dark["secondaryBackgroundColor"])
        rendered = _over(link, panel, opacity=_CAPTION_OPACITY)
        ratio = _contrast_ratio(rendered, panel)
        assert ratio >= _WCAG_AA_TEXT, f"caption link {rendered}: {ratio:.2f}:1"

    def test_alert_text_contrast_meets_wcag_aa(self):
        # Every alert the app draws, on each surface it draws on, whether or
        # not the config restyles its hue: an unset key is stock (1.64's stock
        # red measured 4.40:1 on a #1E1E1E page; 1.67's is 6.38:1). A fill may
        # be translucent, so the text is measured against the fill composited
        # onto the surface.
        dark, sidebar = _load_dark_theme()
        surfaces = {
            "page": dark["backgroundColor"],
            "sidebar": sidebar.get("backgroundColor", dark["secondaryBackgroundColor"]),
        }
        for hue, where in _APP_ALERTS.items():
            # A <hue>Color would derive the unset keys instead of stock.
            assert f"{hue}Color" not in dark, f"{hue}Color is set: model its derivation"
            stock_fill, stock_text = _STOCK_DARK_ALERTS[hue]
            text = dark.get(f"{hue}TextColor", stock_text)
            for name in where:
                fill = _over(
                    dark.get(f"{hue}BackgroundColor", stock_fill), surfaces[name]
                )
                ratio = _contrast_ratio(text, fill)
                assert ratio >= _WCAG_AA_TEXT, f"{hue} alert on {name}: {ratio:.2f}:1"


class TestModelLoading:
    def test_model_created_from_8bit_repo(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model) as mock_klein,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_4b.return_value = "distilled_config"
            streamlit_app._get_model()
            # Both kwargs are asserted because the call site states both. mflux
            # would default model_config to the same value, so this pins the
            # call shape, not a correctness requirement.
            mock_klein.assert_called_once_with(
                model_path=streamlit_app.MODEL_REPO,
                model_config="distilled_config",
            )

    def test_edit_model_created_from_8bit_repo(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_model) as mock_edit,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_4b.return_value = "distilled_config"
            streamlit_app._get_edit_model()
            mock_edit.assert_called_once_with(
                model_path=streamlit_app.MODEL_REPO,
                model_config="distilled_config",
            )

    def test_both_pipelines_share_one_repo(self):
        """Text-to-image and editing load the same weights, so they share one
        download and one HF cache entry (resident memory is not shared)."""
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model) as mock_klein,
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_model) as mock_edit,
            patch("streamlit_app.ModelConfig"),
        ):
            streamlit_app._get_model()
            streamlit_app._get_edit_model()
            assert (
                mock_klein.call_args.kwargs["model_path"]
                == mock_edit.call_args.kwargs["model_path"]
                == streamlit_app.MODEL_REPO
            )

    def test_getters_materialize_the_weights_before_caching(self):
        """Streamlit reruns on new threads and MLX streams are per-thread, so
        lazily loaded weights would work only on the rerun that built them.
        The getters must evaluate them on the loading thread. This pins the
        call; test_smoke's cross-thread test proves it on real weights."""
        txt2img, edit = _make_mock_model(), _make_mock_model()
        streamlit_app, _, _ = _reload_app(txt2img, mock_edit_model=edit)
        with patch("streamlit_app.mx.eval") as mock_eval:
            assert streamlit_app._get_model() is txt2img
            assert streamlit_app._get_edit_model() is edit
        assert [c.args for c in mock_eval.call_args_list] == [
            (txt2img.parameters.return_value,),
            (edit.parameters.return_value,),
        ]


class TestInfer:
    def test_returns_image_and_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            image, seed = streamlit_app.infer("a cat", seed=42)
            assert isinstance(image, Image.Image)
            assert seed == 42

    def test_forwards_args_to_model(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                seed=123,
                width=768,
                height=512,
                num_inference_steps=20,
            )
            mock_model.generate_image.assert_called_once_with(
                seed=123,
                prompt="a cat",
                num_inference_steps=20,
                width=768,
                height=512,
                guidance=streamlit_app.GUIDANCE,
            )

    def test_fixed_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            _, seed = streamlit_app.infer("a cat", seed=99, randomize_seed=False)
            assert seed == 99

    def test_randomized_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            _, seed = streamlit_app.infer("a cat", seed=42, randomize_seed=True)
            assert 0 <= seed <= streamlit_app.MAX_SEED

    def test_default_params(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=4,
                width=1024,
                height=1024,
                guidance=1.0,
            )

    def test_empty_prompt(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            image, seed = streamlit_app.infer("", seed=42)
            assert isinstance(image, Image.Image)
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="",
                num_inference_steps=4,
                width=1024,
                height=1024,
                guidance=1.0,
            )

    def test_unset_params_fall_back_to_module_defaults(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=streamlit_app.DEFAULT_STEPS,
                width=1024,
                height=1024,
                guidance=streamlit_app.GUIDANCE,
            )

    def test_explicit_steps_override_default(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                num_inference_steps=10,
            )
            call_kwargs = mock_model.generate_image.call_args[1]
            assert call_kwargs["num_inference_steps"] == 10
            assert call_kwargs["guidance"] == streamlit_app.GUIDANCE

    def test_infer_takes_no_mode_argument(self):
        """The mode parameter is gone; a stale caller must fail loudly."""
        import inspect

        import streamlit_app

        assert "mode" not in inspect.signature(streamlit_app.infer).parameters

    def test_infer_takes_no_guidance_argument(self):
        """Guidance is pinned, not a parameter; a stale caller must fail loudly
        rather than have its value silently ignored."""
        import inspect

        import streamlit_app

        params = inspect.signature(streamlit_app.infer).parameters
        assert "guidance_scale" not in params
        assert "guidance" not in params

    def test_image_list_uses_edit_model(self):
        mock_model = _make_mock_model()
        mock_edit_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model, mock_edit_model=mock_edit_model)
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_edit_model),
        ):
            streamlit_app.infer("edit this", image_list=images)
            call_kwargs = mock_edit_model.generate_image.call_args[1]
            assert call_kwargs["image_paths"] is images
            # The edit pipeline runs the same CFG-above-1.0 branch.
            assert call_kwargs["guidance"] == streamlit_app.GUIDANCE
            mock_model.generate_image.assert_not_called()

    def test_no_images_uses_txt2img_model(self):
        mock_model = _make_mock_model()
        mock_edit_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model, mock_edit_model=mock_edit_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_edit_model),
        ):
            streamlit_app.infer("a cat")
            mock_model.generate_image.assert_called_once()
            assert "image_paths" not in mock_model.generate_image.call_args[1]
            mock_edit_model.generate_image.assert_not_called()

    def test_progress_callback_registered(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer("a cat", progress_callback=callback)
            mock_model.callbacks.register.assert_called_once()

    def test_no_callback_when_progress_callback_none(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            mock_model.callbacks.register.assert_not_called()

    def test_progress_callback_invoked_with_step_and_total(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 4
            registered.call_in_loop(0, 42, "a cat", None, mock_config, None)
            callback.assert_called_once_with(1, 4)

    def test_progress_callback_step_counts_across_steps(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 4
            for step in range(4):
                registered.call_in_loop(step, 42, "a cat", None, mock_config, None)
            assert callback.call_count == 4
            callback.assert_any_call(1, 4)
            callback.assert_any_call(2, 4)
            callback.assert_any_call(3, 4)
            callback.assert_any_call(4, 4)

    def test_progress_callback_uses_config_not_the_default_step_count(self):
        # Every other reporter test drives 4 steps, which equals DEFAULT_STEPS —
        # so a reporter that ignored config and reported DEFAULT_STEPS would pass
        # them all. Drive a non-default total to pin config.num_inference_steps.
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=50, progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 50
            registered.call_in_loop(0, 42, "a cat", None, mock_config, None)
            callback.assert_called_once_with(1, 50)

    def test_progress_callback_with_image_list(self):
        mock_model = _make_mock_model()
        mock_edit_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model, mock_edit_model=mock_edit_model)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_edit_model),
        ):
            callback = MagicMock()
            streamlit_app.infer(
                "edit this", image_list=images, progress_callback=callback
            )
            call_kwargs = mock_edit_model.generate_image.call_args[1]
            assert call_kwargs["image_paths"] is images
            mock_edit_model.callbacks.register.assert_called_once()

    def test_progress_reporter_removed_after_infer(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat", progress_callback=MagicMock())
            assert mock_model.callbacks.in_loop == []

    def test_reporters_do_not_accumulate_across_runs(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            for _ in range(3):
                streamlit_app.infer("a cat", progress_callback=MagicMock())
            assert mock_model.callbacks.in_loop == []
            assert mock_model.callbacks.register.call_count == 3

    def test_progress_reporter_removed_on_generate_failure(self):
        mock_model = _make_mock_model()
        mock_model.generate_image.side_effect = RuntimeError("boom")
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            with contextlib.suppress(RuntimeError):
                streamlit_app.infer("a cat", progress_callback=MagicMock())
            assert mock_model.callbacks.in_loop == []


class TestDimensionsFromImages:
    def test_square_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (800, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_landscape_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1600, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_portrait_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 512
        assert h == 1024

    def test_rounds_to_multiple_of_32(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1000, 700))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w % 32 == 0
        assert h % 32 == 0

    def test_clamps_min_to_256(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (3000, 500))]
        _, h = streamlit_app._dimensions_from_images(images)
        assert h == 256

    def test_zero_height_returns_default(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (100, 0))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_zero_width_returns_default(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (0, 100))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_uses_first_image_only(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1600, 800)), Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_4_3_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1200, 900))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 768

    def test_16_9_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1920, 1080))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 576

    def test_portrait_3_4_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (900, 1200))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 768
        assert h == 1024

    def test_extreme_panoramic_clamps_height(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (5000, 500))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 256

    def test_extreme_tall_clamps_width(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (500, 5000))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 256
        assert h == 1024


class TestCanvasGeometry:
    """st.image has no height parameter and Streamlit exposes no viewport size,
    so the canvas bounds a picture's drawn height through its width, twice: a
    column share (CANVAS_HEIGHT_RATIO x the main-area width) and a pixel cap
    (CANVAS_MAX_HEIGHT)."""

    # Streamlit 1.64 layout, measured at dpr 1 with the shoot recipe in
    # CLAUDE.md: the canvas starts 152px down (header, top padding, prompt bar,
    # gap), the seed caption needs 30px under the picture (8px gap + 22px
    # line), and the main area is padded 80px each side.
    _CANVAS_TOP, _CAPTION, _MAIN_PAD = 152, 30, 160

    # (viewport width, height, sidebar open): browser windows on Mac displays
    # the picture and its seed caption must fit without scrolling. The known
    # misses -- a collapsed sidebar in any window under 842px tall, e.g.
    # 1280x720 and 1440x790 -- are left out on purpose; CLAUDE.md records them.
    _FOLD_VIEWPORTS = [
        (1280, 720, True),
        (1440, 790, True),
        (1440, 880, True),
        (1512, 860, True),
        (1728, 1000, True),
        (1920, 968, True),
        (2560, 1330, True),
        (1440, 880, False),
        (1512, 860, False),
        (1920, 968, False),
        (1920, 1080, False),
    ]

    def test_column_share_bounds_height_by_the_ratio(self):
        import streamlit_app

        ratio = streamlit_app.CANVAS_HEIGHT_RATIO
        for w, h in [(1024, 1024), (672, 1024), (256, 1024), (1024, 768)]:
            spec = streamlit_app._canvas_spec(w, h)
            assert spec is not None
            left, share, right = spec
            assert left == right
            assert abs(left + share + right - 1) < 1e-9
            assert abs(share / (w / h) - ratio) < 1e-9

    def test_wide_landscapes_span_the_canvas(self):
        import streamlit_app

        ratio = streamlit_app.CANVAS_HEIGHT_RATIO
        assert streamlit_app._canvas_spec(1024, 256) is None
        # Just past the break-even aspect of 1 / ratio.
        assert streamlit_app._canvas_spec(1000, int(1000 * ratio) - 1) is None

    def test_display_width_caps_the_drawn_height(self):
        import streamlit_app

        cap = streamlit_app.CANVAS_MAX_HEIGHT
        for w, h in [(1024, 1024), (672, 1024), (1024, 576), (256, 1024)]:
            drawn = streamlit_app._canvas_display_width(w, h)
            # Within the rounding of the width to a whole pixel.
            assert abs(drawn * h / w - cap) <= h / w

    def test_degenerate_sizes_fall_back(self):
        import streamlit_app

        assert streamlit_app._canvas_spec(0, 1024) is None
        assert streamlit_app._canvas_spec(1024, 0) is None
        cap = streamlit_app.CANVAS_MAX_HEIGHT
        assert streamlit_app._canvas_display_width(0, 0) == cap

    def test_blank_canvas_keeps_the_output_aspect(self):
        import streamlit_app

        for w, h in [(1024, 1024), (672, 1024), (1024, 576), (256, 1024)]:
            canvas = streamlit_app._blank_canvas(w, h)
            assert canvas.width * h == canvas.height * w
            # Translucent, so it reads on stock light and the graphite dark theme.
            assert canvas.mode == "RGBA"
            assert canvas.getpixel((0, 0))[3] < 255

    def _caption_bottom(self, app, vw, sidebar_open, size, display_width):
        """Where the seed caption ends, modelled with the measured offsets."""
        w, h = size
        main_w = vw - self._MAIN_PAD - (app.SIDEBAR_WIDTH if sidebar_open else 0)
        spec = app._canvas_spec(w, h)
        column = main_w * (spec[1] if spec else 1)
        drawn_w = min(column, display_width(w, h))
        return self._CANVAS_TOP + drawn_w * h / w + self._CAPTION

    def test_picture_and_caption_fit_the_measured_viewports(self):
        import streamlit_app as app

        assert 200 <= app.SIDEBAR_WIDTH <= 600  # set_page_config's int range
        for vw, vh, sidebar_open in self._FOLD_VIEWPORTS:
            for size in [(1024, 1024), (672, 1024), (1024, 576)]:
                bottom = self._caption_bottom(
                    app, vw, sidebar_open, size, app._canvas_display_width
                )
                assert bottom <= vh, (vw, vh, sidebar_open, size, bottom)

    def test_the_pixel_cap_is_load_bearing(self):
        # Non-vacuity: the README-hero size with the sidebar collapsed is in
        # the fold list, and a square fits there only because of the cap --
        # without it the caption lands off screen, the regression the cap
        # exists to stop.
        import streamlit_app as app

        assert (1440, 880, False) in self._FOLD_VIEWPORTS
        square = (1024, 1024)
        capped = self._caption_bottom(
            app, 1440, False, square, app._canvas_display_width
        )
        uncapped = self._caption_bottom(app, 1440, False, square, lambda *_: math.inf)
        assert capped <= 880 < uncapped

    def test_layout_constants_are_the_measured_ones(self):
        # The fold model above only catches a retune that grows the picture;
        # a smaller canvas or a sidebar-width change (which re-wraps the
        # example labels -- AppTest cannot see layout) passes it. Pin the
        # measured trio so a retune has to come through here and re-measure
        # with the shoot recipe in CLAUDE.md.
        import streamlit_app as app

        assert (app.SIDEBAR_WIDTH, app.CANVAS_HEIGHT_RATIO, app.CANVAS_MAX_HEIGHT) == (
            320,
            0.6,
            660,
        )


class TestVLMInit:
    def test_vlm_loads_correct_model(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            streamlit_app._get_vlm()
            mock_load.assert_called_once_with("mlx-community/Qwen3-VL-2B-Instruct-8bit")
            mock_lc.assert_called_once_with("mlx-community/Qwen3-VL-2B-Instruct-8bit")

    def test_vlm_returns_triple(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            result = streamlit_app._get_vlm()
            assert result == (mock_vlm_model, mock_vlm_processor, mock_vlm_config)


EXPECTED_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their core subject and intent. Keep the enhanced prompt under 120 "
    "words.\n\n"
    "Guidelines:\n"
    "- Add concrete visual specifics: textures, materials, lighting, "
    "shadows, and spatial relationships.\n"
    "- Only include rendered text the user explicitly asked for; never "
    "invent signs, labels, captions, or titles. When the user does ask for "
    "text, put it in quotation marks - without quotes, the model generates "
    "gibberish.\n\n"
    "Output only the revised prompt and nothing else."
)

EXPECTED_SYSTEM_PROMPT_WITH_IMAGES = (
    "You are an image-editing expert. Convert the user's editing request "
    "into one concise instruction (50-80 words, ~30 for brief requests).\n\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid vague words like "
    '"whimsical" or "cascading")\n'
    "- Specify what changes AND what stays the same (face, lighting, "
    "composition)\n"
    "- Turn negatives into positives "
    '("don\'t change X" becomes "keep X")\n'
    "- Replace abstract adjectives with the specific materials, colours "
    "and lighting they imply\n\n"
    "Output only the final instruction in plain text and nothing else."
)


class TestVlmImages:
    def test_returns_none_without_images(self):
        import streamlit_app

        assert streamlit_app._vlm_images(None) is None
        assert streamlit_app._vlm_images([]) is None

    def test_downscales_to_the_cap(self):
        import streamlit_app

        out = streamlit_app._vlm_images([Image.new("RGB", (4032, 3024))])
        assert max(out[0].size) == streamlit_app.VLM_MAX_IMAGE_SIZE

    def test_leaves_small_images_alone(self):
        import streamlit_app

        out = streamlit_app._vlm_images([Image.new("RGB", (512, 768))])
        assert out[0].size == (512, 768)

    def test_does_not_mutate_the_originals(self):
        """infer() passes the same list to mflux; thumbnail() resizes in place."""
        import streamlit_app

        original = Image.new("RGB", (4032, 3024))
        streamlit_app._vlm_images([original])
        assert original.size == (4032, 3024)


class TestUpsamplePrompt:
    def test_chat_message_format_text_only(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT},
                    {"role": "user", "content": "a cat"},
                ],
                num_images=0,
            )

    def test_chat_message_format_with_images(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("make it blue", image_list=images)
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT_WITH_IMAGES},
                    {"role": "user", "content": "make it blue"},
                ],
                num_images=2,
            )

    def test_images_passed_to_generate(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("edit", image_list=images)
            call_kwargs = mock_gen.call_args[1]
            passed = call_kwargs["image"]
            # Downscaled copies, never the originals: infer() still
            # hands the full-resolution images to mflux.
            assert passed is not images
            assert len(passed) == len(images)
            assert all(a is not b for a, b in zip(passed, images, strict=True))
            assert all(max(i.size) <= streamlit_app.VLM_MAX_IMAGE_SIZE for i in passed)

    def test_no_images_passed_for_text_only(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is None

    def test_generation_kwargs(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["max_tokens"] == 256
            assert call_kwargs["temperature"] == 0.7
            # Qwen3-VL's shipped generation_config values, not SmolVLM's.
            assert call_kwargs["top_p"] == 0.8
            assert call_kwargs["top_k"] == 20
            assert call_kwargs["repetition_penalty"] == 1.05
            # Explicit: mlx-vlm's default window is 20 tokens, too short for
            # a clause-length loop.
            assert call_kwargs["repetition_context_size"] == 64

    def test_grounding_markers_are_stripped(self):
        """mlx-vlm's skip set is only all_special_ids (<|im_end|>/<|endoftext|>
        for Qwen3-VL), so result.text keeps grounding markers; upsample_prompt
        must decode token_ids with the tokenizer's full skip instead."""
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        pieces = [
            "A ",
            "<|object_ref_start|>",
            "cat",
            "<|object_ref_end|>",
            " on a sofa",
            "<|im_end|>",
        ]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult(
                # What mlx-vlm's own detokenizer yields for these ids.
                "A <|object_ref_start|>cat<|object_ref_end|> on a sofa",
                token_ids=pieces,
            )
            assert streamlit_app.upsample_prompt("a cat") == "A cat on a sofa"

    def test_extracts_and_strips_output(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("  A majestic feline  ")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "A majestic feline"

    def test_truncated_generation_returns_original(self):
        """A capped run is a mid-sentence fragment, not a usable FLUX prompt."""
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult(
                "A majestic feline sitting on a windowsill in the",
                finish_reason="length",
            )
            assert streamlit_app.upsample_prompt("a cat") == "a cat"

    def test_natural_stop_is_kept(self):
        """The guard must not reject a normal completion."""
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult(
                "A majestic feline", finish_reason="stop"
            )
            assert streamlit_app.upsample_prompt("a cat") == "A majestic feline"

    def test_empty_output_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_whitespace_only_output_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("   ")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_exception_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
            patch("streamlit_app.st") as mock_st,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.side_effect = RuntimeError("OOM")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"
            mock_st.warning.assert_called_once_with(
                "Prompt enhancement failed. Using original prompt.",
                icon=":material/warning:",
            )

    def test_empty_image_list_uses_text_only_path(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat", image_list=[])
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT},
                    {"role": "user", "content": "a cat"},
                ],
                num_images=0,
            )
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is None


class TestResolvePrompt:
    def test_returns_original_when_auto_enhance_off(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        result, was_enhanced = streamlit_app._resolve_prompt(
            "a cat", None, auto_enhance=False
        )
        assert result == "a cat"
        assert was_enhanced is False

    def test_enhances_when_auto_enhance_on(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True
            )
            assert result == "enhanced prompt"
            assert was_enhanced is True

    def test_enhances_with_images(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "edit this", images, auto_enhance=True
            )
            assert was_enhanced is True
            call_kwargs = mock_gen.call_args[1]
            passed = call_kwargs["image"]
            # Downscaled copies, never the originals: infer() still
            # hands the full-resolution images to mflux.
            assert passed is not images
            assert len(passed) == len(images)
            assert all(a is not b for a, b in zip(passed, images, strict=True))
            assert all(max(i.size) <= streamlit_app.VLM_MAX_IMAGE_SIZE for i in passed)

    def test_falls_back_on_vlm_error(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
            patch("streamlit_app.st"),
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.side_effect = RuntimeError("OOM")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True
            )
            assert result == "a cat"
            assert was_enhanced is False

    def test_returns_not_enhanced_when_vlm_output_matches_input(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with patch("streamlit_app.upsample_prompt") as mock_upsample:
            mock_upsample.return_value = "a cat"
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True
            )
            assert result == "a cat"
            assert was_enhanced is False


class TestStreamlitApp:
    def test_get_model_uses_cache_resource(self):
        """Verify _get_model is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_model, "clear")

    def test_get_edit_model_uses_cache_resource(self):
        """Verify _get_edit_model is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_edit_model, "clear")

    def test_get_vlm_uses_cache_resource(self):
        """Verify _get_vlm is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_vlm, "clear")

    def test_getters_have_loading_spinner_labels(self):
        """All three cached getters label their long first load via show_spinner."""
        captured = []

        def recording_cache_resource(func=None, **kwargs):
            if func is None:
                captured.append(kwargs)
                return lambda f: f
            return func

        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
            patch("streamlit.cache_resource", recording_cache_resource),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)

        assert len(captured) == 3
        assert all(kwargs["show_spinner"].startswith("Loading") for kwargs in captured)

    def test_ui_not_executed_on_import(self):
        """Guards the ``if __name__ == "__main__"`` testability seam: a plain
        import must not render the UI, so these unit tests can import the module
        directly without a Streamlit script context."""
        mock_model = _make_mock_model()
        with (
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.button") as mock_button,
        ):
            _reload_app(mock_model)
            mock_markdown.assert_not_called()
            mock_text_input.assert_not_called()
            mock_button.assert_not_called()

    def test_no_mode_control_rendered(self):
        # The base variant was dropped, so there is nothing to switch between.
        # Use the typed property, NOT at.get("segmented_control"): AppTest names
        # the node "button_group", so the get() form returns [] whether or not a
        # segmented control is on the page — a guard that can never fail.
        with _app_test() as app:
            at = app.run(timeout=10)
            assert len(at.segmented_control) == 0
            assert len(at.get("button_group")) == 0

    def test_uploader_always_present(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            # Unified layout: the optional uploader is always rendered
            assert len(at.get("file_uploader")) == 1

    def test_run_button_present_and_enabled(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            run_buttons = [b for b in at.button if b.label == "Run"]
            assert len(run_buttons) == 1
            # Editing is implicit now, so Run is never disabled
            assert run_buttons[0].disabled is False

    def test_run_is_form_submit_button(self):
        # Run submits the prompt form, so pressing Enter in the prompt box
        # triggers generation (the form's enter_to_submit default).
        with _app_test() as app:
            at = app.run(timeout=10)
            run = next(b for b in at.button if b.label == "Run")
            assert run.proto.is_form_submitter
            assert run.proto.form_id

    def test_prompt_uses_enter_prompt_placeholder(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            assert at.text_input(key="prompt_input").placeholder == "Enter your prompt"

    def test_sliders_seed_the_distilled_defaults(self):
        # Nothing writes this key any more (the mode block used to), so the
        # setdefault seeding is the only thing standing between the slider and
        # its min_value — 1 step would render noise.
        import streamlit_app

        with _app_test() as app:
            at = app.run(timeout=10)
            assert at.slider(key="steps_slider").value == streamlit_app.DEFAULT_STEPS

    def test_widgets_start_at_the_shipped_default_constants(self):
        # eval/run_eval.py reads these constants to run "as shipped"; a widget
        # seeded from a literal instead would let the two drift apart silently.
        import streamlit_app

        with _app_test() as app:
            at = app.run(timeout=10)
            assert at.slider(key="width_slider").value == streamlit_app.DEFAULT_SIZE
            assert at.slider(key="height_slider").value == streamlit_app.DEFAULT_SIZE
            assert (
                at.toggle(key="auto_enhance_toggle").value
                == streamlit_app.DEFAULT_AUTO_ENHANCE
            )

    def test_steps_slider_still_allows_long_runs(self):
        # Losing the Base preset must not lose the ability to run many steps.
        # Assert the value reaches generate_image() — merely reading the slider
        # back would only prove Streamlit stores what you set it to.
        mock_txt2img = _make_mock_model()
        mock_edit = _make_mock_model()
        with _patched_models(mock_txt2img, mock_edit) as (app, _generate):
            at = app.run(timeout=10)
            assert at.slider(key="steps_slider").proto.max >= 50
            at.slider(key="steps_slider").set_value(50).run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert (
                mock_txt2img.generate_image.call_args.kwargs["num_inference_steps"]
                == 50
            )

    def test_advanced_sliders_render_while_collapsed(self):
        # The Advanced settings expander is collapsed by default, yet its three
        # sliders must still instantiate every run: their values feed infer(),
        # so gating the expander body on it being open would leave them unset.
        with _app_test() as app:
            at = app.run(timeout=10)
            for key in ("width_slider", "height_slider", "steps_slider"):
                assert at.slider(key=key).value is not None

    def test_example_buttons_render(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            # Run button + 4 example buttons
            assert len(at.button) >= 5
            assert at.button(key="example_2").label  # capybara text-to-image example

    def test_clicking_example_fills_prompt(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            example = at.button(key="example_2")  # capybara prompt
            example.click().run(timeout=10)
            # Button labels are truncated; clicking sets the full prompt
            value = at.text_input(key="prompt_input").value or ""
            assert "capybara" in value
            assert value.endswith("close up photo")

    def test_edit_example_loads_prompt_and_images(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            edit_example = at.button(key="edit_example_0")
            edit_example.click().run(timeout=10)
            assert "onto the mug" in (at.text_input(key="prompt_input").value or "")
            assert len(at.session_state["example_images"]) == 2


class _FakeSessionState(dict):
    # Minimal st.session_state stand-in: attribute access + item access + pop.
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@contextlib.contextmanager
def _patched_models(txt2img, edit, *, vlm_text=None):
    """Patch the heavy deps and yield (AppTest factory, mlx_vlm.generate mock).

    Pass distinct ``txt2img``/``edit`` mocks to assert which pipeline ran, or
    ``vlm_text`` to make the VLM return a fixed enhanced prompt.
    """
    mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
    vlm_result = (
        _MockGenerationResult() if vlm_text is None else _MockGenerationResult(vlm_text)
    )
    with (
        patch("mflux.models.flux2.variants.Flux2Klein", return_value=txt2img),
        patch("mflux.models.flux2.variants.Flux2KleinEdit", return_value=edit),
        patch("mflux.models.common.config.ModelConfig"),
        patch("mlx_vlm.load", return_value=(mock_vlm_model, mock_vlm_processor)),
        patch("mlx_vlm.generate", return_value=vlm_result) as vlm_generate,
        patch("mlx_vlm.prompt_utils.apply_chat_template"),
        patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
        patch("streamlit.cache_resource", _passthrough_cache_resource),
    ):
        from streamlit.testing.v1 import AppTest

        yield AppTest.from_file(_APP_PATH), vlm_generate


def _find_canvas(at):
    """(centering block, frame): the fixed-width block the canvas picture is
    drawn into (see _canvas), and the block that centers it."""
    stack = [(None, at.main)]
    while stack:
        parent, node = stack.pop()
        children = list(getattr(node, "children", {}).values())
        if any(getattr(child, "type", None) == "image" for child in children):
            return parent, node
        stack.extend((node, child) for child in children)
    raise AssertionError("no canvas picture in the main area")


def _canvas_frame(at):
    """The fixed-width block the canvas picture is drawn into (see _canvas)."""
    return _find_canvas(at)[1]


def _main_position(at, element_type):
    """Index of the top-level main-area slot whose subtree holds that type."""
    for index, child in sorted(at.main.children.items()):
        stack = [child]
        while stack:
            node = stack.pop()
            if getattr(node, "type", None) == element_type:
                return index
            stack.extend(getattr(node, "children", {}).values())
    raise AssertionError(f"no {element_type} in the main area")


@contextlib.contextmanager
def _app_test():
    """Yield an AppTest factory with one shared mock model for both pipelines."""
    mock_model = _make_mock_model()
    with _patched_models(mock_model, mock_model) as (app, _generate):
        yield app


class TestExamplesAndLabels:
    def test_example_prompts_shape(self):
        import streamlit_app

        assert len(streamlit_app.EXAMPLE_PROMPTS) == 4
        assert all(
            isinstance(p, str) and p.strip() for p in streamlit_app.EXAMPLE_PROMPTS
        )
        assert any("capybara" in p for p in streamlit_app.EXAMPLE_PROMPTS)

    def test_edit_examples_shape_and_files_exist(self):
        import streamlit_app

        assert streamlit_app._EXAMPLES_DIR.is_dir()
        assert streamlit_app._EXAMPLES_DIR.name == "examples"
        assert len(streamlit_app.EDIT_EXAMPLES) == 1
        for prompt, images in streamlit_app.EDIT_EXAMPLES:
            assert isinstance(prompt, str) and prompt.strip()
            # Multi-image: the example exists to show compositing across inputs.
            assert isinstance(images, list) and len(images) >= 2
            for image_path in images:
                assert Path(image_path).is_file()

    def test_truncate(self):
        import streamlit_app

        truncate = streamlit_app._truncate
        assert truncate("hi") == "hi"
        assert truncate("x" * 70) == "x" * 70
        long_label = truncate("y" * 71)
        assert long_label.endswith("…")
        assert len(long_label) == 71
        # The slice is rstrip'd before the ellipsis is appended.
        assert truncate("a" * 68 + "  zzzz") == "a" * 68 + "…"
        assert truncate("abcdef", length=3) == "abc…"


class TestExampleCallbacks:
    def test_set_example_prompt_clears_example_images(self):
        import streamlit_app

        fake = _FakeSessionState(example_images=["x"])
        with patch.object(streamlit_app, "st") as mock_st:
            mock_st.session_state = fake
            streamlit_app._set_example_prompt("a cat")
        assert fake["prompt_input"] == "a cat"
        assert "example_images" not in fake

    def test_load_edit_example_sets_prompt_and_copies_images(self):
        import streamlit_app

        src = ["p1", "p2"]
        fake = _FakeSessionState()
        with patch.object(streamlit_app, "st") as mock_st:
            mock_st.session_state = fake
            streamlit_app._load_edit_example("edit me", src)
        assert fake["prompt_input"] == "edit me"
        assert fake["example_images"] == ["p1", "p2"]
        # Stored as a copy so mutating session_state cannot corrupt EDIT_EXAMPLES.
        assert fake["example_images"] is not src

    def test_clear_example_images(self):
        import streamlit_app

        fake = _FakeSessionState(example_images=["x"], prompt_input="keep")
        with patch.object(streamlit_app, "st") as mock_st:
            mock_st.session_state = fake
            streamlit_app._clear_example_images()
            assert "example_images" not in fake
            assert fake["prompt_input"] == "keep"
            streamlit_app._clear_example_images()  # idempotent, must not raise


class TestMoreCoreLogic:
    def test_randomized_seed_forwarded_to_model(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.random.randint", return_value=777),
        ):
            _, seed = streamlit_app.infer("a cat", seed=42, randomize_seed=True)
            assert seed == 777
            assert mock_model.generate_image.call_args[1]["seed"] == 777

    def test_aspect_landing_on_min_floor(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        # aspect 4 -> short side rounds to exactly 256 (the new floor)
        assert streamlit_app._dimensions_from_images(
            [Image.new("RGB", (2048, 512))]
        ) == (
            1024,
            256,
        )
        assert streamlit_app._dimensions_from_images(
            [Image.new("RGB", (512, 2048))]
        ) == (
            256,
            1024,
        )

    def test_long_side_equals_max_image_size(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        for size in [(1600, 800), (800, 1600), (1200, 900), (1920, 1080)]:
            w, h = streamlit_app._dimensions_from_images([Image.new("RGB", size)])
            assert max(w, h) == streamlit_app.MAX_IMAGE_SIZE


class TestUIWidgets:
    def test_app_title_renders(self):
        import streamlit_app

        with _app_test() as app:
            at = app.run(timeout=10)
            assert any(t.value == streamlit_app.APP_TITLE for t in at.title)

    def test_dimension_sliders_range(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            for key in ("width_slider", "height_slider"):
                slider = at.slider(key=key)
                assert slider.min == 256
                assert slider.max == 1024
                assert slider.step == 32

    def test_no_guidance_slider_rendered(self):
        # Guidance is pinned (see GUIDANCE). Assert the exact slider set rather
        # than the absence of one label, so the check can't pass vacuously.
        with _app_test() as app:
            at = app.run(timeout=10)
            assert {s.label for s in at.slider} == {
                "Width",
                "Height",
                "Number of inference steps",
            }

    def test_seed_is_number_input(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            assert len(at.number_input) == 1
            seed = at.number_input[0]
            assert seed.label == "Seed"
            assert seed.min == 0
            assert seed.max == 2_147_483_647
            assert seed.value == 0

    def test_two_examples_sections_render(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            headers = {m.value for m in at.markdown}
            assert "**Examples**" in headers
            assert "**Editing examples**" in headers
            assert at.button(key="edit_example_0").label

    def test_example_buttons_wrap_their_labels(self):
        """Since Streamlit 1.63 an unset ``wrap`` on a button placed directly in
        a column or a horizontal container resolves to one ellipsized line.
        The examples now stack in the sidebar, where they would wrap anyway;
        the explicit opt-in keeps them wrapping wherever they move next.
        AppTest cannot see layout, so assert it on the proto -- ``wrap`` has
        field presence, so unset is detectable.
        """
        import streamlit_app

        keys = [f"example_{i}" for i in range(len(streamlit_app.EXAMPLE_PROMPTS))]
        keys += [f"edit_example_{i}" for i in range(len(streamlit_app.EDIT_EXAMPLES))]
        with _app_test() as app:
            at = app.run(timeout=10)
            for key in keys:
                proto = at.button(key=key).proto
                assert proto.HasField("wrap") and proto.wrap, key

    def test_t2i_example_clears_loaded_edit_images(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            at.button(key="edit_example_0").click().run(timeout=10)
            assert "example_images" in at.session_state
            at.button(key="example_2").click().run(timeout=10)
            assert "example_images" not in at.session_state
            assert "capybara" in (at.text_input(key="prompt_input").value or "")

    def test_clear_example_images_button(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            at.button(key="edit_example_0").click().run(timeout=10)
            assert any(b.label == "Clear example images" for b in at.button)
            next(b for b in at.button if b.label == "Clear example images").click().run(
                timeout=10
            )
            assert "example_images" not in at.session_state
            assert not any(b.label == "Clear example images" for b in at.button)

    def test_example_run_routes_through_edit_model(self):
        mock_txt2img = _make_mock_model()
        mock_edit = _make_mock_model()
        with _patched_models(mock_txt2img, mock_edit) as (app, _generate):
            at = app.run(timeout=10)
            # Load the editing example, then Run — loaded images must route to edit.
            at.button(key="edit_example_0").click().run(timeout=10)
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert mock_edit.generate_image.called
            assert not mock_txt2img.generate_image.called

    def test_empty_run_is_guarded(self):
        mock_txt2img = _make_mock_model()
        mock_edit = _make_mock_model()
        with _patched_models(mock_txt2img, mock_edit) as (app, _generate):
            at = app.run(timeout=10)
            # Run with no prompt and no image must not start inference.
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert not mock_txt2img.generate_image.called
            assert not mock_edit.generate_image.called
            guard = next(w for w in at.warning if "Enter a prompt" in w.value)
            assert guard.icon == ":material/warning:"

    def test_settings_use_toggles(self):
        # Prompt upsampling and Randomize seed are settings, so they render as
        # toggles (not checkboxes).
        with _app_test() as app:
            at = app.run(timeout=10)
            labels = {t.label for t in at.toggle}
            assert "Prompt upsampling" in labels
            assert "Randomize seed" in labels

    def test_infer_failure_is_handled(self):
        mock_model = _make_mock_model()
        mock_model.generate_image.side_effect = RuntimeError("backend exploded")
        with _patched_models(mock_model, mock_model) as (app, _generate):
            at = app.run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            # A generation failure is surfaced, not raised as an uncaught crash.
            assert not at.exception
            failure = next(e for e in at.error if "failed" in e.value.lower())
            assert failure.icon == ":material/error:"
            assert "result_image" not in at.session_state

    def test_size_preserved_when_images_cleared(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            # Loading an editing example sizes the sliders from the image
            # (mug.webp is 768x1024 -> 768x1024).
            at.button(key="edit_example_0").click().run(timeout=10)
            assert at.slider(key="width_slider").value == 768
            assert at.slider(key="height_slider").value == 1024
            # A manual width (distinct from any auto value) must survive clearing
            # the images, not reset to 1024.
            at.slider(key="width_slider").set_value(320).run(timeout=10)
            next(b for b in at.button if b.label == "Clear example images").click().run(
                timeout=10
            )
            assert at.slider(key="width_slider").value == 320

    def test_enhanced_prompt_banner_follows_toggle(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            at.session_state["auto_enhanced_prompt"] = "an enhanced prompt"
            # Upsampling off (default) -> the banner stays hidden.
            at.run(timeout=10)
            assert not any("Enhanced prompt" in i.value for i in at.info)
            # Turn upsampling on -> the banner appears for the stored prompt.
            at.toggle(key="auto_enhance_toggle").set_value(True).run(timeout=10)
            assert any("Enhanced prompt" in i.value for i in at.info)

    def test_placeholder_uses_material_icon(self):
        # The idle placeholder is theme-aware end to end: a Material Symbol
        # inside :gray[...] (an emoji would keep its fixed platform colors).
        with _app_test() as app:
            at = app.run(timeout=10)
            placeholder = next(m for m in at.markdown if "appear here" in m.value)
            assert ":material/image:" in placeholder.value
            assert "🖼" not in placeholder.value

    def test_successful_run_stores_and_renders_result(self):
        import streamlit_app

        with _app_test() as app:
            at = app.run(timeout=10)
            # The placeholder shows before any run.
            assert any("appear here" in m.value for m in at.markdown)
            # Disable randomize + fix the seed for a deterministic caption.
            next(t for t in at.toggle if t.label == "Randomize seed").set_value(
                False
            ).run(timeout=10)
            at.number_input[0].set_value(123).run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert not at.exception
            assert at.session_state["result_image"] is not None
            assert at.session_state["result_seed"] == 123
            # The result frame renders the image's seed caption, not the placeholder.
            assert any("Seed: 123" in c.value for c in at.caption)
            assert not any("appear here" in m.value for m in at.markdown)
            # Drawn under the pixel cap, not merely stretched to the column,
            # centered, with the seed caption inside the frame so it lines up
            # with the picture's edge.
            result = at.session_state["result_image"]
            center, frame = _find_canvas(at)
            assert frame.proto.width_config.pixel_width == (
                streamlit_app._canvas_display_width(*result.size)
            )
            flex = center.proto.flex_container
            assert flex.align == flex.ALIGN_CENTER
            assert any(
                "Seed: 123" in str(getattr(child, "value", ""))
                for child in frame.children.values()
            )

    def test_upsampling_run_uses_enhanced_prompt_and_caches(self):
        mock_model = _make_mock_model()
        with _patched_models(mock_model, mock_model, vlm_text="ENHANCED a cat") as (
            app,
            mock_generate,
        ):
            at = app.run(timeout=10)
            at.toggle(key="auto_enhance_toggle").set_value(True).run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            # The enhanced prompt reaches the model and the banner shows it.
            assert (
                mock_model.generate_image.call_args.kwargs["prompt"] == "ENHANCED a cat"
            )
            assert any("ENHANCED a cat" in i.value for i in at.info)
            # A second Run with the same prompt hits the cache (no new VLM call).
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert mock_generate.call_count == 1

    def test_enhance_cache_is_bounded(self):
        mock_model = _make_mock_model()
        with _patched_models(mock_model, mock_model, vlm_text="ENHANCED") as (
            app,
            _generate,
        ):
            at = app.run(timeout=10)
            # Pre-fill the cache to the cap, then enhance a brand-new prompt.
            at.session_state["_enhance_cache"] = {
                (f"p{i}", ()): f"e{i}" for i in range(32)
            }
            at.toggle(key="auto_enhance_toggle").set_value(True).run(timeout=10)
            at.text_input(key="prompt_input").set_value("fresh prompt")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            cache = at.session_state["_enhance_cache"]
            assert len(cache) == 32  # stays bounded
            assert ("p0", ()) not in cache  # oldest entry evicted
            assert ("fresh prompt", ()) in cache  # newest entry kept

    def test_unreadable_example_images_warn_and_clear(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            # Point example_images at a non-image file -> Image.open raises OSError.
            at.session_state["example_images"] = [__file__]
            at.run(timeout=10)
            assert any(
                "Could not load the example images" in w.value for w in at.warning
            )
            assert "example_images" not in at.session_state

    def test_missing_example_images_warn_and_clear(self):
        """A missing path fails in the preview's st.image first, raising
        MediaFileStorageError rather than OSError; uncaught, it aborted the
        script before the load guard, and re-raised on every rerun because
        the key was never cleared."""
        with _app_test() as app:
            at = app.run(timeout=10)
            at.session_state["example_images"] = [str(_REPO_ROOT / "missing.webp")]
            at.run(timeout=10)
            assert not at.exception
            assert any(
                "Could not load the example images" in w.value for w in at.warning
            )
            assert "example_images" not in at.session_state
            # The rest of the sidebar still renders.
            assert at.button(key="edit_example_0").label

    def test_corrupt_upload_warns(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            at.file_uploader[0].upload("bad.png", b"not a real image").run(timeout=10)
            assert not at.exception
            assert any(
                "Could not load one or more uploaded images" in w.value
                for w in at.warning
            )

    def test_edit_example_overrides_stale_manual_upload(self):
        """Loading an editing example must not be undone by an earlier upload.

        The uploader retains files across reruns, so without the nonce-cycled
        key the manual-upload override would pop the example images on the
        same rerun the example callback set them.
        """
        import streamlit_app

        with _app_test() as app:
            at = app.run(timeout=10)
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, format="PNG")
            at.file_uploader[0].upload("mine.png", buf.getvalue()).run(timeout=10)
            at.button(key="edit_example_0").click().run(timeout=10)
            assert not at.exception
            # A fresh (renamed) uploader renders empty, so the example survives…
            assert (
                at.session_state["example_images"]
                == (streamlit_app.EDIT_EXAMPLES[0][1])
            )
            # …and the input panel was opened via the keyed expander state.
            assert at.session_state["input_expander"] is True

    def test_changing_prompt_clears_enhanced_banner(self):
        """A prompt change must invalidate the stored enhanced prompt.

        The prompt box lives in a form, so typed edits only reach the app on
        submit (AppTest would not model that gating anyway); the production
        path that changes the prompt outside a submit is an example click,
        which pushes ``prompt_input`` via session state.
        """
        with _app_test() as app:
            at = app.run(timeout=10)
            at.session_state["auto_enhanced_prompt"] = "an enhanced prompt"
            at.toggle(key="auto_enhance_toggle").set_value(True).run(timeout=10)
            assert any("Enhanced prompt" in i.value for i in at.info)
            at.button(key="example_2").click().run(timeout=10)
            assert "auto_enhanced_prompt" not in at.session_state
            assert not any("Enhanced prompt" in i.value for i in at.info)

    def test_prompt_and_canvas_in_main_controls_in_sidebar(self):
        # The prompt stays in main: "auto" collapses the sidebar on narrow
        # viewports, and the primary action must not hide behind it.
        import streamlit_app

        keys = [f"example_{i}" for i in range(len(streamlit_app.EXAMPLE_PROMPTS))]
        keys += [f"edit_example_{i}" for i in range(len(streamlit_app.EDIT_EXAMPLES))]
        with _app_test() as app:
            at = app.run(timeout=10)
            main, sidebar = at.main, at.sidebar
            assert [t.key for t in main.text_input] == ["prompt_input"]
            assert [b.label for b in main.button] == ["Run"]
            assert len(sidebar.text_input) == 0
            assert [t.value for t in sidebar.title] == [streamlit_app.APP_TITLE]
            assert {s.key for s in sidebar.slider} == {
                "width_slider",
                "height_slider",
                "steps_slider",
            }
            assert {t.label for t in sidebar.toggle} == {
                "Prompt upsampling",
                "Randomize seed",
            }
            assert [n.label for n in sidebar.number_input] == ["Seed"]
            assert len(sidebar.get("file_uploader")) == 1
            assert set(keys) <= {b.key for b in sidebar.button}

    def test_run_notices_render_in_main(self):
        # Notices must survive a collapsed sidebar, and must stay out of
        # result_slot, which the bottom block overwrites on every run.
        mock_model = _make_mock_model()
        mock_model.generate_image.side_effect = RuntimeError("backend exploded")
        with _patched_models(mock_model, mock_model) as (app, _generate):
            at = app.run(timeout=10)
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert any("Enter a prompt" in w.value for w in at.main.warning)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert any("failed" in e.value.lower() for e in at.main.error)
            assert len(at.sidebar.error) == 0
            # Between the prompt bar and the canvas, not under the picture.
            assert _main_position(at, "error") < _main_position(at, "image")

    def test_enhancement_failure_notice_renders_in_main(self):
        # upsample_prompt warns and falls back when the VLM fails; that
        # warning must land in notices, not in the status inside result_slot,
        # which the bottom block overwrites -- a failed enhancement would then
        # leave no trace once the image renders.
        mock_model = _make_mock_model()
        with _patched_models(mock_model, mock_model) as (app, vlm_generate):
            vlm_generate.side_effect = RuntimeError("vlm down")
            at = app.run(timeout=10)
            at.toggle(key="auto_enhance_toggle").set_value(True).run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert not at.exception
            assert any("Prompt enhancement failed" in w.value for w in at.main.warning)
            assert len(at.sidebar.warning) == 0
            assert _main_position(at, "warning") < _main_position(at, "image")

    def test_failed_enhanced_run_drops_the_banner(self):
        # The enhanced-prompt banner sits under the image it produced. A failed
        # enhanced run must not hang its new prompt under the previous image.
        mock_model = _make_mock_model()
        with _patched_models(mock_model, mock_model, vlm_text="ENHANCED a cat") as (
            app,
            _generate,
        ):
            at = app.run(timeout=10)
            at.toggle(key="auto_enhance_toggle").set_value(True).run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert any("ENHANCED a cat" in i.value for i in at.main.info)
            mock_model.generate_image.side_effect = RuntimeError("backend exploded")
            at.text_input(key="prompt_input").set_value("a dog")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            at.run(timeout=10)
            assert not any("Enhanced prompt" in i.value for i in at.main.info)
            assert any("Seed:" in c.value for c in at.main.caption)

    def test_mid_run_status_sits_under_the_blank_canvas(self):
        # Freeze a Run two steps in (st.stop() leaves the tree as drawn): the
        # swatch holds the canvas geometry, the one status sits under it, and
        # the label carries the step count next to the progress bar.
        from types import SimpleNamespace

        import streamlit as st

        import streamlit_app

        mock_model = _make_mock_model()

        def _two_steps_then_freeze(**kwargs):
            config = SimpleNamespace(num_inference_steps=kwargs["num_inference_steps"])
            for t in range(2):
                for reporter in list(mock_model.callbacks.in_loop):
                    reporter.call_in_loop(
                        t=t,
                        seed=0,
                        prompt="",
                        latents=None,
                        config=config,
                        time_steps=None,
                    )
            st.stop()

        mock_model.generate_image.side_effect = _two_steps_then_freeze
        with _patched_models(mock_model, mock_model) as (app, _generate):
            at = app.run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert not at.exception
            frame = _canvas_frame(at)
            assert frame.proto.width_config.pixel_width == (
                streamlit_app._canvas_display_width(1024, 1024)
            )
            children = list(frame.children.values())
            assert [c.type for c in children[:2]] == ["image", "status"]
            status = children[1]
            assert status.label == "Generating image… step 2/4"
            assert status.proto.expanded
            progress = next(iter(status.children.values()))
            assert progress.type == "progress"
            assert progress.value == 50

    def test_result_frame_follows_the_result_size(self):
        # The result frame is sized from the image itself, not the sliders: a
        # portrait result under the default square sliders draws portrait.
        import streamlit_app

        mock_model = _make_mock_model()
        mock_model.generate_image.return_value = _MockGeneratedImage(
            Image.new("RGB", (672, 1024))
        )
        with _patched_models(mock_model, mock_model) as (app, _generate):
            at = app.run(timeout=10)
            at.text_input(key="prompt_input").set_value("a cat")
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert not at.exception
            assert _canvas_frame(at).proto.width_config.pixel_width == (
                streamlit_app._canvas_display_width(672, 1024)
            )

    def test_enhanced_prompt_renders_in_main(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            at.session_state["auto_enhanced_prompt"] = "an enhanced prompt"
            at.toggle(key="auto_enhance_toggle").set_value(True).run(timeout=10)
            assert any("Enhanced prompt" in i.value for i in at.main.info)
            assert len(at.sidebar.info) == 0

    def test_idle_canvas_previews_the_requested_size(self):
        import streamlit_app

        with _app_test() as app:
            at = app.run(timeout=10)
            placeholder = next(m for m in at.main.markdown if "appear here" in m.value)
            # Non-breaking spaces keep "W × H" whole on a narrow canvas.
            assert "1024 × 1024" in placeholder.value
            assert _canvas_frame(at).proto.width_config.pixel_width == (
                streamlit_app._canvas_display_width(1024, 1024)
            )
            # The portrait editing example resizes the output; the blank
            # canvas follows it and says what is being edited.
            at.button(key="edit_example_0").click().run(timeout=10)
            placeholder = next(m for m in at.main.markdown if "appear here" in m.value)
            assert "768 × 1024" in placeholder.value
            assert "Editing 2 input images" in placeholder.value
            assert _canvas_frame(at).proto.width_config.pixel_width == (
                streamlit_app._canvas_display_width(768, 1024)
            )
