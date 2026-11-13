"""Run the seeded evaluation set, and compare two runs for drift.

The set, the rubric and the gates are in docs/plans/flux3-upgrade.md
(Evaluation). This script measures the app's real path: it calls only
``_resolve_prompt`` and ``infer()``, with the shipped defaults read from the
app's constants, and never a getter or a model-specific keyword, so a model
swap needs no change here beyond what ``infer()``'s signature forces.

It needs real weights, so it is not part of pytest (tests/test_eval.py checks
only the case file and the pure helpers). Outputs never land in the repo:

    uv run python eval/run_eval.py run --label golden-flux2-0.9.0
    uv run python eval/run_eval.py compare golden-flux2-0.9.0 <other label>

``run`` writes one PNG per case plus ``manifest.json`` (prompts, enhanced
prompts, seeds, sizes, the model revision as resolved from the HF cache,
package versions, timings, peak MLX memory and empty score slots) to
``<out-root>/<label>/``. ``compare`` writes ``comparison.json`` (drift figures
and the blind key) and ``contact.html`` (the two runs side by side, unlabeled,
in random left/right order) to ``<out-root>/compare/<a>__vs__<b>/``.
"""

import argparse
import base64
import hashlib
import html
import io
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = Path(__file__).resolve().parent / "cases.json"
DEFAULT_OUT_ROOT = Path.home() / "Library/Application Support/flux-image/eval"
# Mean absolute difference, in percent of the 0-255 range, above which a case
# is flagged for a human look. A starting point: the plan tunes it from the
# measured repeat-run and toolchain-drift figures. Flags never fail anything.
DRIFT_THRESHOLD_PCT = 2.0
DRIFT_SIZE = 256
_VERSIONED = (
    "mflux",
    "mlx",
    "mlx-metal",
    "mlx-vlm",
    "transformers",
    "huggingface-hub",
    "streamlit",
    "pillow",
    "numpy",
)


def load_cases(path=CASES_PATH):
    return json.loads(Path(path).read_text())


def case_seed(doc, case):
    return doc["seed_base"] + case["id"]


def _git(*args):
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _versions():
    out = {"python": platform.python_version()}
    for name in _VERSIONED:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _machine():
    def sysctl(key):
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    mem = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(mem) / 2**30) if mem and mem.isdigit() else None,
        "macos": platform.mac_ver()[0],
    }


def resolved_revision(repo_id):
    """The snapshot SHA the HF cache serves for ``repo_id``, as actually loaded.

    Read even for a model the app doesn't pin: FLUX.2 floats to ``main``, and
    an upstream re-upload would otherwise look like toolchain drift. Read from
    the cache's own layout, because mflux fetches only the weight files and
    ``snapshot_download(local_files_only=True)`` refuses such a partial
    snapshot. ``refs/main`` covers a floating repo; a repo fetched by pinned
    SHA has no ref, so its single snapshot folder is the answer.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        return ref.read_text().strip()
    snapshots = sorted(p.name for p in (repo_dir / "snapshots").glob("*"))
    if len(snapshots) == 1:
        return snapshots[0]
    return snapshots or None  # several and no ref: record them all


def pixel_sha256(image):
    """Hash of the decoded pixels, so PNG encoder changes can't fake drift."""
    h = hashlib.sha256(f"{image.mode}{image.size}".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def drift_pct(a, b):
    """Mean absolute difference of two images, in percent of the 0-255 range,
    on a DRIFT_SIZE grayscale downscale (numpy and Pillow only)."""

    def prep(image):
        small = image.convert("L").resize(
            (DRIFT_SIZE, DRIFT_SIZE), Image.Resampling.BILINEAR
        )
        return np.asarray(small, dtype=np.float64)

    return float(np.abs(prep(a) - prep(b)).mean() / 255 * 100)


def _timed_run(app, prompt, image_list, width, height, seed, auto_enhance):
    """One Run as the user experiences it: _resolve_prompt plus infer()."""
    import mlx.core as mx

    mx.reset_peak_memory()
    start = time.perf_counter()
    final_prompt, was_enhanced = app._resolve_prompt(prompt, image_list, auto_enhance)
    image, used_seed = app.infer(
        final_prompt,
        seed=seed,
        width=width,
        height=height,
        image_list=image_list,
    )
    seconds = time.perf_counter() - start
    peak_gb = mx.get_peak_memory() / 2**30
    return image, used_seed, final_prompt, was_enhanced, seconds, peak_gb


def run(
    label,
    *,
    out_root,
    cases_path,
    only=None,
    latency=True,
    force=False,
    auto_enhance: bool | None = None,
):
    sys.path.insert(0, str(REPO_ROOT))
    import streamlit_app as app

    doc = load_cases(cases_path)
    cases = [c for c in doc["cases"] if not only or c["id"] in only]
    out = Path(out_root) / label
    if out.exists() and not force:
        sys.exit(f"{out} exists; a golden run is never overwritten (pass --force)")
    out.mkdir(parents=True, exist_ok=True)

    size = app.DEFAULT_SIZE
    # As shipped unless overridden: the plan also records an upsampling-off figure.
    if auto_enhance is None:
        auto_enhance = app.DEFAULT_AUTO_ENHANCE
    manifest = {
        "label": label,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "versions": _versions(),
        "machine": _machine(),
        "defaults": {
            "steps": app.DEFAULT_STEPS,
            "size": size,
            "auto_enhance": auto_enhance,
            "guidance": getattr(app, "GUIDANCE", None),
        },
        # Revisions are resolved after the runs: the first Run may download.
        "model": {"repo": app.MODEL_REPO, "revision": None},
        "vlm": {"repo": app.VLM_MODEL_ID, "revision": None} if auto_enhance else None,
        "drift": {"size": DRIFT_SIZE, "threshold_pct": DRIFT_THRESHOLD_PCT},
        "latency": None,
        "cases": [],
    }

    if latency:
        lat = doc["latency"]
        lat_case = next(c for c in doc["cases"] if c["id"] == lat["case"])
        # Includes a download if the weights are not cached yet: check the
        # model's cache before reading this as a load time.
        print(f"latency: case {lat_case['id']}, cold first Run (load + warm-up)...")
        start = time.perf_counter()
        *_, peak = _timed_run(
            app, lat_case["prompt"], None, size, size, 0, auto_enhance
        )
        cold = time.perf_counter() - start
        warm = []
        for seed in lat["seeds"]:
            *_, seconds, _peak = _timed_run(
                app, lat_case["prompt"], None, size, size, seed, auto_enhance
            )
            warm.append(round(seconds, 2))
            print(f"  warm seed {seed}: {seconds:.1f} s")
        manifest["latency"] = {
            "case": lat_case["id"],
            "size": [size, size],
            "cold_first_run_seconds": round(cold, 2),
            "cold_peak_memory_gb": round(peak, 2),
            "warm_seconds": warm,
            "warm_median_seconds": statistics.median(warm),
        }

    # Text-to-image first, then edits. One model instance serves both modes,
    # so there is no second pipeline to load between them.
    ordered = sorted(cases, key=lambda c: (bool(c["references"]), c["id"]))
    for case in ordered:
        refs = case["references"]
        # Opened the way the app opens them (EXIF-upright RGB); _open_image is
        # a pure helper, not a getter.
        opener = getattr(app, "_open_image", Image.open)
        image_list = [opener(app._EXAMPLES_DIR / name) for name in refs] or None
        if image_list:
            width, height = app._dimensions_from_images(image_list)
        else:
            width = height = size
        seed = case_seed(doc, case)
        print(f"case {case['id']:>2} {case['name']} ({width}x{height}, seed {seed})...")
        image, used_seed, final_prompt, was_enhanced, seconds, peak = _timed_run(
            app, case["prompt"], image_list, width, height, seed, auto_enhance
        )
        filename = f"case-{case['id']:02d}-{case['name']}.png"
        image.save(out / filename)
        manifest["cases"].append(
            {
                "id": case["id"],
                "name": case["name"],
                "mode": "edit" if refs else "t2i",
                "prompt": case["prompt"],
                "enhanced_prompt": final_prompt if was_enhanced else None,
                "seed": used_seed,
                "steps": app.DEFAULT_STEPS,
                "width": image.width,
                "height": image.height,
                "references": refs,
                "file": filename,
                "pixel_sha256": pixel_sha256(image),
                "seconds": round(seconds, 2),
                "peak_memory_gb": round(peak, 2),
                "scores": None,
            }
        )
        print(f"   {seconds:.1f} s, peak {peak:.1f} GB")

    manifest["cases"].sort(key=lambda c: c["id"])
    manifest["model"]["revision"] = resolved_revision(app.MODEL_REPO)
    if manifest["vlm"]:
        manifest["vlm"]["revision"] = resolved_revision(app.VLM_MODEL_ID)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    return out


def _data_uri(path, longest=768):
    image = Image.open(path).convert("RGB")
    image.thumbnail((longest, longest))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def compare(
    label_a, label_b, *, out_root, threshold=DRIFT_THRESHOLD_PCT, shuffle_seed=None
):
    root = Path(out_root)
    man_a = json.loads((root / label_a / "manifest.json").read_text())
    man_b = json.loads((root / label_b / "manifest.json").read_text())
    by_id_b = {c["id"]: c for c in man_b["cases"]}
    rng = random.Random(shuffle_seed)

    rows, key = [], {}
    for ca in man_a["cases"]:
        cb = by_id_b.get(ca["id"])
        if cb is None:
            continue
        img_a = Image.open(root / label_a / ca["file"])
        img_b = Image.open(root / label_b / cb["file"])
        same_size = img_a.size == img_b.size
        pct = drift_pct(img_a, img_b) if same_size else None
        prompt_same = ca["enhanced_prompt"] == cb["enhanced_prompt"]
        flagged = (not same_size) or pct is None or pct > threshold or not prompt_same
        rows.append(
            {
                "id": ca["id"],
                "name": ca["name"],
                "bit_identical": ca["pixel_sha256"] == cb["pixel_sha256"],
                "drift_pct": None if pct is None else round(pct, 4),
                "same_size": same_size,
                "enhanced_prompt_same": prompt_same,
                "flagged": flagged,
            }
        )
        key[str(ca["id"])] = rng.choice([[label_a, label_b], [label_b, label_a]])

    out = root / "compare" / f"{label_a}__vs__{label_b}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "a": label_a,
        "b": label_b,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold_pct": threshold,
        "drift_size": DRIFT_SIZE,
        "cases": rows,
        "blind_key": key,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_contact_sheet(out / "contact.html", root, man_a, man_b, key)

    print(f"{'case':<26} {'identical':>9} {'drift %':>8} {'prompt':>7} flag")
    for r in rows:
        pct = "-" if r["drift_pct"] is None else f"{r['drift_pct']:.4f}"
        print(
            f"{r['id']:>2} {r['name']:<23} {str(r['bit_identical']):>9} {pct:>8} "
            f"{'same' if r['enhanced_prompt_same'] else 'DIFF':>7} "
            f"{'FLAG' if r['flagged'] else ''}"
        )
    print(f"wrote {out}")
    return report


def _write_contact_sheet(path, root, man_a, man_b, key):
    """Each case's two outputs side by side, unlabeled and in random order.

    The key lives in comparison.json, never on the page, so "the new one" can't
    score itself up. Enhanced prompts are left off for the same reason: only
    one version may have them.
    """
    by_label = {man_a["label"]: man_a, man_b["label"]: man_b}
    cases_by_label = {
        lbl: {c["id"]: c for c in m["cases"]} for lbl, m in by_label.items()
    }
    examples = REPO_ROOT / "examples"
    rubric = (
        "Score 0-2 each: adherence; preservation (edits only); medium/style; "
        "artifacts. Left/right order is random per case."
    )
    parts = [
        "<!doctype html><meta charset=utf-8><title>Evaluation contact sheet</title>",
        "<style>body{font:14px system-ui;margin:24px;background:#fff;color:#111}"
        "section{margin:0 0 40px}img{max-width:100%;height:auto;display:block}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
        ".refs{display:flex;gap:8px;margin:8px 0}.refs img{height:120px;width:auto}"
        "h2{font-size:16px;margin:0 0 4px}p{margin:4px 0;max-width:80ch}</style>",
        f"<h1>Contact sheet</h1><p>{html.escape(rubric)}</p>",
    ]
    for case in man_a["cases"]:
        order = key.get(str(case["id"]))
        if not order:
            continue
        parts.append(
            f"<section><h2>Case {case['id']}: {html.escape(case['name'])}</h2>"
        )
        parts.append(f"<p>{html.escape(case['prompt'])}</p>")
        if case["references"]:
            parts.append("<div class=refs>")
            for ref in case["references"]:
                parts.append(
                    f"<img alt='reference' src='{_data_uri(examples / ref, 240)}'>"
                )
            parts.append("</div>")
        parts.append("<div class=pair>")
        for side, lbl in zip(("Left", "Right"), order, strict=True):
            file = root / lbl / cases_by_label[lbl][case["id"]]["file"]
            parts.append(f"<figure><img alt='{side}' src='{_data_uri(file)}'>")
            parts.append(f"<figcaption>{side}</figcaption></figure>")
        parts.append("</div></section>")
    Path(path).write_text("\n".join(parts) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the evaluation set")
    p_run.add_argument("--label", required=True, help="e.g. golden-flux2-0.9.0")
    p_run.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_run.add_argument("--cases", type=Path, default=CASES_PATH)
    p_run.add_argument("--only", type=lambda s: {int(x) for x in s.split(",")})
    p_run.add_argument("--no-latency", action="store_true")
    p_run.add_argument("--force", action="store_true", help="overwrite the label")
    p_run.add_argument(
        "--upsampling",
        choices=("on", "off"),
        help="override the shipped DEFAULT_AUTO_ENHANCE (recorded in the manifest)",
    )

    p_cmp = sub.add_parser("compare", help="drift check and blind contact sheet")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_cmp.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_cmp.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD_PCT)
    p_cmp.add_argument("--shuffle-seed", type=int)

    args = parser.parse_args(argv)
    if args.command == "run":
        run(
            args.label,
            out_root=args.out_root,
            cases_path=args.cases,
            only=args.only,
            latency=not args.no_latency,
            force=args.force,
            auto_enhance=None if args.upsampling is None else args.upsampling == "on",
        )
    else:
        compare(
            args.a,
            args.b,
            out_root=args.out_root,
            threshold=args.threshold,
            shuffle_seed=args.shuffle_seed,
        )


if __name__ == "__main__":
    main()
