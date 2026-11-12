"""The evaluation set's case file and the runner's pure helpers.

eval/run_eval.py needs real weights for ``run``, so pytest never calls it.
What can break silently without weights is checked here: a case file that
drifts from the plan's shape, a reference image that was renamed away, and the
drift and blind-key arithmetic that ``compare`` reports.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _REPO_ROOT / "eval"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_eval", _EVAL_DIR / "run_eval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_eval = _load_runner()


class TestCaseFile:
    def test_ten_cases_six_t2i_four_edits(self):
        cases = run_eval.load_cases()["cases"]
        assert [c["id"] for c in cases] == list(range(1, 11))
        assert sum(not c["references"] for c in cases) == 6
        assert sum(bool(c["references"]) for c in cases) == 4
        assert len({c["name"] for c in cases}) == 10  # names become file names

    def test_every_reference_is_a_bundled_example(self):
        # The golden run is the only pre-upgrade baseline; a reference renamed
        # away would make every later comparison of that case impossible.
        for case in run_eval.load_cases()["cases"]:
            for ref in case["references"]:
                assert (_REPO_ROOT / "examples" / ref).is_file(), ref

    def test_seeds_follow_the_plan(self):
        doc = run_eval.load_cases()
        assert doc["seed_base"] == 20261021
        assert run_eval.case_seed(doc, doc["cases"][0]) == 20261022

    def test_latency_case_is_text_to_image(self):
        doc = run_eval.load_cases()
        case = next(c for c in doc["cases"] if c["id"] == doc["latency"]["case"])
        assert not case["references"]
        assert len(doc["latency"]["seeds"]) == 3  # median of 3, per the plan


class TestDrift:
    def test_identical_images_have_zero_drift(self):
        image = Image.new("RGB", (64, 48), (10, 200, 30))
        assert run_eval.drift_pct(image, image.copy()) == 0.0
        assert run_eval.pixel_sha256(image) == run_eval.pixel_sha256(image.copy())

    def test_black_against_white_is_the_full_range(self):
        black = Image.new("RGB", (32, 32), (0, 0, 0))
        white = Image.new("RGB", (32, 32), (255, 255, 255))
        assert run_eval.drift_pct(black, white) == pytest.approx(100.0)

    def test_pixel_hash_sees_one_pixel(self):
        a = Image.new("RGB", (8, 8))
        b = a.copy()
        b.putpixel((3, 3), (1, 0, 0))
        assert run_eval.pixel_sha256(a) != run_eval.pixel_sha256(b)


def _fake_run(root, label, colors, prompts=None):
    out = root / label
    out.mkdir(parents=True)
    cases = []
    for i, color in enumerate(colors, start=1):
        image = Image.new("RGB", (32, 32), color)
        name = f"case-{i:02d}.png"
        image.save(out / name)
        cases.append(
            {
                "id": i,
                "name": f"c{i}",
                "prompt": f"prompt {i}",
                "enhanced_prompt": (prompts or {}).get(i),
                "references": [],
                "file": name,
                "pixel_sha256": run_eval.pixel_sha256(image),
            }
        )
    (out / "manifest.json").write_text(json.dumps({"label": label, "cases": cases}))


class TestCompare:
    def test_flags_only_what_moved(self, tmp_path):
        _fake_run(tmp_path, "a", [(0, 0, 0), (0, 0, 0), (0, 0, 0)])
        _fake_run(
            tmp_path,
            "b",
            [(0, 0, 0), (2, 2, 2), (128, 128, 128)],
            prompts={1: "the model rewrote this"},
        )
        report = run_eval.compare("a", "b", out_root=tmp_path, shuffle_seed=0)
        rows = {r["id"]: r for r in report["cases"]}
        # Case 1: same pixels, but the enhanced prompt changed, so it is flagged.
        assert rows[1]["bit_identical"] and rows[1]["flagged"]
        # Case 2: <1% drift, under the 2% threshold.
        assert not rows[2]["bit_identical"] and not rows[2]["flagged"]
        # Case 3: ~50% drift.
        assert rows[3]["flagged"]

    def test_contact_sheet_is_blind(self, tmp_path):
        _fake_run(tmp_path, "golden-old", [(0, 0, 0)] * 2)
        _fake_run(tmp_path, "candidate-new", [(9, 9, 9)] * 2)
        report = run_eval.compare(
            "golden-old", "candidate-new", out_root=tmp_path, shuffle_seed=1
        )
        out = tmp_path / "compare" / "golden-old__vs__candidate-new"
        page = (out / "contact.html").read_text()
        # The key is in comparison.json, never on the page.
        assert "golden-old" not in page and "candidate-new" not in page
        assert set(report["blind_key"]) == {"1", "2"}
        for order in report["blind_key"].values():
            assert sorted(order) == ["candidate-new", "golden-old"]


class TestResolvedRevision:
    """mflux fetches only weight files, so the revision is read from the
    cache layout; snapshot_download(local_files_only=True) refuses a partial
    snapshot, which left the first golden manifest's revision null."""

    def _repo(self, tmp_path, monkeypatch):
        import huggingface_hub.constants

        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
        return tmp_path / "models--org--model"

    def test_floating_repo_reads_refs_main(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path, monkeypatch)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("abc123\n")
        (repo / "snapshots" / "abc123").mkdir(parents=True)
        (repo / "snapshots" / "old999").mkdir()
        assert run_eval.resolved_revision("org/model") == "abc123"

    def test_pinned_repo_without_a_ref_reads_its_snapshot(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path, monkeypatch)
        (repo / "snapshots" / "def456").mkdir(parents=True)
        assert run_eval.resolved_revision("org/model") == "def456"

    def test_uncached_repo_is_unknown(self, tmp_path, monkeypatch):
        self._repo(tmp_path, monkeypatch)
        assert run_eval.resolved_revision("org/model") is None
