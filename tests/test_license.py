"""Contract tests for the project's license declaration.

The repo shipped for several versions with no license at all — which under
copyright law means *all rights reserved*, so nobody could legally fork or reuse
the public code. It is now MIT, declared in three places that must stay
consistent or the guarantee silently rots:

  * ``LICENSE`` — the canonical MIT text. GitHub's Licensee detector matches the
    wording, so an emptied or reworded file drops the repo's "MIT License" badge
    and blanks the ``licenseInfo`` API field.
  * ``pyproject.toml`` — ``license = "MIT"`` (a PEP 639 SPDX string), so the
    packaging metadata agrees with the file.
  * ``README.md`` — a public-facing License section that links the file and
    credits the permissive upstreams.

These tests tie the three together against one ``_EXPECTED_SPDX`` source of
truth: relicense one without the others and CI fails, exactly as ``test_ci.py``
ties ``.python-version`` to ``requires-python``.
"""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LICENSE = _REPO_ROOT / "LICENSE"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_README = _REPO_ROOT / "README.md"

# The SPDX identifier the whole stack settles on. Pinned once so a drift in
# either the LICENSE file or pyproject is measured against a single truth.
_EXPECTED_SPDX = "MIT"

# Load-bearing sentences of the canonical MIT text. GitHub's detector keys off
# this wording; if it stops appearing, the badge and licenseInfo go with it. Not
# the full file (that would lock trailing-whitespace trivia) — just the phrases
# that identify the license.
_MIT_MARKERS = (
    "MIT License",
    "Permission is hereby granted, free of charge",
    'THE SOFTWARE IS PROVIDED "AS IS"',
    "WITHOUT WARRANTY OF ANY KIND",
)

# "Copyright (c) <year> <holder>" — MIT detection needs a real copyright notice.
# Match the shape (a 4-digit year, optional range, then the holder). The holder
# group is deliberately permissive so a whitespace-only holder still matches and
# is then caught by the .strip() assertion in the test, rather than silently
# failing the match. Not the specific name or year, which may legitimately change.
_COPYRIGHT_RE = re.compile(r"Copyright \(c\) \d{4}(?:-\d{4})? (.*)")


def _pyproject_license():
    """The ``project.license`` value from pyproject.toml (a PEP 639 SPDX string)."""
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["license"]


class TestLicenseFile:
    """``LICENSE`` must exist and be recognizably the MIT license."""

    def test_license_file_exists_and_nonempty(self):
        assert _LICENSE.is_file(), "LICENSE file is missing"
        assert _LICENSE.read_text().strip(), "LICENSE file is empty"

    def test_license_is_mit(self):
        text = _LICENSE.read_text()
        for marker in _MIT_MARKERS:
            assert marker in text, f"LICENSE is missing MIT marker: {marker!r}"

    def test_license_carries_a_copyright_notice(self):
        # A blanked holder or missing year would still read as "MIT-ish" prose
        # but fail GitHub's detection, so assert the notice is well-formed.
        match = _COPYRIGHT_RE.search(_LICENSE.read_text())
        assert match, "LICENSE has no 'Copyright (c) <year> <holder>' line"
        assert match.group(1).strip(), "LICENSE copyright holder is empty"


class TestPyprojectLicense:
    """``pyproject.toml`` must declare the license as a PEP 639 SPDX string."""

    def test_declares_mit_spdx_string(self):
        license_field = _pyproject_license()
        # PEP 639 uses a bare SPDX string; the deprecated table form
        # (``{ text = "..." }``) or a removed field must fail here.
        assert isinstance(license_field, str), (
            f"license must be a PEP 639 SPDX string, got {type(license_field).__name__}"
        )
        assert license_field == _EXPECTED_SPDX, (
            f"pyproject license is {license_field!r}, expected {_EXPECTED_SPDX!r}"
        )


class TestLicenseConsistency:
    """The file and the metadata must name the *same* license — the anti-drift
    guard. Relicensing means editing both; editing one alone fails here."""

    def test_pyproject_matches_license_file(self):
        assert _pyproject_license() == _EXPECTED_SPDX
        assert f"{_EXPECTED_SPDX} License" in _LICENSE.read_text(), (
            f"pyproject declares {_EXPECTED_SPDX} but LICENSE is not the "
            f"{_EXPECTED_SPDX} text"
        )


class TestReadmeAttribution:
    """The public README must declare the project's own license in its License
    section (anchored, so upstream-credit license names can't satisfy it)."""

    def test_readme_declares_project_license(self):
        # Anchor to the markdown link that appears ONLY in the project's own
        # declaration ("[MIT License](LICENSE)"), not any occurrence of the SPDX
        # token. README also names licenses in its upstream-credits list ("mflux
        # and mlx-vlm — MIT"), so a bare `_EXPECTED_SPDX in text` would stay green
        # after a partial relicense that updated LICENSE + pyproject but forgot
        # this section. The link welds the token, "License", and the file target
        # into one project-scoped assertion (subsuming the old link/token checks).
        text = _README.read_text()
        assert "## License" in text, "README dropped its License section"
        assert f"[{_EXPECTED_SPDX} License](LICENSE)" in text, (
            f"README's License section no longer declares the project as "
            f"{_EXPECTED_SPDX} via a [{_EXPECTED_SPDX} License](LICENSE) link"
        )


_requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="tracked-file check needs git"
)


@_requires_git
class TestLicenseTracked:
    """An untracked (e.g. gitignored) LICENSE silently defeats the point: GitHub
    only detects a license committed on the default branch."""

    def test_license_is_git_tracked(self):
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "LICENSE"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        assert out.strip() == "LICENSE", "LICENSE is not git-tracked"


class TestVendoredPromptNotice:
    """BFL's prompts are Apache-2.0, and JSON cannot carry a notice, so a
    LICENSE + NOTICE sidecar travels with them and names the source commit
    that streamlit_app.py's loader comment cites."""

    _DIR = _REPO_ROOT / "prompts"
    _SHA = "bfc8889a4497f846416f253a3972a1941ac698b9"

    def test_sidecar_files_exist_and_name_the_source(self):
        license_text = (self._DIR / "LICENSE").read_text()
        assert "Apache License" in license_text
        assert "Version 2.0" in license_text
        notice = (self._DIR / "NOTICE").read_text()
        assert self._SHA in notice
        assert "black-forest-labs/flux-3-image-klein" in notice

    def test_the_loader_cites_the_same_commit(self):
        app = (_REPO_ROOT / "streamlit_app.py").read_text()
        assert self._SHA[:8] in app

    def test_sidecar_is_git_tracked(self):
        if shutil.which("git") is None:
            pytest.skip("git not available")
        tracked = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "prompts"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        for name in ("LICENSE", "NOTICE", "pup_prompts_external_vlm.json"):
            assert f"prompts/{name}" in tracked, name
