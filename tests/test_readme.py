"""Contract tests for the local image assets README.md embeds.

The README leads with a dark-theme screenshot (``docs/screenshot-dark.png``)
and illustrates the editing example with its bundled inputs (``examples/*.webp``).
These are *local* references: unlike the shields.io badges (remote URLs), they
render on GitHub only when the file is committed at the referenced path. A
deleted, renamed, or gitignored asset turns the README hero into a broken-image
icon with no other signal — so this file locks every local image the README
points at to a file that exists and is git-tracked, the same anti-drift stance
``test_license`` takes on the LICENSE / pyproject / README trio.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_README = _REPO_ROOT / "README.md"

# The three ways an image reaches the rendered README: Markdown ``![alt](path)``,
# HTML ``<img src="path">`` (the hero and editing-example inputs use HTML so
# they can set width/alt), and a ``<picture>``'s ``<source srcset="...">``, which
# is how the hero serves its light variant to a light-mode reader.
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_HTML_IMG_RE = re.compile(r"""<img\b[^>]*?\bsrc=["']([^"']+)["']""", re.IGNORECASE)
_SRCSET_RE = re.compile(r"""<source\b[^>]*?\bsrcset=["']([^"']+)["']""", re.IGNORECASE)

# The hero pair, one per color scheme; lock both by name so neither can
# silently vanish while other images keep the suite green.
_HERO_SCREENSHOTS = ("docs/screenshot-dark.png", "docs/screenshot-light.png")


def _local_image_refs():
    """Every image ``src`` in README.md, with remote URLs and anchors dropped."""
    text = _README.read_text()
    # A Markdown image may carry a title — ``![alt](path "title")`` — after the
    # path, so strip anything past the first whitespace. An HTML ``src`` capture
    # is already the bare attribute value (and could legitimately contain a
    # space), so it must NOT be split.
    refs = [m.strip().split()[0] for m in _MD_IMG_RE.findall(text) if m.strip()]
    refs += [m.strip() for m in _HTML_IMG_RE.findall(text) if m.strip()]
    # A srcset is a comma-separated candidate list, each "url [descriptor]".
    for srcset in _SRCSET_RE.findall(text):
        refs += [c.split()[0] for c in srcset.split(",") if c.strip()]
    local = []
    for ref in refs:
        ref = ref.split("#", 1)[0]  # drop any ``#anchor``
        if ref.startswith(("http://", "https://", "data:", "//")):
            continue  # remote (badges) or inline data URI — not a repo file
        local.append(ref)
    return local


_requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="tracked-file check needs git"
)


class TestReadmeImages:
    """Every local image the README embeds must exist and be committed."""

    def test_local_images_exist(self):
        refs = _local_image_refs()
        # Non-vacuity: a parser regression that returned nothing would make the
        # exist-check below pass without checking a single file.
        assert refs, "found no local image references in README (parser drift?)"
        missing = [r for r in refs if not (_REPO_ROOT / r).is_file()]
        assert not missing, f"README references missing local images: {missing}"

    def test_hero_screenshots_referenced(self):
        # The studio hero is the deliverable these tests reflect; assert it is
        # embedded via a real image tag (not merely a substring of the prose),
        # then that it is present on disk.
        refs = _local_image_refs()
        for asset in _HERO_SCREENSHOTS:
            assert asset in refs, f"README no longer embeds {asset}"
            assert (_REPO_ROOT / asset).is_file(), f"{asset} is missing on disk"

    @_requires_git
    def test_local_images_are_tracked(self):
        # A local-only (untracked or gitignored) image renders on the author's
        # machine but 404s on GitHub — the same silent failure ``test_license``
        # guards LICENSE against. Assert every referenced image is tracked.
        refs = _local_image_refs()
        assert refs, "found no local image references in README (parser drift?)"
        listed = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "-z", "--", *refs],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        tracked = {p for p in listed.split("\0") if p}
        untracked = [r for r in refs if r not in tracked]
        assert not untracked, f"README images not git-tracked: {untracked}"
