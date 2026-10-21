"""Guard tests: no real secret may reach a git-tracked file.

Context: the CLAUDE.md audit found a live Hugging Face access token sitting in a
local ``.env``. It was correctly gitignored and blocked by ``guard-paths.sh``,
so it never reached the repo — but "never reached the repo" was luck plus one
convenience hook, not something CI could prove. These tests turn that into an
enforced invariant:

  * no tracked file may contain a recognizable secret (HF / GitHub token, PEM
    private key, AWS access-key id);
  * the secret-bearing files themselves (``.env`` / non-template ``.env.*``,
    ``secrets.toml``) must never be tracked.

They scan ``git ls-files`` (so a gitignored local ``.env`` is invisible here,
exactly as it should be) and run in CI on every push/PR. This is a *detective*
control: a push reaches GitHub before CI runs, so on a public repo treat any hit
as an already-published secret to rotate — the guard makes that impossible to
miss and fails any later commit that reintroduces it. The detector is
self-tested below so a broken (or newly added, untested) regex can't let the
guard pass vacuously.

Note: files are scanned as raw bytes, so a single non-UTF-8 byte can't hide an
ASCII secret by making the whole file undecodable; and the sample secrets in the
detector self-test are built by concatenation, so no contiguous secret literal
exists in this file — which ``test_no_secret_in_tracked_files`` also scans.
"""

import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ``git ls-files`` is the source of truth for "in the repo": it excludes the
# gitignored local .env and every other untracked file, so this guard sees only
# what a push would actually publish.
_requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="secret scan needs git to list tracked files"
)

# High-signal secret shapes: fixed prefixes + fixed lengths keep false positives
# near zero, so a match is almost certainly a real leak. Byte patterns, so the
# scan works on any file without a decode step. Add new shapes here — and add a
# firing sample to _DETECTOR_SAMPLES below, which the self-test enforces.
_SECRET_PATTERNS = (
    # Hugging Face user access token: "hf_" + 34 base62 chars — the exact shape
    # the audit found in a local .env.
    ("hugging-face-token", re.compile(rb"hf_[A-Za-z0-9]{34,}")),
    # GitHub token: ghp_/gho_/ghs_/ghu_/ghr_ + 36 chars (classic PAT / OAuth /
    # server / user / refresh), plus fine-grained PATs. The credential this repo
    # actually uses — it runs on GitHub Actions and mandates the gh CLI.
    ("github-token", re.compile(rb"gh[opsur]_[A-Za-z0-9]{36,}")),
    ("github-fine-grained-pat", re.compile(rb"github_pat_[0-9A-Za-z_]{82,}")),
    # PEM private-key block header (RSA / EC / OPENSSH / DSA / PGP or unlabelled).
    ("private-key-block", re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    # AWS access-key id: fixed "AKIA" prefix + 16 upper/digits.
    ("aws-access-key-id", re.compile(rb"AKIA[0-9A-Z]{16}")),
)

# One firing sample per pattern name, built by concatenation so no contiguous
# secret literal lands in this file (which the scanner also reads). The self-test
# asserts these keys match _SECRET_PATTERNS exactly, so a new pattern without a
# sample — or a sample its own regex misses — fails CI.
_DETECTOR_SAMPLES = {
    "hugging-face-token": b"hf_" + b"a1B2c3D4e5" * 3 + b"F6g7",  # "hf_" + 34
    "github-token": b"ghp_" + b"A1b2C3d4E5" * 3 + b"F6g7h8",  # "ghp_" + 36
    "github-fine-grained-pat": b"github_pat_" + b"A1b2C3d4E5" * 8 + b"AB",  # + 82
    "private-key-block": b"-----BEGIN " + b"OPENSSH PRIVATE KEY" + b"-----",
    "aws-access-key-id": b"AKIA" + b"ABCDEFGH12345678",  # "AKIA" + 16
}

# Mirror guard-paths.sh: real dotenv files hold secrets and must stay untracked,
# but any committed *template* (which carries no secrets) is allowed.
_ALLOWED_ENV_TEMPLATES = frozenset(
    {".env.example", ".env.sample", ".env.template", ".env.dist"}
)


def _tracked_files():
    """Every git-tracked path, repo-relative and POSIX-separated (git's format)."""
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def _read_bytes_or_none(rel_path):
    """Raw bytes of a tracked file, or None if it is gone (e.g. staged deletion)."""
    try:
        return (_REPO_ROOT / rel_path).read_bytes()
    except OSError:
        return None


@_requires_git
class TestNoSecretsInRepo:
    """No recognizable secret — and no secret-bearing file — may be tracked."""

    def test_no_secret_in_tracked_files(self):
        tracked = _tracked_files()
        assert tracked, "git ls-files returned nothing; scan would pass vacuously"

        scanned = 0
        offenders = []
        for rel_path in tracked:
            data = _read_bytes_or_none(rel_path)
            if data is None:  # file gone from the working tree — nothing to scan
                continue
            scanned += 1
            for name, pattern in _SECRET_PATTERNS:
                if pattern.search(data):
                    offenders.append(f"{rel_path}: {name}")
        assert scanned, "no tracked files were scanned; guard is vacuous"
        assert not offenders, (
            "secret-shaped strings found in tracked files: " + "; ".join(offenders)
        )

    def test_secret_files_are_not_tracked(self):
        offenders = []
        for rel_path in _tracked_files():
            base = PurePosixPath(rel_path).name
            if (
                base == "secrets.toml"
                or base == ".env"
                or (base.startswith(".env.") and base not in _ALLOWED_ENV_TEMPLATES)
            ):
                offenders.append(rel_path)
        assert not offenders, f"secret-bearing files must never be tracked: {offenders}"


class TestSecretDetector:
    """The scan is only as good as its patterns, so prove they fire (and don't
    fire on lookalikes) — a broken or untested regex must not pass vacuously."""

    def test_every_pattern_has_a_firing_sample(self):
        # Pinned per-pattern (not any()-across-all): every pattern must match its
        # OWN sample, so a newly added pattern with a typo or bad quantifier fails
        # here instead of silently never catching its secret.
        pattern_names = {name for name, _ in _SECRET_PATTERNS}
        assert set(_DETECTOR_SAMPLES) == pattern_names, (
            "keep _DETECTOR_SAMPLES in sync with _SECRET_PATTERNS"
        )
        for name, pattern in _SECRET_PATTERNS:
            assert pattern.search(_DETECTOR_SAMPLES[name]), (
                f"pattern {name!r} does not match its own sample"
            )

    def test_patterns_ignore_benign_lookalikes(self):
        # Strings that legitimately live in the repo (doc URLs, the max_tokens
        # kwarg, a test name) must not trip the guard, or it cries wolf.
        benign = (
            b"https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
            b"https://github.com/darylalim/flux-image",
            b"max_tokens=256",
            b"test_extracts_and_strips_output",
        )
        for text in benign:
            assert not any(rx.search(text) for _, rx in _SECRET_PATTERNS), text
