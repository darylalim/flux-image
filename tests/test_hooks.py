"""Tests for the Claude Code hooks under .claude/.

The hooks are plain bash scripts wired up in ``.claude/settings.json``, so this
suite drives them the way Claude Code does: the tool payload on stdin, asserting
on the exit code and — for the guard — the JSON decision on stdout. A stubbed
``uv`` on ``PATH`` locks which toolchain commands each hook would run, and in
what order, without paying for a real ruff/ty/pytest run.

Two rules keep this file from growing back into a transcription of the scripts:

  * A test must be able to fail for a reason the script's own source does not
    already make obvious. Reading a ``case`` statement back as one parametrized
    case per arm moves in lockstep with the thing it mirrors and cannot detect
    drift. Prefer external truth (real ruff, real git), cross-file locks, or
    invariants no single script can express.
  * Rationale lives in the hook script's header, not duplicated here. A test
    comment earns its place only by naming the *failure* the test catches.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HOOKS_DIR = _REPO_ROOT / ".claude" / "hooks"
_SETTINGS = _HOOKS_DIR.parent / "settings.json"
_HOOK_PREFIX = "${CLAUDE_PROJECT_DIR}/.claude/hooks/"

_requires_shell = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="hook scripts require bash and jq",
)
_requires_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="mark-tests-pending.sh derives its filter from git",
)
_requires_uv = pytest.mark.skipif(
    shutil.which("uv") is None, reason="needs the project's ruff"
)

_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["x  =  1\n"],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _exec(script, stdin, *, project_dir, env=None):
    child_env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)}
    if env:
        child_env.update(env)
    return subprocess.run(
        ["bash", str(_HOOKS_DIR / script)],
        input=stdin,
        capture_output=True,
        text=True,
        env=child_env,
        timeout=60,
    )


def _run_hook(script, file_path, *, project_dir, env=None):
    """Invoke a tool-event hook the way Claude Code does: payload on stdin."""
    payload = json.dumps(
        {"tool_name": "Edit", "tool_input": {"file_path": str(file_path)}}
    )
    return _exec(script, payload, project_dir=project_dir, env=env)


def _run_bash_hook(script, command, *, project_dir, env=None):
    """Invoke a Bash-matched PreToolUse hook: the command on stdin."""
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return _exec(script, payload, project_dir=project_dir, env=env)


def _run_stop_hook(script, *, project_dir, env=None):
    """Invoke a Stop hook: no tool payload (Stop receives none)."""
    return _exec(script, "", project_dir=project_dir, env=env)


def _decision(result):
    return json.loads(result.stdout)["hookSpecificOutput"]


def _uv_stub_env(tmp_path, *, exit_code=0, fail_on=None):
    """A fake ``uv`` on ``PATH`` that logs its args and exits ``exit_code``.

    ``fail_on`` fails only the matching subcommand: run-tests.sh chains several
    gates, so a blanket non-zero would always trip the first and leave the rest
    untestable. Returns ``(env, log_path)``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = bin_dir / "uv.log"
    log.write_text("")
    stub = bin_dir / "uv"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_UV_LOG"\n'
        'if [[ -n "${STUB_UV_FAIL_ON:-}" && "$*" == *"$STUB_UV_FAIL_ON"* ]]; then\n'
        "  exit 1\n"
        "fi\n"
        'exit "${STUB_UV_EXIT:-0}"\n'
    )
    stub.chmod(0o755)
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "STUB_UV_LOG": str(log),
        "STUB_UV_EXIT": str(exit_code),
        "STUB_UV_FAIL_ON": fail_on or "",
    }, log


def _no_jq_env(tmp_path):
    """A PATH with bash/cat/basename but NOT jq, for the degraded-guard path."""
    bin_dir = tmp_path / "nojq-bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in ("bash", "cat", "basename"):
        src = shutil.which(tool)
        if src and not (bin_dir / tool).exists():
            (bin_dir / tool).symlink_to(src)
    return {"PATH": str(bin_dir)}


def _real_ruff(*args):
    """Run the project's actual ruff (never the stub), always cache-free."""
    return subprocess.run(
        ["uv", "run", "ruff", args[0], "--no-cache", *args[1:]],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _project(tmp_path):
    """A throwaway project dir with a .claude/ subdir (for the marker)."""
    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True, exist_ok=True)
    return proj


def _git_project(tmp_path):
    """A throwaway project that is a real git repo with the repo's own ignores.

    mark-tests-pending.sh asks ``git check-ignore`` rather than consulting a
    hardcoded list, so its tests must ask real git too — that externality is the
    whole point of the rewrite.
    """
    proj = _project(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=proj, check=True, capture_output=True, timeout=30
    )
    (proj / ".gitignore").write_text(
        ".venv/\n__pycache__/\n.env\n.env.*\n!.env.example\nsecrets.toml\n"
        ".claude/settings.local.json\n.claude/.tests-pending\n"
    )
    return proj


def _commands_by_event():
    """{event: {script basename, ...}} across settings.json."""
    settings = json.loads(_SETTINGS.read_text())
    return {
        event: {Path(h["command"]).name for g in groups for h in g["hooks"]}
        for event, groups in settings["hooks"].items()
    }


def _all_commands():
    settings = json.loads(_SETTINGS.read_text())
    return [
        h["command"]
        for groups in settings["hooks"].values()
        for g in groups
        for h in g["hooks"]
    ]


def _matchers_for(event, scripts):
    """Matchers of the ``event`` groups that run any of ``scripts``."""
    return [
        g.get("matcher")
        for g in json.loads(_SETTINGS.read_text())["hooks"][event]
        if {Path(h["command"]).name for h in g["hooks"]} & scripts
    ]


class TestHookWiring:
    """settings.json must reference real, executable scripts, bound to events."""

    def test_scripts_are_bound_to_the_correct_events(self):
        # The one assertion that can catch a hook *disappearing*. Every other
        # test iterates whatever settings.json lists, so a removed entry is
        # invisible to them: delete the "Stop" block and the rest of this file
        # stays green while ruff, ty and pytest go silently dead. Not the
        # transcription this file avoids elsewhere — that mirrored a list inside
        # the script under test; this pins config against the scripts on disk.
        by_event = _commands_by_event()
        assert by_event["PreToolUse"] == {"guard-paths.sh", "guard-cache.sh"}
        assert by_event["PostToolUse"] == {"ruff-format.sh", "mark-tests-pending.sh"}
        assert by_event["Stop"] == {"run-tests.sh"}

    def test_commands_use_the_project_dir_prefix(self):
        # A ${CLAUDE_PROJ_DIR} typo disables the whole system with no error.
        for cmd in _all_commands():
            assert cmd.startswith(_HOOK_PREFIX), cmd

    def test_referenced_scripts_exist_and_are_executable(self):
        # A lost +x bit on a fresh clone silently disables a hook.
        commands = _all_commands()
        assert commands, "no hook commands configured"
        for cmd in commands:
            script = _HOOKS_DIR / Path(cmd).name
            assert script.exists(), f"{cmd} -> missing {script}"
            assert os.access(script, os.X_OK), f"{script} is not executable"

    @pytest.mark.parametrize(
        ("event", "scripts"),
        [
            ("PreToolUse", {"guard-paths.sh"}),
            ("PostToolUse", {"ruff-format.sh", "mark-tests-pending.sh"}),
        ],
    )
    def test_tool_matchers_cover_edit_write_multiedit(self, event, scripts):
        # A dropped MultiEdit would silently disable the hooks on that tool.
        matchers = _matchers_for(event, scripts)
        assert matchers, f"{event} has no matcher groups for {scripts}"
        assert all(
            {"Edit", "Write", "MultiEdit"} <= set(m.split("|")) for m in matchers
        ), matchers

    def test_cache_guard_matches_bash(self):
        # The cache-wiping commands only ever arrive through the Bash tool.
        matchers = _matchers_for("PreToolUse", {"guard-cache.sh"})
        assert matchers
        assert all("Bash" in m.split("|") for m in matchers), matchers

    def test_no_whole_project_gate_runs_per_edit(self):
        # PostToolUse fires once per edit, so a whole-project command there
        # re-derives one fact N times a turn and surfaces a transient failure on
        # every intermediate edit of a multi-file refactor. Greps what the wired
        # scripts actually run, so it can fail for a reason the config does not show.
        for cmd in _commands_by_event()["PostToolUse"]:
            body = "\n".join(
                line
                for line in (_HOOKS_DIR / cmd).read_text().splitlines()
                if not line.lstrip().startswith("#")
            )
            for needle in ("pytest", "ty check", "ruff check --fix .", "ruff format ."):
                assert needle not in body, f"{cmd} runs '{needle}' per edit"


@_requires_shell
class TestGuardPaths:
    """PreToolUse guard: deny writes to protected files, allow everything else.

    One case per *case-statement arm* — `.env` (literal), `.env.*` (glob), and
    the literal names — not one per filename.
    """

    @pytest.mark.parametrize("name", [".env", ".env.local", "uv.lock", "secrets.toml"])
    def test_protected_files_are_denied(self, name, tmp_path):
        result = _run_hook("guard-paths.sh", tmp_path / name, project_dir=tmp_path)
        assert result.returncode == 0  # the decision rides in the JSON
        decision = _decision(result)
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert name in decision["permissionDecisionReason"]

    def test_matching_is_case_insensitive(self, tmp_path):
        # macOS's filesystem is case-insensitive, so UV.LOCK opens the real file.
        result = _run_hook("guard-paths.sh", tmp_path / "UV.LOCK", project_dir=tmp_path)
        assert _decision(result)["permissionDecision"] == "deny"

    def test_secret_free_template_is_allowed(self, tmp_path):
        # The template arm must be reached before the .env.* deny arm.
        result = _run_hook(
            "guard-paths.sh", tmp_path / ".env.example", project_dir=tmp_path
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_ordinary_file_is_allowed(self, tmp_path):
        # "notes.env" proves only true dotenv files are blocked, not any *.env.
        result = _run_hook(
            "guard-paths.sh", tmp_path / "notes.env", project_dir=tmp_path
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_denies_protected_paths_outside_the_project(self, tmp_path):
        # Deliberate: this is the only hook with no ${CLAUDE_PROJECT_DIR}
        # containment check. An out-of-repo .env is still a secrets file git
        # cannot restore, and adding containment could only remove protection.
        result = _run_hook(
            "guard-paths.sh", "/somewhere/else/.env", project_dir=tmp_path / "proj"
        )
        assert _decision(result)["permissionDecision"] == "deny"

    @pytest.mark.parametrize("name", [".env", "uv.lock"])
    def test_still_denies_protected_paths_without_jq(self, name, tmp_path):
        # Degraded but not disabled: the path is extracted with bash string ops.
        result = _run_hook(
            "guard-paths.sh",
            tmp_path / name,
            project_dir=tmp_path,
            env=_no_jq_env(tmp_path),
        )
        assert _decision(result)["permissionDecision"] == "deny"

    def test_missing_jq_does_not_block_ordinary_files(self, tmp_path):
        # It used to deny EVERY Edit/Write/MultiEdit without jq — a protection
        # surface of four filenames with a blast radius of 100% of edits.
        result = _run_hook(
            "guard-paths.sh",
            tmp_path / "streamlit_app.py",
            project_dir=tmp_path,
            env=_no_jq_env(tmp_path),
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_unreadable_payload_without_jq_is_denied(self, tmp_path):
        # The surviving fail-closed case, and an inverted default worth pinning:
        # no path means a protected write cannot be ruled out, so deny. The
        # natural refactor (`command -v jq || exit 0`) flips this to permit.
        result = _exec(
            "guard-paths.sh",
            "not json at all",
            project_dir=tmp_path,
            env=_no_jq_env(tmp_path),
        )
        assert _decision(result)["permissionDecision"] == "deny"
        assert "jq" in _decision(result)["permissionDecisionReason"]


_COMMIT_MENTIONING_THE_RULES = (
    "git commit -m \"$(cat <<'EOF'\n"
    "harden(hooks): deny hf cache prune; deny uv cache clean\n"
    "\n"
    "rm -rf ~/.cache/huggingface is denied too.\n"
    "EOF\n"
    ')"'
)


@_requires_shell
class TestGuardCache:
    """PreToolUse guard on Bash: the Disk plan's "never" commands.

    The deny and allow lists are the Disk plan's (docs/plans/flux3-upgrade.md),
    not the script's: each case is a command a session would plausibly type,
    and the interesting ones are the wrappers and spellings a naive substring
    or first-word match would get wrong in one direction or the other.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "hf cache prune",
            "hf cache prune -y",
            "hf cache prune --no-dry-run",
            "hf cache prune --dry-run --no-dry-run",
            "uv run hf cache prune",
            "uvx --from huggingface_hub hf cache prune",
            "HF_HUB_OFFLINE=1 hf cache prune",
            "cd /tmp && hf cache prune",
            "uv cache clean",
            "uv cache clean mlx",
            "uv --no-progress cache clean",
            "sudo uv cache clean",
            "rm -rf ~/.cache/huggingface",
            'rm -r "$HOME/.cache/huggingface/hub/models--mlx-community--x"',
            "rm -Rf $HF_HOME/hub",
            "rm --recursive ~/.cache/uv",
            "rm -rf /some/where/hub/blobs/abc",
            "find ~/.cache/huggingface -name '*.incomplete' -delete",
        ],
    )
    def test_cache_wiping_commands_are_denied(self, command, tmp_path):
        result = _run_bash_hook("guard-cache.sh", command, project_dir=tmp_path)
        assert result.returncode == 0  # the decision rides in the JSON
        decision = _decision(result)
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert "Disk plan" in decision["permissionDecisionReason"]

    @pytest.mark.parametrize(
        "command",
        [
            "hf cache ls",
            "hf cache rm mlx-community/flux2-klein-4b-8bit",
            "hf cache prune --dry-run",
            "uv run hf cache prune --dry-run",
            "hf cache prune --help",
            "uv cache prune",
            "uv cache dir",
            "du -sh ~/.cache/huggingface",
            "df -h ~",
            "find ~/.cache/huggingface -name '*.incomplete'",
            "rm ~/.cache/huggingface/stored_tokens",
            "rm -rf build/",
            # Text, not commands: the phrase sits inside one argument of git/echo.
            'git commit -m "harden: deny hf cache prune; uv cache clean"',
            _COMMIT_MENTIONING_THE_RULES,
            'echo "rm -rf ~/.cache/huggingface"',
        ],
    )
    def test_prescribed_and_unrelated_commands_are_allowed(self, command, tmp_path):
        result = _run_bash_hook("guard-cache.sh", command, project_dir=tmp_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    @pytest.mark.parametrize(
        "command", ["HF Cache Prune", "rm -rf ~/.Cache/HuggingFace/hub"]
    )
    def test_matching_is_case_insensitive(self, command, tmp_path):
        # On macOS's case-insensitive filesystem both of these hit the real thing.
        result = _run_bash_hook("guard-cache.sh", command, project_dir=tmp_path)
        assert _decision(result)["permissionDecision"] == "deny"

    def test_missing_jq_fails_closed_and_names_the_fix(self, tmp_path):
        # Without jq the raw payload is matched anywhere, so even a commit
        # message that only mentions the phrase is denied; the reason must say
        # how to get out of that state.
        result = _run_bash_hook(
            "guard-cache.sh",
            _COMMIT_MENTIONING_THE_RULES,
            project_dir=tmp_path,
            env=_no_jq_env(tmp_path),
        )
        decision = _decision(result)
        assert decision["permissionDecision"] == "deny"
        assert "install jq" in decision["permissionDecisionReason"]

    def test_missing_jq_still_allows_ordinary_commands(self, tmp_path):
        result = _run_bash_hook(
            "guard-cache.sh",
            "uv run pytest",
            project_dir=tmp_path,
            env=_no_jq_env(tmp_path),
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""


@_requires_shell
class TestRuffFormatHook:
    """PostToolUse: format the edited in-project file, else no-op."""

    @pytest.mark.parametrize("name", ["mod.py", "doc.md", "nb.ipynb"])
    def test_formats_scoped_to_that_file_without_lint_fixing(self, name, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        target = proj / name
        target.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", target, project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        assert "ruff format" in logged
        assert str(target) in logged  # scoped to the edited file, not the repo
        # `--fix` deletes code and would race a multi-edit refactor; it lives in
        # the Stop hook, which formats again after it.
        assert "ruff check" not in logged

    @pytest.mark.parametrize(
        "name", ["settings.json", "notes.txt", "README.MD", "NB.IPYNB", "STUB.PYI"]
    )
    def test_skips_what_ruff_would_destroy(self, name, tmp_path):
        # Ruff picks its parser from the extension it is handed and parses
        # anything unrecognized as Python — and JSON is valid Python expression
        # syntax, so a settings.json handed to it is silently rewritten rather
        # than rejected. The mis-cased names are the same hazard reached through
        # macOS's case-insensitive filesystem, where NB.IPYNB opens the notebook.
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        (proj / name).write_text("# hi\n")
        result = _run_hook("ruff-format.sh", proj / name, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""  # uv never invoked

    def test_skips_nonexistent_file(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook(
            "ruff-format.sh", proj / "ghost.py", project_dir=proj, env=env
        )
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_skips_file_outside_the_project(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        outside = tmp_path / "outside.py"
        outside.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", outside, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_is_non_blocking(self, tmp_path):
        # The deliberate opposite of the Stop hook's exit 2.
        proj = _project(tmp_path)
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        (proj / "mod.py").write_text("x=1\n")
        result = _run_hook("ruff-format.sh", proj / "mod.py", project_dir=proj, env=env)
        assert result.returncode == 0


@_requires_uv
class TestRuffHandlesTheHookedExtensions:
    """The real ruff must behave as ruff-format.sh and run-tests.sh assume.

    Every other test here drives a stubbed ``uv``: they lock *that* ruff is
    invoked, not what it does once it is. If a future ruff dropped Markdown or
    notebook formatting, those would stay green while the hook silently did
    nothing and CI's ``ruff format --check .`` diverged from it all over again.
    The only tests in this file exercising something the repo does not control.
    """

    @pytest.mark.parametrize(
        ("name", "source", "expected"),
        [
            # Markdown is why the hook was widened: ruff >=0.16 formats Python
            # fenced inside it, and CI's `ruff format --check .` walks the root.
            ("doc.md", "# t\n\n```python\nx  =  1\n```\n", "x = 1"),
            ("mod.py", "x  =  1\n", "x = 1"),
            ("stub.pyi", "def f(x:int)->int: ...\n", "def f(x: int) -> int: ..."),
        ],
    )
    def test_ruff_reformats(self, name, source, expected, tmp_path):
        target = tmp_path / name
        target.write_text(source)
        _real_ruff("format", str(target))
        assert expected in target.read_text()

    def test_ruff_reformats_notebook_cells_in_place(self, tmp_path):
        # Assert the envelope still parses, not just the cell text: ruff handed
        # a notebook under an unrecognized extension rewrites the JSON as a
        # Python dict literal, leaving the substring present but the file dead.
        nb = tmp_path / "nb.ipynb"
        nb.write_text(json.dumps(_NOTEBOOK))
        _real_ruff("format", str(nb))
        loaded = json.loads(nb.read_text())  # still a notebook, not Python
        assert "".join(loaded["cells"][0]["source"]).strip() == "x = 1"

    def test_lint_fix_must_run_before_format(self, tmp_path):
        # The verified bug the ruff split exists to prevent. `ruff check --fix`
        # removing an unused import strands the blank line that followed it, and
        # `ruff format --check` — CI's gate — then rejects the file. Formatter
        # first is the red build the old PostToolUse hook caused silently.
        target = tmp_path / "mod.py"
        source = "import contextlib\n\nx = 1\n"
        fix = ("check", "--fix", "--select", "F401", str(target))
        fmt = ("format", str(target))

        def rejected_by_ci():
            return _real_ruff("format", "--check", str(target)).returncode != 0

        target.write_text(source)
        _real_ruff(*fmt)
        _real_ruff(*fix)
        assert rejected_by_ci(), (
            "format-then-fix no longer strands a blank line; the Stop hook's "
            "ordering comment and this guard can be revisited"
        )

        target.write_text(source)
        _real_ruff(*fix)
        _real_ruff(*fmt)
        assert not rejected_by_ci()


@_requires_shell
@_requires_git
class TestMarkTestsPendingHook:
    """PostToolUse: arm the end-of-turn marker, per real ``git check-ignore``."""

    def _marker(self, proj):
        return proj / ".claude" / ".tests-pending"

    @pytest.mark.parametrize(
        "rel",
        [
            "streamlit_app.py",
            # A tracked path in a directory the old thirteen globs never listed.
            # Verified against the old script: this did not arm. (A new module
            # under tests/ *did*, via `*/tests/*` — the fail-open case was new
            # directories, not new files in known ones.)
            "assets/logo.svg",
            # Tracked, so test_secrets.py scans it; the old list wrongly
            # excluded it on the claim that no test reads CLAUDE.md.
            "CLAUDE.md",
        ],
    )
    def test_arms_for_tracked_paths(self, rel, tmp_path):
        proj = _git_project(tmp_path)
        result = _run_hook("mark-tests-pending.sh", proj / rel, project_dir=proj)
        assert result.returncode == 0
        assert self._marker(proj).exists()

    @pytest.mark.parametrize(
        "rel",
        [
            ".claude/settings.local.json",
            ".env",
            "__pycache__/streamlit_app.cpython-312.pyc",
        ],
    )
    def test_no_marker_for_gitignored_path(self, rel, tmp_path):
        proj = _git_project(tmp_path)
        result = _run_hook("mark-tests-pending.sh", proj / rel, project_dir=proj)
        assert result.returncode == 0
        assert not self._marker(proj).exists()

    def test_arms_when_git_cannot_answer(self, tmp_path):
        # Fail-closed. `git check-ignore` exits 128 outside a repo; an
        # unanswerable question must arm the suite, not skip it. The natural
        # refactor (treat any non-zero as "skip") inverts this silently.
        proj = _project(tmp_path)  # deliberately NOT a git repo
        result = _run_hook(
            "mark-tests-pending.sh", proj / "streamlit_app.py", project_dir=proj
        )
        assert result.returncode == 0
        assert self._marker(proj).exists()

    def test_no_marker_for_git_internals(self, tmp_path):
        proj = _git_project(tmp_path)
        result = _run_hook(
            "mark-tests-pending.sh", proj / ".git" / "config", project_dir=proj
        )
        assert result.returncode == 0
        assert not self._marker(proj).exists()

    def test_no_marker_for_file_outside_project(self, tmp_path):
        proj = _git_project(tmp_path)
        result = _run_hook(
            "mark-tests-pending.sh", tmp_path / "streamlit_app.py", project_dir=proj
        )
        assert result.returncode == 0
        assert not self._marker(proj).exists()


@_requires_shell
class TestRunTestsHook:
    """Stop hook: format every turn; --fix, ty and pytest once, iff pending."""

    def _marker(self, proj):
        return proj / ".claude" / ".tests-pending"

    def test_runs_every_gate_when_pending(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        for gate in ("ruff check --fix .", "ruff format .", "ty check .", "pytest"):
            assert gate in logged, f"{gate} not run"
        assert not self._marker(proj).exists()  # consumed -> loop-safe

    def test_a_format_pass_always_follows_the_lint_fix(self, tmp_path):
        # `--fix` strands a blank line where it removed an unused import, so a
        # format must come AFTER it or CI's `ruff format --check .` rejects the
        # result. `rindex`: an unconditional format also runs before the gate,
        # so the invariant is that one *follows* the fix, not that none precedes.
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        self._marker(proj).write_text("")
        _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        logged = log.read_text()
        assert logged.rindex("ruff format .") > logged.index("ruff check --fix .")

    def test_formats_but_skips_the_expensive_gates_when_not_pending(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        assert "ruff format ." in logged  # layout only: safe unvalidated
        # `--fix` deletes code, and on a turn with no marker ty and pytest never
        # run, so nothing would validate a tree-wide rewrite.
        assert "ruff check --fix" not in logged
        assert "pytest" not in logged
        assert "ty check" not in logged

    def test_ty_failure_blocks_with_exit_2_and_clears_marker(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path, fail_on="ty check")
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 2  # surfaced back to Claude as an error
        # Exit-code-agnostic wording: an env failure isn't mislabeled a type error.
        assert "ty check failed" in result.stderr
        assert "pytest" not in log.read_text()  # fails fast, before the 14s suite
        assert not self._marker(proj).exists()

    def test_pytest_failure_blocks_with_exit_2_and_clears_marker(self, tmp_path):
        proj = _project(tmp_path)
        env, _ = _uv_stub_env(tmp_path, fail_on="pytest")
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 2
        assert "pytest run failed" in result.stderr
        # Cleared even on failure, so a no-edit follow-up turn won't loop.
        assert not self._marker(proj).exists()
