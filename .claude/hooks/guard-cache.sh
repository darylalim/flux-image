#!/usr/bin/env bash
# PreToolUse hook (matcher: Bash): deny the cache-wiping commands that the
# Disk plan in docs/plans/flux3-upgrade.md forbids.
#
#   hf cache prune           (unless --dry-run) — prune deletes every revision
#                            no branch or tag points at, and a model downloaded
#                            by pinned SHA (as this app loads its weights) has
#                            no ref, so whole repos go (66.2 GB in the dry-run
#                            on the owner's Mac, 2026-11-12)
#   uv cache clean           every form; a bare one empties the uv cache for
#                            every project on the machine
#   rm -r / find -delete     on the HF or uv caches, a hub/blobs path or a
#                            models--* folder (removing a models--* folder by
#                            hand orphans blobs shared through the pool)
#
# Allowed: hf cache ls/rm, hf cache prune --dry-run, uv cache prune/dir, du,
# df and find without -delete, which is what the Disk plan prescribes instead.
#
# Matching is on COMMANDS, not text. The command line is split, quote-aware,
# into segments at ; & | ( ) ` and newlines, and a pattern counts only as a
# segment's command word, past any VAR=value, sudo/env/exec-style wrapper,
# shell keyword, `uv run` or `uvx` prefix. So `git commit -m "... hf cache
# prune ..."` is allowed: the phrase sits inside one quoted argument of git.
# Matching is case-insensitive throughout, because on macOS's case-insensitive
# filesystem `HF` finds the hf binary and ~/.Cache/HuggingFace is the real cache.
#
# Missing jq: the command cannot be parsed out of the JSON payload reliably, so
# the raw payload is searched anywhere instead. That over-denies (a commit
# message mentioning the phrase is blocked too), which is the fail-closed
# direction, and the reason names the fix: install jq.
#
# Written for bash 3.2, the bash macOS ships: no ${var,,}, no associative arrays.
#
# Scope note: a backstop, not a security boundary. An alias, a script, `bash
# -c "..."`, `xargs rm`, or Python's shutil.rmtree all bypass it, and so does
# anything run outside Claude Code.
set -uo pipefail

_deny() {
  if command -v jq >/dev/null 2>&1; then
    jq -n --arg r "$1" \
      '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
  else
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$1"
  fi
  exit 0
}

_RULE="see the Disk plan in docs/plans/flux3-upgrade.md"
_PRUNE="Blocked by project hook: 'hf cache prune' deletes every revision no branch or tag points at, which includes every model downloaded by pinned SHA ($_RULE). Use 'hf cache prune --dry-run' to look, then 'hf cache rm <revision>' for what you mean to free."
_CLEAN="Blocked by project hook: 'uv cache clean' empties the uv cache for every project on this machine ($_RULE). Use 'uv cache prune' instead."
_RM="Blocked by project hook: recursive deletion inside the Hugging Face or uv cache ($_RULE). Removing a models--* folder or blobs by hand orphans shared blobs; use 'hf cache rm <repo or revision>'."

shopt -s nocasematch

_is_cache_path() {
  local t=$1
  [[ $t == *".cache/huggingface"* ||
    $t == *".cache/uv" || $t == *".cache/uv/"* ||
    $t == "~/.cache" || $t == "~/.cache/" ||
    $t == */.cache || $t == */.cache/ ||
    $t == *'$hf_home'* || $t == *'${hf_home}'* ||
    $t == *'$hf_hub_cache'* || $t == *'${hf_hub_cache}'* ||
    $t == *'$uv_cache_dir'* || $t == *'${uv_cache_dir}'* ||
    $t == *'$xdg_cache_home'* || $t == *'${xdg_cache_home}'* ||
    $t == *"hub/blobs"* || $t == *"models--"* ]]
}

input=$(cat)

if ! command -v jq >/dev/null 2>&1; then
  _NOJQ=" (This hook could not parse the command without 'jq' and matched the raw text, which can over-deny: install jq, e.g. 'brew install jq'.)"
  if [[ $input == *"hf cache prune"* && $input != *"--dry-run"* ]]; then
    _deny "$_PRUNE$_NOJQ"
  fi
  [[ $input == *"uv cache clean"* ]] && _deny "$_CLEAN$_NOJQ"
  if [[ $input == *"rm "* || $input == *"-delete"* ]] &&
    [[ $input == *".cache/huggingface"* || $input == *".cache/uv"* ||
      $input == *"hf_home"* || $input == *"hub/blobs"* || $input == *"models--"* ]]; then
    _deny "$_RM$_NOJQ"
  fi
  exit 0
fi

cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
[[ -n $cmd ]] || exit 0

# Options of uv / uv run / uvx that take a separate value, so the value is not
# mistaken for the command word.
_takes_value() {
  case "$1" in
    --from | --with | --with-editable | --with-requirements | --python | -p | \
      --project | --directory | --package | --env-file | --cache-dir | \
      --config-file | --index | --default-index | --extra | --group | -w)
      return 0
      ;;
  esac
  return 1
}

TOK=()

# Check the segment held in TOK.
_check_segment() {
  local n=${#TOK[@]} s=0 t w j
  ((n)) || return 0
  while ((s < n)); do
    t=${TOK[s]}
    # Leading VAR=value assignments.
    if [[ $t =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      ((s++))
      continue
    fi
    case "$t" in
      sudo | env | command | exec | nohup | time | builtin | nice | \
        "{" | "!" | then | do | else | elif | if | while | until)
        ((s++))
        while ((s < n)) && [[ ${TOK[s]} == -* ]]; do ((s++)); done
        continue
        ;;
    esac
    w=${t##*/} # /usr/bin/rm, .venv/bin/hf
    if [[ $w == uv ]]; then
      j=$((s + 1))
      while ((j < n)) && [[ ${TOK[j]} == -* ]]; do
        _takes_value "${TOK[j]}" && ((j++))
        ((j++))
      done
      if [[ ${TOK[j]:-} == run ]]; then
        ((j++))
        while ((j < n)) && [[ ${TOK[j]} == -* ]]; do
          _takes_value "${TOK[j]}" && ((j++))
          ((j++))
        done
        s=$j
        continue
      fi
      if [[ ${TOK[j]:-} == cache && ${TOK[j + 1]:-} == clean ]]; then
        _deny "$_CLEAN"
      fi
      return 0
    fi
    if [[ $w == uvx ]]; then
      j=$((s + 1))
      while ((j < n)) && [[ ${TOK[j]} == -* ]]; do
        _takes_value "${TOK[j]}" && ((j++))
        ((j++))
      done
      s=$j
      continue
    fi
    break
  done
  ((s < n)) || return 0
  w=${TOK[s]##*/}

  case "$w" in
    hf)
      if [[ ${TOK[s + 1]:-} == cache && ${TOK[s + 2]:-} == prune ]]; then
        # --dry-run / --no-dry-run is a click flag pair: the last one wins.
        local dry=0
        for ((j = s + 3; j < n; j++)); do
          case "${TOK[j]}" in
            --help | -h) return 0 ;;
            --dry-run) dry=1 ;;
            --no-dry-run) dry=0 ;;
          esac
        done
        ((dry)) || _deny "$_PRUNE"
      fi
      ;;
    rm)
      local recursive=0 opts=1 hit=0
      for ((j = s + 1; j < n; j++)); do
        t=${TOK[j]}
        if ((opts)) && [[ $t == -- ]]; then
          opts=0
        elif ((opts)) && [[ $t == --recursive ]]; then
          recursive=1
        elif ((opts)) && [[ $t == -[!-]* ]]; then
          [[ $t == *r* ]] && recursive=1
        elif _is_cache_path "$t"; then
          hit=1
        fi
      done
      ((recursive && hit)) && _deny "$_RM"
      ;;
    find)
      local del=0 hit=0
      for ((j = s + 1; j < n; j++)); do
        t=${TOK[j]}
        [[ $t == -delete ]] && del=1
        _is_cache_path "$t" && hit=1
      done
      ((del && hit)) && _deny "$_RM"
      ;;
  esac
  return 0
}

# Quote-aware split into segments of tokens. Quotes are removed from tokens and
# separators inside quotes are literal, which is what keeps a commit message
# from being read as a command.
tok="" have=0 quote=""
_flush() {
  if ((have)); then TOK+=("$tok"); fi
  tok="" have=0
}
len=${#cmd}
for ((i = 0; i < len; i++)); do
  c=${cmd:i:1}
  if [[ -n $quote ]]; then
    if [[ $c == "$quote" ]]; then
      quote=""
    elif [[ $quote == '"' && $c == '\' ]]; then
      ((i++))
      tok+=${cmd:i:1}
    else
      tok+=$c
    fi
    continue
  fi
  case "$c" in
    "'" | '"')
      quote=$c
      have=1
      ;;
    '\')
      ((i++))
      # A backslash-newline is a line continuation, not a character.
      [[ ${cmd:i:1} == $'\n' ]] || { tok+=${cmd:i:1}; have=1; }
      ;;
    ' ' | $'\t')
      _flush
      ;;
    ';' | '&' | '|' | '(' | ')' | '`' | $'\n')
      _flush
      _check_segment
      TOK=()
      ;;
    *)
      tok+=$c
      have=1
      ;;
  esac
done
_flush
_check_segment
exit 0
