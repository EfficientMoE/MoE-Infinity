#!/usr/bin/env bash
# leak_scan.sh — scan staged (or given) files for credential-shaped strings
# before any commit/push. Exit 1 on hit. Usage:
#   ./scripts/leak_scan.sh --staged          # scan `git diff --cached` paths
#   ./scripts/leak_scan.sh FILE [FILE...]    # scan explicit files
# NOTE: pattern deliberately avoids literal single-quote chars (BSD/GNU
# quoting portability); [[:punct:]] covers them where relevant.
set -u
PAT='(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [A-Za-z0-9._-]{20,}|api[_-]?key[[:space:]]*[[:punct:]=:]+[A-Za-z0-9_-]{16,}|password[[:space:]]*[[:punct:]=:]+[^[:space:]]{8,}|postgres(ql)?://[^[:space:]@]+@)'
if [ "${1:-}" = "--staged" ]; then
  shift
  files=$(git diff --cached --name-only -- .)
  [ -z "$files" ] && { echo "leak_scan: nothing staged"; exit 0; }
  # shellcheck disable=SC2086
  set -- $files
fi
[ $# -eq 0 ] && { echo "usage: leak_scan.sh --staged | FILE..." >&2; exit 2; }
rc=0
for f in "$@"; do
  [ -f "$f" ] || continue
  # drop own pattern-def lines to avoid self-match (line numbers relative to filtered view)
  if sed "/^[[:space:]]*PAT=/d" "$f" | grep -nEi "$PAT"; then
    echo "leak_scan: POTENTIAL LEAK in $f (see match above)" >&2
    rc=1
  fi
done
[ $rc -eq 0 ] && echo "leak_scan: CLEAN ($# files)"
exit $rc
