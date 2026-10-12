#!/usr/bin/env bash
set -euo pipefail

# Safely apply a source archive previously uploaded to gpu-remote-179.
# The exclusive sync lock prevents code replacement while cooperative GenRec
# jobs (started through run-locked-179.sh) hold the corresponding shared lock.

REPO_DIR="${GENREC_REMOTE_REPO:-/root/workspace/code/genrec}"
UPLOAD_ROOT="${GENREC_UPLOAD_ROOT:-/root/workspace/uploads}"
LOCK_DIR="${GENREC_LOCK_DIR:-/root/workspace/locks}"
SYNC_LOCK="$LOCK_DIR/genrec-sync.lock"
LAST_APPLIED="$LOCK_DIR/genrec-last-upload"
LAST_APPLIED_SHA="$LOCK_DIR/genrec-last-upload.sha256"
LAST_APPLIED_GIT_SHA="$LOCK_DIR/genrec-last-upload.git-sha"
LOCK_WAIT="${GENREC_SYNC_LOCK_WAIT:-30}"

usage() {
  echo "Usage: bash scripts/apply-upload-179.sh ARCHIVE.tar.gz EXPECTED_SHA256 [--force-stale]" >&2
}

if (($# < 2 || $# > 3)); then
  usage
  exit 2
fi

ARCHIVE="$1"
EXPECTED_SHA="$2"
FORCE_STALE=0
if (($# == 3)); then
  [[ "$3" == "--force-stale" ]] || { usage; exit 2; }
  FORCE_STALE=1
fi

if [[ ! "$EXPECTED_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid SHA256: $EXPECTED_SHA" >&2
  exit 2
fi

ARCHIVE_REAL="$(readlink -f "$ARCHIVE")"
UPLOAD_REAL="$(readlink -f "$UPLOAD_ROOT")"
case "$ARCHIVE_REAL" in
  "$UPLOAD_REAL"/*) ;;
  *) echo "Refusing archive outside $UPLOAD_REAL: $ARCHIVE_REAL" >&2; exit 1 ;;
esac
[[ -f "$ARCHIVE_REAL" ]] || { echo "Archive not found: $ARCHIVE_REAL" >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA="$(sha256sum "$ARCHIVE_REAL" | awk '{print $1}')"
else
  ACTUAL_SHA="$(shasum -a 256 "$ARCHIVE_REAL" | awk '{print $1}')"
fi
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "SHA256 mismatch: expected $EXPECTED_SHA, got $ACTUAL_SHA" >&2
  exit 1
fi

mkdir -p "$LOCK_DIR"
exec {SYNC_FD}>"$SYNC_LOCK"
if ! flock -x -w "$LOCK_WAIT" "$SYNC_FD"; then
  echo "Could not acquire GenRec sync lock within ${LOCK_WAIT}s; a task or another upload owns it" >&2
  exit 75
fi

ARCHIVE_NAME="$(basename "$ARCHIVE_REAL")"
if [[ "$FORCE_STALE" -eq 0 && -s "$LAST_APPLIED" ]]; then
  PREVIOUS="$(cat "$LAST_APPLIED")"
  if [[ "$ARCHIVE_NAME" < "$PREVIOUS" || "$ARCHIVE_NAME" == "$PREVIOUS" ]]; then
    echo "Refusing stale/already-applied bundle: $ARCHIVE_NAME (last: $PREVIOUS)" >&2
    exit 75
  fi
fi

# Catch non-cooperative jobs that did not use run-locked-179.sh.
RUNNING="$(ps -eo pid=,args= | grep -E 'genrec[./]trainers|genrec\.trainers|scripts/run_matrix\.py' | grep -v -E 'grep|apply-upload-179' || true)"
if [[ -n "$RUNNING" ]]; then
  echo "Refusing to replace code while a GenRec job is running:" >&2
  echo "$RUNNING" >&2
  exit 75
fi

STAGE_DIR="$(mktemp -d /root/workspace/genrec-sync.XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

while IFS= read -r member; do
  case "$member" in
    /*|../*|*/../*) echo "Unsafe archive member: $member" >&2; exit 1 ;;
  esac
done < <(tar -tzf "$ARCHIVE_REAL")

tar -xzf "$ARCHIVE_REAL" -C "$STAGE_DIR"
[[ -f "$STAGE_DIR/pyproject.toml" && -d "$STAGE_DIR/genrec" ]] || {
  echo "Archive does not look like a GenRec source bundle" >&2
  exit 1
}

mkdir -p "$REPO_DIR"
rsync -a \
  --exclude='.git/' \
  --exclude='venv/' \
  --exclude='dataset/' \
  --exclude='models_hub/' \
  --exclude='out/' \
  --exclude='wandb/' \
  "$STAGE_DIR/" "$REPO_DIR/"

printf '%s\n' "$ARCHIVE_NAME" > "$LAST_APPLIED"
printf '%s\n' "$ACTUAL_SHA" > "$LAST_APPLIED_SHA"
if [[ "$ARCHIVE_NAME" =~ -([0-9a-f]{7,40})(-dirty)?-[0-9]+\.tar\.gz$ ]]; then
  printf '%s\n' "${BASH_REMATCH[1]}" > "$LAST_APPLIED_GIT_SHA"
else
  rm -f "$LAST_APPLIED_GIT_SHA"
fi
echo "Applied $ARCHIVE_NAME to $REPO_DIR"
echo "SHA256 $ACTUAL_SHA"
