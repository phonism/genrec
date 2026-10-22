#!/usr/bin/env bash
set -euo pipefail

# Run one GenRec command while holding cooperative per-GPU locks and a shared
# source-sync lock. Use this inside every gpu-remote-179 task_submit command.

LOCK_DIR="${GENREC_LOCK_DIR:-/root/workspace/locks}"
SYNC_LOCK="$LOCK_DIR/genrec-sync.lock"
LAST_APPLIED="$LOCK_DIR/genrec-last-upload"
LAST_APPLIED_SHA="$LOCK_DIR/genrec-last-upload.sha256"
LAST_APPLIED_GIT_SHA="$LOCK_DIR/genrec-last-upload.git-sha"
LOCK_WAIT="${GENREC_GPU_LOCK_WAIT:-5}"
GPU_DRAIN_WAIT="${GENREC_GPU_DRAIN_WAIT:-15}"
ALLOW_OCCUPIED="${GENREC_ALLOW_OCCUPIED_GPU:-0}"
HOLD_MAX_MB="${GENREC_HOLD_MAX_MB:-2048}"

if [[ ! "$GPU_DRAIN_WAIT" =~ ^[0-9]+$ ]]; then
  echo "GENREC_GPU_DRAIN_WAIT must be a non-negative integer" >&2
  exit 2
fi

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run-locked-179.sh --gpus GPU[,GPU...]|none --cell CELL_ID -- COMMAND [ARG...]

Example:
  scripts/run-locked-179.sh --gpus 2 --cell sasrec.beauty.loo.s42 -- \
    venv/bin/python -m genrec.trainers.sasrec_trainer config/sasrec/amazon.gin

CPU-only example:
  scripts/run-locked-179.sh --gpus none --cell data.amazon2014.raw.prepare -- \
    venv/bin/python scripts/prepare_amazon2014.py --splits beauty sports toys
EOF
}

if (($# < 6)) || [[ "$1" != "--gpus" || "$3" != "--cell" ]]; then
  usage
  exit 2
fi
GPU_CSV="$2"
CELL_ID="$4"
shift 4
[[ "$1" == "--" ]] || { usage; exit 2; }
shift

if [[ ! "$CELL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$ ]]; then
  echo "Invalid cell ID: $CELL_ID" >&2
  exit 2
fi
CELL_LOCK_NAME="${CELL_ID//:/_}"

NORMALIZED_GPUS=()
if [[ "$GPU_CSV" == "none" ]]; then
  NORMALIZED_CSV="none"
  unset CUDA_VISIBLE_DEVICES
else
  IFS=',' read -r -a REQUESTED_GPUS <<< "$GPU_CSV"
  if ((${#REQUESTED_GPUS[@]} == 0)); then
    echo "At least one GPU is required (or pass --gpus none)" >&2
    exit 2
  fi
  for gpu in "${REQUESTED_GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-7]$ ]] || { echo "Invalid physical GPU index: $gpu" >&2; exit 2; }
    NORMALIZED_GPUS+=("$gpu")
  done
  IFS=$'\n' NORMALIZED_GPUS=($(printf '%s\n' "${NORMALIZED_GPUS[@]}" | sort -n -u))
  unset IFS
  NORMALIZED_CSV="$(IFS=,; echo "${NORMALIZED_GPUS[*]}")"

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$NORMALIZED_CSV" ]]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES does not match --gpus $NORMALIZED_CSV" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$NORMALIZED_CSV"
fi

# Propagate the cooperative reservation identity into results.json.  The cell
# is stable across MCP task retries and is therefore the primary execution
# provenance key; task registries can be joined back to it without guessing
# from an output directory.
export GENREC_CELL_ID="$CELL_ID"
export GENREC_PHYSICAL_GPUS="$NORMALIZED_CSV"

mkdir -p "$LOCK_DIR"

# Shared for the whole run: code uploads require the exclusive counterpart.
exec {SYNC_FD}>"$SYNC_LOCK"
if ! flock -s -w "$LOCK_WAIT" "$SYNC_FD"; then
  echo "Could not acquire GenRec source lock; an upload is being applied" >&2
  exit 75
fi

# Attach the exact uploaded source bundle to every results.json. A git SHA is
# insufficient when the research worktree intentionally contains uncommitted
# changes.
if [[ -r "$LAST_APPLIED" ]]; then
  export GENREC_SOURCE_BUNDLE_NAME="$(sed -n '1p' "$LAST_APPLIED")"
fi
if [[ -r "$LAST_APPLIED_SHA" ]]; then
  source_bundle_sha="$(sed -n '1p' "$LAST_APPLIED_SHA")"
  if [[ "$source_bundle_sha" =~ ^[0-9a-f]{64}$ ]]; then
    export GENREC_SOURCE_BUNDLE_SHA256="$source_bundle_sha"
  fi
fi
if [[ -r "$LAST_APPLIED_GIT_SHA" ]]; then
  source_git_sha="$(sed -n '1p' "$LAST_APPLIED_GIT_SHA")"
  if [[ "$source_git_sha" =~ ^[0-9a-f]{7,40}$ ]]; then
    export GENREC_SOURCE_GIT_SHA="$source_git_sha"
  fi
fi

# Prevent two Codex sessions from running the same experiment cell on
# different GPUs after both observed an apparently empty task list.
CELL_LOCK="$LOCK_DIR/genrec-cell-${CELL_LOCK_NAME}.lock"
exec {CELL_FD}>"$CELL_LOCK"
if ! flock -x -w "$LOCK_WAIT" "$CELL_FD"; then
  echo "Experiment cell $CELL_ID is already running" >&2
  exit 75
fi
printf 'cell=%s pid=%s started=%s cwd=%s command=%q\n' \
  "$CELL_ID" "$$" "$(date -u +%FT%TZ)" "$PWD" "$*" >&"$CELL_FD"

GPU_LOCK_FDS=()
for gpu in "${NORMALIZED_GPUS[@]}"; do
  lock_file="$LOCK_DIR/genrec-gpu-${gpu}.lock"
  exec {gpu_fd}>"$lock_file"
  if ! flock -x -w "$LOCK_WAIT" "$gpu_fd"; then
    echo "GPU $gpu is already reserved by another cooperative task" >&2
    exit 75
  fi
  printf 'pid=%s started=%s cwd=%s command=%q\n' \
    "$$" "$(date -u +%FT%TZ)" "$PWD" "$*" >&"$gpu_fd"
  GPU_LOCK_FDS+=("$gpu_fd")
done

# The OS lock closes the check-vs-submit race among cooperating Codex sessions.
# This second check catches external/non-cooperative CUDA processes. 179's
# growth-llm hold.py workers are intentionally invisible in this container's
# PID namespace (nvidia-smi reports "[Not Found]") and auto-yield to real work;
# ignore only that low-memory signature. Any named or larger process still
# blocks the run. Set GENREC_HOLD_MAX_MB=0 to disable this exception.
#
# nvidia-smi reports host PIDs, so a hold.py PID can collide with an unrelated
# PID in this container and resolve to a real name. A container process whose
# CUDA_VISIBLE_DEVICES excludes this GPU cannot be the occupant; treat that
# collision like "[Not Found]". An unset or unreadable mask still blocks.
pid_masks_out_gpu() {
  local pid="$1" gpu="$2" mask
  mask="$(tr '\0' '\n' 2>/dev/null < "/proc/$pid/environ" \
    | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')" || return 1
  [[ -n "$mask" && ",$mask," != *",$gpu,"* ]]
}

if [[ "$ALLOW_OCCUPIED" != "1" ]]; then
  for gpu in "${NORMALIZED_GPUS[@]}"; do
    drain_remaining="$GPU_DRAIN_WAIT"
    while true; do
      occupants="$(nvidia-smi -i "$gpu" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
      blocking_occupants=""
      ignored_hold=""
      while IFS= read -r occupant; do
        [[ -n "$occupant" ]] || continue
        IFS=',' read -r occupant_pid occupant_name occupant_mb <<< "$occupant"
        occupant_pid="${occupant_pid//[[:space:]]/}"
        occupant_name="${occupant_name#${occupant_name%%[![:space:]]*}}"
        occupant_name="${occupant_name%${occupant_name##*[![:space:]]}}"
        occupant_mb="${occupant_mb//[[:space:]]/}"
        hold_like=0
        if [[ "$occupant_name" == "[Not Found]" ]] \
            || pid_masks_out_gpu "$occupant_pid" "$gpu"; then
          hold_like=1
        fi
        if [[ "$hold_like" == 1 \
              && "$occupant_mb" =~ ^[0-9]+$ \
              && "$HOLD_MAX_MB" =~ ^[0-9]+$ \
              && "$HOLD_MAX_MB" -gt 0 \
              && "$occupant_mb" -le "$HOLD_MAX_MB" ]]; then
          ignored_hold+="${occupant}"$'\n'
        else
          blocking_occupants+="${occupant}"$'\n'
        fi
      done <<< "$occupants"

      if [[ -z "$blocking_occupants" ]]; then
        if [[ -n "$ignored_hold" ]]; then
          echo "GPU $gpu: allowing auto-yield hold.py signature:" >&2
          printf '%s' "$ignored_hold" >&2
        fi
        break
      fi
      if ((drain_remaining <= 0)); then
        echo "GPU $gpu has existing CUDA processes; refusing to share:" >&2
        printf '%s' "$blocking_occupants" >&2
        exit 75
      fi

      drain_delay=2
      if ((drain_remaining < drain_delay)); then
        drain_delay="$drain_remaining"
      fi
      echo "GPU $gpu still has a CUDA process after lock handoff; " \
           "waiting ${drain_delay}s for teardown" >&2
      sleep "$drain_delay"
      drain_remaining=$((drain_remaining - drain_delay))
    done
  done
fi

if [[ "$NORMALIZED_CSV" == "none" ]]; then
  echo "Locked CPU-only cell $CELL_ID for PID $$"
else
  echo "Locked cell $CELL_ID and physical GPU(s) $NORMALIZED_CSV for PID $$"
fi
exec "$@"
