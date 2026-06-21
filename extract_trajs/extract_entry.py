#!/usr/bin/env bash
# Remote-side ONLY: read raw mini-swe traj trees and write evaluation *input* files.
# Does NOT run Pass@1, ContextBench metrics, pattern metrics, or final report.
#
# Outputs (under ContextBench repo root):
#   preds_merged/<model>/preds_*.jsonl, preds_select500_merged.jsonl, build_preds_manifest.json
#   miniswe_preds/<model-slug>/*_patches_from_miniswe.*
#
# Copy preds_merged/<model>/ and miniswe_preds/<model-slug>/ to the eval machine
# (rsync/scp). Optional --pack if tools/pack_eval_inputs_bundle.sh exists.
#
# Traj layout (same as contextbench.run / run_contextbench_miniswe.sh -o):
#   <MINISWE_OUTPUT_ROOT>/<output-run>/miniswe/{Verified,Pro,Poly,Multi}/<id>/<id>.traj.json
#
# Usage:
#   ./extract_trajs/extract_entry.sh \
#     --model codestral-22b \
#     --output-run output_vllm/select_500_codestral_22b
#
#   ./extract_trajs/extract_entry.sh \
#     --model codestral-22b \
#     --traj-root agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/output_vllm/select_500_codestral_22b/miniswe
#
set -euo pipefail

EXTRACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXTRACT_DIR/.." && pwd)"
# Must match contextbench.run MINISWE_OUTPUT_ROOT
MINISWE_OUTPUT_ROOT="$REPO_ROOT/agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src"

cd "$REPO_ROOT"

MODEL=""
TRAJ_ROOT=""
OUTPUT_RUN=""
PACK=0
PYTHON="${PYTHON:-}"

if [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
  PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python3}"
else
  PYTHON="${PYTHON:-python3}"
fi

usage() {
  cat <<'EOF'
Extract evaluation input files from raw mini-swe traj (remote machine).

  --model NAME           e.g. codestral-22b  → preds_merged/<model>/
  --output-run PATH      Same -o as run_contextbench_miniswe.sh (relative to mini-swe src/)
                         Resolves traj-root to: <src>/<path>/miniswe
  --traj-root PATH       Explicit miniswe root (Verified/ Pro/ Poly/ Multi/ underneath)
  --pack                 Optional tarball if tools/pack_eval_inputs_bundle.sh exists

Provide exactly one of --output-run or --traj-root.

Requires: ContextBench + Python (contextbench.process_trajectories for preds JSONL).
Does NOT run official harness eval (--run-eval is not passed to miniswe_eval.py).
EOF
  exit "${1:-0}"
}

resolve_traj_root() {
  if [[ -n "$TRAJ_ROOT" ]]; then
    if [[ "$TRAJ_ROOT" != /* ]]; then
      TRAJ_ROOT="$REPO_ROOT/$TRAJ_ROOT"
    fi
    TRAJ_ROOT="$(cd "$TRAJ_ROOT" && pwd)"
    return 0
  fi
  if [[ -z "$OUTPUT_RUN" ]]; then
    return 1
  fi
  local base="$MINISWE_OUTPUT_ROOT"
  if [[ "$OUTPUT_RUN" = /* ]]; then
    TRAJ_ROOT="$(cd "$OUTPUT_RUN/miniswe" && pwd)"
  else
    TRAJ_ROOT="$(cd "$base/$OUTPUT_RUN/miniswe" && pwd)"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --traj-root) TRAJ_ROOT="$2"; shift 2 ;;
    --output-run) OUTPUT_RUN="$2"; shift 2 ;;
    --pack) PACK=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "[ERROR] Unknown: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "[ERROR] --model required" >&2; exit 2; }

if [[ -n "$TRAJ_ROOT" && -n "$OUTPUT_RUN" ]]; then
  echo "[ERROR] Use only one of --traj-root or --output-run" >&2
  exit 2
fi
if [[ -z "$TRAJ_ROOT" && -z "$OUTPUT_RUN" ]]; then
  echo "[ERROR] --traj-root or --output-run required" >&2
  usage 1
fi

if ! resolve_traj_root; then
  echo "[ERROR] Could not resolve traj root (need --traj-root or --output-run)" >&2
  exit 2
fi
[[ -d "$TRAJ_ROOT" ]] || {
  echo "[ERROR] traj root not found: $TRAJ_ROOT" >&2
  echo "  Hint: run miniswe first, e.g. tools/run_contextbench_miniswe.sh -o <output-run> ..." >&2
  exit 2
}

MODEL_SLUG="${MODEL//./-}"
PREDS_DIR="$REPO_ROOT/preds_merged/$MODEL"
PATCH_DIR="$REPO_ROOT/miniswe_preds/$MODEL_SLUG"
BUILD_PREDS="$EXTRACT_DIR/build_preds_per_bench.py"
MINISWE_EVAL="$EXTRACT_DIR/miniswe_eval.py"

for f in "$BUILD_PREDS" "$MINISWE_EVAL"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing: $f" >&2; exit 2; }
done

echo "[extract] repo=$REPO_ROOT"
echo "[extract] traj-root=$TRAJ_ROOT"
echo "[extract] model=$MODEL (slug=$MODEL_SLUG)"

echo "[1/2] traj → preds JSONL ($BUILD_PREDS)"
"$PYTHON" "$BUILD_PREDS" \
  --traj-root "$TRAJ_ROOT" \
  --output-dir "$PREDS_DIR" \
  --merge

echo "[2/2] traj → patch files ($MINISWE_EVAL, extract only)"
mkdir -p "$PATCH_DIR"
"$PYTHON" "$MINISWE_EVAL" \
  --traj-root "$TRAJ_ROOT" \
  --benchmark multi poly pro verified \
  --output-dir "$PATCH_DIR"

echo ""
echo "[OK] Extracted inputs for model=$MODEL"
echo "  preds:  $PREDS_DIR"
echo "  patch:  $PATCH_DIR"
ls -la "$PREDS_DIR"/preds_*.jsonl 2>/dev/null || true
ls -la "$PATCH_DIR"/*_patches* 2>/dev/null || true

if [[ "$PACK" -eq 1 ]]; then
  PACK_SCRIPT="$REPO_ROOT/tools/pack_eval_inputs_bundle.sh"
  if [[ -x "$PACK_SCRIPT" ]]; then
    exec "$PACK_SCRIPT" --model "$MODEL"
  else
    echo "[WARN] --pack requested but not found: $PACK_SCRIPT (copy dirs manually)" >&2
  fi
fi
