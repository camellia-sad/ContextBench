#!/usr/bin/env bash

set -euo pipefail

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$_SCRIPTS_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$_SCRIPTS_DIR/env_vllm_chat.sh"

MINISWE_OUTPUT_ROOT="$REPO_ROOT/agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src"
TASK_CSV="${TASK_CSV:-data/selected_500_instances.csv}"
# 整实例墙钟上限（秒）；Codestral/长 context 建议 ≥ 10800
DEFAULT_TIMEOUT="${DEFAULT_TIMEOUT:-10800}"
# 单步 LLM 响应上限（秒）。历史 Codestral 全是 StepResponseTimeout@600，故默认提到 1800
DEFAULT_STEP_TIMEOUT="${DEFAULT_STEP_TIMEOUT:-1800}"

usage() {
  cat <<'EOF'
Usage:
  tools/run_contextbench_miniswe.sh --bench <Verified|Pro|Poly|Multi> --output <output_dir> --workers <N> --rerun <true|false> [--instances <id1,id2,...>] [--debug] [--once]

  Default (no --once): retry loop until every instance has info.exit_status=Submitted
  (includes missing traj, timeouts, and other failures). No max retry count; Ctrl+C
  prints remaining non-Submitted instances.

  --once          Single pass only (old behavior; no auto-retry loop).
  --debug         Forward to contextbench.run: print agent subprocess cwd/cmd and live output.

  Timeouts (env override, seconds):
    DEFAULT_TIMEOUT       whole-instance wall clock (default 10800)
    DEFAULT_STEP_TIMEOUT  single LLM step response (default 1800)
    # StepResponseTimeout means one model call exceeded DEFAULT_STEP_TIMEOUT

Examples:
  tools/run_contextbench_miniswe.sh --bench Verified --output output_vllm/select_500_qwen25_32B_coder --workers 2 --rerun true
  tools/run_contextbench_miniswe.sh --bench Pro --output output_vllm/select_500_qwen25_32B_coder --workers 4 --rerun true --once
  tools/run_contextbench_miniswe.sh --bench Verified --output output_vllm/select_500_qwen25_32B_coder --workers 1 --rerun true --instances "astropy__astropy-14995"

  # 覆盖 vLLM 模型名（需与 start_vllm.sh 的 VLLM_MODEL / /v1/models 一致）:
  export MINISWE_VLLM_MODEL=hosted_vllm/Qwen/Qwen3-8B

  # Codestral 更长超时示例:
  export DEFAULT_STEP_TIMEOUT=3600 DEFAULT_TIMEOUT=14400
  export MINISWE_VLLM_MODEL=hosted_vllm/mistralai/Codestral-22B-v0.1
EOF
}

BENCH=""
OUTPUT=""
WORKERS=""
RERUN=""
INSTANCES=""
DEBUG=0
ONCE=0
ROUND=0
_INTERRUPTED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench)
      BENCH="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:-}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:-}"
      shift 2
      ;;
    --rerun)
      RERUN="${2:-}"
      shift 2
      ;;
    --instances)
      INSTANCES="${2:-}"
      shift 2
      ;;
    --debug)
      DEBUG=1
      shift 1
      ;;
    --once)
      ONCE=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run_contextbench_miniswe] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$BENCH" || -z "$OUTPUT" || -z "$WORKERS" || -z "$RERUN" ]]; then
  echo "[run_contextbench_miniswe] Missing required arguments." >&2
  usage
  exit 2
fi

case "$BENCH" in
  Verified|Pro|Poly|Multi)
    ;;
  *)
    echo "[run_contextbench_miniswe] Invalid --bench: $BENCH (expected Verified|Pro|Poly|Multi)" >&2
    exit 2
    ;;
esac

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -le 0 ]]; then
  echo "[run_contextbench_miniswe] --workers must be a positive integer, got: $WORKERS" >&2
  exit 2
fi

case "${RERUN,,}" in
  true|1|yes|y)
    RERUN_FLAG="--rerun"
    ;;
  false|0|no|n)
    RERUN_FLAG=""
    ;;
  *)
    echo "[run_contextbench_miniswe] --rerun must be true/false (or 1/0), got: $RERUN" >&2
    exit 2
    ;;
esac

_resolve_output_abs() {
  if [[ "$OUTPUT" = /* ]]; then
    echo "$OUTPUT"
  else
    echo "$MINISWE_OUTPUT_ROOT/$OUTPUT"
  fi
}

_resolve_traj_root() {
  echo "$(_resolve_output_abs)/miniswe/$BENCH"
}

# Print comma-separated instance ids that are not Submitted (missing traj counts too).
# Honors SCOPE_INSTANCES env (same as initial --instances filter).
_list_not_submitted_ids() {
  TRAJ_ROOT="$(_resolve_traj_root)" \
  REPO_ROOT="$REPO_ROOT" \
  TASK_CSV="$TASK_CSV" \
  BENCH="$BENCH" \
  SCOPE_INSTANCES="${SCOPE_INSTANCES:-$INSTANCES}" \
  python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
csv_path = repo / os.environ["TASK_CSV"]
bench = os.environ["BENCH"]
traj_root = Path(os.environ["TRAJ_ROOT"])
scope_raw = os.environ.get("SCOPE_INSTANCES", "").strip()
scope = {t.strip() for t in scope_raw.split(",") if t.strip()} if scope_raw else None

if not csv_path.is_file():
    raise SystemExit(f"CSV not found: {csv_path}")

# index: dirname -> exit_status (best effort)
status_by_dir: dict[str, str] = {}
if traj_root.is_dir():
    for p in traj_root.rglob("*.traj.json"):
        dname = p.parent.name
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            status = str(data.get("info", {}).get("exit_status", "") or "").strip()
        except Exception:
            status = "<parse_error>"
        prev = status_by_dir.get(dname)
        if prev is None or status == "Submitted":
            status_by_dir[dname] = status

pending: list[str] = []
with csv_path.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if (row.get("bench") or "").strip() != bench:
            continue
        iid = (row.get("instance_id") or "").strip()
        oid = (row.get("original_inst_id") or "").strip()
        run_id = oid or iid
        if not run_id:
            continue
        if scope is not None and iid not in scope and oid not in scope and run_id not in scope:
            continue

        names = {x for x in (iid, oid, run_id) if x}
        st = None
        for n in names:
            if n in status_by_dir:
                st = status_by_dir[n]
                break
        if st != "Submitted":
            pending.append(run_id)

seen: set[str] = set()
out: list[str] = []
for x in pending:
    if x in seen:
        continue
    seen.add(x)
    out.append(x)
print(",".join(out))
PY
}

# Human-readable report for Ctrl+C / final summary.
_report_not_submitted() {
  TRAJ_ROOT="$(_resolve_traj_root)" \
  REPO_ROOT="$REPO_ROOT" \
  TASK_CSV="$TASK_CSV" \
  BENCH="$BENCH" \
  SCOPE_INSTANCES="${SCOPE_INSTANCES:-$INSTANCES}" \
  python3 - <<'PY'
import csv
import json
import os
from collections import Counter
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
csv_path = repo / os.environ["TASK_CSV"]
bench = os.environ["BENCH"]
traj_root = Path(os.environ["TRAJ_ROOT"])
scope_raw = os.environ.get("SCOPE_INSTANCES", "").strip()
scope = {t.strip() for t in scope_raw.split(",") if t.strip()} if scope_raw else None

status_by_dir: dict[str, str] = {}
if traj_root.is_dir():
    for p in traj_root.rglob("*.traj.json"):
        dname = p.parent.name
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            status = str(data.get("info", {}).get("exit_status", "") or "").strip() or "<empty>"
        except Exception:
            status = "<parse_error>"
        prev = status_by_dir.get(dname)
        if prev is None or status == "Submitted":
            status_by_dir[dname] = status

pending: list[tuple[str, str]] = []
with csv_path.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if (row.get("bench") or "").strip() != bench:
            continue
        iid = (row.get("instance_id") or "").strip()
        oid = (row.get("original_inst_id") or "").strip()
        run_id = oid or iid
        if not run_id:
            continue
        if scope is not None and iid not in scope and oid not in scope and run_id not in scope:
            continue
        names = {x for x in (iid, oid, run_id) if x}
        st = None
        for n in names:
            if n in status_by_dir:
                st = status_by_dir[n]
                break
        if st != "Submitted":
            pending.append((run_id, st or "<missing_traj>"))

if not pending:
    print("[run_contextbench_miniswe] All in-scope instances are Submitted.")
    raise SystemExit(0)

ctr = Counter(st for _, st in pending)
print("")
print("=== Not Submitted (in scope) ===")
print(f"Bench: {bench}")
print(f"Traj root: {traj_root}")
print(f"Count: {len(pending)}")
print("By status:")
for st, n in ctr.most_common():
    print(f"  {st}: {n}")
print("")
print("Instance ids (comma-separated, for --instances):")
print(",".join(iid for iid, _ in pending))
PY
}

_on_interrupt() {
  _INTERRUPTED=1
  echo "" >&2
  echo "[run_contextbench_miniswe] Interrupted (round=${ROUND}). Pending instances:" >&2
  _report_not_submitted >&2 || true
  exit 130
}

trap _on_interrupt INT TERM

# Mirror routing rules:
case "$BENCH" in
  Poly)
    export MSWEA_DOCKER_IMAGE_REGISTRY="ghcr.nju.edu.cn"
    ;;
  Multi|Pro|Verified)
    export MSWEA_DOCKER_IMAGE_REGISTRY="fczi514j9ggm7b.xuanyuan.run"
    ;;
esac
export MSWEA_POLY_GHCR_REGISTRY="ghcr.nju.edu.cn"
unset MSWEA_GHCR_MIRROR

if [[ -d /workspace/hf_cache ]]; then
  export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
else
  export HF_HOME="${HF_HOME:-/home/dataset-local/hf_cache}"
fi
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

_preload_miniswe_hf_dataset() {
  local dataset_id split
  case "$BENCH" in
    Verified) dataset_id="princeton-nlp/SWE-bench_Verified"; split="test" ;;
    Pro) dataset_id="ScaleAI/SWE-bench_Pro"; split="test" ;;
    Poly) dataset_id="AmazonScience/SWE-PolyBench"; split="test" ;;
    Multi) dataset_id="ByteDance-Seed/Multi-SWE-bench"; split="train" ;;
    *) return 0 ;;
  esac
  echo "[run_contextbench_miniswe] Preloading ${dataset_id} (split=${split}) → ${HUGGINGFACE_HUB_CACHE}" >&2
  if [[ "$BENCH" == "Multi" ]]; then
    python3 -c "
from datasets import load_dataset
it = iter(load_dataset('${dataset_id}', split='${split}', streaming=True))
next(it)
print('[run_contextbench_miniswe] preload ok (streaming probe):', '${dataset_id}')
" || {
      echo "[run_contextbench_miniswe] Multi preload failed. Try: export HF_ENDPOINT=https://huggingface.co" >&2
      exit 1
    }
    return 0
  fi
  python3 -c "
from datasets import load_dataset
load_dataset('${dataset_id}', split='${split}')
print('[run_contextbench_miniswe] preload ok:', '${dataset_id}')
" || {
    echo "[run_contextbench_miniswe] Preload failed. Try: export HF_ENDPOINT=https://huggingface.co" >&2
    echo "[run_contextbench_miniswe] Or wait and retry mirror; reduce --workers during first download." >&2
    exit 1
  }
}
_preload_miniswe_hf_dataset

export MINISWE_VLLM_MODEL="${MINISWE_VLLM_MODEL:-hosted_vllm/Qwen/Qwen3-8B}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"

# Fixed scope for retry loop (initial --instances, if any).
SCOPE_INSTANCES="$INSTANCES"

_run_contextbench_batch() {
  local batch_instances="$1"
  local -a cmd=(
    python -m contextbench.run
    --agent miniswe
    --task-csv "$TASK_CSV"
    --bench "$BENCH"
    --timeout "$DEFAULT_TIMEOUT"
    --miniswe-step-response-timeout "$DEFAULT_STEP_TIMEOUT"
    -o "$OUTPUT"
    --workers "$WORKERS"
    --rerun
  )

  if [[ -n "$batch_instances" ]]; then
    cmd+=(--instances "$batch_instances")
  fi

  if [[ "$DEBUG" -eq 1 ]]; then
    cmd+=(--debug)
  fi

  echo "[run_contextbench_miniswe] Running: ${cmd[*]}" >&2
  "${cmd[@]}" || true
}

echo "[run_contextbench_miniswe] BENCH=$BENCH"
echo "[run_contextbench_miniswe] OUTPUT=$OUTPUT (abs=$(_resolve_output_abs)) WORKERS=$WORKERS RERUN=$RERUN"
echo "[run_contextbench_miniswe] MODE=$([[ "$ONCE" -eq 1 ]] && echo once || echo until-submitted)"
if [[ -n "$INSTANCES" ]]; then
  echo "[run_contextbench_miniswe] SCOPE_INSTANCES=$INSTANCES"
fi
echo "[run_contextbench_miniswe] DEBUG=$DEBUG"
echo "[run_contextbench_miniswe] MSWEA_DOCKER_IMAGE_REGISTRY=${MSWEA_DOCKER_IMAGE_REGISTRY}"
echo "[run_contextbench_miniswe] MSWEA_POLY_GHCR_REGISTRY=${MSWEA_POLY_GHCR_REGISTRY}"
echo "[run_contextbench_miniswe] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[run_contextbench_miniswe] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[run_contextbench_miniswe] MINISWE_VLLM_MODEL=${MINISWE_VLLM_MODEL} (vLLM serve: ${VLLM_MODEL})"
echo "[run_contextbench_miniswe] timeouts: instance=${DEFAULT_TIMEOUT}s step=${DEFAULT_STEP_TIMEOUT}s"

cd "$REPO_ROOT"

if [[ "$ONCE" -eq 1 ]]; then
  CMD=(
    python -m contextbench.run
    --agent miniswe
    --task-csv "$TASK_CSV"
    --bench "$BENCH"
    --timeout "$DEFAULT_TIMEOUT"
    --miniswe-step-response-timeout "$DEFAULT_STEP_TIMEOUT"
    -o "$OUTPUT"
    --workers "$WORKERS"
  )
  if [[ -n "$RERUN_FLAG" ]]; then
    CMD+=("$RERUN_FLAG")
  fi
  if [[ -n "$INSTANCES" ]]; then
    CMD+=(--instances "$INSTANCES")
  fi
  if [[ "$DEBUG" -eq 1 ]]; then
    CMD+=(--debug)
  fi
  echo "[run_contextbench_miniswe] Running: ${CMD[*]}"
  exec "${CMD[@]}"
fi

# until-submitted loop: always --rerun failed/missing instances until all Submitted.
while true; do
  ROUND=$((ROUND + 1))
  pending="$(_list_not_submitted_ids || true)"
  if [[ -z "$pending" ]]; then
    echo "[run_contextbench_miniswe] All in-scope instances Submitted (rounds=${ROUND})." >&2
    _report_not_submitted >&2 || true
    exit 0
  fi

  pending_count="$(awk -F, '{print NF}' <<<"$pending")"
  echo "[run_contextbench_miniswe] Round ${ROUND}: ${pending_count} not Submitted → rerun" >&2
  _run_contextbench_batch "$pending"
done
