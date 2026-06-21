#!/usr/bin/env bash

set -euo pipefail

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$_SCRIPTS_DIR/env_vllm_chat.sh"

usage() {
  cat <<'EOF'
Usage:
  tools/run_contextbench_miniswe.sh --bench <Verified|Pro|Poly|Multi> --output <output_dir> --workers <N> --rerun <true|false> [--instances <id1,id2,...>] [--debug]

  --debug   Forward to contextbench.run: print agent subprocess cwd/cmd and live output (no capture).

Examples:
  tools/run_contextbench_miniswe.sh --bench Pro --output output_vllm/select_500_qwen25_32B_instruct --workers 12 --rerun true
  tools/run_contextbench_miniswe.sh --bench Multi --output output_vllm/select_500_qwen25_32B_instruct --workers 8 --rerun false
  tools/run_contextbench_miniswe.sh --bench Multi --output output_vllm/select_500_qwen25_32B_instruct --workers 8 --rerun true --instances "cli__cli-5047,sveltejs__svelte-10608"
  tools/run_contextbench_miniswe.sh --bench Pro --output output_vllm/select_500_qwen25_32B_instruct --workers 1 --rerun false --debug

  # 覆盖 vLLM 模型名（需与 start_vllm.sh 的 VLLM_MODEL / /v1/models 一致）:
  export MINISWE_VLLM_MODEL=hosted_vllm/Qwen/Qwen2.5-32B-Instruct
EOF
}

BENCH=""
OUTPUT=""
WORKERS=""
RERUN=""
INSTANCES=""
DEBUG=0

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

# Accept common boolean spellings for rerun switch.
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

# Mirror routing rules:
# - Multi: use Xuanyuan private registry for mswebench images.
# - Poly: use NJU ghcr mirror.
# - Pro/Verified: default to Xuanyuan (avoid direct Docker Hub timeout).
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

# Hugging Face（HF_ENDPOINT 由 env_vllm_chat.sh 探测；强制镜像: export HF_ENDPOINT=https://hf-mirror.com）
if [[ -d /workspace/hf_cache ]]; then
  export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
else
  export HF_HOME="${HF_HOME:-/home/dataset-local/hf_cache}"
fi
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# 每个 miniswe 子进程都会 load_dataset；并行时打镜像易 429。先单机预载进 hub 缓存。
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
    # Multi-SWE-bench 各实例 test 字段 schema 不一致，全量 load_dataset 会 PyArrow cast 失败；
    # miniswe 子进程用 streaming（见 swebench_context_aware._load_multiswe_dataset_safely）。
    python -c "
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
  python -c "
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

# LiteLLM hosted_vllm/* 须与 tools/start_vllm.sh 当前服务的模型 ID 一致
export MINISWE_VLLM_MODEL="${MINISWE_VLLM_MODEL:-hosted_vllm/Qwen/Qwen2.5-32B-Instruct}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

CMD=(
  python -m contextbench.run
  --agent miniswe
  --task-csv data/selected_500_instances.csv
  --bench "$BENCH"
  --timeout 3600
  --miniswe-step-response-timeout 600
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

echo "[run_contextbench_miniswe] BENCH=$BENCH"
echo "[run_contextbench_miniswe] OUTPUT=$OUTPUT WORKERS=$WORKERS RERUN=$RERUN"
if [[ -n "$INSTANCES" ]]; then
  echo "[run_contextbench_miniswe] INSTANCES=$INSTANCES"
fi
echo "[run_contextbench_miniswe] DEBUG=$DEBUG"
echo "[run_contextbench_miniswe] MSWEA_DOCKER_IMAGE_REGISTRY=${MSWEA_DOCKER_IMAGE_REGISTRY}"
echo "[run_contextbench_miniswe] MSWEA_POLY_GHCR_REGISTRY=${MSWEA_POLY_GHCR_REGISTRY}"
echo "[run_contextbench_miniswe] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[run_contextbench_miniswe] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[run_contextbench_miniswe] MINISWE_VLLM_MODEL=${MINISWE_VLLM_MODEL} (vLLM serve: ${VLLM_MODEL})"
echo "[run_contextbench_miniswe] Running: ${CMD[*]}"

exec "${CMD[@]}"
