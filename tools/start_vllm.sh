#!/bin/bash
# 在 GPU 机器或带 GPU 的 Docker 容器内执行；ContextBench / miniswe 通过 api_base 连本机 :8000
# 正常启动：bash tools/start_vllm.sh
#   会先打印一行可复制的「export …; vllm serve …」，随后**会执行** vLLM（exec）。
# 仅打印、不执行：bash tools/start_vllm.sh --print-cmd   或   -p
#   只输出上述一行命令后立刻退出，**不会**安装依赖、**不会**启动 vLLM。
#
# 默认：facebook/cwm（原生 max_position_embeddings=131072，无需 YaRN）。
#
# CWM（门禁）示例（能直连 huggingface.co）：
#   export VLLM_MODEL=facebook/cwm
#   export VLLM_TP_SIZE=4
#   export HF_TOKEN=hf_...
#   export HF_ENDPOINT=https://huggingface.co
#
# CWM 离线（GPU 机无代理、不能连 huggingface.co）：
#   在能上网的机器用 HF_TOKEN 下载后，把整个 hub 缓存目录拷到本机，例如：
#     rsync -avz ~/.cache/huggingface/hub/models--facebook--cwm/  GPU机:$HF_HOME/hub/models--facebook--cwm/
#   然后在 GPU 机：
#     export VLLM_HF_OFFLINE=1
#     export VLLM_PRELOAD_HF_MODEL=0
#     export HF_HOME=/path/to/hf_cache
#     bash tools/start_vllm.sh
#   （hf-mirror 不能下 gated 的 cwm；勿指望镜像 + token）
#
# Codestral（YaRN 32768→131072）示例：
#   export VLLM_MODEL=mistralai/Codestral-22B-v0.1
#   export VLLM_TP_SIZE=1
#   export VLLM_MAX_MODEL_LEN=131072
#   export VLLM_NATIVE_MAX_MODEL_LEN=32768
#
# Qwen 等示例：
#   export VLLM_MODEL=Qwen/Qwen2.5-Coder-7B
#   export VLLM_TP_SIZE=1
#   export VLLM_MAX_MODEL_LEN=32768

set -euo pipefail

_PRINT_CMD_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-cmd | -p) _PRINT_CMD_ONLY=1; shift ;;
    --help | -h)
      cat <<'EOF'
[start_vllm] 用法
  bash tools/start_vllm.sh
      按当前环境变量安装依赖（若缺）并启动 vLLM。
      启动前会打印一行可复制的「export …; vllm serve …」，随后**会真正执行** vLLM（当前 shell 被替换为服务进程）。

  bash tools/start_vllm.sh --print-cmd
  bash tools/start_vllm.sh -p
      **仅打印**上述「export …; vllm serve …」一行命令，然后**立即退出**。
      **不会**安装依赖，**不会**启动或 exec vLLM。可先 export VLLM_MODEL 等再运行本选项核对参数。

  bash tools/start_vllm.sh --help
      显示本说明。

常用环境变量见脚本头部注释（VLLM_MODEL、VLLM_TP_SIZE、VLLM_MAX_MODEL_LEN、HF_TOKEN 等）。
EOF
      exit 0
      ;;
    *)
      echo "[start_vllm] 未知参数: $1（支持 --print-cmd / -p / --help）" >&2
      exit 2
      ;;
  esac
done

# 仅当需要超过模型 config 里的原生 context（如 Qwen2.5 的 32k）时再手动 export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1（RoPE 外推有风险）。
if [[ "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}" == "1" ]]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$_TOOLS_DIR/env_vllm_chat.sh"

# FlashInfer 的 top-k/top-p 采样会在首次运行时调用 nvcc JIT；仅含 CUDA runtime 的镜像无 nvcc 会失败。
# 使用 cuda devel / 已装 toolkit 且需要该路径时再 export VLLM_USE_FLASHINFER_SAMPLER=1。
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

if [[ "$VLLM_MODEL" == "facebook/cwm" ]]; then
  VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
  VLLM_NATIVE_LEN="${VLLM_NATIVE_MAX_MODEL_LEN:-131072}"
  VLLM_MAX_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
  VLLM_GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
  VLLM_PRELOAD_HF_MODEL="${VLLM_PRELOAD_HF_MODEL:-1}"
elif [[ "$VLLM_MODEL" == *Codestral* ]] || [[ "$VLLM_MODEL" == mistralai/Codestral* ]]; then
  VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
  VLLM_NATIVE_LEN="${VLLM_NATIVE_MAX_MODEL_LEN:-32768}"
  VLLM_MAX_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
  VLLM_GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
  VLLM_PRELOAD_HF_MODEL="${VLLM_PRELOAD_HF_MODEL:-1}"
else
  VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
  VLLM_NATIVE_LEN="${VLLM_NATIVE_MAX_MODEL_LEN:-32768}"
  VLLM_MAX_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
  VLLM_GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
fi

# facebook/cwm：门禁；hf-mirror 无法替代官网 token 下载
_hf_hub_cache_slug() {
  local repo="$1"
  echo "models--${repo//\//--}"
}

_hf_model_cached_locally() {
  local repo="$1"
  local hub="${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-}/hub}"
  [[ -n "$hub" ]] || return 1
  local snap="${hub}/$(_hf_hub_cache_slug "$repo")/snapshots"
  compgen -G "${snap}/*/config.json" >/dev/null 2>&1
}

VLLM_HF_CACHE_HIT=0
if [[ "$VLLM_MODEL" == "facebook/cwm" ]]; then
  export VLLM_USE_CONCAT_CHAT_TEMPLATE="${VLLM_USE_CONCAT_CHAT_TEMPLATE:-1}"
  if _hf_model_cached_locally "facebook/cwm"; then
    VLLM_HF_CACHE_HIT=1
    if [[ "${VLLM_HF_OFFLINE:-0}" == "1" ]] || [[ "${VLLM_SKIP_HF_PRELOAD:-0}" == "1" ]]; then
      export HF_HUB_OFFLINE=1
      VLLM_PRELOAD_HF_MODEL=0
      if [[ "${_PRINT_CMD_ONLY}" != "1" ]]; then
        echo "[start_vllm] 检测到本地 HF 缓存 models--facebook--cwm，离线启动（HF_HUB_OFFLINE=1，跳过预下载）" >&2
      fi
    fi
  fi
  if [[ "${VLLM_HF_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    VLLM_PRELOAD_HF_MODEL=0
  fi
  if [[ -n "${VLLM_HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="${VLLM_HF_ENDPOINT}"
  elif [[ "${VLLM_HF_CACHE_HIT}" == "1" ]] && [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    : # 离线：不强制改 HF_ENDPOINT
  else
    export HF_ENDPOINT="https://huggingface.co"
  fi
  if [[ "${HF_ENDPOINT:-}" == *hf-mirror* ]] && [[ "${_PRINT_CMD_ONLY}" != "1" ]]; then
    echo "[start_vllm] 警告: facebook/cwm 不要用 hf-mirror（会 GatedRepoError）" >&2
  fi
  _hf_tok="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  if [[ -z "$_hf_tok" ]] && [[ "${VLLM_HF_CACHE_HIT}" == "0" ]] && [[ "${_PRINT_CMD_ONLY}" != "1" ]]; then
    echo "[start_vllm] 错误: 无本地缓存时需 HF_TOKEN + https://huggingface.co/facebook/cwm Accept" >&2
    exit 1
  fi
  if [[ "${VLLM_HF_CACHE_HIT}" == "0" ]] && [[ "${VLLM_PRELOAD_HF_MODEL:-0}" == "1" ]] \
      && ! curl -fsS -m 8 -o /dev/null https://huggingface.co 2>/dev/null \
      && [[ "${_PRINT_CMD_ONLY}" != "1" ]]; then
    echo "[start_vllm] 错误: 本机无法访问 huggingface.co，且未找到本地 models--facebook--cwm 缓存。" >&2
    echo "[start_vllm] 请在能访问官网的机器下载后拷入 \$HF_HOME/hub/models--facebook--cwm/ ，再:" >&2
    echo "  export VLLM_HF_OFFLINE=1 VLLM_PRELOAD_HF_MODEL=0" >&2
    exit 1
  fi
fi

# 国内环境：优先使用清华 PyPI 镜像（可通过外部 PIP_INDEX_URL 覆盖）
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

if [[ "${_PRINT_CMD_ONLY}" != "1" ]]; then
  # 解析 python 命令：容器镜像可能只有 python3 没有 python
  PYTHON_BIN="${PYTHON_BIN:-}"
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[start_vllm] python 未找到，将安装 python3/pip（并使用国内源）..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ca-certificates curl build-essential
    # shellcheck disable=SC2034
    PYTHON_BIN="python3"
  fi

  # 首次 /chat/completions 时 Triton 可能 JIT 编译 cuda_utils.c；缺 Python.h 会导致 EngineDeadError
  _py_ver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.10")"
  if [[ ! -f "/usr/include/python${_py_ver}/Python.h" ]] && ! compgen -G "/usr/include/python*/Python.h" >/dev/null; then
    echo "[start_vllm] 安装 python3-dev + build-essential（Triton JIT 需要 Python.h）..." >&2
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends python3-dev build-essential
  fi

  # 如果容器里没有 vllm / torch，则先安装依赖
  if ! command -v vllm >/dev/null 2>&1; then
    echo "[start_vllm] vllm 未找到，使用国内源安装依赖（torch + vllm）..."
    "$PYTHON_BIN" -m pip install -U pip setuptools wheel
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
    "$PYTHON_BIN" -m pip install --index-url "$TORCH_INDEX" torch
    "$PYTHON_BIN" -m pip install vllm>=0.6.0
  fi

  # 让 flashinfer / nvcc 能找到 CUDA 库（在 base/conda 环境下）
  if [ -d /opt/conda/lib64 ]; then
    export LD_LIBRARY_PATH="/opt/conda/lib64:/opt/conda/lib64/stubs:${LD_LIBRARY_PATH:-}"
  fi

  if [[ "${VLLM_PRELOAD_HF_MODEL:-0}" == "1" ]] && _hf_model_cached_locally "${VLLM_MODEL}"; then
    echo "[start_vllm] 本地已有 ${VLLM_MODEL} 缓存，跳过预下载（可设 VLLM_PRELOAD_HF_MODEL=0 静默）" >&2
  elif [[ "${VLLM_PRELOAD_HF_MODEL:-0}" == "1" ]]; then
    echo "[start_vllm] 预下载 ${VLLM_MODEL}（HF_ENDPOINT=${HF_ENDPOINT:-} HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}）…" >&2
    if ! "$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null; then
      "$PYTHON_BIN" -m pip install -q huggingface_hub
    fi
    "$PYTHON_BIN" -c "
import os
from huggingface_hub import snapshot_download
tok = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
snapshot_download('${VLLM_MODEL}', token=tok, resume_download=True)
" || {
      echo "[start_vllm] 预下载失败。" >&2
      if [[ "$VLLM_MODEL" == "facebook/cwm" ]]; then
        echo "  门禁 cwm：勿用 hf-mirror。能上网的机器执行:" >&2
        echo "    export HF_ENDPOINT=https://huggingface.co HF_TOKEN=hf_... HF_HUB_DISABLE_XET=1" >&2
        echo "    huggingface-cli download facebook/cwm --token \$HF_TOKEN" >&2
        echo "  再把 hub/models--facebook--cwm/ 拷到本机 \$HF_HOME/hub/ ，然后:" >&2
        echo "    export VLLM_HF_OFFLINE=1 VLLM_PRELOAD_HF_MODEL=0" >&2
      else
        echo "  export HF_ENDPOINT=https://huggingface.co  或配置离线缓存 + VLLM_HF_OFFLINE=1" >&2
      fi
      exit 1
    }
  fi
fi

CHAT_ARGS=()
if [[ "${VLLM_USE_CONCAT_CHAT_TEMPLATE:-0}" == "1" ]]; then
  if [[ ! -f "${CHAT_TEMPLATE:-}" ]]; then
    echo "[start_vllm] VLLM_USE_CONCAT_CHAT_TEMPLATE=1 but CHAT_TEMPLATE missing: ${CHAT_TEMPLATE:-}" >&2
    exit 1
  fi
  CHAT_ARGS=(--chat-template "$CHAT_TEMPLATE")
fi

# --hf-overrides：把 config 里的原生 context（如 32768）用 YaRN 等扩到 --max-model-len（如 131072）
HF_OVERRIDE_ARGS=()
if [[ -n "${VLLM_HF_OVERRIDES:-}" ]]; then
  HF_OVERRIDE_ARGS=(--hf-overrides "$VLLM_HF_OVERRIDES")
  if [[ "$VLLM_MAX_LEN" -gt "$VLLM_NATIVE_LEN" ]]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  fi
elif [[ "$VLLM_MAX_LEN" -gt "$VLLM_NATIVE_LEN" ]]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  # factor = 目标长度 / 原生长度（Codestral 32768 -> 131072 即 4.0）
  _rope_factor="${VLLM_ROPE_YARN_FACTOR:-}"
  if [[ -z "$_rope_factor" ]]; then
    _rope_factor="$(awk -v m="$VLLM_MAX_LEN" -v n="$VLLM_NATIVE_LEN" 'BEGIN { printf "%.4f", m/n }')"
  fi
  _hf_json="{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":${_rope_factor},\"original_max_position_embeddings\":${VLLM_NATIVE_LEN}},\"max_position_embeddings\":${VLLM_MAX_LEN}}"
  HF_OVERRIDE_ARGS=(--hf-overrides "$_hf_json")
  if [[ "${_PRINT_CMD_ONLY}" != "1" ]]; then
    echo "[start_vllm] YaRN hf-overrides: native=${VLLM_NATIVE_LEN} max=${VLLM_MAX_LEN} factor=${_rope_factor}" >&2
  fi
fi

_VLLM_LAUNCH_CMD=(
  vllm serve "$VLLM_MODEL"
  --host 0.0.0.0
  --port 8000
  --tensor-parallel-size "$VLLM_TP_SIZE"
  --max-model-len "$VLLM_MAX_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEM"
  --trust-remote-code
  --max-num-seqs 4
  --enable-chunked-prefill
  --enforce-eager
)
if ((${#HF_OVERRIDE_ARGS[@]} > 0)); then
  _VLLM_LAUNCH_CMD+=("${HF_OVERRIDE_ARGS[@]}")
fi
if ((${#CHAT_ARGS[@]} > 0)); then
  _VLLM_LAUNCH_CMD+=("${CHAT_ARGS[@]}")
fi

_print_copyable_cmd_line() {
  echo "[start_vllm] 可复制的一条命令（按需改 export / 参数后再执行）："
  printf '%s' "export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
  if [[ "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}" == "1" ]]; then
    printf '%s' "; export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  fi
  if [[ -n "${VLLM_HF_OVERRIDES:-}" ]]; then
    printf '%s' "; export VLLM_HF_OVERRIDES=$(printf '%q' "$VLLM_HF_OVERRIDES")"
  elif [[ "$VLLM_MAX_LEN" -gt "$VLLM_NATIVE_LEN" ]]; then
    printf '%s' "; export VLLM_NATIVE_MAX_MODEL_LEN=${VLLM_NATIVE_LEN}"
  fi
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    printf '%s' "; export HF_ENDPOINT=${HF_ENDPOINT}"
  fi
  if [[ "${VLLM_USE_CONCAT_CHAT_TEMPLATE:-0}" == "1" ]]; then
    printf '%s' "; export VLLM_USE_CONCAT_CHAT_TEMPLATE=1"
  fi
  if [[ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
    printf '%s' "; export HF_TOKEN=\${HF_TOKEN:-<你的_hf_token>}"
  elif [[ "$VLLM_MODEL" == "facebook/cwm" ]]; then
    printf '%s' "; export HF_TOKEN=<你的_hf_token>"
  fi
  printf '%s' "; "
  printf '%q ' "${_VLLM_LAUNCH_CMD[@]}"
  echo
}

if [[ "${_PRINT_CMD_ONLY}" == "1" ]]; then
  echo "[start_vllm] --print-cmd：只打印下面一行命令，不执行 vLLM、不 exec（model=$VLLM_MODEL tp=$VLLM_TP_SIZE native=$VLLM_NATIVE_LEN max_len=$VLLM_MAX_LEN gpu_mem=$VLLM_GPU_MEM）"
  _print_copyable_cmd_line
  exit 0
fi

echo "[start_vllm] model=$VLLM_MODEL tp=$VLLM_TP_SIZE native=$VLLM_NATIVE_LEN max_len=$VLLM_MAX_LEN gpu_mem=$VLLM_GPU_MEM hf_overrides=${#HF_OVERRIDE_ARGS[@]} concat_template=${VLLM_USE_CONCAT_CHAT_TEMPLATE:-0}（随后将 exec 启动 vLLM；若只想看命令请加 --print-cmd）"
_print_copyable_cmd_line

exec "${_VLLM_LAUNCH_CMD[@]}"
