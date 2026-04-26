#!/bin/bash
# 由 tools/gen_vllm_configs.py 生成，上下文窗口 1000000
# 在 GPU 机器或带 GPU 的 Docker 容器内执行，启动后 ContextBench 通过 api_base 连接
# 缓存到数据盘，断开重连后不会重复下载

set -euo pipefail

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/dataset-local/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# 国内环境：优先使用清华 PyPI 镜像（可通过外部 PIP_INDEX_URL 覆盖）
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

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

# 如果容器里没有 vllm / torch，则先安装依赖
if ! command -v vllm >/dev/null 2>&1; then
  echo "[start_vllm] vllm 未找到，使用国内源安装依赖（torch + vllm）..."
  "$PYTHON_BIN" -m pip install -U pip setuptools wheel
  # CUDA 12.2 对应 cu121 轮子；若平台有专门源，可通过 TORCH_INDEX 覆盖
  TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
  "$PYTHON_BIN" -m pip install --index-url "$TORCH_INDEX" torch
  "$PYTHON_BIN" -m pip install vllm>=0.6.0
fi

# 让 flashinfer / nvcc 能找到 CUDA 库（在 base/conda 环境下）
if [ -d /opt/conda/lib64 ]; then
  export LD_LIBRARY_PATH="/opt/conda/lib64:/opt/conda/lib64/stubs:${LD_LIBRARY_PATH:-}"
fi

vllm serve deepseek-ai/deepseek-coder-33b-instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --enforce-eager \
  --hf-overrides '{
"max_position_embeddings":131072,
"rope_scaling":{"type":"yarn","factor":8.0},
"original_max_position_embeddings":16384
}'
# Optional: alternate rope override example (keep commented):
#  --hf-overrides '{"max_position_embeddings":131072,"rope_scaling":{"type":"yarn","factor":4.0},"original_max_position_embeddings":32768}'
