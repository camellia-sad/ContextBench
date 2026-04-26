#!/usr/bin/env bash
# 在 GPU 容器内执行（例如 nvidia/cuda:12.2.0-runtime-ubuntu22.04）。
# 依赖与 pip 缓存写入挂载的数据盘目录（默认 /workspace，对应宿主机 docker_workspace）。
#
# 典型用法（宿主机）：
# Docker Hub 不可达时，镜像可写 xuanyuan 全路径，例如：
#   fczi514j9ggm7b.xuanyuan.run/docker.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04
#   docker run -it --gpus all \
#     -v /home/dataset-assist-0/vllm_workspace/docker_workspace:/workspace \
#     -v /home/dataset-assist-0/vllm_workspace/ContextBench:/workspace/ContextBench \
#     -w /workspace \
#     ... 镜像 bash
#   容器内：
#     bash /workspace/ContextBench/tools/install_vllm_docker_deps.sh
#
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/pip_cache}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE"

VENV="${VENV:-$WORKSPACE/.venv-vllm}"

if [ "$(id -u)" -eq 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl
fi

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel --cache-dir "$PIP_CACHE_DIR"

# CUDA 12.x 运行时镜像：与 cu121 预编译 wheel 通常兼容
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
pip install --cache-dir "$PIP_CACHE_DIR" torch --index-url "$TORCH_INDEX"
pip install --cache-dir "$PIP_CACHE_DIR" "vllm>=0.6.0"

echo
echo "安装完成。"
echo "  虚拟环境: $VENV"
echo "  pip 缓存:   $PIP_CACHE_DIR"
echo "  HF 缓存:    $HUGGINGFACE_HUB_CACHE"
echo
echo "启动前请： source $VENV/bin/activate"
echo "然后执行： bash /workspace/ContextBench/tools/start_vllm.sh"
echo "（若仓库不在 /workspace/ContextBench，请把 start_vllm.sh 拷到当前目录或设置 HF_HOME/见脚本内说明）"
