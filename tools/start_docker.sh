#!/usr/bin/env bash
# 启动开发容器；默认使用 GPU 列表 GPU_DEVICES（默认 0,1,2,3）。
#
# 用法：bash tools/start_docker.sh
#       export GPU_DEVICES=0,1
#       export DOCKER_RESUME_CONTAINER=容器名
#
# GPU 与 daemon 报错「cannot set both Count and DeviceIDs」：
#   部分平台在 --gpus "device=..." 时会与内部再注入的 GPU 请求冲突。
#   默认使用 --gpus all，仅在容器内 export CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES
#   为 GPU_DEVICES（宿主机恰好 4 卡时，即等价使用 0–3 号卡）。
#   若你确认环境支持按设备号申请，可：
#     export DOCKER_GPUS_STRATEGY=device
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_DOCKER_IMAGE="${DEFAULT_DOCKER_IMAGE:-fczi514j9ggm7b.xuanyuan.run/nvidia/cuda:12.2.0-runtime-ubuntu22.04}"
DOCKER_IMAGE="${DOCKER_IMAGE:-$DEFAULT_DOCKER_IMAGE}"
DOCKER_AUTO_PULL="${DOCKER_AUTO_PULL:-1}"

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
# all = --gpus all（规避 Count+DeviceIDs）；device = --gpus "device=${GPU_DEVICES}"
DOCKER_GPUS_STRATEGY="${DOCKER_GPUS_STRATEGY:-all}"

DOCKER_WORKSPACE_HOST="${DOCKER_WORKSPACE_HOST:-/home/dataset-assist-0/docker_workspace}"
DOCKER_WORKSPACE_CONTAINER="${DOCKER_WORKSPACE_CONTAINER:-/workspace}"
DOCKER_RESUME_CONTAINER="${DOCKER_RESUME_CONTAINER:-}"
DOCKER_CONTAINER_NAME="${DOCKER_CONTAINER_NAME:-cb_workspace}"
HOST_PORT="${HOST_PORT:-8000}"

if [[ -n "${DOCKER_RESUME_CONTAINER}" ]]; then
  docker container inspect "${DOCKER_RESUME_CONTAINER}" &>/dev/null || {
    echo "[start_docker] 无此容器: ${DOCKER_RESUME_CONTAINER}"; exit 1; }
  exec docker start -ai "${DOCKER_RESUME_CONTAINER}"
fi

if [[ "${DOCKER_REQUIRE_LOCAL_IMAGE:-0}" == "1" ]] && ! docker image inspect "${DOCKER_IMAGE}" &>/dev/null; then
  echo "[start_docker] 本地无镜像: ${DOCKER_IMAGE}"; exit 1
fi

if [[ "${DOCKER_AUTO_PULL}" == "1" ]] && ! docker image inspect "${DOCKER_IMAGE}" &>/dev/null; then
  docker pull "${DOCKER_IMAGE}"
fi

mkdir -p "${DOCKER_WORKSPACE_HOST}/hf_cache/hub"

if [[ "${DOCKER_GPUS_STRATEGY}" == "device" ]]; then
  GPUS_ARG="device=${GPU_DEVICES}"
  echo "[start_docker] DOCKER_GPUS_STRATEGY=device  ->  --gpus \"${GPUS_ARG}\""
else
  GPUS_ARG="all"
  echo "[start_docker] DOCKER_GPUS_STRATEGY=all     ->  --gpus all；容器内 CUDA/NVIDIA 可见设备: ${GPU_DEVICES}"
fi

exec env \
  -u NVIDIA_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u DOCKER_NVIDIA_VISIBLE_DEVICES -u NV_GPU \
  docker run --rm -it \
  --name "${DOCKER_CONTAINER_NAME}" \
  --gpus 4 \
  -e GPU_DEVICES="${GPU_DEVICES}" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  -e PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}" \
  -e HF_HOME="${DOCKER_WORKSPACE_CONTAINER}/hf_cache" \
  -e HUGGINGFACE_HUB_CACHE="${DOCKER_WORKSPACE_CONTAINER}/hf_cache/hub" \
  -v "${DOCKER_WORKSPACE_HOST}:${DOCKER_WORKSPACE_CONTAINER}" \
  -v "${REPO_ROOT}:${DOCKER_WORKSPACE_CONTAINER}/ContextBench" \
  -w "${DOCKER_WORKSPACE_CONTAINER}" \
  -p "${HOST_PORT}:${HOST_PORT}" \
  "${DOCKER_IMAGE}" \
  bash -lc "export CUDA_VISIBLE_DEVICES='${GPU_DEVICES}'; export NVIDIA_VISIBLE_DEVICES='${GPU_DEVICES}'; exec bash"
