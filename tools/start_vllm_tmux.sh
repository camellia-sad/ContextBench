#!/usr/bin/env bash
# 辅助「恢复」已有 tmux 会话（默认不结束、不覆盖 vllm 会话）。
#
# 推荐手动流程（不依赖本脚本的自动起服务）:
#   1) 宿主机: tmux attach -t vllm   （或 tmux ls 看名字）
#   2) 在会话里: docker exec -it <容器名> bash
#   3) 容器内: cd /workspace/ContextBench && bash tools/start_vllm.sh
#
# 本脚本行为:
#   - 若已存在会话 ${VLLM_TMUX_SESSION:-vllm}: 仅提示/尝试 attach，绝不 kill-session
#   - 若不存在: 打印上述手动步骤；不自动起 vLLM
#   - 仅当需要「无人值守新建 detached 会话并 docker exec 跑 start_vllm」时:
#       export VLLM_TMUX_AUTO_START=1
#       （仍会检查 DOCKER_CONTAINER_NAME / 容器运行状态）
#
# 可选环境变量:
#   DOCKER_CONTAINER_NAME   仅 VLLM_TMUX_AUTO_START=1 时使用，默认 cb_workspace
#   VLLM_TMUX_SESSION       默认 vllm
#   HF_TOKEN / HUGGINGFACE_HUB_TOKEN  仅自动模式传入容器

set -euo pipefail

CONTAINER="${DOCKER_CONTAINER_NAME:-cb_workspace}"
SESSION="${VLLM_TMUX_SESSION:-vllm}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[start_vllm_tmux] 会话 ${SESSION} 已存在，不会结束该会话。" >&2
  if [[ -t 0 ]] && [[ -t 1 ]]; then
    echo "[start_vllm_tmux] 正在恢复（attach）…" >&2
    exec tmux attach -t "$SESSION"
  fi
  echo "[start_vllm_tmux] 请在交互终端执行: tmux attach -t ${SESSION}" >&2
  exit 0
fi

if [[ "${VLLM_TMUX_AUTO_START:-}" != "1" ]]; then
  echo "[start_vllm_tmux] 没有名为 ${SESSION} 的 tmux 会话（未自动新建）。" >&2
  echo "[start_vllm_tmux] 恢复/创建后手动部署 vLLM，例如:" >&2
  echo "  tmux new -s ${SESSION}    # 若还没有会话" >&2
  echo "  tmux attach -t ${SESSION}" >&2
  echo "  docker exec -it <容器名> bash" >&2
  echo "  cd /workspace/ContextBench && bash tools/start_vllm.sh" >&2
  echo "[start_vllm_tmux] 若仍要脚本自动新建 detached 会话并 exec 容器内 start_vllm: export VLLM_TMUX_AUTO_START=1 后重跑本脚本。" >&2
  exit 0
fi

if ! docker inspect "${CONTAINER}" &>/dev/null; then
  echo "[start_vllm_tmux] 无此容器: ${CONTAINER}" >&2
  echo "[start_vllm_tmux] 当前运行中的容器:" >&2
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null || true
  echo "[start_vllm_tmux] 启动默认开发容器: cd <ContextBench> && bash tools/start_docker.sh" >&2
  echo "[start_vllm_tmux] 或: export DOCKER_CONTAINER_NAME=…" >&2
  exit 1
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" != "true" ]]; then
  echo "[start_vllm_tmux] 容器未运行: ${CONTAINER}" >&2
  echo "[start_vllm_tmux] 请先: bash tools/start_docker.sh  或  docker start ${CONTAINER}" >&2
  exit 1
fi

HF="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
tmux new-session -d -s "$SESSION" "docker exec -e HF_TOKEN=\"${HF}\" -e HUGGINGFACE_HUB_TOKEN=\"${HF}\" \"${CONTAINER}\" bash -lc 'cd /workspace/ContextBench && exec bash tools/start_vllm.sh'"

echo "[start_vllm_tmux] 已新建 detached 会话 ${SESSION}（VLLM_TMUX_AUTO_START=1）。" >&2
echo "[start_vllm_tmux] 查看: tmux attach -t ${SESSION}" >&2
