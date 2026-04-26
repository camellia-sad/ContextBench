#!/usr/bin/env bash
# 从「仍在的」cb_workspace（或你指定的）容器恢复可见配置，并说明为何改 docker run 后要重装环境。
#
# 用法：
#   bash tools/inspect_vllm_docker_config.sh
#   DOCKER_CONTAINER_NAME=other_name bash tools/inspect_vllm_docker_config.sh
#
set -euo pipefail

NAME="${DOCKER_CONTAINER_NAME:-${VLLM_CONTAINER_NAME:-cb_workspace}}"

echo "========== 1) 容器是否存在 =========="
if ! docker container inspect "$NAME" &>/dev/null; then
  echo "没有名为 '$NAME' 的容器。可能已被删除或未在本机创建。"
  echo "若曾用相同名字重建，旧容器的文件系统（含未挂载路径下的 venv）已随容器删除。"
  exit 1
fi

echo "========== 2) docker inspect（镜像、命令、环境变量、挂载）=========="
docker inspect "$NAME" --format 'Image: {{.Config.Image}}
Cmd:    {{json .Config.Cmd}}
Entrypoint: {{json .Config.Entrypoint}}
WorkingDir: {{.Config.WorkingDir}}
'
echo "--- Env (节选 HF/PIP/NVIDIA) ---"
docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(HF_|PIP_|NVIDIA|CUDA|LD_)' || true

echo ""
echo "--- Mounts ---"
docker inspect "$NAME" --format '{{range .Mounts}}{{println .Source " -> " .Destination}}{{end}}'

echo ""
echo "========== 3) 等价 docker run 草稿（需按你当前镜像与路径核对）=========="
IMG=$(docker inspect "$NAME" --format '{{.Config.Image}}')
echo "# 反推镜像: $IMG"
echo "完整挂载列表见上节 Mounts；自行拼成:"
echo "  docker run -d --name <name> --gpus '\"device=0,1,2,3\"' -p ...:8000 -v <src>:<dst> ... \"$IMG\" sleep infinity"

echo ""
echo "========== 4) 为何每次改 docker run 都要「重装环境」？ =========="
cat <<'EOF'
- 容器可写层里的内容（例如默认在 / 下建的 venv、pip install 到系统目录）只存在于「这一份」容器文件系统里。
- 你若用新 docker run 新建容器、或 rm 后重建，未挂载到宿主机的目录会清空，看起来像「又要装一遍」。
- 解决办法（你仓库里 install_vllm_docker_deps.sh 已按此设计）：
  - 把 WORKSPACE（如 /workspace）挂到宿主机固定目录，venv 放在 /workspace/.venv-vllm；
  - pip / HF 缓存也指向挂载目录（PIP_CACHE_DIR、HF_HOME）。
这样只要 HOST_WORKSPACE 不变，换镜像或重建容器后只需 activate 原 venv，一般不必重装。

EOF

echo "========== 5) 宿主机上可能还保留的旧配置 =========="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
for f in \
  "$REPO_ROOT/tools/start_vllm.sh" \
  "$HOME/.bash_history" \
  /home/dataset-assist-0/vllm_workspace/docker_workspace/.venv-vllm/pyvenv.cfg
  do
  if [[ -f "$f" ]]; then
    echo "[exists] $f"
  fi
done
echo "（可在 ~/.bash_history 里搜: docker run.*cb_workspace）"
