#!/usr/bin/env bash
# 在当前机器 / 容器中安装 Python 及项目依赖（使用国内源），以便后续运行 ContextBench 相关命令。
#
# 用法（在仓库根或任意位置）：
#   bash tools/install_python_requirements.sh
#   # 或指定 ContextBench 根目录：
#   CONTEXTBENCH_ROOT=/path/to/ContextBench bash tools/install_python_requirements.sh
#
# 环境变量（可选覆盖默认国内源）：
#   PIP_INDEX_URL        默认 https://pypi.tuna.tsinghua.edu.cn/simple
#   PIP_TRUSTED_HOST     默认 pypi.tuna.tsinghua.edu.cn
#   HF_ENDPOINT          默认 https://hf-mirror.com
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXTBENCH_ROOT="${CONTEXTBENCH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

echo "[install_python_requirements] ContextBench 根目录: ${CONTEXTBENCH_ROOT}"

REQ_FILE="${CONTEXTBENCH_ROOT}/requirements.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[install_python_requirements] 未找到 requirements.txt: ${REQ_FILE}" >&2
  exit 1
fi

# 国内镜像源（可被外部 export 覆盖）
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[install_python_requirements] 使用 pip 源: ${PIP_INDEX_URL} (trusted: ${PIP_TRUSTED_HOST})"
echo "[install_python_requirements] Hugging Face 源: ${HF_ENDPOINT}"

# 判断 python 可用性，必要时安装 python3/pip（适用于 CUDA 运行时基础镜像）
PYTHON_BIN="${PYTHON_BIN:-}"
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[install_python_requirements] python 未找到，尝试通过 apt 安装 python3 / pip ..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl build-essential
  PYTHON_BIN="python3"
fi

echo "[install_python_requirements] 使用 Python 解释器: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m pip install -U pip setuptools wheel
echo "[install_python_requirements] 安装 ContextBench requirements.txt ..."
"${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"

echo "[install_python_requirements] 完成。你现在可以运行例如："
echo "  cd ${CONTEXTBENCH_ROOT}"
echo "  ${PYTHON_BIN} -m contextbench.run --help"

