# ContextBench + 本机 vLLM（StarCoder 等无 tokenizer chat_template）共用环境
# 用法（在 ContextBench 仓库根目录）:
#   source tools/env_vllm_chat.sh
# 或由 tools/start_vllm.sh / tools/run_contextbench_miniswe.sh 自动 source。
#
# 勿在此文件内使用 set -e，以免中断调用方的 shell。

_TOOLS_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONTEXTBENCH_ROOT="$(cd "$_TOOLS_DIR_ENV/.." && pwd)"

# vLLM --chat-template（修复 transformers>=4.44 下 chat/completions 400）
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-$_TOOLS_DIR_ENV/vllm_concat_messages.jinja}"

# Hugging Face（可在外部覆盖 HF_ENDPOINT）
# 默认：能直连 huggingface.co 则用官网（避免 hf-mirror 高并发 429）；否则回退镜像
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if curl -sf --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then
    HF_ENDPOINT="https://huggingface.co"
  else
    HF_ENDPOINT="https://hf-mirror.com"
  fi
fi
export HF_ENDPOINT
export HF_HOME="${HF_HOME:-/home/dataset-local/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || true

# 国内/不稳定链路：禁用 Xet(CAS/xethub) 大文件通道，改用经典 LFS，减少 read timeout
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

# LiteLLM / OpenAI 兼容客户端（与 configs 里 api_base 一致；可按需改远程）
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
