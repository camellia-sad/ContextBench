# ContextBench 工作交接记录

> 机器：`ide-ba435ad50bfc4bafa81ae392ed179379-006312`  
> 仓库：`/home/dataset-assist-0/vllm_workspace/ContextBench`  
> Git 分支：`backup/wip-before-qwen32-miniswe-20260620`（有未提交改动）  
> 记录日期：2026-08-05  
> 目的：离开本机前，把**做过什么、改了哪些执行函数/工具、产出在哪、踩过什么坑**写清楚，方便后续接续。

---

## 1. 总体在做什么

本机工作围绕 **ContextBench + mini-SWE-agent（miniswe）** 跑 select_500 评测流水线：

```text
启动 vLLM  →  contextbench.run / run_contextbench_miniswe.sh 跑 agent
          →  写出 *.traj.json
          →  extract_trajs/extract_entry.sh 抽 preds / patches
          →  （后续）官方 harness / ContextBench 指标评测
```

近期重点模型：

| 阶段 | 模型 | 输出目录（miniswe 根） |
|------|------|------------------------|
| 较早调试 | Qwen2.5-Coder / 32B 系列 | `output_vllm/select_500_qwen25_32B_coder` 等 |
| 当前主跑 | **Qwen3-8B** | `.../src/output_vllm/select_500_qwen_8b/miniswe/` |

`select_500_qwen_8b` 已有 traj 数量（约）：

- Verified：174  
- Pro：54  
- Poly：116  
- Multi：156  
- 提取合并后：`preds_select500_merged.jsonl` = **500 行**

---

## 2. 本机未提交代码改动一览（`git status`）

当前相对 HEAD 有改动的文件：

| 文件 | 性质 |
|------|------|
| `mini-swe-agent/.../run/extra/swebench_context_aware.py` | **关键**：Multi 数据集本地 jsonl 加载 |
| `mini-swe-agent/.../models/litellm_model.py` | `auto_max_tokens` / 剩余窗口算 `max_tokens` |
| `mini-swe-agent/.../agents/context_aware.py` | 去掉 EXPLORE_CONTEXT 强制重试相关逻辑 |
| `configs/swebench_following_context_strict.yaml` | 默认模型改 Qwen3-8B + auto_max_tokens + thinking |
| `configs/swebench_multi_strict.yaml` | 同上（Multi 专用） |
| `contextbench/run.py` | 默认 `MINISWE_VLLM_MODEL` → Qwen3-8B |
| `tools/start_vllm.sh` | 多模型 profile、YaRN、`--print-cmd` 等 |
| `tools/run_contextbench_miniswe.sh` | 默认超时、自动重试到 Submitted、`--once` |

另有大量 **未跟踪产出**：`preds_merged/`、`miniswe_preds/`、各类 `*_error.txt`、`docker_info.txt`、ckpt json 等。

---

## 3. 重点：修改了哪些「执行函数」（代码级）

### 3.1 Multi-SWE 数据集加载（本会话直接修通的阻塞点）

**文件**：  
`agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/minisweagent/run/extra/swebench_context_aware.py`

**背景问题**：

- `load_dataset("ByteDance-Seed/Multi-SWE-bench")` 会因不同 repo 的嵌套 test 字段 schema 不一致触发  
  `DatasetGenerationError` / Arrow `Couldn't cast ...`
- 环境变量 `HF_HOME` 曾指向 `/home/dataset-local/.hf_cache`，而完整 jsonl 在  
  `/home/dataset-local/hf_cache/hub/datasets--ByteDance-Seed--Multi-SWE-bench/`
- `HF_HUB_ENABLE_HF_TRANSFER=1` 但未装 `hf_transfer` 也会导致下载失败

**新增 / 改写函数**：

1. **`_find_multiswe_local_snapshot(dataset_path)`**  
   - 在 `HUGGINGFACE_HUB_CACHE` / `HF_HOME/hub` /  
     `/home/dataset-local/hf_cache/hub` / `/home/dataset-local/.hf_cache/hub` 等路径中  
     查找 `datasets--ByteDance-Seed--Multi-SWE-bench/snapshots/*/.../*.jsonl`

2. **`_load_multiswe_from_local_jsonl(snapshot_dir)`**  
   - 直接 `json.loads` 读所有 jsonl  
   - 每行调用已有的 **`_simplify_multiswe_instance`**（去掉易炸的 nested test 字段）

3. **`_load_multiswe_dataset_safely(dataset_path, split)`**（重写优先级）  
   - **优先本地 jsonl**  
   - 失败才回退 Hub `streaming` / 普通 `load_dataset`  
   - 本地路径验证可加载 **1632** 条

**调用点**：`main()` 里 `BenchmarkType.MULTI_SWE` 分支仍调用 `_load_multiswe_dataset_safely`。

---

### 3.2 LiteLLM 调用侧：`auto_max_tokens`（防超长生成 / 超窗）

**文件**：  
`.../minisweagent/models/litellm_model.py`

**新增逻辑**：

- `_LOCAL_MODEL_KWARG_KEYS`：本地消费、不转发给 OpenAI API 的键  
  （`auto_max_tokens`, `max_model_len`, `max_tokens_reserve`, `max_tokens_cap`）
- `_estimate_prompt_tokens(...)`：用 litellm.token_counter，失败则 `chars//4`
- **`LitellmModel._prepare_completion_kwargs(...)`**：  
  当 `auto_max_tokens=true` 时：  
  `max_tokens = max_model_len - prompt_tokens - reserve`，并可选 `max_tokens_cap` 软截断

**动机（与本机现象对应）**：

- 长 traj 多轮后 context 膨胀，单次 completion 若 `max_tokens` 很大 → vLLM `Running: 1 reqs`、KV cache 持续上涨、`verified_error.log` 长时间无新输出  
- `max_tokens_cap: 4096` 用来限制单步生成长度

**环境变量后备**：`MSWEA_AUTO_MAX_TOKENS`、`VLLM_MAX_MODEL_LEN`、`MSWEA_MAX_TOKENS_RESERVE`、`MSWEA_MAX_TOKENS_CAP`

---

### 3.3 ContextAwareAgent：去掉 EXPLORE_CONTEXT 强制执法

**文件**：  
`.../minisweagent/agents/context_aware.py`

**删除 / 弱化**：

- `_bash_prints_file_content`、`_explore_violation_user_message` 等一系列「读文件必须带 EXPLORE_CONTEXT」的强制校验  
- 配置项 `explore_context_retry_limit`  
- 异常类 `ExploreContextEnforcementExceeded`

**含义**：agent 仍可按 prompt 写 `<EXPLORE_CONTEXT>`，但运行时不再因格式问题连续拦截并终止实例。  
（ContextBench 提取 traj 时仍优先解析标签；没有标签才从 bash 猜测，见第 7 节。）

---

### 3.4 PolyBench Docker 镜像策略（已有代码，本会话诊断过）

**相关**：

- `PolyBenchStrategy.get_docker_config`：镜像名  
  `ghcr.io/timesler/swe-polybench.eval.x86_64.<instance_id.lower()>:latest`
- `apply_registry_mirror_prefix`（`docker_image_registry.py`）+  
  `contextbench/run.py` 设置 `MSWEA_DOCKER_IMAGE_REGISTRY=fczi514j9ggm7b.xuanyuan.run`

**本会话结论**：

- Poly 报错核心是镜像站 **404 manifest unknown**（不是 Docker daemon 挂掉）  
- 直连 GHCR 的 manifest 可能存在；xuanyuan 前缀路径未同步时需 **pull + tag** 或 **save/load**

---

## 4. Tools / 脚本：改了什么、用了什么

### 4.1 `tools/start_vllm.sh`（已大幅扩展）

用途：在 GPU / Docker 内启动 OpenAI 兼容 vLLM（`:8000`）。

当前能力要点：

- 默认模型：`Qwen/Qwen3-8B`
- 多 profile：`Qwen3-8B` / `Qwen2.5-32B-Instruct` / Codestral / `facebook/cwm` / 其它
- YaRN：`max_model_len` > native 时自动 `--hf-overrides` rope_scaling
- `--print-cmd` / `-p`：只打印可复制命令不 exec
- 依赖安装、HF 预下载、离线缓存探测等

相关：`tools/env_vllm_chat.sh`、`tools/start_vllm_tmux.sh`

**本会话诊断过的旧短配置问题**：早期超长 `--max-model-len 131072` + 无单步 cap → 单请求长时间 decode、日志像卡死。

---

### 4.2 `tools/run_contextbench_miniswe.sh`（批跑入口）

相对旧版增强：

- 默认整实例超时 `DEFAULT_TIMEOUT=10800`，单步 `DEFAULT_STEP_TIMEOUT=1800`
- **默认循环重试**直到 `info.exit_status=Submitted`（可用 `--once` 关闭）
- 解析 traj 根：`$MINISWE_OUTPUT_ROOT/$OUTPUT/miniswe/$BENCH`
- 默认模型提示改为 Qwen3-8B

底层仍调用：`python -m contextbench.run --agent miniswe ...`

---

### 4.3 `extract_trajs/`（轨迹提取，本会话讲透并已对 qwen-8b 跑通）

入口：

```bash
./extract_trajs/extract_entry.sh \
  --model qwen-8b \
  --output-run output_vllm/select_500_qwen_8b
```

两步：

1. **`build_preds_per_bench.py`** → `contextbench.process_trajectories.convert`  
   - 产出 `preds_merged/<model>/preds_{Verified,Pro,Poly,Multi}.jsonl`  
   - + `preds_select500_merged.jsonl` + `build_preds_manifest.json`
2. **`miniswe_eval.py`**（不带 `--run-eval`）  
   - 产出官方 harness 格式 patches 到 `miniswe_preds/<model-slug>/`

**注意**：Verified 的 `model_name_or_path`、Pro 的 `prefix` 写死为 `"mini_swe_agent"`（标 agent 系统，不是 HF 模型名）。

---

### 4.4 其它常用 tools（目录内已有，本机常用）

| 脚本 | 作用 |
|------|------|
| `tools/analyze_traj_info.py` | 分析 traj / 与 CSV 对比 |
| `tools/traj_recover_submission.py` | 从 traj 恢复 submission |
| `tools/pull_*docker_images*.sh` | 按 CSV/traj 拉评测镜像 |
| `tools/start_docker.sh` | Docker 相关启动 |
| `docker_image_registry.py` | xuanyuan 镜像前缀 |

---

## 5. 本会话排查过的运行时问题（结论摘要）

### 5.1 「log 不刷 + KV cache 上涨」

- **不是** agent 死锁；Python 在等 HTTP；vLLM `Running:1` 持续 decode  
- 原因：多轮历史上下文变长 + 单次 `max_tokens` 过大  
- 缓解方向：`auto_max_tokens` + `max_tokens_cap`、必要时减小 `max_model_len`、控制并发

### 5.2 Poly Docker `manifest unknown`

- 拉的是：  
  `fczi514j9ggm7b.xuanyuan.run/ghcr.io/timesler/swe-polybench.eval.x86_64.<id>:latest`  
- 404 = 镜像站没有该 tag；需直连 GHCR 后 `docker tag` 成带前缀名，或 `save/load`

### 5.3 Multi `DatasetGenerationError`

- 根因：Arrow schema 冲突，不是单纯断网  
- **已用本地 jsonl 加载修通**（见 §3.1）

### 5.4 HF 缓存路径

| 路径 | 说明 |
|------|------|
| `/home/dataset-local/hf_cache/hub` | Multi-SWE jsonl 完整缓存所在 |
| `/home/dataset-local/.hf_cache` | 另一套 HF_HOME，曾导致「以为有缓存其实没有」 |

建议跑 Multi 前：

```bash
export HF_HOME=/home/dataset-local/hf_cache
export HUGGINGFACE_HUB_CACHE=/home/dataset-local/hf_cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ENDPOINT=https://hf-mirror.com   # 若仍需联网
# 本地加载已优先，可不依赖 Hub
```

---

## 6. 产出物位置（可带走/可 rsync）

### 6.1 原始 traj

```text
agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/output_vllm/select_500_qwen_8b/miniswe/
  Verified|Pro|Poly|Multi/<instance_id>/<instance_id>.traj.json
```

### 6.2 ContextBench 统一 preds

```text
preds_merged/qwen-8b/
  preds_Verified.jsonl
  preds_Pro.jsonl
  preds_Poly.jsonl
  preds_Multi.jsonl
  preds_select500_merged.jsonl
  build_preds_manifest.json
```

每行大致：

```json
{
  "instance_id": "...",
  "model_patch": "<git diff>",
  "traj_data": {
    "pred_steps": [{"files": [...], "spans": {...}}, ...],
    "pred_files": [...],
    "pred_spans": {"path": [{"type":"line","start":1,"end":50}, ...]}
  }
}
```

### 6.3 官方 harness patches

```text
miniswe_preds/qwen-8b/
  swebench_verified_patches_from_miniswe.jsonl
  poly_patches_from_miniswe.jsonl
  pro_patches_from_miniswe.json
  multi_patches_from_miniswe.jsonl
```

### 6.4 调试日志（历史）

- `verified_error.log` / `verified_error.txt`  
- `poly_error.txt` / `pro_error.txt` / `docker_info.txt`  
- 各类 `mini_swe_agent.qwen3_8b_ckpt60_b*.json`

---

## 7. traj_data 提取规则（接续评测必懂）

实现：`contextbench/agents/minisweagent/extract.py`

1. **步骤 `pred_steps`**：优先解析消息里的 `<explore_context>` / `<EXPLORE_CONTEXT>`（File + Lines）  
2. **若整条 traj 完全没有该标签**：再从 assistant 的 `` ```bash `` `` 里用正则猜 `nl|sed -n|head|cat|grep` 等读文件命令  
3. **最终 `pred_files` / `pred_spans`**：只认 `<PATCH_CONTEXT>`，bash fallback **不影响**最终上下文  
4. **`model_patch`**：来自 traj `info.submission`（git diff），与 traj_data 独立——可能有步骤无 patch，或有 patch 无 explore 标签

---

## 8. 推荐接续操作清单

1. **提交或备份当前 git 改动**（尤其是 Multi 本地加载与 litellm auto_max_tokens），避免丢在 WIP 分支。  
2. 若换机器跑 Multi：确保拷贝  
   `/home/dataset-local/hf_cache/hub/datasets--ByteDance-Seed--Multi-SWE-bench/`  
   或依赖已改过的本地 jsonl 加载逻辑。  
3. 评测：把 `preds_merged/qwen-8b/` 与 `miniswe_preds/qwen-8b/` 拷到评测机，再跑各 harness / ContextBench evaluate。  
4. Poly 镜像：按需对失败 instance 做 `docker pull ghcr.io/...` + `docker tag` 到 xuanyuan 前缀名。  
5. 启动模型：优先用当前 `tools/start_vllm.sh`；agent 默认已指向 Qwen3-8B。

---

## 9. 关键命令速查

```bash
# 启动 vLLM（容器/GPU 机）
bash tools/start_vllm.sh
# 或只看命令
bash tools/start_vllm.sh --print-cmd

# 跑某一 bench
tools/run_contextbench_miniswe.sh \
  --bench Multi \
  --output output_vllm/select_500_qwen_8b \
  --workers 2 \
  --rerun true

# 提取 traj → preds/patches
./extract_trajs/extract_entry.sh \
  --model qwen-8b \
  --output-run output_vllm/select_500_qwen_8b

# 验证 Multi 本地加载
cd agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src
python -c "from minisweagent.run.extra.swebench_context_aware import _load_multiswe_dataset_safely as f; print(len(f('ByteDance-Seed/Multi-SWE-bench','train')))"
```

---

## 10. 一句话总结

本机把 **Qwen3-8B + miniswe select_500** 跑通并完成 **traj→preds 提取**；代码上最关键的执行侧改动是：  
**(1) Multi 改为优先本地 jsonl 加载**；**(2) LiteLLM 增加 auto_max_tokens / cap 防长请求卡死**；**(3) 启动/批跑脚本与默认模型切到 Qwen3-8B**。  
评测输入已在 `preds_merged/qwen-8b/` 与 `miniswe_preds/qwen-8b/`。
