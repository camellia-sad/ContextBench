#!/usr/bin/env python3
"""
从 mini-swe-agent 产生的 traj.json 中提取 submission（git diff 补丁），
并按不同 benchmark 的官方评测脚本期望的格式生成预测文件，
同时提供一个统一入口可调用各 benchmark 的评测脚本。

支持的 benchmark 及映射：

- Multi-SWE-bench
  - traj 目录:  <traj_root>/Multi/<org>__<repo>-<number>/<same>.traj.json
  - 预测文件:   JSONL，每行:
        {"org": "...", "repo": "...", "number": 1234, "fix_patch": "<git diff>"}
  - 对应评测脚本:
        python -m multi_swe_bench.harness.run_evaluation \
          --mode evaluation \
          --patch_files PATCH_JSONL \
          --dataset_files DATA_JSONL ...

- SWE-PolyBench
  - traj 目录:  <traj_root>/Poly/<instance_id>/<instance_id>.traj.json
  - 预测文件:   JSONL，每行:
        {"instance_id": "<instance_id>", "model_patch": "<git diff>"}
  - 对应评测脚本:
        python -m poly_bench_evaluation.run_evaluation \
          --dataset-path DATASET \
          --predictions-path PRED_JSONL ...

- SWE-bench_Pro
  - traj 目录:  <traj_root>/Pro/<instance_id>/<instance_id>.traj.json
    其中 instance_id 形如:
      instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan
  - 预测文件:   JSON 数组:
        [
          {
            "instance_id": "<instance_id>",
            "patch": "<git diff>",
            "prefix": "mini_swe_agent"
          },
          ...
        ]
  - 对应评测脚本:
        python swe_bench_pro_eval.py \
          --raw_sample_path DATA_CSV \
          --patch_path PATCH_JSON \
          --output_dir OUTPUT_DIR \
          --scripts_dir run_scripts \
          --num_workers 100 \
          --dockerhub_username YOUR_DOCKERHUB \
          [--use_local_docker]

- SWE-bench-verified
  - traj 目录:  <traj_root>/Verified/<instance_id>/<instance_id>.traj.json
  - 预测文件:   JSONL，每行:
        {"instance_id": "<instance_id>", "model_patch": "<git diff>"}
  - 对应评测脚本:
        python -m swebench.harness.run_evaluation \
          --dataset_name SWE-bench/SWE-bench_Verified \
          --split test \
          --predictions_path PRED_JSONL \
          --max_workers N \
          --run_id RUN_ID

本脚本只做两件事：
1. 从 traj 里抽取 submission，生成各 benchmark 需要的预测/补丁文件；
2. （可选）调用各自评测脚本，把生成的预测文件喂进去。

用法示例：

  # 仅从 traj 生成三个 benchmark 的预测文件
  python miniswe_eval.py \
    --traj-root ./path/to/miniswe_traj_root \
    --benchmark multi poly pro verified \
    --output-dir ./miniswe_preds

  # 生成 Multi-SWE-bench 补丁并直接评测
  python miniswe_eval.py \
    --traj-root ./path/to/miniswe_traj_root \
    --benchmark multi \
    --output-dir ./miniswe_preds \
    --run-eval \
    --multi-dataset-files /path/to/Multi-SWE-bench_mini.jsonl \
    --multi-workdir /tmp/multi-workdir \
    --multi-output-dir /tmp/multi-output \
    --multi-repo-dir ./path/to/multi_repos

  # 生成 Poly 补丁并评测
  python miniswe_eval.py \
    --traj-root ./path/to/miniswe_traj_root \
    --benchmark poly \
    --output-dir ./miniswe_preds \
    --run-eval \
    --poly-dataset-path /path/to/SWE-PolyBench_500.jsonl \
    --poly-result-path /tmp/poly-result \
    --poly-repo-path ./path/to/poly_repos

  # 生成 Pro 补丁并评测
  python miniswe_eval.py \
    --traj-root ./path/to/miniswe_traj_root \
    --benchmark pro \
    --output-dir ./miniswe_preds \
    --run-eval \
    --pro-raw-sample-path /path/to/data.csv \
    --pro-output-dir /tmp/pro-output \
    --pro-scripts-dir ./path/to/pro_run_scripts \
    --pro-dockerhub-username your_dockerhub_name \
    --pro-num-workers 64 \
    --pro-use-local-docker

  # 生成 SWE-bench-verified 补丁并评测
  python miniswe_eval.py \
    --traj-root ./path/to/miniswe_traj_root \
    --benchmark verified \
    --output-dir ./miniswe_preds \
    --run-eval \
    --verified-run-id miniswe_verified_run \
    --verified-max-workers 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BENCH_MULTI = "multi"
BENCH_POLY = "poly"
BENCH_PRO = "pro"
BENCH_VERIFIED = "verified"

# Benchmark harness outputs default under this repo (not system /tmp).
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_RUNS = REPO_ROOT / "eval_runs"

# Default upstream harness clones: siblings of ContextBench (same parent directory).
# Override with --multi-root / --poly-root / etc. if your layout differs.
_UPSTREAM_SIBLING = REPO_ROOT.parent
DEFAULT_MULTI_ROOT = _UPSTREAM_SIBLING / "multi-swe-bench"
DEFAULT_POLY_ROOT = _UPSTREAM_SIBLING / "SWE-PolyBench"
DEFAULT_POLY_REPO_PATH = DEFAULT_POLY_ROOT / "repos"
DEFAULT_PRO_ROOT = _UPSTREAM_SIBLING / "SWE-bench_Pro-os"
DEFAULT_VERIFIED_ROOT = _UPSTREAM_SIBLING / "SWE-bench"


@dataclass
class TrajSubmission:
    instance_id: str
    submission: str
    exit_status: str
    traj_path: Path


def load_traj_submission(path: Path) -> Optional[TrajSubmission]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load traj {path}: {exc}")
        return None

    info = data.get("info") or {}
    submission = info.get("submission") or ""
    exit_status = info.get("exit_status") or ""
    instance_id = data.get("instance_id") or path.stem

    # 即使 submission 为空，也保留记录；后续可以决定是否跳过空补丁。
    return TrajSubmission(
        instance_id=instance_id,
        submission=submission,
        exit_status=exit_status,
        traj_path=path,
    )


def iter_traj_submissions(traj_dir: Path) -> Iterable[TrajSubmission]:
    """
    遍历给定目录下的所有 *.traj.json，产出 TrajSubmission。
    假设结构为：
      traj_dir/
        <instance_id>/
          <instance_id>.traj.json
        ...
    """
    if not traj_dir.exists():
        return

    for child in sorted(traj_dir.iterdir()):
        if not child.is_dir():
            continue
        # 跳过 exit_statuses_xxx.yaml 等文件夹名以外的条目
        traj_files = list(child.glob("*.traj.json"))
        if not traj_files:
            continue
        traj_path = traj_files[0]
        ts = load_traj_submission(traj_path)
        if ts is None:
            continue
        yield ts


# ---------------- Multi-SWE-bench ----------------


def build_multi_predictions_from_traj(traj_root: Path, output_dir: Path) -> Path:
    """
    从 <traj_root>/Multi 提取 submission，生成 Multi-SWE-bench 所需 JSONL。
    每行：
      {"org": "...", "repo": "...", "number": 1234, "fix_patch": "<git diff>"}
    """
    multi_dir = traj_root / "Multi"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "multi_patches_from_miniswe.jsonl"

    records: List[Dict[str, object]] = []

    for ts in iter_traj_submissions(multi_dir):
        # 目录名 / instance_id 形如 alibaba__fastjson2-2775
        # 解析 org, repo, number
        name = ts.instance_id
        # 某些 run 中 instance_id 可能不带 org__repo-；此时直接跳过。
        if "__" not in name or "-" not in name:
            print(f"[WARN] Unexpected Multi instance_id format, skip: {name}")
            continue
        org_repo, num_str = name.rsplit("-", 1)
        org, repo = org_repo.split("__", 1)
        try:
            number = int(num_str)
        except ValueError:
            print(f"[WARN] Cannot parse PR number from {name}, skip")
            continue

        patch = ts.submission or ""
        rec = {
            "org": org,
            "repo": repo,
            "number": number,
            "fix_patch": patch,
        }
        records.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Multi-SWE-bench predictions written to {out_path} ({len(records)} instances)")
    return out_path


def run_multi_evaluation(
    swe_multi_root: Path,
    patch_file: Path,
    dataset_files: Sequence[str],
    workdir: Path,
    output_dir: Path,
    repo_dir: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    """
    调用 multi_swe_bench.harness.run_evaluation 进行评测。

    相当于：
      python -m multi_swe_bench.harness.run_evaluation \
        --mode evaluation \
        --patch_files PATCH \
        --dataset_files DATA ... \
        --workdir WORKDIR \
        --output_dir OUTPUT_DIR \
        --log_dir LOG_DIR \
        [--repo_dir REPO_DIR] \
        [extra_args...]

    ``LOG_DIR`` 默认为 ``<output_dir>/logs``（上游 harness 要求 ``--log_dir`` 不能省略）。

    会对 ``patch_file`` / ``workdir`` / ``output_dir`` 做 ``resolve()``，避免同一仓库在
    ``/home/...`` 与 ``/ssd-disk/...`` 等多挂载点下出现两套评测目录或日志分裂。
    """
    patch_file = patch_file.expanduser().resolve()
    workdir = workdir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    cmd: List[str] = [
        sys.executable,
        "-m",
        "multi_swe_bench.harness.run_evaluation",
        "--mode",
        "evaluation",
        "--patch_files",
        str(patch_file),
        "--dataset_files",
    ]
    cmd.extend(str(Path(p).expanduser().resolve()) for p in dataset_files)
    log_dir = (output_dir / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(
        [
            "--workdir",
            str(workdir),
            "--output_dir",
            str(output_dir),
            "--log_dir",
            str(log_dir),
        ]
    )
    if repo_dir is not None:
        cmd.extend(["--repo_dir", str(Path(repo_dir).expanduser().resolve())])
    if extra_args:
        cmd.extend(list(extra_args))

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running Multi-SWE-bench eval:\n  {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(swe_multi_root))


# ---------------- SWE-PolyBench ----------------


def build_poly_predictions_from_traj(traj_root: Path, output_dir: Path) -> Path:
    """
    从 <traj_root>/Poly 提取 submission，生成 SWE-PolyBench 所需 JSONL。
    每行：
      {"instance_id": "<instance_id>", "model_patch": "<git diff>"}
    """
    poly_dir = traj_root / "Poly"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "poly_patches_from_miniswe.jsonl"

    records: List[Dict[str, object]] = []

    for ts in iter_traj_submissions(poly_dir):
        patch = ts.submission or ""
        rec = {
            "instance_id": ts.instance_id,
            "model_patch": patch,
        }
        records.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] SWE-PolyBench predictions written to {out_path} ({len(records)} instances)")
    return out_path


def _validate_poly_dataset_path(dataset_path: Path) -> None:
    """Reject Multi (or annotation-only) JSONL passed by mistake as --poly-dataset-path."""
    p = dataset_path.expanduser().resolve()
    if not p.is_file():
        print(f"[ERROR] Poly --poly-dataset-path not found: {p}", file=sys.stderr)
        sys.exit(2)
    first_line = ""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                first_line = line.strip()
                break
    if not first_line:
        print(f"[ERROR] Poly dataset file is empty: {p}", file=sys.stderr)
        sys.exit(2)
    try:
        obj: object = json.loads(first_line)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Poly dataset is not valid JSONL (line 1): {e}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(obj, dict):
        print("[ERROR] Poly dataset first line must be a JSON object.", file=sys.stderr)
        sys.exit(2)
    if "org" in obj and "repo" in obj and "number" in obj:
        print(
            "[ERROR] --poly-dataset-path points at Multi-SWE-bench-style JSONL "
            f"(org/repo/number): {p}\n"
            "Poly needs SWE-PolyBench task JSONL (columns include language, patch, Dockerfile, …).\n"
            "Example: .../ContextBench/hf_csv/poly_patches_from_miniswe_local_subset.jsonl\n"
            "Generate with: tools/export_poly_hf_subset_from_patches.py "
            "--dataset AmazonScience/SWE-PolyBench --split test\n"
            "For Multi evaluation use --multi-dataset-files, not --poly-dataset-path.",
            file=sys.stderr,
        )
        sys.exit(2)
    if "language" not in obj or "patch" not in obj:
        keys = sorted(obj.keys())
        print(
            "[ERROR] --poly-dataset-path is not SWE-PolyBench task JSONL "
            "(first row must include 'language' and 'patch').\n"
            f"  Path: {p}\n"
            f"  Keys (sample): {keys[:30]}{' …' if len(keys) > 30 else ''}\n"
            "Do not use data/annotations.jsonl or multi-swe-bench-dataset.jsonl here.",
            file=sys.stderr,
        )
        sys.exit(2)


def run_poly_evaluation(
    swe_poly_root: Path,
    predictions_path: Path,
    dataset_path: Path,
    result_path: Path,
    repo_path: Optional[Path] = None,
    num_threads: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    """
    调用 poly_bench_evaluation.run_evaluation 进行评测。

    相当于：
      python -m poly_bench_evaluation.run_evaluation \
        --dataset-path DATASET \
        --predictions-path PRED_JSONL \
        --result-path RESULT_PATH \
        [--repo-path REPO_PATH] \
        [--num-threads N] \
        [extra_args...]
    """
    _validate_poly_dataset_path(dataset_path)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "poly_bench_evaluation.run_evaluation",
        "--dataset-path",
        str(dataset_path.expanduser().resolve()),
        "--predictions-path",
        str(predictions_path),
        "--result-path",
        str(result_path),
    ]
    if repo_path is not None:
        cmd.extend(["--repo-path", str(repo_path)])
    if num_threads is not None:
        cmd.extend(["--num-threads", str(num_threads)])
    if extra_args:
        cmd.extend(list(extra_args))

    result_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running SWE-PolyBench eval:\n  {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(swe_poly_root))


# ---------------- SWE-bench Pro ----------------


def build_pro_predictions_from_traj(traj_root: Path, output_dir: Path) -> Path:
    """
    从 <traj_root>/Pro 提取 submission，生成 SWE-bench Pro 所需 JSON 数组。
    结构：
      [
        {"instance_id": "...", "patch": "<git diff>", "prefix": "mini_swe_agent"},
        ...
      ]
    """
    pro_dir = traj_root / "Pro"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pro_patches_from_miniswe.json"

    records: List[Dict[str, object]] = []

    for ts in iter_traj_submissions(pro_dir):
        patch = ts.submission or ""
        rec = {
            "instance_id": ts.instance_id,
            "patch": patch,
            "prefix": "mini_swe_agent",
        }
        records.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[INFO] SWE-bench Pro patches written to {out_path} ({len(records)} instances)")
    return out_path


def run_pro_evaluation(
    swe_pro_root: Path,
    patch_path: Path,
    raw_sample_path: Path,
    output_dir: Path,
    scripts_dir: Path,
    dockerhub_username: str,
    num_workers: int = 100,
    use_local_docker: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    """
    调用 swe_bench_pro_eval.py 进行评测。

    相当于：
      python swe_bench_pro_eval.py \
        --raw_sample_path DATA_CSV \
        --patch_path PATCH_JSON \
        --output_dir OUTPUT_DIR \
        --scripts_dir run_scripts \
        --num_workers N \
        --dockerhub_username USER \
        [--use_local_docker] \
        [extra_args...]
    """
    cmd: List[str] = [
        sys.executable,
        "swe_bench_pro_eval.py",
        "--raw_sample_path",
        str(raw_sample_path),
        "--patch_path",
        str(patch_path),
        "--output_dir",
        str(output_dir),
        "--scripts_dir",
        str(scripts_dir),
        "--num_workers",
        str(num_workers),
        "--dockerhub_username",
        dockerhub_username,
    ]
    if use_local_docker:
        cmd.append("--use_local_docker")
    if extra_args:
        cmd.extend(list(extra_args))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running SWE-bench Pro eval:\n  {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(swe_pro_root))


# ---------------- SWE-bench Verified ----------------


def build_verified_predictions_from_traj(traj_root: Path, output_dir: Path) -> Path:
    """
    从 <traj_root>/Verified 提取 submission，生成 SWE-bench Verified 所需 JSONL。
    每行：
      {"instance_id": "<instance_id>", "model_patch": "<git diff>"}
    """
    verified_dir = traj_root / "Verified"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "swebench_verified_patches_from_miniswe.jsonl"

    records: List[Dict[str, object]] = []
    for ts in iter_traj_submissions(verified_dir):
        records.append(
            {
                "instance_id": ts.instance_id,
                "model_patch": ts.submission or "",
                "model_name_or_path": "mini_swe_agent",
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] SWE-bench Verified predictions written to {out_path} ({len(records)} instances)")
    return out_path


def run_verified_evaluation(
    swebench_root: Path,
    predictions_path: Path,
    run_id: str,
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
    max_workers: int = 4,
    namespace: Optional[str] = "swebench",
    timeout: int = 1800,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    """
    调用 SWE-bench 官方 harness 评测 Verified。

    相当于：
      python -m swebench.harness.run_evaluation \
        --dataset_name SWE-bench/SWE-bench_Verified \
        --split test \
        --predictions_path PRED_JSONL \
        --max_workers N \
        --run_id RUN_ID \
        [--namespace ...] \
        [--timeout ...]
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
    ]
    if namespace is not None:
        cmd.extend(["--namespace", namespace])
    if extra_args:
        cmd.extend(list(extra_args))

    print(f"[INFO] Running SWE-bench Verified eval:\n  {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(swebench_root))


# ---------------- CLI ----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract patches from mini-swe-agent trajs and run multi-benchmark evaluations.",
    )
    parser.add_argument(
        "--traj-root",
        type=Path,
        default=None,
        help="Root directory of mini-swe-agent trajectories (required unless --eval-only).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="+",
        choices=[BENCH_MULTI, BENCH_POLY, BENCH_PRO, BENCH_VERIFIED],
        required=True,
        help="Which benchmarks to process: multi, poly, pro, verified (can be multiple).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write generated prediction/patch files (required unless --eval-only).",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="If set, also call each benchmark's official evaluation script.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip reading traj; use --multi-patch-file / --poly-patch-file / --pro-patch-file / "
        "--verified-patch-file for each selected --benchmark, then run evaluation (implies --run-eval).",
    )
    parser.add_argument(
        "--multi-patch-file",
        type=Path,
        default=None,
        help="Existing Multi-SWE-bench JSONL (--eval-only for benchmark multi).",
    )
    parser.add_argument(
        "--poly-patch-file",
        type=Path,
        default=None,
        help="Existing SWE-PolyBench predictions JSONL (--eval-only for benchmark poly).",
    )
    parser.add_argument(
        "--pro-patch-file",
        type=Path,
        default=None,
        help="Existing SWE-bench Pro patch JSON (--eval-only for benchmark pro).",
    )
    parser.add_argument(
        "--verified-patch-file",
        type=Path,
        default=None,
        help="Existing SWE-bench Verified predictions JSONL (--eval-only for benchmark verified).",
    )

    # Multi-SWE-bench eval options
    parser.add_argument(
        "--multi-root",
        type=Path,
        default=DEFAULT_MULTI_ROOT,
        help=f"Root directory of multi-swe-bench repo (default: {DEFAULT_MULTI_ROOT}).",
    )
    parser.add_argument(
        "--multi-dataset-files",
        type=str,
        nargs="*",
        help="Dataset JSONL files for Multi-SWE-bench (paths passed to --dataset_files).",
    )
    parser.add_argument(
        "--multi-workdir",
        type=Path,
        default=DEFAULT_EVAL_RUNS / "multi" / "workdir",
        help="Workdir for Multi-SWE-bench evaluation (default: <repo>/eval_runs/multi/workdir).",
    )
    parser.add_argument(
        "--multi-output-dir",
        type=Path,
        default=DEFAULT_EVAL_RUNS / "multi" / "output",
        help="Output directory for Multi-SWE-bench evaluation (default: <repo>/eval_runs/multi/output).",
    )
    parser.add_argument(
        "--multi-repo-dir",
        type=Path,
        default=None,
        help="Repo directory for Multi-SWE-bench (--repo_dir). Layout must be <repo_dir>/<org>/<repo> "
        "(not ContextBench/repos github.com__* layout). Default: <multi-root>/repos (created if missing).",
    )
    parser.add_argument(
        "--multi-specifics",
        type=str,
        nargs="*",
        default=None,
        help="Forwarded to multi-swe-bench --specifics: only run dataset/instance ids that match "
        '(substring match). Example: --multi-specifics "vuejs/core:pr-8538".',
    )
    parser.add_argument(
        "--multi-skips",
        type=str,
        nargs="*",
        default=None,
        help="Forwarded to multi-swe-bench --skips (substring match on instance id).",
    )

    # SWE-PolyBench eval options
    parser.add_argument(
        "--poly-root",
        type=Path,
        default=DEFAULT_POLY_ROOT,
        help=f"Root directory of SWE-PolyBench repo (default: {DEFAULT_POLY_ROOT}).",
    )
    parser.add_argument(
        "--poly-dataset-path",
        type=Path,
        help="Dataset path for SWE-PolyBench (--dataset-path).",
    )
    parser.add_argument(
        "--poly-result-path",
        type=Path,
        default=DEFAULT_EVAL_RUNS / "poly" / "result",
        help="Result directory for SWE-PolyBench (--result-path) (default: <repo>/eval_runs/poly/result).",
    )
    parser.add_argument(
        "--poly-repo-path",
        type=Path,
        default=DEFAULT_POLY_REPO_PATH,
        help=f"Repo path for SWE-PolyBench evaluation (--repo-path) (default: {DEFAULT_POLY_REPO_PATH}).",
    )
    parser.add_argument(
        "--poly-num-threads",
        type=int,
        help="Number of threads for SWE-PolyBench (--num-threads).",
    )

    # SWE-bench Pro eval options
    parser.add_argument(
        "--pro-root",
        type=Path,
        default=DEFAULT_PRO_ROOT,
        help=f"Root directory of SWE-bench_Pro-os repo (default: {DEFAULT_PRO_ROOT}).",
    )
    parser.add_argument(
        "--pro-raw-sample-path",
        type=Path,
        help="CSV file containing SWE-bench Pro samples (--raw_sample_path).",
    )
    parser.add_argument(
        "--pro-output-dir",
        type=Path,
        default=DEFAULT_EVAL_RUNS / "pro" / "output",
        help="Output directory for SWE-bench Pro evaluation (--output_dir) (default: <repo>/eval_runs/pro/output).",
    )
    parser.add_argument(
        "--pro-scripts-dir",
        type=Path,
        default=None,
        help="Scripts directory for SWE-bench Pro (--scripts_dir). Default: <pro-root>/run_scripts.",
    )
    parser.add_argument(
        "--pro-dockerhub-username",
        type=str,
        default="jefzda",
        help="Docker Hub username for SWE-bench Pro images (default: jefzda, official sweap-images).",
    )
    parser.add_argument(
        "--pro-num-workers",
        type=int,
        default=100,
        help="Number of workers for SWE-bench Pro evaluation (--num_workers).",
    )
    parser.add_argument(
        "--pro-use-local-docker",
        action="store_true",
        help="Use local Docker instead of Modal for SWE-bench Pro (--use_local_docker).",
    )

    # SWE-bench verified eval options
    parser.add_argument(
        "--verified-root",
        type=Path,
        default=DEFAULT_VERIFIED_ROOT,
        help=f"Root directory of SWE-bench repo (default: {DEFAULT_VERIFIED_ROOT}).",
    )
    parser.add_argument(
        "--verified-dataset-name",
        type=str,
        default="SWE-bench/SWE-bench_Verified",
        help="Dataset name for SWE-bench harness (--dataset_name).",
    )
    parser.add_argument(
        "--verified-split",
        type=str,
        default="test",
        help="Dataset split for SWE-bench harness (--split).",
    )
    parser.add_argument(
        "--verified-run-id",
        type=str,
        default="miniswe_verified_eval",
        help="Run id for SWE-bench harness (--run_id).",
    )
    parser.add_argument(
        "--verified-max-workers",
        type=int,
        default=4,
        help="Number of workers for SWE-bench verified evaluation (--max_workers).",
    )
    parser.add_argument(
        "--verified-timeout",
        type=int,
        default=1800,
        help="Per-instance timeout seconds for SWE-bench verified evaluation (--timeout).",
    )
    parser.add_argument(
        "--verified-namespace",
        type=str,
        default="swebench",
        help='Docker image namespace for SWE-bench harness (--namespace). Set to "none" to disable namespace.',
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.pro_scripts_dir is None:
        args.pro_scripts_dir = Path(args.pro_root) / "run_scripts"

    if args.eval_only:
        args.run_eval = True

    if not args.eval_only and args.traj_root is None:
        print("[ERROR] --traj-root is required unless --eval-only.")
        return 2

    if not args.eval_only and args.output_dir is None:
        print("[ERROR] --output-dir is required unless --eval-only.")
        return 2

    traj_root: Optional[Path] = args.traj_root
    output_dir: Optional[Path] = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    bench_set = set(args.benchmark)

    multi_patch_path: Optional[Path] = None
    poly_patch_path: Optional[Path] = None
    pro_patch_path: Optional[Path] = None
    verified_patch_path: Optional[Path] = None

    if args.eval_only:
        if BENCH_MULTI in bench_set:
            multi_patch_path = args.multi_patch_file
            if multi_patch_path is None or not multi_patch_path.is_file():
                print("[ERROR] --eval-only with benchmark multi requires existing --multi-patch-file.")
                return 2
        if BENCH_POLY in bench_set:
            poly_patch_path = args.poly_patch_file
            if poly_patch_path is None or not poly_patch_path.is_file():
                print("[ERROR] --eval-only with benchmark poly requires existing --poly-patch-file.")
                return 2
        if BENCH_PRO in bench_set:
            pro_patch_path = args.pro_patch_file
            if pro_patch_path is None or not pro_patch_path.is_file():
                print("[ERROR] --eval-only with benchmark pro requires existing --pro-patch-file.")
                return 2
        if BENCH_VERIFIED in bench_set:
            verified_patch_path = args.verified_patch_file
            if verified_patch_path is None or not verified_patch_path.is_file():
                print("[ERROR] --eval-only with benchmark verified requires existing --verified-patch-file.")
                return 2
    else:
        assert traj_root is not None
        assert output_dir is not None
        if BENCH_MULTI in bench_set:
            multi_patch_path = build_multi_predictions_from_traj(traj_root, output_dir)

        if BENCH_POLY in bench_set:
            poly_patch_path = build_poly_predictions_from_traj(traj_root, output_dir)

        if BENCH_PRO in bench_set:
            pro_patch_path = build_pro_predictions_from_traj(traj_root, output_dir)

        if BENCH_VERIFIED in bench_set:
            verified_patch_path = build_verified_predictions_from_traj(traj_root, output_dir)

    if not args.run_eval:
        # 只生成补丁文件，不跑评测。
        return 0

    # ---- 运行各自评测脚本 ----
    ret_codes: List[Tuple[str, int]] = []

    if BENCH_MULTI in bench_set:
        if not multi_patch_path:
            print("[ERROR] Multi-SWE-bench patch file not generated; skip evaluation.")
        else:
            if not args.multi_dataset_files:
                print(
                    "[WARN] Missing --multi-dataset-files; skip Multi-SWE-bench evaluation.",
                )
            else:
                multi_repo_dir = args.multi_repo_dir or (args.multi_root / "repos")
                multi_repo_dir = multi_repo_dir.expanduser().resolve()
                if multi_repo_dir.exists() and not multi_repo_dir.is_dir():
                    print(
                        f"[ERROR] Multi-SWE-bench --repo_dir is not a directory: {multi_repo_dir}",
                        file=sys.stderr,
                    )
                    ret_codes.append((BENCH_MULTI, 2))
                else:
                    if not multi_repo_dir.exists():
                        multi_repo_dir.mkdir(parents=True, exist_ok=True)
                        print(
                            f"[INFO] Created Multi-SWE-bench repo dir (empty): {multi_repo_dir}\n"
                            "[INFO] Harness default --need_clone clones inside Docker; "
                            "if you use local COPY mode, pre-populate this directory per multi-swe-bench docs.",
                            file=sys.stderr,
                        )
                    multi_extra: List[str] = []
                    if args.multi_specifics:
                        multi_extra.extend(["--specifics", *args.multi_specifics])
                    if args.multi_skips:
                        multi_extra.extend(["--skips", *args.multi_skips])
                    code = run_multi_evaluation(
                        swe_multi_root=args.multi_root,
                        patch_file=multi_patch_path,
                        dataset_files=args.multi_dataset_files,
                        workdir=args.multi_workdir,
                        output_dir=args.multi_output_dir,
                        repo_dir=multi_repo_dir,
                        extra_args=multi_extra if multi_extra else None,
                    )
                    ret_codes.append((BENCH_MULTI, code))

    if BENCH_POLY in bench_set:
        if not poly_patch_path:
            print("[ERROR] SWE-PolyBench patch file not generated; skip evaluation.")
        else:
            if not args.poly_dataset_path:
                print(
                    "[WARN] Missing --poly-dataset-path; skip SWE-PolyBench evaluation.",
                )
            else:
                code = run_poly_evaluation(
                    swe_poly_root=args.poly_root,
                    predictions_path=poly_patch_path,
                    dataset_path=args.poly_dataset_path,
                    result_path=args.poly_result_path,
                    repo_path=args.poly_repo_path,
                    num_threads=args.poly_num_threads,
                )
                ret_codes.append((BENCH_POLY, code))

    if BENCH_PRO in bench_set:
        if not pro_patch_path:
            print("[ERROR] SWE-bench Pro patch file not generated; skip evaluation.")
        else:
            if not args.pro_raw_sample_path:
                print(
                    "[WARN] Missing --pro-raw-sample-path; skip SWE-bench Pro evaluation.",
                )
            else:
                code = run_pro_evaluation(
                    swe_pro_root=args.pro_root,
                    patch_path=pro_patch_path,
                    raw_sample_path=args.pro_raw_sample_path,
                    output_dir=args.pro_output_dir,
                    scripts_dir=args.pro_scripts_dir,
                    dockerhub_username=args.pro_dockerhub_username,
                    num_workers=args.pro_num_workers,
                    use_local_docker=args.pro_use_local_docker,
                )
                ret_codes.append((BENCH_PRO, code))

    if BENCH_VERIFIED in bench_set:
        if not verified_patch_path:
            print("[ERROR] SWE-bench Verified patch file not generated; skip evaluation.")
        else:
            namespace_value: Optional[str]
            if args.verified_namespace.lower() == "none":
                namespace_value = None
            else:
                namespace_value = args.verified_namespace

            code = run_verified_evaluation(
                swebench_root=args.verified_root,
                predictions_path=verified_patch_path,
                dataset_name=args.verified_dataset_name,
                split=args.verified_split,
                run_id=args.verified_run_id,
                max_workers=args.verified_max_workers,
                namespace=namespace_value,
                timeout=args.verified_timeout,
            )
            ret_codes.append((BENCH_VERIFIED, code))

    # 打印各 benchmark 的返回码
    if ret_codes:
        print("[INFO] Evaluation return codes:")
        for name, code in ret_codes:
            print(f"  {name}: {code}")

    # 返回非 0 表示至少有一个评测失败
    for _, code in ret_codes:
        if code != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

