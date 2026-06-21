#!/usr/bin/env python3
"""
从 select_500（或任意）miniswe 根目录 **按 benchmark 子目录** 各生成一份 preds JSONL，
并可选合并为 `preds_select500_merged.jsonl`（按 instance_id 去重，顺序保留先出现的行）。

目录约定与 `contextbench.run` / `build_preds_jsonl_from_trajs.py` 一致：

  <traj-root>/Verified/.../*.traj.json
  <traj-root>/Pro/...
  <traj-root>/Poly/...
  <traj-root>/Multi/...

单目录转换逻辑与 `contextbench.process_trajectories.cmd_convert` 相同。

示例：

  python3 extract_trajs/build_preds_per_bench.py \\
    --traj-root agent-frameworks/mini-swe-agent/.../src/output_vllm/select_500_codestral_22b/miniswe \\
    --output-dir ./preds_merged/codestral-22b \\
    --merge

  # 只构建其中几个 split：
  python3 extract_trajs/build_preds_per_bench.py \\
    --traj-root agent-frameworks/mini-swe-agent/.../src/output_vllm/select_500_codestral_22b/miniswe \\
    --output-dir ./preds_merged/kimik2.5 \\
    --benches Verified,Pro \\
    --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

DEFAULT_BENCHES = ("Verified", "Pro", "Poly", "Multi")
MERGED_NAME = "preds_select500_merged.jsonl"


def _ensure_repo_on_path() -> Path:
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _merge_jsonl_unique_instance(paths: list[Path], out: Path) -> dict:
    """Concatenate JSONLs; skip later lines whose instance_id was already seen."""
    seen: set[str] = set()
    kept = 0
    skipped_dup = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        for p in paths:
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = str(obj.get("instance_id") or obj.get("original_inst_id") or "")
                if not iid:
                    wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    continue
                if iid in seen:
                    skipped_dup += 1
                    continue
                seen.add(iid)
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    return {"merged_instances": kept, "skipped_duplicate_ids": skipped_dup}


def main() -> int:
    _ensure_repo_on_path()
    from contextbench.process_trajectories import cmd_convert

    ap = argparse.ArgumentParser(
        description="Build preds_<Bench>.jsonl per split under traj-root; optional merged JSONL.",
    )
    ap.add_argument(
        "--traj-root",
        type=Path,
        required=True,
        help="miniswe 根目录（其下含 Verified / Pro / Poly / Multi 子目录）。",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录：写入 preds_Verified.jsonl 等。",
    )
    ap.add_argument(
        "--benches",
        type=str,
        default=",".join(DEFAULT_BENCHES),
        help=f"逗号分隔，默认 {','.join(DEFAULT_BENCHES)}。",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help=f"合并各 split 到 {MERGED_NAME}（instance_id 去重）。",
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="传给 convert（mini-swe 一般可省略）。",
    )
    args = ap.parse_args()

    root = args.traj_root.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        return 2

    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)

    built: dict[str, str] = {}
    skipped: list[str] = []
    out_paths_for_merge: list[Path] = []

    for bench in benches:
        sub = root / bench
        out_path = out_dir / f"preds_{bench}.jsonl"
        if not sub.is_dir():
            print(f"[WARN] Skip {bench}: missing directory {sub}", file=sys.stderr)
            skipped.append(bench)
            continue
        ns = SimpleNamespace(
            input=[str(sub)],
            out=str(out_path),
            agent="mini-swe-agent",
            recursive=bool(args.recursive),
        )
        rc = cmd_convert(ns)
        if rc != 0:
            print(f"[ERROR] convert failed for bench={bench} rc={rc}", file=sys.stderr)
            return rc
        built[bench] = str(out_path)
        out_paths_for_merge.append(out_path)
        n_lines = sum(1 for _ in out_path.open(encoding="utf-8") if _.strip())
        print(f"[OK] {bench}: {n_lines} preds -> {out_path}", file=sys.stderr)

    manifest = {
        "traj_root": str(root),
        "output_dir": str(out_dir),
        "benches_requested": benches,
        "built": built,
        "skipped_missing_dir": skipped,
    }

    if args.merge and out_paths_for_merge:
        merged_path = out_dir / MERGED_NAME
        stats = _merge_jsonl_unique_instance(out_paths_for_merge, merged_path)
        manifest["merged"] = str(merged_path)
        manifest["merge_stats"] = stats
        print(
            f"[OK] merged -> {merged_path} ({stats['merged_instances']} rows, "
            f"skipped_dup={stats['skipped_duplicate_ids']})",
            file=sys.stderr,
        )

    man_path = out_dir / "build_preds_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] manifest -> {man_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
