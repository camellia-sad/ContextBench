#!/usr/bin/env python3
"""
Analyze traj.json files under a directory:

- Recursively find all *.traj.json
- For each file, read info.exit_status and info.submission
- Count how many trajectories have each distinct exit_status
- Check whether submission looks like a valid diff patch
  (heuristic: non-empty string whose first non-whitespace characters start with 'diff --git ')

Optional: compare against a task CSV (e.g. Verified rows in data/selected_500_instances.csv)
to list instance IDs that have no trajectory yet.

Example:
    python tools/analyze_traj_info.py \
        agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/output_vllm/select_500_qwen3_coder_next/miniswe/Verified

    python tools/analyze_traj_info.py <root> --compare-csv
    # Bench for CSV filtering is inferred from root (e.g. .../miniswe/Verified -> Verified).
    python tools/analyze_traj_info.py <root> --compare-csv data/selected_500_instances.csv --bench Pro

    # Only print comma-separated missing ids (for shell):
    python tools/analyze_traj_info.py <root> --compare-csv --instances-only

    # Print sample instance ids by status:
    python tools/analyze_traj_info.py <root> --show-status-samples InternalServerError

    # Only output comma-separated ids for one/more statuses:
    python tools/analyze_traj_info.py <root> --status-instances-only InternalServerError StepResponseTimeout

    # Union status ids with "missing traj" ids from CSV:
    python tools/analyze_traj_info.py <root> --status-instances-only InternalServerError \
        --compare-csv --include-missing-from-csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_CSV = REPO_ROOT / "data" / "selected_500_instances.csv"

# Same canonical names as ContextBench CSV `bench` column / output layout (.../<agent>/<Bench>/).
KNOWN_BENCHES = frozenset({"Verified", "Pro", "Poly", "Multi"})
BENCH_ALIASES = {
    "verified": "Verified",
    "pro": "Pro",
    "poly": "Poly",
    "multi": "Multi",
}
# contextbench.run uses these directory names under the output root
AGENT_OUTPUT_SEGMENTS = frozenset({"miniswe", "agentless", "sweagent", "openhands"})


@dataclass
class ScanStats:
    total: int = 0
    status_counter: Counter[str] = field(default_factory=Counter)
    valid_patch_counter: Counter[bool] = field(default_factory=Counter)
    per_status_valid: Counter[Tuple[str, bool]] = field(default_factory=Counter)
    status_instances: Dict[str, list[str]] = field(default_factory=dict)


def normalize_bench_token(token: str) -> str | None:
    t = token.strip()
    if t in KNOWN_BENCHES:
        return t
    return BENCH_ALIASES.get(t.lower())


def infer_bench_from_output_path(root: Path) -> str | None:
    """Infer CSV bench from traj scan root, e.g. .../miniswe/Verified or .../agentless/Pro."""
    try:
        resolved = root.resolve()
    except OSError:
        resolved = root
    if nb := normalize_bench_token(resolved.name):
        return nb
    parts = resolved.parts
    for i, seg in enumerate(parts):
        if seg.lower() in AGENT_OUTPUT_SEGMENTS and i + 1 < len(parts):
            if nb := normalize_bench_token(parts[i + 1]):
                return nb
    return None


def looks_like_diff_patch(submission: str) -> bool:
    """Heuristic: treat as valid patch if it starts with 'diff --git ' after stripping leading whitespace."""
    if not isinstance(submission, str):
        return False
    s = submission.lstrip()
    return s.startswith("diff --git ")


def load_expected_ids_from_csv(csv_path: Path, bench: str) -> Tuple[Set[str], list[dict[str, str]]]:
    """Return (set of canonical instance ids, list of row dicts for missing reporting)."""
    bench_norm = bench.strip()
    expected: Set[str] = set()
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = (row.get("bench") or "").strip()
            if b != bench_norm:
                continue
            oid = (row.get("original_inst_id") or "").strip()
            iid = (row.get("instance_id") or "").strip()
            canonical = oid or iid
            if not canonical:
                continue
            expected.add(canonical)
            rows.append(
                {
                    "instance_id": iid,
                    "original_inst_id": oid,
                    "canonical": canonical,
                }
            )
    return expected, rows


def collect_traj_instance_ids(root: Path) -> Set[str]:
    """Instance id = parent directory name of each *.traj.json (matches mini-swe layout)."""
    found: Set[str] = set()
    for p in root.rglob("*.traj.json"):
        found.add(p.parent.name)
    return found


def _csv_rel_or_abs(csv_path: Path) -> str:
    try:
        return str(csv_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(csv_path)


def print_rerun_instances_hint(
    missing: list[str],
    *,
    csv_path: Path,
    bench: str,
) -> None:
    """Print --instances value and a copy-paste contextbench.run example.

    Uses canonical ids (original_inst_id or instance_id from CSV); contextbench matches either column.
    """
    if not missing:
        return
    joined = ",".join(missing)
    print()
    print("=== Rerun: contextbench.run --instances ===")
    print(
        "These ids match CSV `original_inst_id` or `instance_id` (same as used for traj dir names where applicable)."
    )
    print()
    print(f"  --instances {shlex.quote(joined)}")
    print()
    print("Example (adjust -o and other flags):")
    csv_disp = _csv_rel_or_abs(csv_path)
    ex = (
        f"python -m contextbench.run --agent miniswe "
        f"--task-csv {shlex.quote(csv_disp)} --bench {shlex.quote(bench)} "
        f"--timeout 3600 --miniswe-step-response-timeout 300 "
        f"-o output_vllm/select_500_deepseek_coder_33b --instances {shlex.quote(joined)}"
        f"--workers 12 --rerun "
    )
    print(f"  {ex}")


def compare_csv_coverage(
    root: Path,
    csv_path: Path,
    bench: str,
    *,
    bench_source: str,
    print_rerun_hint: bool = True,
) -> list[str]:
    """Return sorted list of missing canonical instance ids."""
    if not csv_path.exists():
        raise SystemExit(f"CSV does not exist: {csv_path}")
    expected, detail_rows = load_expected_ids_from_csv(csv_path, bench)
    found = collect_traj_instance_ids(root)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    by_canonical = {r["canonical"]: r for r in detail_rows}

    print()
    print("=== CSV vs trajectories ===")
    print(f"CSV path: {csv_path}")
    print(f"Bench filter: {bench!r} ({bench_source})")
    print(f"Expected instances (from CSV): {len(expected)}")
    print(f"Trajectory directories found: {len(found)}")
    print(f"Missing (in CSV, no *.traj.json under root): {len(missing)}")
    if missing:
        for mid in missing:
            row = by_canonical.get(mid, {})
            extra_cols = []
            if row.get("instance_id") and row["instance_id"] != mid:
                extra_cols.append(f"csv.instance_id={row['instance_id']}")
            print(f"  {mid}")
            if extra_cols:
                print(f"    ({', '.join(extra_cols)})")
    print(f"Extra (traj dirs not in CSV selection): {len(extra)}")
    if extra:
        for eid in extra[:50]:
            print(f"  {eid}")
        if len(extra) > 50:
            print(f"  ... and {len(extra) - 50} more")

    if print_rerun_hint and missing:
        print_rerun_instances_hint(missing, csv_path=csv_path, bench=bench)
    return missing


def compare_csv_missing_ids_only(
    root: Path,
    csv_path: Path,
    bench: str,
) -> list[str]:
    """Compare coverage without printing; for --instances-only."""
    if not csv_path.exists():
        raise SystemExit(f"CSV does not exist: {csv_path}")
    expected, _ = load_expected_ids_from_csv(csv_path, bench)
    found = collect_traj_instance_ids(root)
    return sorted(expected - found)


def _resolve_instance_id(data: dict, path: Path) -> str:
    iid = str(data.get("instance_id", "") or "").strip()
    return iid or path.parent.name


def collect_scan_stats(root: Path) -> ScanStats:
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    traj_files = sorted(root.rglob("*.traj.json"))
    if not traj_files:
        return ScanStats()

    stats = ScanStats()
    by_status: Dict[str, list[str]] = {}

    for p in traj_files:
        stats.total += 1
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            status = f"<parse_error:{type(e).__name__}>"
            iid = p.parent.name
            stats.status_counter[status] += 1
            stats.per_status_valid[(status, False)] += 1
            by_status.setdefault(status, []).append(iid)
            continue

        info = data.get("info", {})
        iid = _resolve_instance_id(data, p)
        status = str(info.get("exit_status", "") or "").strip() or "<empty>"
        submission = str(info.get("submission", "") or "")

        stats.status_counter[status] += 1

        is_valid = looks_like_diff_patch(submission)
        stats.valid_patch_counter[is_valid] += 1
        stats.per_status_valid[(status, is_valid)] += 1
        by_status.setdefault(status, []).append(iid)

    stats.status_instances = by_status
    return stats


def _collect_status_instance_ids(stats: ScanStats, requested_statuses: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for status in requested_statuses:
        for iid in stats.status_instances.get(status, []):
            if iid in seen:
                continue
            seen.add(iid)
            out.append(iid)
    return out


def _print_status_samples(stats: ScanStats, statuses: list[str], limit: int) -> None:
    print()
    print("Status sample instances:")
    for status in statuses:
        ids = stats.status_instances.get(status, [])
        print(f"  {status!r}: {len(ids)}")
        if not ids:
            continue
        shown = ids if limit <= 0 else ids[:limit]
        for iid in shown:
            print(f"    {iid}")
        if limit > 0 and len(ids) > len(shown):
            print(f"    ... and {len(ids) - len(shown)} more")


def analyze_root(root: Path, *, sample_statuses: list[str] | None = None, sample_limit: int = 20) -> ScanStats:
    stats = collect_scan_stats(root)
    if stats.total == 0:
        print(f"No traj.json files found under: {root}")
        return stats

    print(f"Scan root: {root}")
    print(f"Total traj.json files: {stats.total}")
    print()

    print("Exit status counts:")
    for status, count in stats.status_counter.most_common():
        print(f"  {status!r}: {count}")

    print()
    print("Submission looks like valid diff patch (by heuristic):")
    print(f"  True : {stats.valid_patch_counter[True]}")
    print(f"  False: {stats.valid_patch_counter[False]}")

    print()
    print("Exit status x valid_diff matrix:")
    for (status, is_valid), count in sorted(stats.per_status_valid.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        label = "valid_diff" if is_valid else "not_diff"
        print(f"  {status!r:30s} {label:10s}: {count}")

    if sample_statuses:
        _print_status_samples(stats, sample_statuses, sample_limit)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze traj.json info.exit_status and submission diff presence.")
    ap.add_argument(
        "root",
        type=Path,
        help="Root directory to scan recursively for *.traj.json files",
    )
    ap.add_argument(
        "--compare-csv",
        type=Path,
        nargs="?",
        const=DEFAULT_TASK_CSV,
        default=None,
        metavar="CSV",
        help=(
            "Compare traj coverage to CSV rows (bench inferred from root path unless --bench). "
            f"Omit path to use {DEFAULT_TASK_CSV.relative_to(REPO_ROOT)}"
        ),
    )
    ap.add_argument(
        "--bench",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Override CSV `bench` column filter. "
            "Default: infer from root (e.g. .../miniswe/Verified -> Verified)."
        ),
    )
    ap.add_argument(
        "--no-rerun-hint",
        action="store_true",
        help="With --compare-csv, do not print --instances / contextbench rerun example.",
    )
    ap.add_argument(
        "--instances-only",
        action="store_true",
        help=(
            "With --compare-csv: only print comma-separated missing instance ids to stdout "
            "(for scripts). Skips traj stats. Implies no rerun hint on stderr."
        ),
    )
    ap.add_argument(
        "--show-status-samples",
        type=str,
        nargs="+",
        default=None,
        metavar="STATUS",
        help=(
            "After stats, print sample instance names for given exit_status values "
            "(exact match, e.g. InternalServerError)."
        ),
    )
    ap.add_argument(
        "--status-sample-limit",
        type=int,
        default=20,
        metavar="N",
        help="Max samples per status for --show-status-samples (<=0 means all).",
    )
    ap.add_argument(
        "--status-instances-only",
        type=str,
        nargs="+",
        default=None,
        metavar="STATUS",
        help=(
            "Print comma-separated instance ids whose info.exit_status is in STATUS list. "
            "Useful for rerun --instances."
        ),
    )
    ap.add_argument(
        "--include-missing-from-csv",
        action="store_true",
        help=(
            "With --status-instances-only and --compare-csv: union status-selected ids with "
            "CSV-missing ids."
        ),
    )
    args = ap.parse_args()

    if args.instances_only and args.compare_csv is None:
        raise SystemExit("--instances-only requires --compare-csv")
    if args.instances_only and args.status_instances_only:
        raise SystemExit("--instances-only cannot be used with --status-instances-only")
    if args.include_missing_from_csv and args.compare_csv is None:
        raise SystemExit("--include-missing-from-csv requires --compare-csv")

    def resolve_bench() -> tuple[str, str]:
        if args.bench is not None:
            return args.bench.strip(), "from --bench"
        inferred = infer_bench_from_output_path(args.root)
        if not inferred:
            raise SystemExit(
                "Could not infer bench from root path (expected e.g. .../miniswe/Verified). "
                "Pass --bench Verified|Pro|Poly|Multi explicitly."
            )
        return inferred, "inferred from output path"

    if args.instances_only:
        bench, _src = resolve_bench()
        missing = compare_csv_missing_ids_only(args.root, args.compare_csv, bench)
        sys.stdout.write(",".join(missing))
        if missing:
            sys.stdout.write("\n")
        return

    if args.status_instances_only:
        stats = collect_scan_stats(args.root)
        selected = _collect_status_instance_ids(stats, args.status_instances_only)
        if args.include_missing_from_csv:
            bench, _src = resolve_bench()
            missing = compare_csv_missing_ids_only(args.root, args.compare_csv, bench)
            selected = sorted(set(selected).union(missing))
        sys.stdout.write(",".join(selected))
        if selected:
            sys.stdout.write("\n")
        return

    analyze_root(
        args.root,
        sample_statuses=args.show_status_samples,
        sample_limit=args.status_sample_limit,
    )
    if args.compare_csv is not None:
        bench, bench_source = resolve_bench()
        compare_csv_coverage(
            args.root,
            args.compare_csv,
            bench,
            bench_source=bench_source,
            print_rerun_hint=not args.no_rerun_hint,
        )


if __name__ == "__main__":
    main()
