#!/usr/bin/env python3
"""
Recover or synthesize `info.submission` (model patch) from mini-swe-agent `.traj.json`
by scanning **message history**, simulating what a careful reader would take as the
final git unified diff when `info.submission` is empty or non-diff (e.g. context errors).

Typical cases:
- Success: `info.submission` already holds `git diff`; nothing to do.
- Failure: submission is a placeholder, but earlier **user** observations contain
  `<output>...diff --git...</output>` from `git diff` / `git diff --cached` runs.

Limitations:
- Truncated observations (`<output_head>` / `<output_tail>` only) may miss hunks;
  this script does not reassemble truncated diffs.
- Merging per-file hunks assumes non-overlapping edits; weird cases may need manual fix.

Examples:
  # Preview recovered patch for one traj
  python tools/traj_recover_submission.py --traj path/to/instance.traj.json

  # Write back into traj (backup recommended)
  python tools/traj_recover_submission.py --traj path/to/instance.traj.json --write-back

  # Scan all Pro trajs under a root, only print stats
  python tools/traj_recover_submission.py --traj-root .../miniswe/Pro --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _looks_like_unified_diff(s: str) -> bool:
    t = (s or "").lstrip()
    return t.startswith("diff --git ") and "\n--- " in t and "\n+++ " in t


def _extract_output_bodies(content: str) -> List[str]:
    """Pull observation bodies from mini-swe user messages."""
    if not content:
        return []
    bodies: List[str] = []
    for m in re.finditer(r"<output>\s*([\s\S]*?)\s*</output>", content):
        bodies.append(m.group(1))
    if not bodies and "diff --git" in content:
        bodies.append(content)
    return bodies


def _iter_diff_hunks(text: str) -> List[str]:
    """Split `text` into zero or more unified-diff hunks (each starts with diff --git)."""
    text = (text or "").strip()
    if not text.startswith("diff --git"):
        return []
    hunks = re.findall(
        r"(?ms)^diff --git .+?(?=^diff --git |\Z)",
        text + "\n",
    )
    return [h.strip() for h in hunks if h.strip().startswith("diff --git")]


def _hunk_file_key(hunk: str) -> str:
    """Stable key from first line `diff --git a/X b/X` (fallback: first line)."""
    first = (hunk or "").strip().splitlines()[0] if hunk.strip() else ""
    return first


def collect_diff_hunks_from_messages(messages: list) -> List[str]:
    """
    Walk messages in order; from each **user** message extract `<output>` bodies
    and split into diff hunks. Returns a flat list of hunks in chronological order.
    """
    ordered: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for body in _extract_output_bodies(msg.get("content") or ""):
            ordered.extend(_iter_diff_hunks(body))
    return ordered


def merge_hunks_last_wins(hunks: List[str]) -> str:
    """Later hunks overwrite earlier ones with the same diff --git first line."""
    by_key: dict[str, str] = {}
    key_order: List[str] = []
    for h in hunks:
        k = _hunk_file_key(h)
        if k not in by_key:
            key_order.append(k)
        by_key[k] = h
    return "\n".join(by_key[k] for k in key_order)


def longest_diff_blob(messages: list) -> str:
    """Return the longest contiguous substring starting with diff --git from any user body."""
    best = ""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for body in _extract_output_bodies(msg.get("content") or ""):
            if "diff --git" not in body:
                continue
            for h in _iter_diff_hunks(body):
                if len(h) > len(best):
                    best = h
            # also consider whole body if multiple hunks concatenated in one blob
            m = re.search(r"(?ms)^diff --git .+", body)
            if m:
                blob = m.group(0).strip()
                if len(blob) > len(best):
                    best = blob
    return best


def recover_submission(
    data: dict,
    *,
    strategy: str,
) -> Tuple[str, str]:
    """
    Returns (patch, note) where note explains the source.
    """
    info = data.get("info") or {}
    submission = (info.get("submission") or "").strip()

    if strategy in ("submission", "auto") and _looks_like_unified_diff(submission):
        return submission, "kept_info.submission"

    hunks = collect_diff_hunks_from_messages(data.get("messages") or [])

    if strategy == "longest":
        patch = longest_diff_blob(data.get("messages") or [])
        if patch:
            return patch, "longest_diff_blob_in_messages"
        if hunks:
            merged = merge_hunks_last_wins(hunks)
            if merged:
                return merged, "fallback_merge_after_longest_empty"
        return "", "no_diff_found"

    if strategy in ("merge", "auto"):
        if hunks:
            merged = merge_hunks_last_wins(hunks)
            if merged:
                return merged, f"merge_last_wins({len(hunks)}_hunks_from_messages)"
        if strategy == "auto" and _looks_like_unified_diff(submission):
            return submission, "fallback_info.submission"
        return "", "no_diff_hunks_in_messages"

    return "", "unknown_strategy"


def load_traj(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Cannot load {path}: {exc}")
        return None


def iter_traj_files(traj_root: Path) -> Iterable[Path]:
    if not traj_root.is_dir():
        return
    for child in sorted(traj_root.iterdir()):
        if not child.is_dir():
            continue
        for p in sorted(child.glob("*.traj.json")):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=Path, help="Single .traj.json file")
    ap.add_argument("--traj-root", type=Path, help="Directory of instance dirs (each containing *.traj.json)")
    ap.add_argument(
        "--strategy",
        choices=("auto", "submission", "merge", "longest"),
        default="auto",
        help="auto: use info.submission if valid diff; else merge hunks from messages. "
        "merge: only from message hunks. longest: single largest diff blob. "
        "submission: only trust info.submission.",
    )
    ap.add_argument("--write-back", action="store_true", help="Write recovered patch into info.submission")
    ap.add_argument("--dry-run", action="store_true", help="With --traj-root, only print summary per file")
    ap.add_argument("--emit-patch", type=Path, help="Write recovered diff to this file (single --traj only)")
    args = ap.parse_args()

    if bool(args.traj) == bool(args.traj_root):
        ap.error("Specify exactly one of --traj or --traj-root")

    if args.traj:
        path = args.traj
        data = load_traj(path)
        if data is None:
            return 1
        patch, note = recover_submission(data, strategy=args.strategy)
        cur = ((data.get("info") or {}).get("submission") or "").strip()
        print(f"file: {path}")
        print(f"strategy: {args.strategy} -> {note}")
        print(f"current submission length: {len(cur)}")
        print(f"recovered patch length: {len(patch)}")
        if args.emit_patch:
            args.emit_patch.write_text(patch, encoding="utf-8")
            print(f"wrote patch to {args.emit_patch}")
        if args.write_back:
            if not patch:
                print("[ERROR] Empty patch; refuse --write-back")
                return 1
            data.setdefault("info", {})["submission"] = patch
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"updated info.submission in {path}")
        return 0

    # traj-root mode
    n_ok = n_skip = n_empty = 0
    for path in iter_traj_files(args.traj_root):
        data = load_traj(path)
        if data is None:
            n_skip += 1
            continue
        patch, note = recover_submission(data, strategy=args.strategy)
        cur = ((data.get("info") or {}).get("submission") or "").strip()
        if not patch:
            n_empty += 1
            print(f"[empty] {path.name} ({note})")
            continue
        if _looks_like_unified_diff(cur) and args.strategy == "auto" and note.startswith("kept"):
            n_ok += 1
            if not args.dry_run:
                print(f"[ok] {path.name} already valid submission")
            continue
        n_ok += 1
        print(f"[recovered] {path.name} len={len(patch)} ({note})")
        if args.write_back and not args.dry_run:
            data.setdefault("info", {})["submission"] = patch
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
    print(f"done: recovered/ok={n_ok} empty={n_empty} load_fail={n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
