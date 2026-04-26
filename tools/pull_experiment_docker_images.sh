#!/usr/bin/env bash
# Pull docker images required by this experiment (based on existing traj.json metadata).
#
# What it does:
# 1) Scan miniswe trajectories under:
#    agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/output_vllm/select_500_qwen25_32B_coder/miniswe/
# 2) Extract values of JSON keys named "image" (docker image refs)
# 3) Create a unique list and run `docker pull` for each.
#
# Typical usage:
#   bash tools/pull_experiment_docker_images.sh
#   bash tools/pull_experiment_docker_images.sh --list-only
#
# Options:
#   --root <path>            override scan root
#   --list-only              only print images, do not docker pull
#   --include-running       include images from currently running containers
#
set -euo pipefail

ROOT_DIR_DEFAULT="/home/dataset-assist-0/vllm_workspace/ContextBench/agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/output_vllm/select_500_qwen25_32B_coder/miniswe"

ROOT_DIR="$ROOT_DIR_DEFAULT"
LIST_ONLY=0
INCLUDE_RUNNING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    --list-only)
      LIST_ONLY=1
      shift 1
      ;;
    --include-running)
      INCLUDE_RUNNING=1
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ROOT_DIR not found: $ROOT_DIR" >&2
  exit 1
fi

OUT_LIST="$ROOT_DIR/docker_images_to_pull.txt"

python3 - <<'PY' "$ROOT_DIR" "$OUT_LIST" "$INCLUDE_RUNNING" "$LIST_ONLY"
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_list = Path(sys.argv[2])
include_running = sys.argv[3] == "1"
list_only = sys.argv[4] == "1"

images = set()

EXCLUDE_PREFIXES = (
    "polybench-temp-",
    "polybench_temp-",
    "minisweagent-",
)

# Exclude some non-image placeholders just in case.
EXCLUDE_SUBSTRINGS = (
    "unfinished",
    "None",
)

def looks_like_image_ref(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    if any(s.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    if any(x in s for x in EXCLUDE_SUBSTRINGS):
        return False
    # Must look like docker ref: contains '/' and ':' (tag or digest-like).
    # (We keep it permissive; most swebench images are ".../...:latest".)
    return ("/" in s) and (":" in s)

def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "image" and isinstance(v, str) and looks_like_image_ref(v):
                images.add(v.strip())
            walk(v)
    elif isinstance(obj, list):
        for it in obj:
            walk(it)

traj_files = list(root.rglob("*.traj.json"))
if not traj_files:
    print(f"No traj.json found under: {root}", file=sys.stderr)

for p in traj_files:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    try:
        walk(d)
    except Exception:
        continue

if include_running:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "docker ps --format '{{.Image}}' || true"],
            text=True,
        )
        for line in out.splitlines():
            line = line.strip()
            if line and looks_like_image_ref(line):
                images.add(line)
    except Exception:
        pass

imgs_sorted = sorted(images)
out_list.parent.mkdir(parents=True, exist_ok=True)
out_list.write_text("\n".join(imgs_sorted) + ("\n" if imgs_sorted else ""), encoding="utf-8")

print(f"Scan root: {root}")
print(f"Found traj.json: {len(traj_files)}")
print(f"Unique docker image refs extracted: {len(imgs_sorted)}")
print(f"Written: {out_list}")

if list_only:
    for img in imgs_sorted:
        print(img)
    sys.exit(0)

failures = 0
for img in imgs_sorted:
    print(f"Pulling: {img}")
    try:
        subprocess.run(["docker", "pull", img], check=True)
    except subprocess.CalledProcessError:
        failures += 1
        print(f"FAILED: {img}", file=sys.stderr)

print(f"Done. failures={failures}")
sys.exit(1 if failures else 0)
PY

echo "Image list: $OUT_LIST"
