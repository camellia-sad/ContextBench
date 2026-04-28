#!/usr/bin/env bash
# Pull docker images required for your selected instances list,
# based on the *same naming logic as mini-swe-agent* (not on existing traj.json).
#
# Default targets:
# - Verified: docker.io/swebench/sweb.eval.x86_64.<iid_with__mapped_to_1776_>:latest
# - Poly:     ghcr.io/timesler/swe-polybench.eval.x86_64.<iid>:latest
#   With MSWEA_DOCKER_IMAGE_REGISTRY=ghcr.nju.edu.cn: ghcr.nju.edu.cn/timesler/...
#
# Notes:
# - Pro/Multi require additional metadata (repo/pr_number) that is not present in
#   ContextBench/data/selected_500_instances.csv (repo column is empty), so this script
#   skips them unless you extend it.
#
# Usage:
#   cd /home/dataset-assist-0/vllm_workspace/ContextBench
#   bash tools/pull_images_from_selected_csv.sh --bench Verified --list-only
#   bash tools/pull_images_from_selected_csv.sh --bench Verified
#
# Options:
#   --csv <path>          default: data/selected_500_instances.csv
#   --bench <name,...>   default: Verified
#   --root-out <dir>      default: ./agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/output_vllm/select_500_qwen25_32B_coder/miniswe
#   --list-only           only print and exit
#   --include-running     also include images from current running containers
#
set -euo pipefail

ROOT_OUT_DEFAULT="/home/dataset-assist-0/vllm_workspace/ContextBench/agent-frameworks/mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/output_vllm/select_500_qwen25_32B_coder/miniswe"
CSV_DEFAULT="/home/dataset-assist-0/vllm_workspace/ContextBench/data/selected_500_instances.csv"

CSV_PATH="$CSV_DEFAULT"
BENCHES="Verified"
ROOT_OUT="$ROOT_OUT_DEFAULT"
LIST_ONLY=0
INCLUDE_RUNNING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV_PATH="$2"; shift 2 ;;
    --bench) BENCHES="$2"; shift 2 ;;
    --root-out) ROOT_OUT="$2"; shift 2 ;;
    --list-only) LIST_ONLY=1; shift 1 ;;
    --include-running) INCLUDE_RUNNING=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV not found: $CSV_PATH" >&2
  exit 1
fi

OUT_LIST="$ROOT_OUT/docker_images_to_pull_from_selected_csv.txt"
mkdir -p "$ROOT_OUT"

python3 - <<'PY' "$CSV_PATH" "$BENCHES" "$OUT_LIST" "$ROOT_OUT" "$INCLUDE_RUNNING" 
import csv, os, sys, subprocess
from pathlib import Path

csv_path = Path(sys.argv[1])
benches = set(b.strip() for b in sys.argv[2].split(",") if b.strip())
out_list = Path(sys.argv[3])
root_out = Path(sys.argv[4])
include_running = sys.argv[5] == "1"

def normalize_registry_prefix(raw: str) -> str:
    p = (raw or "").strip()
    if not p:
        return ""
    p = p.removeprefix("https://").removeprefix("http://").rstrip("/")
    return p

def apply_registry_prefix(uri: str) -> str:
    if not uri:
        return uri
    prefix = normalize_registry_prefix(os.environ.get("MSWEA_DOCKER_IMAGE_REGISTRY", ""))
    if not prefix:
        return uri
    if uri == prefix or uri.startswith(prefix + "/"):
        return uri
    rest = uri
    if rest.startswith("docker.io/"):
        rest = rest[len("docker.io/"):]
    else:
        first = rest.split("/", 1)[0]
        if "." in first:
            return uri
    return f"{prefix}/{rest}"


def apply_registry_mirror_prefix(full_image_uri: str) -> str:
    """Same as minisweagent: ghcr.io/org:tag -> <mirror>/org:tag."""
    if not full_image_uri:
        return full_image_uri
    prefix = normalize_registry_prefix(os.environ.get("MSWEA_DOCKER_IMAGE_REGISTRY", ""))
    if not prefix:
        return full_image_uri
    if full_image_uri == prefix or full_image_uri.startswith(prefix + "/"):
        return full_image_uri
    if full_image_uri.startswith("ghcr.io/"):
        return f"{prefix}/{full_image_uri[len('ghcr.io/') :]}"
    return full_image_uri

images = set()

def add_image(img: str):
    if img and ("/" in img) and (":" in img):
        images.add(apply_registry_prefix(img).lower())

with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bench = (row.get("bench") or "").strip()
        if bench not in benches:
            continue
        iid = (row.get("original_inst_id") or row.get("instance_id") or "").strip()
        if not iid:
            continue

        if bench == "Verified":
            # mirror naming logic in get_swebench_docker_image_name()
            id_docker_compatible = iid.replace("__", "_1776_")
            img = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest"
            add_image(img)
        elif bench == "Poly":
            img = apply_registry_mirror_prefix(
                f"ghcr.io/timesler/swe-polybench.eval.x86_64.{iid}:latest"
            )
            add_image(img)
        else:
            # Pro/Multi not solvable from this CSV alone (repo/pr_number missing)
            continue

if include_running:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "docker ps --format '{{.Image}}' || true"],
            text=True,
        )
        for line in out.splitlines():
            line = line.strip()
            if line:
                # keep only image refs that look like docker tags
                if ("/" in line) and (":" in line):
                    images.add(apply_registry_prefix(line).lower())
    except Exception:
        pass

imgs_sorted = sorted(images)
out_list.write_text("\n".join(imgs_sorted) + ("\n" if imgs_sorted else ""), encoding="utf-8")

print(f"CSV: {csv_path}")
print(f"Bench filter: {','.join(sorted(benches))}")
print(f"Extracted unique image refs: {len(imgs_sorted)}")
print(f"Written: {out_list}")
PY

if [[ "$LIST_ONLY" -eq 1 ]]; then
  cat "$OUT_LIST"
  exit 0
fi

fail=0
while IFS= read -r img; do
  [[ -z "$img" ]] && continue
  echo "Pulling: $img"
  if ! docker pull "$img"; then
    echo "FAILED: $img" >&2
    fail=1
  fi
done < "$OUT_LIST"

exit $fail

