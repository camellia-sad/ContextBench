import os
import csv
import sys
from typing import Dict, Set, Optional, List

from datasets import load_dataset

MINISWE_SRC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "agent-frameworks",
    "mini-swe-agent",
    "multi-poly-pro-verified",
    "mini-swe-agent",
    "src",
)
MINISWE_SRC = os.path.abspath(MINISWE_SRC)
sys.path.insert(0, MINISWE_SRC)

from minisweagent.run.extra.docker_image_registry import (
    apply_docker_image_registry_prefix,
    apply_registry_mirror_prefix,
)


def get_swebench_docker_image_name(instance: dict) -> str:
    """Mirror mini-swe-agent's swebench.get_swebench_docker_image_name() without importing extra modules."""
    image_name = instance.get("image_name")
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        dataset_name = instance.get("dataset_name", "")
        if "PolyBench" in dataset_name or any(
            repo in iid
            for repo in ["mui__", "sveltejs__", "prettier__", "serverless__", "microsoft__vscode"]
        ):
            image_name = f"ghcr.io/timesler/swe-polybench.eval.x86_64.{iid}:latest"
        else:
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if image_name.startswith("ghcr.io/"):
        return apply_registry_mirror_prefix(image_name)
    return apply_docker_image_registry_prefix(image_name)


def get_dockerhub_image_uri(instance_id: str, repo_name: str = "") -> str:
    """Mirror DockerConfigExtractor.get_dockerhub_image_uri() for SWE-bench Pro."""
    if not repo_name or "/" not in repo_name:
        return ""
    repo_base, repo_name_only = repo_name.lower().split("/", 1)
    hsh = instance_id.replace("instance_", "")

    # Handle special cases (copied from mini-swe-agent)
    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    return apply_docker_image_registry_prefix(f"jefzda/sweap-images:{tag}")


def get_multiswe_image_uri(instance_id: str, repo_name: str = "", pr_number: str = "") -> str:
    """Mirror DockerConfigExtractor.get_multiswe_image_uri() for Multi-SWE-bench."""
    if not repo_name or "/" not in repo_name:
        return ""

    org, repo = repo_name.lower().split("/", 1)
    org_clean = org.replace(".", "_")
    repo_clean = repo.replace(".", "_")

    if not pr_number:
        # Try to extract from instance_id patterns
        if "__" in instance_id:
            parts = instance_id.split("__")
            if len(parts) >= 2 and "-" in parts[-1]:
                pr_number = parts[-1].split("-")[-1]
        elif "_pr" in instance_id:
            pr_number = instance_id.split("_pr")[-1]
        elif "-" in instance_id:
            pr_number = instance_id.split("-")[-1]

    if not pr_number or not pr_number.isdigit():
        return ""

    return apply_docker_image_registry_prefix(f"mswebench/{org_clean}_m_{repo_clean}:pr-{pr_number}")


def read_desired_instance_ids(csv_path: str, benches: Set[str]) -> Dict[str, Set[str]]:
    # csv contains: bench, instance_id, original_inst_id, ...
    # mini-swe-agent uses dataset instance_id, which aligns with CSV original_inst_id for these subsets.
    mapping: Dict[str, Set[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            bench = (row.get("bench") or "").strip()
            if bench not in benches:
                continue
            original = (row.get("original_inst_id") or "").strip()
            if not original:
                continue
            mapping.setdefault(bench, set()).add(original)
    return mapping


SUBSET_TO_DATASET = {
    # mini-swe-agent uses these names in swebench.py
    "Verified": ("verified", "princeton-nlp/SWE-Bench_Verified"),
    "Pro": ("pro", "ScaleAI/SWE-bench_Pro"),
    "Poly": ("poly", "AmazonScience/SWE-PolyBench"),
    # contextbench/run.py passes --subset "multi-swe-bench" for Multi.
    # mini-swe-agent maps it to ByteDance-Seed/Multi-SWE-bench (and converts split=test/dev -> train).
    "Multi": ("multi", "ByteDance-Seed/Multi-SWE-bench"),
}

def compute_image_for_instance(bench: str, inst: dict) -> Optional[str]:
    iid = inst.get("instance_id") or ""
    if bench == "Verified":
        return get_swebench_docker_image_name(inst)
    if bench == "Poly":
        # PolyBenchStrategy: ghcr; mirror with MSWEA_DOCKER_IMAGE_REGISTRY (e.g. ghcr.nju.edu.cn)
        return apply_registry_mirror_prefix(
            f"ghcr.io/timesler/swe-polybench.eval.x86_64.{iid}:latest"
        )
    if bench == "Pro":
        # SWE-bench Pro dataset typically stores repo as "org/repo".
        # Be defensive in case the schema differs.
        repo_name = inst.get("repo") or ""
        if not repo_name:
            org = inst.get("org") or ""
            repo_only = inst.get("repo_name") or inst.get("repo_only") or ""
            if org and repo_only:
                repo_name = f"{org}/{repo_only}"
        return get_dockerhub_image_uri(iid, repo_name)
    if bench == "Multi":
        # Multi-SWE-bench schema may use either:
        # - repo="org/repo" and number=PR, or
        # - org=... and repo=repo_name, number=PR
        repo_name = inst.get("repo") or ""
        if not repo_name:
            org = inst.get("org") or ""
            repo_only = inst.get("repo_name") or inst.get("repo_only") or ""
            if org and repo_only:
                repo_name = f"{org}/{repo_only}"
        pr_number = inst.get("number") or inst.get("pr_number") or inst.get("pr") or ""
        return get_multiswe_image_uri(iid, repo_name, str(pr_number))
    return None


def main():
    csv_path = os.environ["CSV_PATH"]
    benches = set(x.strip() for x in os.environ["BENCHES"].split(",") if x.strip())
    split = os.environ.get("SPLIT", "test")

    desired_by_bench = read_desired_instance_ids(csv_path, benches)

    # Load dataset per bench and collect base_image references
    all_images: Set[str] = set()

    for bench in sorted(desired_by_bench.keys()):
        desired = desired_by_bench[bench]
        if not desired:
            continue

        subset_name, dataset_path = SUBSET_TO_DATASET[bench]
        found = set()
        found_with_image = set()
        print(f"[resolve] bench={bench} dataset={dataset_path} split={split} need={len(desired)}", file=sys.stderr)

        # streaming to avoid huge memory
        ds_split = split
        if bench == "Multi" and split in ["test", "dev"]:
            ds_split = "train"
        ds = load_dataset(dataset_path, split=ds_split, streaming=True)
        for inst in ds:
            iid = inst.get("instance_id") or ""
            if iid in desired:
                img = compute_image_for_instance(bench, inst)
                if img:
                    all_images.add(img)
                    found_with_image.add(iid)
                else:
                    # Most common reason: missing repo/org metadata in dataset instance.
                    # We'll warn once to avoid spamming stderr.
                    if len(found_with_image) == 0:
                        print(f"[resolve][warn] bench={bench} instance_id={iid} computed_image_is_empty", file=sys.stderr)
                found.add(iid)
                if len(found) >= len(desired):
                    break

        missing = desired - found_with_image
        if missing:
            # Fatal to "pull all needed images" but dataset scanning likely succeeded.
            print(f"[resolve][warn] bench={bench} instances_missing_computed_images={len(missing)} e.g. {next(iter(missing))}", file=sys.stderr)

    for img in sorted(all_images):
        print(img)


if __name__ == "__main__":
    main()
