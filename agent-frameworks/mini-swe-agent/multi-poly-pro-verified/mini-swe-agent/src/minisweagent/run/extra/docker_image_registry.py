"""Optional registry host prefix for Docker image references (SWE-Bench pulls).

Set ``MSWEA_DOCKER_IMAGE_REGISTRY`` to a hostname (no scheme), e.g.
``fczi514j9ggm7b.xuanyuan.run``, so implicit Hub refs become
``<host>/jefzda/sweap-images:tag`` and ``docker.io/swebench/...`` become
``<host>/swebench/...``.

Refs that already use another registry host (e.g. ``ghcr.io/...``) are left
unchanged. Unset the variable to restore default image names.

For GHCR (Poly pre-built, etc.) use ``apply_registry_mirror_prefix``. Host is taken
from **``MSWEA_POLY_GHCR_REGISTRY``** (alias: ``MSWEA_GHCR_MIRROR``) first, e.g. ``ghcr.nju.edu.cn``;
if unset, falls back to ``MSWEA_DOCKER_IMAGE_REGISTRY`` (backward compatible).

**Split mirrors:** set ``MSWEA_DOCKER_IMAGE_REGISTRY`` to your xuanyuan (or other) host
for ``docker.io/swebench/...`` / Hub-style images only, and
``MSWEA_GHCR_MIRROR=ghcr.nju.edu.cn`` for ``ghcr.io/...`` → ``ghcr.nju.edu.cn/.../``,
so the three non-Poly-style pulls keep xuanyuan while Poly GHCR uses NJU.

To revert: unset these variables or see apply_* call sites in ``swebench*``.
"""

from __future__ import annotations

import os


def _ghcr_mirror_host() -> str:
    """Host that replaces ``ghcr.io`` for ``apply_registry_mirror_prefix`` (Poly pre-built, etc.)."""
    p = _normalize_registry_prefix(
        os.environ.get("MSWEA_POLY_GHCR_REGISTRY", "")
        or os.environ.get("MSWEA_GHCR_MIRROR", "")
    )
    if p:
        return p
    return _normalize_registry_prefix(os.environ.get("MSWEA_DOCKER_IMAGE_REGISTRY", ""))


def _normalize_registry_prefix(raw: str) -> str:
    p = raw.strip()
    if not p:
        return ""
    p = p.removeprefix("https://").removeprefix("http://").rstrip("/")
    return p


def apply_docker_image_registry_prefix(uri: str) -> str:
    """If ``MSWEA_DOCKER_IMAGE_REGISTRY`` is set, prefix Docker Hub–style refs."""
    if not uri:
        return uri
    prefix = _normalize_registry_prefix(os.environ.get("MSWEA_DOCKER_IMAGE_REGISTRY", ""))
    if not prefix:
        return uri
    if uri == prefix or uri.startswith(prefix + "/"):
        return uri

    rest = uri
    if rest.startswith("docker.io/"):
        rest = rest[len("docker.io/") :]
    else:
        first = rest.split("/", 1)[0]
        if "." in first:
            return uri

    return f"{prefix}/{rest}"


def apply_registry_mirror_prefix(full_image_uri: str) -> str:
    """Replace ``ghcr.io`` with ``_ghcr_mirror_host()`` (``MSWEA_GHCR_MIRROR`` or legacy env).

    ``ghcr.io/timesler/foo:bar`` → ``<mirror>/timesler/foo:bar`` (not ``<mirror>/ghcr.io/...``).

    If the env var is unset or empty, or the URI is not a ``ghcr.io/...`` ref, returns
    ``full_image_uri`` unchanged (after normalizing an already-rewritten URI to avoid dup).
    """
    if not full_image_uri:
        return full_image_uri
    prefix = _ghcr_mirror_host()
    if not prefix:
        return full_image_uri
    if full_image_uri == prefix or full_image_uri.startswith(prefix + "/"):
        return full_image_uri
    if full_image_uri.startswith("ghcr.io/"):
        return f"{prefix}/{full_image_uri[len('ghcr.io/') :]}"
    return full_image_uri
