"""Helpers for creating a :class:`docker.DockerClient`.

Some environments run the `opa-docker-authz` (or similar) authz plugin with a path-based
rule set that **allows** versioned engine routes such as ``GET /v1.49/...`` but **denies**
the legacy unversioned ``GET /version`` endpoint.

`docker-py` ``from_env(version=None)`` (the default) probes the daemon with that unversioned
call to detect the API version, so it fails with *403 Forbidden* even when the Docker CLI
and ``curl /v1.xx/version`` work. Setting an explicit API version (or ``DOCKER_API_VERSION``)
avoids the probe.
"""

from __future__ import annotations

import os
from typing import Any

import docker
from docker.constants import DEFAULT_DOCKER_API_VERSION


def docker_from_env(**kwargs: Any) -> docker.DockerClient:
    """
    Like :func:`docker.from_env`, but if ``version`` is omitted, use ``DOCKER_API_VERSION`` or
    a library default so the unversioned ``/version`` probe is not used.

    To keep docker-py's automatic detection, pass ``version="auto"`` explicitly.
    """
    if kwargs.get("version", None) is None:
        raw = (os.environ.get("DOCKER_API_VERSION") or DEFAULT_DOCKER_API_VERSION).strip()
        if not raw or raw.lower() == "auto":
            raw = DEFAULT_DOCKER_API_VERSION
        kwargs["version"] = raw
    return docker.from_env(**kwargs)
