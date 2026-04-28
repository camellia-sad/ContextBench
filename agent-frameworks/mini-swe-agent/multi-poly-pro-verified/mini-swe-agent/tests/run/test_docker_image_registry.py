import os

import pytest

from minisweagent.run.extra.docker_image_registry import (
    apply_docker_image_registry_prefix,
    apply_registry_mirror_prefix,
)


@pytest.fixture(autouse=True)
def clear_registry_env(monkeypatch):
    monkeypatch.delenv("MSWEA_DOCKER_IMAGE_REGISTRY", raising=False)
    monkeypatch.delenv("MSWEA_GHCR_MIRROR", raising=False)
    monkeypatch.delenv("MSWEA_POLY_GHCR_REGISTRY", raising=False)


def test_no_env_unchanged():
    assert apply_docker_image_registry_prefix("jefzda/sweap-images:foo.bar-baz") == "jefzda/sweap-images:foo.bar-baz"
    assert apply_docker_image_registry_prefix("") == ""


def test_prefix_hub_style(monkeypatch):
    monkeypatch.setenv("MSWEA_DOCKER_IMAGE_REGISTRY", "mirror.example.com")
    assert apply_docker_image_registry_prefix("jefzda/sweap-images:t") == "mirror.example.com/jefzda/sweap-images:t"
    assert apply_docker_image_registry_prefix("docker.io/swebench/img:latest") == "mirror.example.com/swebench/img:latest"


def test_prefix_strips_scheme(monkeypatch):
    monkeypatch.setenv("MSWEA_DOCKER_IMAGE_REGISTRY", "https://mirror.example.com/")
    assert apply_docker_image_registry_prefix("jefzda/sweap-images:t") == "mirror.example.com/jefzda/sweap-images:t"


def test_other_registry_untouched(monkeypatch):
    monkeypatch.setenv("MSWEA_DOCKER_IMAGE_REGISTRY", "mirror.example.com")
    assert apply_docker_image_registry_prefix("ghcr.io/org/img:tag") == "ghcr.io/org/img:tag"


def test_idempotent_when_already_prefixed(monkeypatch):
    monkeypatch.setenv("MSWEA_DOCKER_IMAGE_REGISTRY", "mirror.example.com")
    doubled = "mirror.example.com/jefzda/sweap-images:t"
    assert apply_docker_image_registry_prefix(doubled) == doubled


def test_mirror_prefix_no_env_unchanged():
    assert apply_registry_mirror_prefix("ghcr.io/org/img:tag") == "ghcr.io/org/img:tag"


def test_mirror_prefix_with_env(monkeypatch):
    monkeypatch.setenv("MSWEA_DOCKER_IMAGE_REGISTRY", "fczi514j9ggm7b.xuanyuan.run")
    assert apply_registry_mirror_prefix("ghcr.io/timesler/foo:latest") == (
        "fczi514j9ggm7b.xuanyuan.run/timesler/foo:latest"
    )


def test_mirror_prefix_nju_style_direct_host_replace(monkeypatch):
    monkeypatch.setenv("MSWEA_POLY_GHCR_REGISTRY", "ghcr.nju.edu.cn")
    assert apply_registry_mirror_prefix("ghcr.io/timesler/swe-polybench.eval.x86_64.x__y:latest") == (
        "ghcr.nju.edu.cn/timesler/swe-polybench.eval.x86_64.x__y:latest"
    )


def test_ghcr_mirror_prefers_poly_env_over_docker_mswea(monkeypatch):
    monkeypatch.setenv("MSWEA_DOCKER_IMAGE_REGISTRY", "fczi514j9ggm7b.xuanyuan.run")
    monkeypatch.setenv("MSWEA_POLY_GHCR_REGISTRY", "ghcr.nju.edu.cn")
    assert apply_registry_mirror_prefix("ghcr.io/timesler/p:latest") == "ghcr.nju.edu.cn/timesler/p:latest"
    assert apply_docker_image_registry_prefix("docker.io/swebench/x:1") == (
        "fczi514j9ggm7b.xuanyuan.run/swebench/x:1"
    )


def test_mirror_prefix_idempotent(monkeypatch):
    monkeypatch.setenv("MSWEA_DOCKER_IMAGE_REGISTRY", "mirror.example.com")
    already = "mirror.example.com/timesler/foo:latest"
    assert apply_registry_mirror_prefix(already) == already
