from unittest.mock import patch

from docker.constants import DEFAULT_DOCKER_API_VERSION


@patch("minisweagent.run.extra.docker_client.docker.from_env")
def test_docker_from_env_sets_version_when_omitted(mock_from_env, monkeypatch):
    monkeypatch.delenv("DOCKER_API_VERSION", raising=False)
    from minisweagent.run.extra.docker_client import docker_from_env

    docker_from_env(timeout=5)
    mock_from_env.assert_called_once()
    _, kwargs = mock_from_env.call_args
    assert kwargs["timeout"] == 5
    assert kwargs["version"] == DEFAULT_DOCKER_API_VERSION


@patch("minisweagent.run.extra.docker_client.docker.from_env")
def test_docker_from_env_respects_docker_api_version_env(mock_from_env, monkeypatch):
    monkeypatch.setenv("DOCKER_API_VERSION", "1.49")
    from minisweagent.run.extra.docker_client import docker_from_env

    mock_from_env.reset_mock()
    docker_from_env()
    assert mock_from_env.call_args[1]["version"] == "1.49"


@patch("minisweagent.run.extra.docker_client.docker.from_env")
def test_docker_from_env_does_not_override_explicit_version(mock_from_env, monkeypatch):
    monkeypatch.setenv("DOCKER_API_VERSION", "1.49")
    from minisweagent.run.extra.docker_client import docker_from_env

    docker_from_env(version="1.30")
    assert mock_from_env.call_args[1]["version"] == "1.30"
