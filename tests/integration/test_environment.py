"""F3.5 -- prerequisites. With ``FRAUDAI_DOCKER_TESTS=1`` these FAIL when absent.

The ``docker_env`` fixture fails (never skips) if the daemon is unreachable or the
sandbox image is missing; the versions are also printed in the pytest header so
Linux CI logs carry them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.integration.conftest import DockerEnv


def test_docker_daemon_and_sandbox_image_are_present(docker_env: DockerEnv) -> None:
    version = docker_env.client.version()
    image = docker_env.client.images.get(docker_env.image)

    assert version["Os"] == "linux"  # containers are Linux even on Docker Desktop
    assert image.attrs["Config"]["User"] not in ("", "root", "0")
