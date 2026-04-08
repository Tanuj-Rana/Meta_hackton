from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from meta_grid_env.models import (
    LoadBalancerAction,
    LoadBalancerObservation,
    LoadBalancerState,
    ResetResponse,
    StepResponse,
)


@dataclass
class _ContainerHandle:
    container_id: str
    base_url: str


class SmartGridEnv:
    def __init__(self, client: httpx.AsyncClient, container: _ContainerHandle | None = None) -> None:
        self._client = client
        self._container = container

    @classmethod
    async def from_url(cls, base_url: str) -> "SmartGridEnv":
        client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30.0)
        await cls._wait_until_ready(client)
        return cls(client=client)

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str]) -> "SmartGridEnv":
        if not image_name:
            raise ValueError("IMAGE_NAME or LOCAL_IMAGE_NAME must be set for from_docker_image().")
        run_proc = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "-d",
            "-P",
            image_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await run_proc.communicate()
        if run_proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {(stderr or stdout).decode().strip()}")
        container_id = stdout.decode().strip()

        port_proc = await asyncio.create_subprocess_exec(
            "docker",
            "port",
            container_id,
            "8000/tcp",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        port_stdout, port_stderr = await port_proc.communicate()
        if port_proc.returncode != 0:
            raise RuntimeError(f"docker port failed: {(port_stderr or port_stdout).decode().strip()}")
        host_port = port_stdout.decode().strip().rsplit(":", maxsplit=1)[-1]
        base_url = f"http://127.0.0.1:{host_port}"
        client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        await cls._wait_until_ready(client)
        return cls(client=client, container=_ContainerHandle(container_id=container_id, base_url=base_url))

    @staticmethod
    async def _wait_until_ready(client: httpx.AsyncClient) -> None:
        for _ in range(40):
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
        raise TimeoutError("Environment server did not become ready in time.")

    async def reset(self, task_name: str | None = None) -> ResetResponse:
        response = await self._client.post("/reset", json={"task_name": task_name, "seed": 0})
        response.raise_for_status()
        return ResetResponse.model_validate(response.json())

    async def step(self, action: LoadBalancerAction) -> StepResponse:
        response = await self._client.post("/step", json={"action": action.model_dump(mode="json")})
        response.raise_for_status()
        return StepResponse.model_validate(response.json())

    async def state(self) -> LoadBalancerState:
        response = await self._client.get("/state")
        response.raise_for_status()
        payload = response.json()
        return LoadBalancerState.model_validate(payload["state"])

    async def close(self) -> None:
        await self._client.aclose()
        if self._container:
            with contextlib.suppress(Exception):
                stop_proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "rm",
                    "-f",
                    self._container.container_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await stop_proc.communicate()


def action_to_log_string(action: LoadBalancerAction) -> str:
    compact = json.dumps(action.model_dump(mode="json"), separators=(",", ":"))
    return compact.replace(" ", "")
