"""Process orchestration for the LM Studio backend."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from .config import AISettings
from .lmstudio import LMStudioClient


class LMStudioNotInstalledError(RuntimeError):
    """Raised when the `lms` CLI cannot be located on PATH."""


@dataclass
class ProcessState:
    process: Optional[subprocess.Popen] = None
    started: bool = False


class LMStudioProcessManager:
    """Spawn and health-check LM Studio on demand."""

    def __init__(self, settings: AISettings, client: LMStudioClient) -> None:
        self._settings = settings
        self._client = client
        self._state = ProcessState()
        self._lock = asyncio.Lock()

    async def ensure_running(self) -> None:
        if not self._settings.auto_start:
            return
        async with self._lock:
            if self._state.started and await self._client.health():
                return
            if await self._client.health():
                self._state.started = True
                return
            if shutil.which("lms") is None:
                raise LMStudioNotInstalledError(
                    "LM Studio CLI (`lms`) not found on PATH. Install LM Studio and retry."
                )
            cmd = [
                "lms",
                "server",
                "start",
                "--port",
                str(self._settings.port),
                "--cors",
            ]
            self._state.process = subprocess.Popen(cmd)
            self._state.started = True
        await self._wait_for_health()

    async def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self._settings.poll_timeout_seconds
        while time.monotonic() < deadline:
            if await self._client.health():
                return
            await asyncio.sleep(self._settings.poll_interval_seconds)
        raise TimeoutError(
            "LM Studio failed to respond to /models within the configured timeout"
        )

    def terminate(self) -> None:
        if self._state.process and self._state.process.poll() is None:
            self._state.process.terminate()


__all__ = [
    "LMStudioNotInstalledError",
    "LMStudioProcessManager",
]
