"""Host-local Polar runtime for SearchR1 smoke tests.

This runtime intentionally executes harness commands on the gateway host instead
of starting Docker/Apptainer.  It is useful when the rollout task only needs the
current Python environment and local services (gateway, retrieval server, VERL
SGLang endpoint).  Do not use it for untrusted tasks.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec


class LocalRuntime(BaseRuntime):
    """A minimal runtime that maps runtime paths directly to host paths."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        self.runtime_session_dir = str(session_dir)
        self.runtime_artifacts_dir = str(self.artifacts_dir)
        self.runtime_logs_dir = str(session_dir / "logs")
        self.runtime_agent_log_dir = str(session_dir / "logs" / "agent")

    @property
    def runtime_id(self) -> str:
        return f"local:{self.session_id}"

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("local runtime was already destroyed")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "logs" / "agent").mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        self._destroyed = True

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        effective_env = {**os.environ, **self.spec.env, **(env or {})}
        effective_cwd = cwd or self.spec.workdir or self.runtime_session_dir
        Path(effective_cwd).mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=effective_cwd,
            env=effective_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._active_process = process
        try:
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await process.wait()
                except ProcessLookupError:
                    pass
                return ExecResult(return_code=-1)
        finally:
            self._active_process = None
        return ExecResult(
            stdout=stdout.decode(errors="replace") if stdout else None,
            stderr=stderr.decode(errors="replace") if stderr else None,
            return_code=process.returncode or 0,
        )

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        target = Path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        target = Path(remote_path)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_path, target)

    async def download_file(self, remote_path: str, local_path: str) -> None:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, target)

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        target = Path(local_path)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(remote_path, target)
