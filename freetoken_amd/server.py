"""Launch and supervise one llama-server instance (loopback only)."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import Settings


@dataclass
class ServerConfig:
    label: str
    args: list[str] = field(default_factory=list)  # extra llama-server args (placement, spec decode, threads...)
    ctx: int | None = None
    env: dict[str, str] = field(default_factory=dict)  # e.g. LLAMA_ARG_* passthrough

    def to_dict(self) -> dict:
        return {"label": self.label, "args": self.args, "ctx": self.ctx, "env": self.env}


@dataclass
class LoadStats:
    ready_seconds: float = 0.0
    gpu_model_mib: float | None = None
    host_model_mib: float | None = None
    offloaded: str | None = None
    log_path: str = ""


class LlamaServer:
    def __init__(self, settings: Settings, cfg: ServerConfig, log_path: str):
        self.settings = settings
        self.cfg = cfg
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.stats = LoadStats(log_path=log_path)

    @property
    def base(self) -> str:
        return f"http://{self.settings.host}:{self.settings.port}"

    def argv(self) -> list[str]:
        s = self.settings
        return [
            s.llama_server, "--model", s.model, "--host", s.host, "--port", str(s.port),
            "--no-webui", "-np", "1", "-c", str(self.cfg.ctx or s.ctx), "--log-verbosity", "4",
            *self.cfg.args,
        ]

    def start(self, timeout: float = 600) -> LoadStats:
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        env = self.settings.child_env()
        env.update(self.cfg.env)
        log = open(self.log_path, "ab")
        t0 = time.time()
        self.proc = subprocess.Popen(self.argv(), stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log_path}\n{self.tail()}")
            if self.healthy():
                self.stats.ready_seconds = time.time() - t0
                self._parse_log()
                return self.stats
            time.sleep(1.0)
        self.stop()
        raise TimeoutError(f"llama-server not healthy after {timeout}s; see {self.log_path}")

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                return json.loads(r.read()).get("status") == "ok"
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def completion(self, prompt: str, n_predict: int, timeout: float = 900, **kw) -> dict:
        body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": True, **kw}).encode()
        req = urllib.request.Request(f"{self.base}/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def stop(self) -> None:
        """Shut down gently: SIGINT first so llama-server tears its Vulkan context down
        itself; escalate only if it does not exit. Abrupt kills of a process holding
        GPU buffers are exactly the moment amdgpu resets go wrong on some cards."""
        if self.proc and self.proc.poll() is None:
            for sig, grace in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                    self.proc.wait(timeout=grace)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.proc = None

    def tail(self, n: int = 15) -> str:
        try:
            return "".join(open(self.log_path, errors="replace").readlines()[-n:])
        except OSError:
            return ""

    def _parse_log(self) -> None:
        try:
            text = open(self.log_path, errors="replace").read()
        except OSError:
            return
        m = re.search(r"Vulkan0 model buffer size =\s+([\d.]+) MiB", text)
        if m:
            self.stats.gpu_model_mib = float(m.group(1))
        host = [float(x) for x in re.findall(r"(?:Vulkan_Host|CPU_Mapped|CPU) model buffer size =\s+([\d.]+) MiB", text)]
        if host:
            self.stats.host_model_mib = sum(host)
        m = re.search(r"offloaded (\d+/\d+) layers to GPU", text)
        if m:
            self.stats.offloaded = m.group(1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
