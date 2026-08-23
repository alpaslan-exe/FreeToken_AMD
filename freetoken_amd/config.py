"""Settings: where llama-server and the Vulkan backend live, and which model to serve."""

from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass, field

_LLAMA_CANDIDATES = (
    "/usr/local/lib/ollama/llama-server",  # Ollama ships an upstream llama-server with Vulkan libs
    "/usr/local/bin/llama-server",
    "/usr/bin/llama-server",
)
_BACKEND_GLOBS = (
    "/usr/local/lib/ollama/vulkan/libggml-vulkan.so",
    "/usr/local/lib/ollama/*/libggml-vulkan.so",
    "/usr/local/lib/libggml-vulkan.so",
    "/usr/lib/*/libggml-vulkan.so",
)


def find_llama_server() -> str:
    env = os.environ.get("FTA_LLAMA_SERVER")
    if env and os.path.exists(env):
        return env
    for c in _LLAMA_CANDIDATES:
        if os.path.exists(c):
            return c
    return shutil.which("llama-server") or ""


def find_backend_lib() -> str:
    env = os.environ.get("FTA_BACKEND_LIB")
    if env and os.path.exists(env):
        return env
    for pattern in _BACKEND_GLOBS:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return ""


@dataclass
class Settings:
    llama_server: str = field(default_factory=find_llama_server)
    backend_lib: str = field(default_factory=find_backend_lib)
    model: str = field(default_factory=lambda: os.environ.get("FTA_MODEL", ""))
    host: str = "127.0.0.1"  # loopback only: llama-server has no auth
    port: int = int(os.environ.get("FTA_PORT", "18436"))
    ctx: int = int(os.environ.get("FTA_CTX", "32768"))
    results_dir: str = field(default_factory=lambda: os.environ.get("FTA_RESULTS", "results"))

    def child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.backend_lib:
            # llama.cpp loads exactly the backend library named here (a file path, not a directory)
            env["GGML_BACKEND_PATH"] = self.backend_lib
            env.setdefault("LD_LIBRARY_PATH", os.path.dirname(os.path.dirname(self.backend_lib)))
        return env
