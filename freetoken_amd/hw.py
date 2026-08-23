"""Hardware and runtime probing (static facts; bandwidth is learned from benchmarks)."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

from .config import Settings


@dataclass
class GPU:
    name: str
    backend: str
    total_mib: int
    free_mib: int


@dataclass
class HostInfo:
    cpu_model: str = ""
    threads: int = 0
    cores: int = 0
    cpu_flags: list[str] = field(default_factory=list)
    ram_total_mib: int = 0
    ram_avail_mib: int = 0
    gpus: list[GPU] = field(default_factory=list)
    vram_used_mib_sysfs: int | None = None
    render_access: bool = False
    llama_server: str = ""
    llama_server_version: str = ""
    backend_lib: str = ""
    ollama_resident: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


_INTERESTING_FLAGS = ("avx", "avx2", "avx512f", "avx512_bf16", "amx_tile", "fma", "f16c", "sse4_2")


def _cpuinfo() -> tuple[str, list[str]]:
    model, flags = "", []
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                have = set(line.split(":", 1)[1].split())
                flags = [f for f in _INTERESTING_FLAGS if f in have]
    except OSError:
        pass
    return model, flags


def _meminfo() -> tuple[int, int]:
    total = avail = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024
    except OSError:
        pass
    return total, avail


def _cores() -> int:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
        m = re.search(r"Core\(s\) per socket:\s+(\d+)", out)
        s = re.search(r"Socket\(s\):\s+(\d+)", out)
        if m:
            return int(m.group(1)) * (int(s.group(1)) if s else 1)
    except OSError:
        pass
    return os.cpu_count() or 0


def list_devices(settings: Settings) -> list[GPU]:
    """Ask llama-server which ggml devices it can see (Vulkan via GGML_BACKEND_PATH)."""
    if not settings.llama_server:
        return []
    try:
        out = subprocess.run(
            [settings.llama_server, "--list-devices"],
            capture_output=True, text=True, env=settings.child_env(), timeout=60, check=False,
        ).stdout + ""
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.splitlines():
        m = re.match(r"\s*(\w+?)(\d*):\s+(.+?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB free\)", line)
        if m:
            gpus.append(GPU(name=m.group(3), backend=m.group(1), total_mib=int(m.group(4)), free_mib=int(m.group(5))))
    return gpus


def ollama_resident() -> list[str]:
    if not shutil.which("ollama"):
        return []
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.split()[0] for line in out.splitlines()[1:] if line.strip()]


def probe(settings: Settings) -> HostInfo:
    model, flags = _cpuinfo()
    total, avail = _meminfo()
    info = HostInfo(cpu_model=model, threads=os.cpu_count() or 0, cores=_cores(), cpu_flags=flags,
                    ram_total_mib=total, ram_avail_mib=avail)
    info.llama_server = settings.llama_server
    info.backend_lib = settings.backend_lib
    if settings.llama_server:
        try:
            v = subprocess.run([settings.llama_server, "--version"], capture_output=True, text=True, timeout=20, check=False)
            info.llama_server_version = (v.stdout + v.stderr).strip().splitlines()[0] if (v.stdout or v.stderr) else ""
        except (OSError, subprocess.TimeoutExpired):
            pass
    info.render_access = any(os.access(p, os.R_OK | os.W_OK) for p in glob.glob("/dev/dri/renderD*"))
    used = glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
    if used:
        try:
            info.vram_used_mib_sysfs = int(open(used[0]).read()) // (1024 * 1024)
        except (OSError, ValueError):
            pass
    info.gpus = list_devices(settings)
    info.ollama_resident = ollama_resident()
    return info


def describe(info: HostInfo) -> str:
    lines = [
        f"CPU        {info.cpu_model}  ({info.cores} cores / {info.threads} threads)",
        f"CPU flags  {' '.join(info.cpu_flags) or '-'}",
        f"RAM        {info.ram_total_mib} MiB total, {info.ram_avail_mib} MiB available",
        f"llama      {info.llama_server or 'NOT FOUND'}  {info.llama_server_version}",
        f"backend    {info.backend_lib or 'NOT FOUND'}",
        f"render     {'ok' if info.render_access else 'NO ACCESS to /dev/dri/renderD* (add user to the render group)'}",
    ]
    if info.gpus:
        for g in info.gpus:
            lines.append(f"GPU        {g.backend}: {g.name}  {g.total_mib} MiB total, {g.free_mib} MiB free")
    else:
        lines.append("GPU        none visible to llama-server (backend lib / render access?)")
    if info.vram_used_mib_sysfs is not None:
        lines.append(f"VRAM used  {info.vram_used_mib_sysfs} MiB (sysfs)")
    if info.ollama_resident:
        lines.append(f"ollama     resident models: {', '.join(info.ollama_resident)}  (they hold VRAM!)")
    if "avx2" not in info.cpu_flags:
        lines.append("note       CPU lacks AVX2: expert math on the CPU will be slow; expect low-teens tok/s at best")
    return "\n".join(lines)
