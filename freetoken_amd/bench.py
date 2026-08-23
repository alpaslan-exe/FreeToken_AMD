"""Benchmark harness: run a grid of llama-server configurations and record timings.

Every configuration gets a fresh server (placement is decided at load time).
Timings come from llama-server's /completion response, not from log scraping.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

from .config import Settings
from .hw import ollama_resident
from .server import LlamaServer, ServerConfig

# Three workloads: plain prose (no repetition -> spec decode must earn it), code
# (structured, repetitive -> drafts accept), and a long prompt for prefill.
PROMPTS = {
    "prose": ("Write a 200 word essay about the ocean and why it matters to coastal cities.", 160),
    "code": ("Write a Python class `LRUCache` with get/put methods and O(1) operations, with docstrings and a small "
             "usage example at the bottom. Only output code.", 200),
    "prefill": ("Summarize the following in one sentence:\n" + ("The quick brown fox jumps over the lazy dog. " * 180), 32),
}


@dataclass
class RunResult:
    label: str
    args: list[str]
    ctx: int
    ready_seconds: float
    gpu_model_mib: float | None
    host_model_mib: float | None
    offloaded: str | None
    prompts: dict[str, dict] = field(default_factory=dict)
    error: str | None = None

    @property
    def decode_tps(self) -> float:
        """Mean decode tok/s over prose+code (prefill prompt excluded)."""
        vals = [p["predicted_per_second"] for k, p in self.prompts.items() if k != "prefill" and p.get("predicted_per_second")]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def prefill_tps(self) -> float:
        p = self.prompts.get("prefill") or {}
        return float(p.get("prompt_per_second") or 0.0)


def _timings(resp: dict) -> dict:
    t = resp.get("timings", {})
    keys = ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second",
            "draft_n", "draft_n_accepted")
    return {k: t.get(k) for k in keys}


def load_results(path: str) -> list["RunResult"]:
    rows = json.load(open(path))
    out = []
    for r in rows:
        out.append(RunResult(label=r["label"], args=r["args"], ctx=r["ctx"], ready_seconds=r["ready_seconds"],
                             gpu_model_mib=r.get("gpu_model_mib"), host_model_mib=r.get("host_model_mib"),
                             offloaded=r.get("offloaded"), prompts=r.get("prompts", {}), error=r.get("error")))
    return out


def gpu_power_always_on() -> None:
    """Keep amdgpu out of runtime suspend while benchmarking.

    On headless AMD cards the driver can runtime-suspend the GPU (BACO/D3cold)
    between requests; resume races mid-inference have hard-hung machines. The
    durable fix is `amdgpu.runpm=0` on the kernel command line; this just flips
    the sysfs knob for the current boot when we are allowed to.
    """
    import glob
    for p in glob.glob("/sys/class/drm/card*/device/power/control"):
        try:
            if open(p).read().strip() != "on" and os.access(p, os.W_OK):
                open(p, "w").write("on")
                print(f"set {p} = on (amdgpu runtime PM disabled for this boot)")
        except OSError:
            pass


def guard_ollama(allow_resident: bool = False) -> None:
    resident = ollama_resident()
    if resident and not allow_resident:
        sys.exit(f"ollama has models resident ({', '.join(resident)}); they hold VRAM and would skew results. "
                 f"Run `ollama stop <model>` or pass --allow-resident.")


def wait_for_ollama_idle(max_wait_s: int = 1200, poll_s: int = 30) -> None:
    """Block while Ollama holds models: two processes sharing an 8 GB card is how you get
    VRAM thrash (and, on this class of box, hangs). We never evict the user's models;
    we wait for keep-alive to expire."""
    waited = 0
    while waited < max_wait_s:
        resident = ollama_resident()
        if not resident:
            return
        print(f"   ollama has {', '.join(resident)} resident; waiting for it to unload ({waited}s)...", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def run_config(settings: Settings, cfg: ServerConfig, log_dir: str, prompts: dict | None = None,
               warmup: bool = True, settle_s: float = 10.0) -> RunResult:
    prompts = prompts or PROMPTS
    wait_for_ollama_idle()
    log_path = os.path.join(log_dir, f"{cfg.label}.log")
    srv = LlamaServer(settings, cfg, log_path)
    res = RunResult(label=cfg.label, args=cfg.args, ctx=cfg.ctx or settings.ctx, ready_seconds=0.0,
                    gpu_model_mib=None, host_model_mib=None, offloaded=None)
    try:
        st = srv.start()
        res.ready_seconds, res.gpu_model_mib, res.host_model_mib, res.offloaded = (
            st.ready_seconds, st.gpu_model_mib, st.host_model_mib, st.offloaded)
        if warmup:
            srv.completion("Say hello.", 8)
        for name, (prompt, n) in prompts.items():
            resp = srv.completion(prompt, n)
            res.prompts[name] = _timings(resp)
    except Exception as exc:  # noqa: BLE001 - record and move on to the next config
        res.error = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        srv.stop()
        time.sleep(settle_s)  # let the driver fully release the previous context before the next load
    return res


def fmt_row(r: RunResult) -> str:
    if r.error:
        return f"{r.label:<34} ERROR {r.error.splitlines()[0][:80]}"
    pr = r.prompts.get("prose", {})
    co = r.prompts.get("code", {})
    acc = ""
    dn = (pr.get("draft_n") or 0) + (co.get("draft_n") or 0)
    da = (pr.get("draft_n_accepted") or 0) + (co.get("draft_n_accepted") or 0)
    if dn:
        acc = f" acc {100 * da / dn:.0f}%"
    gpu = f"{r.gpu_model_mib:.0f}" if r.gpu_model_mib else "?"
    return (f"{r.label:<34} decode {r.decode_tps:6.2f} t/s (prose {pr.get('predicted_per_second', 0) or 0:5.2f}, "
            f"code {co.get('predicted_per_second', 0) or 0:5.2f}){acc}  prefill {r.prefill_tps:6.1f} t/s  "
            f"gpu {gpu} MiB  load {r.ready_seconds:.0f}s")


def save_results(results: list[RunResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) | {"decode_tps": r.decode_tps, "prefill_tps": r.prefill_tps} for r in results], f, indent=2)


def best(results: list[RunResult]) -> RunResult | None:
    ok = [r for r in results if not r.error and r.decode_tps > 0]
    return max(ok, key=lambda r: r.decode_tps) if ok else None
