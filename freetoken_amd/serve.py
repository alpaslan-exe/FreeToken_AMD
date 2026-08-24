"""Run the chosen configuration as a plain OpenAI-compatible llama-server (foreground)."""

from __future__ import annotations

import json
import os
import signal
import sys

from .config import Settings
from .server import LlamaServer, ServerConfig


# FreeToken keeps recurrent state at semantic boundaries so later turns reuse
# earlier compute. llama-server's nearest equivalents, all safe to always-on:
#   --cache-reuse N   reuse cached KV even when the prefix SHIFTS (agent history
#                     grows/truncates), in >=N-token chunks, via KV shifting
#   --slot-save-path  persist per-slot KV so a reconnecting client resumes
#                     without recomputing its prefix
# Prompt caching itself is already on by default (cache_prompt in requests).
CACHE_ARGS = ["--cache-reuse", "256"]


def load_best(path: str, cache: bool = True) -> ServerConfig:
    data = json.load(open(path))
    rows = [r for r in data if not r.get("error") and r.get("decode_tps")]
    if not rows:
        sys.exit(f"no successful runs in {path}")
    r = max(rows, key=lambda r: r["decode_tps"])
    args = list(r["args"])
    if cache and "--cache-reuse" not in args:
        args += CACHE_ARGS
    return ServerConfig(label=r["label"], args=args, ctx=r.get("ctx"))


def serve(settings: Settings, cfg: ServerConfig, log_path: str) -> int:
    srv = LlamaServer(settings, cfg, log_path)
    print(f"[freetoken-amd] starting {cfg.label}: {' '.join(srv.argv())}", file=sys.stderr)
    st = srv.start()
    print(f"[freetoken-amd] ready in {st.ready_seconds:.0f}s; GPU model buffer {st.gpu_model_mib} MiB, "
          f"host {st.host_model_mib} MiB, offloaded {st.offloaded}", file=sys.stderr)
    print(f"[freetoken-amd] OpenAI-compatible API: {srv.base}/v1  (loopback only; put a token proxy in front for LAN use)",
          file=sys.stderr)

    def _stop(signum, frame):
        srv.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        return srv.proc.wait() if srv.proc else 1
    finally:
        srv.stop()


def systemd_unit(settings: Settings, cfg: ServerConfig, user: str) -> str:
    argv = LlamaServer(settings, cfg, os.devnull).argv()
    env = settings.child_env()
    lines = [
        "[Unit]", "Description=FreeToken_AMD llama-server (loopback)", "After=network.target", "",
        "[Service]", f"User={user}", "SupplementaryGroups=render video",
        f'Environment="GGML_BACKEND_PATH={env.get("GGML_BACKEND_PATH", "")}"',
        f'Environment="LD_LIBRARY_PATH={env.get("LD_LIBRARY_PATH", "")}"',
        "ExecStart=" + " ".join(argv), "Restart=on-failure", "", "[Install]", "WantedBy=multi-user.target", "",
    ]
    return "\n".join(lines)
