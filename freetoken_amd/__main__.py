"""Command line: probe | plan | bench | serve | systemd."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

from . import __version__
from .bench import PROMPTS, best, fmt_row, gpu_power_always_on, guard_ollama, load_results, run_config, save_results
from .config import Settings
from .gguf import human, read_gguf
from .hw import describe, probe
from .plan import calibrate, candidate_n_cpu_moe, make_plan, max_gpu_layers
from .report import render
from .serve import load_best, serve, systemd_unit
from .server import ServerConfig


def _settings(args) -> Settings:
    s = Settings()
    if getattr(args, "model", None):
        s.model = args.model
    if getattr(args, "port", None):
        s.port = args.port
    if getattr(args, "ctx", None):
        s.ctx = args.ctx
    if getattr(args, "llama_server", None):
        s.llama_server = args.llama_server
    if getattr(args, "backend_lib", None):
        s.backend_lib = args.backend_lib
    return s


def cmd_probe(args) -> int:
    s = _settings(args)
    info = probe(s)
    if args.json:
        print(info.to_json())
    else:
        print(describe(info))
    return 0


def cmd_plan(args) -> int:
    s = _settings(args)
    if not s.model:
        sys.exit("--model (or FTA_MODEL) is required")
    info = read_gguf(s.model)
    vram = args.vram_mib
    if not vram:
        gpus = probe(s).gpus
        vram = gpus[0].total_mib if gpus else 0
    if not vram:
        sys.exit("could not determine VRAM; pass --vram-mib")
    print(f"model   {s.model}")
    print(f"arch    {info.arch}  layers {info.n_layers}  experts {info.n_experts} (top-{info.n_experts_used})  "
          f"total {human(info.total_bytes())}")
    plan = make_plan(info, vram, s.ctx, kv_type_bytes=args.kv_bytes)
    print(plan.describe())
    print(f"llama-server args: {' '.join(plan.llama_args())}")
    print(f"grid ladder for --n-cpu-moe: {candidate_n_cpu_moe(plan)}")
    if args.calibrate:
        cal = calibrate(args.calibrate)
        if not cal:
            print("calibration: not enough measured points in that results file")
            return 0
        n = max_gpu_layers(cal, vram, args.headroom_mib, plan.kv_bytes / 2**20, plan.compute_reserve_bytes / 2**20)
        n_layers = len(plan.expert_layer_bytes)
        print(f"calibrated from {args.calibrate}: fixed {cal['fixed_mib']:.0f} MiB on GPU + {cal['per_layer_mib']:.0f} MiB per expert layer "
              f"(points: {cal['points']})")
        print(f"=> with {args.headroom_mib} MiB headroom: up to {n} expert layer(s) on GPU -> --n-cpu-moe {max(0, n_layers - n)}")
    return 0


def _grid(args, plan_n_cpu_moe: list[int]) -> list[ServerConfig]:
    specs = {
        "none": [],
        "ngram": ["--spec-type", "ngram-simple"],
        "mtp2": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        "mtp3": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        "mtp4": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
        "mtp2+ngram": ["--spec-type", "draft-mtp,ngram-simple", "--spec-draft-n-max", "2"],
    }
    spec_names = args.spec.split(",") if args.spec else ["none", "ngram", "mtp2", "mtp3"]
    threads = [int(t) for t in args.threads.split(",")] if args.threads else [os.cpu_count() or 8]
    cfgs = []
    for n_cpu_moe, spec, th in itertools.product(plan_n_cpu_moe, spec_names, threads):
        if spec not in specs:
            sys.exit(f"unknown spec '{spec}'; choose from {', '.join(specs)}")
        label = f"ncmoe{n_cpu_moe}_{spec}_t{th}"
        a = ["--n-gpu-layers", "999", "--n-cpu-moe", str(n_cpu_moe), "-t", str(th), *specs[spec], *args.extra]
        cfgs.append(ServerConfig(label=label, args=a))
    return cfgs


def cmd_bench(args) -> int:
    s = _settings(args)
    if not s.model:
        sys.exit("--model (or FTA_MODEL) is required")
    guard_ollama(args.allow_resident)
    info = read_gguf(s.model)
    gpus = probe(s).gpus
    vram = args.vram_mib or (gpus[0].total_mib if gpus else 0)
    plan = make_plan(info, vram, s.ctx)
    ladder = [int(x) for x in args.n_cpu_moe.split(",")] if args.n_cpu_moe else candidate_n_cpu_moe(plan)
    ladder = sorted(ladder, reverse=True)  # most experts on CPU first: safest configs produce data first
    if vram and not args.no_vram_guard:
        per_layer = sorted(plan.expert_layer_bytes.values())
        headroom = args.vram_headroom_mib * 1024 * 1024
        keep = []
        for n in ladder:
            est = plan.non_expert_bytes + plan.kv_bytes + plan.compute_reserve_bytes + sum(per_layer[n:])
            if est + headroom <= vram * 1024 * 1024:
                keep.append(n)
            else:
                print(f"skip --n-cpu-moe {n}: est. {est / 2**20:.0f} MiB + {args.vram_headroom_mib} MiB headroom > {vram} MiB VRAM "
                      f"(RADV starts paging VRAM over PCIe near the limit; that both tanks speed and has hung this class of box)")
        ladder = keep
    cfgs = _grid(args, ladder)
    if args.fit_baseline:
        cfgs.insert(0, ServerConfig(label="fit_default", args=["--fit", "on", *args.extra]))
    os.makedirs(s.results_dir, exist_ok=True)
    results = []
    if args.resume:
        out = args.resume
        stamp = os.path.basename(out)[6:-5]
        results = load_results(out)
        done = {r.label for r in results}
        cfgs = [c for c in cfgs if c.label not in done]
        print(f"resuming {out}: {len(results)} done, {len(cfgs)} to go")
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(s.results_dir, f"bench-{stamp}.json")
    log_dir = os.path.join(s.results_dir, f"logs-{stamp}")
    os.makedirs(log_dir, exist_ok=True)
    gpu_power_always_on()
    print(f"{len(cfgs)} configurations; model {os.path.basename(s.model)}; ctx {s.ctx}; results -> {out}")
    for i, cfg in enumerate(cfgs, 1):
        print(f"[{i}/{len(cfgs)}] {cfg.label} ...", flush=True)
        r = run_config(s, cfg, log_dir)
        results.append(r)
        print("   " + fmt_row(r), flush=True)
        save_results(results, out)
    b = best(results)
    if b:
        print(f"\nbest: {b.label}  decode {b.decode_tps:.2f} t/s  prefill {b.prefill_tps:.1f} t/s")
        print(f"args: {' '.join(b.args)}")
    return 0


def cmd_serve(args) -> int:
    s = _settings(args)
    if args.from_results:
        cfg = load_best(args.from_results)
    else:
        cfg = ServerConfig(label="manual", args=args.extra)
    if not s.model:
        sys.exit("--model (or FTA_MODEL) is required")
    os.makedirs(s.results_dir, exist_ok=True)
    return serve(s, cfg, os.path.join(s.results_dir, "serve.log"))


def cmd_report(args) -> int:
    print(render(args.results, args.top))
    return 0


def cmd_systemd(args) -> int:
    s = _settings(args)
    cfg = load_best(args.from_results) if args.from_results else ServerConfig(label="manual", args=args.extra)
    print(systemd_unit(s, cfg, args.user))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="freetoken-amd", description="FreeToken-style MoE serving on AMD/Vulkan via llama.cpp")
    p.add_argument("--version", action="version", version=f"freetoken-amd {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="GGUF path (or FTA_MODEL)")
    common.add_argument("--port", type=int, help="loopback port (default 18436)")
    common.add_argument("--ctx", type=int, help="context size (default 32768)")
    common.add_argument("--llama-server", help="path to llama-server (auto-detected)")
    common.add_argument("--backend-lib", help="path to libggml-vulkan.so (auto-detected)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", parents=[common], help="show what this machine can do")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_probe)

    sp = sub.add_parser("plan", parents=[common], help="read the GGUF and propose expert placement")
    sp.add_argument("--vram-mib", type=int)
    sp.add_argument("--kv-bytes", type=float, default=1.0, help="bytes/elem of KV cache (1.0 q8_0, 2.0 f16)")
    sp.add_argument("--calibrate", help="bench JSON: fit real GPU bytes (fixed + per expert layer) from measured runs")
    sp.add_argument("--headroom-mib", type=int, default=700)
    sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("bench", parents=[common], help="benchmark a grid of placements and spec-decode modes")
    sp.add_argument("--n-cpu-moe", help="comma list to override the planner ladder, e.g. 40,36,34")
    sp.add_argument("--spec", help="comma list of: none,ngram,mtp2,mtp3,mtp4,mtp2+ngram")
    sp.add_argument("--threads", help="comma list, e.g. 6,12")
    sp.add_argument("--vram-mib", type=int)
    sp.add_argument("--fit-baseline", action="store_true", help="also run llama-server's own --fit on")
    sp.add_argument("--allow-resident", action="store_true", help="do not abort when ollama holds models")
    sp.add_argument("--resume", help="bench JSON to continue (skips configs already recorded)")
    sp.add_argument("--vram-headroom-mib", type=int, default=700, help="skip placements estimated closer than this to full VRAM")
    sp.add_argument("--no-vram-guard", action="store_true")
    sp.add_argument("extra", nargs=argparse.REMAINDER, help="extra llama-server args after --")
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("serve", parents=[common], help="run the best (or given) configuration in the foreground")
    sp.add_argument("--from-results", help="bench JSON to pick the best config from")
    sp.add_argument("extra", nargs=argparse.REMAINDER)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("report", help="render a bench JSON as a Markdown table")
    sp.add_argument("results")
    sp.add_argument("--top", type=int)
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("systemd", parents=[common], help="print a systemd unit for the chosen config (not installed)")
    sp.add_argument("--from-results")
    sp.add_argument("--user", default=os.environ.get("USER", "llama"))
    sp.add_argument("extra", nargs=argparse.REMAINDER)
    sp.set_defaults(fn=cmd_systemd)

    args = p.parse_args(argv)
    if hasattr(args, "extra") and args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
