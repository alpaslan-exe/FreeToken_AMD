"""Expert placement planner.

FreeToken keeps attention/routing on the GPU and experts in host RAM, then
fills leftover VRAM with a cache of experts. llama.cpp cannot move tensors at
run time, so the closest static equivalent is:

    --n-gpu-layers 999      everything that is not an expert goes to the GPU
    --n-cpu-moe N           experts of the first N layers stay in host RAM
                            (so the LAST n_layers-N layers' experts are on GPU)
    --override-tensor ...   finer: pin individual expert tensors by regex

The planner reads the GGUF, estimates what must live on the GPU (non-expert
weights + KV cache + compute buffers), and turns the remaining VRAM into a
number of expert layers that fit. The bandwidth-adaptive split of the paper
(q* = m * B_pcie / B_host) becomes an empirical search in `bench`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gguf import GGUFInfo, human

MIB = 1024 * 1024


@dataclass
class Plan:
    n_layers: int
    expert_layer_bytes: dict[int, int]
    non_expert_bytes: int
    kv_bytes: int
    compute_reserve_bytes: int
    vram_budget_bytes: int
    n_gpu_expert_layers: int
    n_cpu_moe: int
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_expert_bytes(self) -> int:
        layers = sorted(self.expert_layer_bytes)
        return sum(self.expert_layer_bytes[l] for l in layers[self.n_cpu_moe:])

    @property
    def gpu_bytes_estimate(self) -> int:
        return self.non_expert_bytes + self.kv_bytes + self.compute_reserve_bytes + self.gpu_expert_bytes

    def llama_args(self) -> list[str]:
        return ["--n-gpu-layers", "999", "--n-cpu-moe", str(self.n_cpu_moe)]

    def describe(self) -> str:
        exps = self.expert_layer_bytes
        per_layer = (sum(exps.values()) / len(exps)) if exps else 0
        lines = [
            f"layers            {self.n_layers}  ({len(exps)} with experts, ~{human(per_layer)} experts/layer)",
            f"non-expert        {human(self.non_expert_bytes)}  -> GPU",
            f"experts total     {human(sum(exps.values()))}  -> host RAM (minus what fits below)",
            f"KV cache est.     {human(self.kv_bytes)}  (ctx-dependent)",
            f"compute reserve   {human(self.compute_reserve_bytes)}",
            f"VRAM budget       {human(self.vram_budget_bytes)}",
            f"=> --n-cpu-moe {self.n_cpu_moe}: experts of {self.n_gpu_expert_layers} layer(s) pinned on GPU "
            f"({human(self.gpu_expert_bytes)}), est. GPU use {human(self.gpu_bytes_estimate)}",
        ]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def make_plan(info: GGUFInfo, vram_mib: int, ctx: int, kv_type_bytes: float = 1.0,
              compute_reserve_mib: int = 768, safety_mib: int = 256) -> Plan:
    exps = info.expert_bytes_per_layer()
    non_expert = info.non_expert_bytes()
    kv = info.kv_bytes_per_token(kv_type_bytes) * ctx
    budget = vram_mib * MIB
    reserve = compute_reserve_mib * MIB
    free = budget - non_expert - kv - reserve - safety_mib * MIB
    notes = []
    if not info.is_moe:
        notes.append("model is not MoE: nothing to place, use plain --n-gpu-layers")
    if free < 0:
        notes.append("non-expert weights + KV do not fit: reduce --ctx or accept partial offload")
    layers = sorted(exps)
    n_gpu_layers = 0
    remaining = free
    # fill from the LAST layer backwards, matching what --n-cpu-moe N can express
    for l in reversed(layers):
        if exps[l] <= remaining:
            remaining -= exps[l]
            n_gpu_layers += 1
        else:
            break
    n_cpu_moe = len(layers) - n_gpu_layers
    return Plan(n_layers=info.n_layers, expert_layer_bytes=exps, non_expert_bytes=non_expert, kv_bytes=kv,
                compute_reserve_bytes=reserve, vram_budget_bytes=budget, n_gpu_expert_layers=n_gpu_layers,
                n_cpu_moe=n_cpu_moe, notes=notes)


def candidate_n_cpu_moe(plan: Plan, extra: int = 2) -> list[int]:
    """A small ladder around the planner's pick for the benchmark grid."""
    n = len(plan.expert_layer_bytes)
    picks = {n, plan.n_cpu_moe}
    for k in range(1, extra + 1):
        picks.add(min(n, plan.n_cpu_moe + k * 2))
        picks.add(max(0, plan.n_cpu_moe - k * 2))
    return sorted(p for p in picks if 0 <= p <= n)


def override_tensor_regex(layers_on_gpu: list[int], parts: tuple[str, ...] = ("ffn_up_exps", "ffn_gate_exps", "ffn_down_exps")) -> str:
    """Build an --override-tensor value pinning chosen layers' expert tensors to the GPU.

    Example: layers [38, 39] -> 'blk\\.(38|39)\\.ffn_(up|gate|down)_exps\\.weight=Vulkan0'
    Everything else is left to --n-cpu-moe / default placement.
    """
    if not layers_on_gpu:
        return ""
    alt = "|".join(str(l) for l in sorted(layers_on_gpu))
    return rf"blk\.({alt})\.ffn_(up|gate|down)_exps\.weight=Vulkan0"
