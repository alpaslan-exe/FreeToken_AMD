# FreeToken_AMD

**FreeToken-style edge MoE serving on AMD GPUs, built on llama.cpp's Vulkan backend.**

Run large Mixture-of-Experts models (Qwen3.6-35B-A3B and friends) on an AMD card
with a few GB of VRAM by putting attention, routing and the KV cache on the GPU,
keeping expert weights in system RAM, filling the leftover VRAM with as many
experts as fit, and using the model's built-in MTP head for speculative decoding.

## What this is / what this is not

[FreeToken](https://github.com/FlashML-org/FreeToken) (Shuo Yang et al., UC Berkeley /
FlashML, Apache-2.0) is a CUDA inference engine for NVIDIA RTX 30/40/50 GPUs. Its
results — Qwen3.6-35B at 39 tok/s on an 8 GB RTX 4060 laptop — come from four things:
a static GPU/CPU split of the model, a **dynamic LRU cache of hot experts in VRAM**,
**bandwidth-adaptive** CPU-vs-PCIe execution of cache misses, and AVX-512/AMX CPU
kernels under CUDA graphs.

This repository is **not a port of that engine**. It is a small, dependency-free
Python tool that reproduces the *recipe* on AMD hardware using what already exists in
upstream [llama.cpp](https://github.com/ggml-org/llama.cpp):

| FreeToken idea | FreeToken_AMD equivalent | Status |
| --- | --- | --- |
| Attention/router on GPU, experts in host RAM | `--n-gpu-layers 999 --n-cpu-moe N` (planner picks N from the GGUF + VRAM) | yes |
| Fill spare VRAM with experts | last `40-N` layers' experts pinned on GPU; `--override-tensor` regex for finer pinning | yes (static) |
| Bandwidth-adaptive split (q\* = m·B\_pcie/B\_host) | empirical: `bench` sweeps N and keeps the fastest | yes (empirical) |
| Speculative decoding across agent turns | `--spec-type draft-mtp` (Qwen3.6's baked-in MTP head) or `ngram-simple` | yes |
| Prefix/KV reuse between turns | llama-server prompt cache + `--cache-reuse` | yes (upstream) |
| Dynamic LRU expert cache in VRAM | not possible in llama.cpp (tensor placement is fixed at load) | no |
| CUDA graphs, AMX/AVX-512 expert kernels | Vulkan (RADV) + whatever SIMD the CPU has | no |

So expect the *shape* of FreeToken's results — a 24 GB model running on an 8 GB AMD card
with usable interactive speed — not its absolute numbers, which also need a modern CPU
and memory bus. See [Results](#results) for what this does on real hardware.

## Requirements

- Linux, an AMD GPU with a working Vulkan driver (Mesa RADV), and your user in the
  `render` group (`sudo usermod -aG render,video $USER`, then log in again — if you use
  SSH multiplexing, also `ssh -O exit host` to drop the old master connection).
- A llama.cpp `llama-server` with the Vulkan backend. **If you run Ollama you already
  have one**: Ollama ships upstream `llama-server` at `/usr/local/lib/ollama/llama-server`
  with `vulkan/libggml-vulkan.so` next to it; this tool auto-detects both and loads the
  backend via `GGML_BACKEND_PATH`.
- A GGUF of a MoE model. Ollama's blob store works directly (`ollama show <model> --modelfile`
  prints the `FROM /path/to/blob` line).
- Python 3.10+. No third-party packages.

## Install

```bash
git clone https://github.com/alpaslan-exe/FreeToken_AMD.git
cd FreeToken_AMD
python3 -m freetoken_amd --version     # runs from the checkout, nothing to install
# optional: pip install -e .            # gives you the `freetoken-amd` command
```

## Usage

```bash
export FTA_MODEL=/srv/ollama/blobs/sha256-...      # the GGUF (FROM line of `ollama show --modelfile`)

python3 -m freetoken_amd probe        # GPU seen by llama-server, VRAM, CPU flags, RAM, Ollama residents
python3 -m freetoken_amd plan         # read the GGUF, estimate what fits, propose --n-cpu-moe
python3 -m freetoken_amd bench        # grid: placement x spec-decode x threads; JSON + table
python3 -m freetoken_amd serve --from-results results/bench-<stamp>.json   # OpenAI API on 127.0.0.1:18436
python3 -m freetoken_amd systemd --from-results results/bench-<stamp>.json --user alp   # prints a unit, does not install it
```

`bench` refuses to run while Ollama holds models in VRAM (they would skew the numbers) —
`ollama stop <model>` first, or pass `--allow-resident` if you know what you are doing.
Everything binds to **127.0.0.1 only**: llama-server has no authentication; put a token
proxy in front for LAN use.

## Results

_(filled in from the benchmark harness; see `results/`)_

## Credits

This project exists because of **FreeToken** by Shuo Yang and collaborators at UC Berkeley
(FlashML-org). Their system design — GPU-resident attention with host-resident experts,
bandwidth-adaptive execution and semantic-aware caching — is what this tool approximates
with llama.cpp primitives, and their technical report is the reference for every design
choice here. Thank you for open-sourcing it.

- Code: https://github.com/FlashML-org/FreeToken (Apache-2.0)
- Paper: *FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution*,
  arXiv:2608.16157 — https://arxiv.org/abs/2608.16157
- Announcement: https://x.com/Andy_ShuoYang/status/2090856976880472439

```bibtex
@article{yang2026freetoken,
  title   = {FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author  = {Yang, Shuo and others},
  journal = {arXiv preprint arXiv:2608.16157},
  year    = {2026}
}
```

All the heavy lifting at run time is done by [llama.cpp](https://github.com/ggml-org/llama.cpp)
(ggml-org) and Mesa's RADV Vulkan driver. No code from FreeToken is used here.

## License

Apache-2.0, same as FreeToken. See [LICENSE](LICENSE).
