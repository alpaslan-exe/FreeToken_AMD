"""Minimal GGUF header reader (no dependencies).

Reads metadata and the tensor table, then summarizes how many bytes live in
MoE expert tensors vs. everything else, per layer. That split is what decides
what can be pinned on the GPU.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

# ggml type id -> bytes per element (block-quant formats expressed as averages)
_TYPE_BYTES = {
    0: 4.0, 1: 2.0, 2: 18 / 32, 3: 20 / 32, 6: 22 / 32, 7: 24 / 32, 8: 34 / 32, 9: 36 / 32,
    10: 84 / 256, 11: 110 / 256, 12: 144 / 256, 13: 176 / 256, 14: 210 / 256, 15: 256 / 256,
    16: 2 / 8, 17: 2 / 8, 18: 2 / 8, 19: 0.5, 20: 2 / 8, 21: 2 / 8, 22: 2 / 8, 23: 2 / 8,
    24: 1.0, 25: 2.0, 26: 4.0, 27: 8.0, 28: 1.0, 29: 0.5, 30: 2.0, 31: 1.0, 32: 1.0,
    33: 1.0, 34: 1.0, 35: 1.0, 36: 1.0, 37: 1.0, 38: 0.5, 39: 0.5,
}
_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 38: "MXFP4",
}
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@dataclass
class Tensor:
    name: str
    dims: list[int]
    type_id: int
    offset: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return int(self.n_elements * _TYPE_BYTES.get(self.type_id, 1.0))

    @property
    def type_name(self) -> str:
        return _TYPE_NAMES.get(self.type_id, f"type{self.type_id}")

    @property
    def layer(self) -> int | None:
        parts = self.name.split(".")
        if len(parts) > 1 and parts[0] == "blk" and parts[1].isdigit():
            return int(parts[1])
        return None

    @property
    def is_expert(self) -> bool:
        return "_exps." in self.name or self.name.endswith("_exps.weight")


@dataclass
class GGUFInfo:
    path: str
    version: int
    metadata: dict[str, Any]
    tensors: list[Tensor] = field(default_factory=list)

    @property
    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", "?"))

    def meta(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.arch}.{key}", default)

    @property
    def n_layers(self) -> int:
        return int(self.meta("block_count", 0))

    @property
    def n_experts(self) -> int:
        return int(self.meta("expert_count", 0))

    @property
    def n_experts_used(self) -> int:
        return int(self.meta("expert_used_count", 0))

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    def expert_bytes_per_layer(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for t in self.tensors:
            if t.is_expert and t.layer is not None:
                out[t.layer] = out.get(t.layer, 0) + t.nbytes
        return out

    def non_expert_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors if not t.is_expert)

    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def expert_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors if t.is_expert)

    def n_full_attention_layers(self) -> int:
        """Layers that keep a KV cache (hybrid SSM/attention models alternate)."""
        interval = self.meta("full_attention_interval")
        n = self.n_layers
        if interval:
            return n // int(interval)
        return n

    def kv_bytes_per_token(self, kv_type_bytes: float = 1.0) -> int:
        """Rough KV-cache bytes per context token (K + V)."""
        n_head_kv = self.meta("attention.head_count_kv", self.meta("attention.head_count", 0))
        if isinstance(n_head_kv, list):
            n_head_kv = max(n_head_kv) if n_head_kv else 0
        head_dim = self.meta("attention.key_length") or (
            int(self.meta("embedding_length", 0)) // max(1, int(self.meta("attention.head_count", 1)))
        )
        return int(2 * self.n_full_attention_layers() * int(n_head_kv) * int(head_dim) * kv_type_bytes)


def _read_string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_value(f: BinaryIO, type_id: int) -> Any:
    if type_id == 8:
        return _read_string(f)
    if type_id == 9:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, elem_type) for _ in range(n)]
    fmt = _SCALARS[type_id]
    (v,) = struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))
    return v


def read_gguf(path: str) -> GGUFInfo:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_string(f)
            (type_id,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_value(f, type_id)
        info = GGUFInfo(path=path, version=version, metadata=metadata)
        for _ in range(n_tensors):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (type_id,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            info.tensors.append(Tensor(name, dims, type_id, offset))
    return info


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"
