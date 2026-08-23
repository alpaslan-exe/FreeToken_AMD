"""Render benchmark JSON as a Markdown table (for README / issues)."""

from __future__ import annotations

import json
import os


def _label_parts(label: str) -> tuple[str, str, str]:
    # ncmoe32_mtp2_t12 -> ("32", "mtp2", "12"); fit_default -> ("fit", "-", "-")
    if label.startswith("ncmoe"):
        parts = label.split("_")
        return parts[0][5:], parts[1] if len(parts) > 1 else "-", parts[2][1:] if len(parts) > 2 else "-"
    return label, "-", "-"


def render(path: str, top: int | None = None) -> str:
    rows = json.load(open(path))
    ok = [r for r in rows if not r.get("error")]
    ok.sort(key=lambda r: -r.get("decode_tps", 0))
    if top:
        ok = ok[:top]
    lines = [
        f"Source: `{os.path.basename(path)}`",
        "",
        "| config | experts on CPU (layers) | spec decode | threads | decode tok/s (prose / code) | draft accept | prefill tok/s | GPU model MiB | load s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in ok:
        n, spec, th = _label_parts(r["label"])
        pr, co = r["prompts"].get("prose", {}), r["prompts"].get("code", {})
        dn = (pr.get("draft_n") or 0) + (co.get("draft_n") or 0)
        da = (pr.get("draft_n_accepted") or 0) + (co.get("draft_n_accepted") or 0)
        acc = f"{100 * da / dn:.0f}%" if dn else "-"
        gpu = f"{r['gpu_model_mib']:.0f}" if r.get("gpu_model_mib") else "?"
        lines.append(
            f"| `{r['label']}` | {n} | {spec} | {th} | **{r['decode_tps']:.2f}** ({pr.get('predicted_per_second', 0) or 0:.2f} / "
            f"{co.get('predicted_per_second', 0) or 0:.2f}) | {acc} | {r['prefill_tps']:.1f} | {gpu} | {r['ready_seconds']:.0f} |"
        )
    errs = [r for r in rows if r.get("error")]
    if errs:
        lines += ["", f"{len(errs)} configuration(s) failed to load (usually: did not fit in VRAM):", ""]
        lines += [f"- `{r['label']}`: {r['error'].splitlines()[0][:120]}" for r in errs]
    return "\n".join(lines)
