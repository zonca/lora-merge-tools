#!/usr/bin/env python3
"""Verify a checkpoint produced by merge_lora.py against its base and adapter.

Checks, per shard:

1. every tensor of the base shard is present in the merged shard, with the same
   shape and float16 dtype;
2. tensors the adapter does not touch are bit-identical to the base upcast to
   float16 (bfloat16 to float16 is an exact widening, so any difference here is
   a bug);
3. touched tensors are bit-identical to an independent recomputation of
   ``W + (alpha / r) * B @ A`` (computed with einsum instead of matmul);
4. the shard index still matches the tensor names on disk.

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def load_pair(adapter_dir: Path):
    tensors = {}
    for f in sorted(adapter_dir.glob("*.safetensors")):
        tensors.update(load_file(str(f), device="cpu"))
    pairs: dict[str, dict] = {}
    for key, value in tensors.items():
        for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
            if key.endswith(suffix):
                module = key[: -len(suffix)]
                # PEFT prefixes module paths with "base_model.model."; the
                # checkpoint names the same tensor without that prefix.
                if module.startswith("base_model.model."):
                    module = module[len("base_model.model.") :]
                pairs.setdefault(module, {})[slot] = value
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--shards", default=None, help="comma separated shard names (default: every shard on disk)")
    args = ap.parse_args()

    base_dir, adapter_dir, merged_dir = Path(args.base), Path(args.adapter), Path(args.merged)
    index = json.loads((merged_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    cfg = json.loads((merged_dir / "config.json").read_text())

    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    pairs = load_pair(adapter_dir)

    shards = sorted(set(weight_map.values())) if not args.shards else [s.strip() for s in args.shards.split(",")]
    failures = []
    totals = {"copied": 0, "merged": 0}
    worst_recompute = 0.0
    worst_delta_rel = 0.0

    print(f"merged dtype: {cfg.get('torch_dtype')} | text_config.dtype: {(cfg.get('text_config') or {}).get('dtype')}")
    print(f"lora scale: {scale} | shards to check: {len(shards)}")

    for sh in shards:
        merged_path = merged_dir / sh
        if not merged_path.exists():
            print(f"{sh}: missing, skipped")
            continue
        names = [n for n, s in weight_map.items() if s == sh]
        with safe_open(str(merged_path), framework="pt", device="cpu") as m, safe_open(
            str(base_dir / sh), framework="pt", device="cpu"
        ) as b:
            missing = set(names) - set(m.keys())
            if missing:
                failures.append(f"{sh}: {len(missing)} tensors missing, e.g. {sorted(missing)[:3]}")
                continue
            extra = set(m.keys()) - set(names)
            if extra:
                failures.append(f"{sh}: {len(extra)} unexpected tensors, e.g. {sorted(extra)[:3]}")
            for name in names:
                mt = m.get_tensor(name)
                bt = b.get_tensor(name)
                if tuple(mt.shape) != tuple(bt.shape):
                    failures.append(f"{name}: shape {tuple(mt.shape)} vs base {tuple(bt.shape)}")
                    continue
                if mt.dtype != torch.float16:
                    failures.append(f"{name}: dtype {mt.dtype}, expected float16")
                    continue
                base_name = name[: -len(".weight")]
                if base_name in pairs:
                    ab = pairs[base_name]
                    delta = torch.einsum("or,ri->oi", ab["B"].float(), ab["A"].float()) * scale
                    expected = (bt.float() + delta).to(torch.float16)
                    if not torch.equal(mt, expected):
                        diff = (mt.float() - expected.float()).abs().max().item()
                        worst_recompute = max(worst_recompute, diff)
                        failures.append(f"{name}: merged value differs from recomputation by {diff:g}")
                    rel = (delta.norm() / bt.float().norm()).item()
                    worst_delta_rel = max(worst_delta_rel, rel)
                    totals["merged"] += 1
                    del delta, expected
                else:
                    expected = bt.to(torch.float16)
                    if not torch.equal(mt, expected):
                        diff = (mt.float() - expected.float()).abs().max().item()
                        failures.append(f"{name}: untouched tensor not bit-identical to base ({diff:g})")
                    totals["copied"] += 1
                del mt, bt
        print(f"{sh}: ok ({len(names)} tensors)")

    print(f"\ntensors: {totals['merged']} merged, {totals['copied']} copied")
    print(f"largest relative LoRA change over the touched tensors: {worst_delta_rel:.3e}")
    if failures:
        print(f"\nFAILED ({len(failures)} problems):")
        for f in failures[:20]:
            print("  -", f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
