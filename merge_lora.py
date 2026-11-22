#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a sharded Hugging Face checkpoint, in float16.

Why this script exists
----------------------
`PeftModel.merge_and_unload()` needs the whole base model in RAM (about 55 GB for
a 27B model) and by default writes the result back as bfloat16. Neither is
acceptable here:

* the merge runs on machines that cannot hold 55 GB, so it works one shard at a
  time and never holds more than a couple of tensors;
* the result must be written as **float16**. A LoRA delta is tiny (around 1e-4)
  against weights of order 0.3, which is below bfloat16's resolution at that
  magnitude: saving the merge in bfloat16 silently throws away most of the
  adapter. float16 keeps it.

Untouched tensors are upcast from bfloat16 to float16 exactly (bfloat16 has a
wider exponent but a shorter mantissa, so every normal bfloat16 value is
representable in float16).

Usage
-----
    python merge_lora.py --base Qwen/Qwen3.8-27B \
                         --adapter harness-hub/Qwen3.8-27B-pi-adapted \
                         --out Qwen3.8-27B-pi-adapted-merged

`--base` and `--adapter` accept a local directory or a Hugging Face repo id.
`--lora-scale` defaults to `lora_alpha / r` from the adapter config, which is
PEFT's convention (2.0 for the pi adapter: alpha 64, r 32).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


# ---------------------------------------------------------------- safetensors

_TORCH_TO_ST_DTYPE = {
    torch.float16: "F16",
    torch.float32: "F32",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.int32: "I32",
    torch.int64: "I64",
}


def _write_header(fh, entries, metadata):
    """entries: list of (name, st_dtype_str, shape tuple, nbytes)."""
    header = {}
    offset = 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    fh.write(len(blob).to_bytes(8, "little"))
    fh.write(blob)
    return offset


def write_shard_streaming(path, src_path, names, deltas, scale, metadata=None):
    """Merge one input shard into one float16 output shard, tensor by tensor.

    Peak memory is one source tensor plus its float32 delta, not the shard.
    Returns a dict with per-tensor bookkeeping for verification.
    """
    entries = []
    with safe_open(src_path, framework="pt", device="cpu") as src:
        for name in names:
            if name not in deltas:
                shape = tuple(src.get_slice(name).get_shape())
                dtype = "F16"
                nbytes = 2 * int(torch.tensor(shape).prod().item())
            else:
                # delta is materialised once below and reused for the size pass
                shape = deltas[name]["shape"]
                dtype = "F16"
                nbytes = 2 * int(torch.tensor(shape).prod().item())
            entries.append((name, dtype, shape, nbytes))

        tmp_path = str(path) + ".tmp"
        with open(tmp_path, "wb") as out:
            _write_header(out, entries, metadata)
            report = {"merged": [], "copied": 0}
            for name in names:
                slice_ = src.get_slice(name)
                tensor = src.get_tensor(name)
                if name in deltas:
                    d = deltas[name]
                    delta = (d["B"].float() @ d["A"].float()) * scale
                    if tuple(delta.shape) != tuple(tensor.shape):
                        raise RuntimeError(
                            f"shape mismatch for {name}: delta {tuple(delta.shape)} "
                            f"vs weight {tuple(tensor.shape)}"
                        )
                    out_t = (tensor.float() + delta).to(torch.float16)
                    report["merged"].append(
                        {
                            "name": name,
                            "delta_absmax": float(delta.abs().max()),
                            "delta_rms": float(delta.pow(2).mean().sqrt()),
                            "weight_absmax": float(tensor.float().abs().max()),
                        }
                    )
                else:
                    if not tensor.is_floating_point():
                        raise RuntimeError(
                            f"{name}: source dtype {tensor.dtype} is not floating point. "
                            "This script converts every tensor to float16; handle integer "
                            "tensors (quantization scales, indices) separately."
                        )
                    out_t = tensor.to(torch.float16)
                    report["copied"] += 1
                del tensor, slice_
                out.write(out_t.contiguous().numpy().tobytes())
                del out_t
        os.replace(tmp_path, path)
    return report


# --------------------------------------------------------------------- lora

def load_adapter_deltas(adapter_path: Path, base_names: set[str], scale: float):
    """Return {base_tensor_name: {"A": ..., "B": ..., "shape": ...}}."""
    files = sorted(adapter_path.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no *.safetensors found in {adapter_path}")
    tensors = {}
    for f in files:
        tensors.update(load_file(str(f), device="cpu"))

    pairs: dict[str, dict] = {}
    for key, value in tensors.items():
        if key.endswith(".lora_A.weight"):
            base = key[: -len(".lora_A.weight")]
            pairs.setdefault(base, {})["A"] = value
        elif key.endswith(".lora_B.weight"):
            base = key[: -len(".lora_B.weight")]
            pairs.setdefault(base, {})["B"] = value
        else:
            raise SystemExit(f"unexpected adapter tensor: {key}")

    # PEFT stores keys as "base_model.model.<model tensor path>"; the base
    # checkpoint uses "<model tensor path>.weight".
    deltas = {}
    unmatched = []
    for base, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise SystemExit(f"incomplete LoRA pair for {base}")
        name = base[len("base_model.model.") :] + ".weight" if base.startswith("base_model.model.") else base + ".weight"
        if name not in base_names:
            unmatched.append(name)
            continue
        r, in_features = ab["A"].shape
        out_features, r_b = ab["B"].shape
        assert r == r_b, f"{base}: A rank {r} != B rank {r_b}"
        deltas[name] = {
            "A": ab["A"],
            "B": ab["B"],
            "shape": (out_features, in_features),
            "rank": r,
        }
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} adapter modules have no matching base tensor, "
            f"first ones: {unmatched[:5]}"
        )
    return deltas


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model directory or repo id")
    ap.add_argument("--adapter", required=True, help="adapter directory or repo id")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--lora-scale", type=float, default=None, help="default: lora_alpha / r")
    ap.add_argument("--shards", default=None, help="comma separated shard file names to merge (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="re-merge shards that already exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base)
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out)

    if not base_dir.is_dir():
        raise SystemExit("--base must be a local directory (download the base first)")
    if not adapter_dir.is_dir():
        raise SystemExit("--adapter must be a local directory")

    cfg = json.loads((base_dir / "config.json").read_text())
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = adapter_cfg["r"]
    alpha = adapter_cfg["lora_alpha"]
    if adapter_cfg.get("use_rslora"):
        raise SystemExit("rsLoRA adapters need scale = alpha / sqrt(r); not supported here")
    scale = args.lora_scale if args.lora_scale is not None else alpha / r

    print(f"base: {base_dir} ({len(shards)} shards, {len(weight_map)} tensors)")
    print(f"adapter: {adapter_dir} (r={r}, alpha={alpha}, scale={scale})")

    deltas = load_adapter_deltas(adapter_dir, set(weight_map), scale)
    touched = sorted(deltas)
    print(f"loRA pairs matched: {len(deltas)}")
    seen = set().union(*({n for n, s in weight_map.items() if s == sh} for sh in shards))
    orphan = set(deltas) - seen
    if orphan:
        raise SystemExit(f"adapter tensors missing from the index: {sorted(orphan)[:5]}")

    selected = shards if not args.shards else [s.strip() for s in args.shards.split(",")]
    for sh in selected:
        if sh not in shards:
            raise SystemExit(f"unknown shard {sh}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # small files first, so a partially merged directory is still understandable
    for name in os.listdir(base_dir):
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        src = base_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
    shutil.copy2(base_dir / "model.safetensors.index.json", out_dir / "model.safetensors.index.json")
    cfg_out = json.loads((out_dir / "config.json").read_text())
    for section in ("text_config", "vision_config", "audio_config"):
        if isinstance(cfg_out.get(section), dict) and "dtype" in cfg_out[section]:
            cfg_out[section]["dtype"] = "float16"
    cfg_out["torch_dtype"] = "float16"
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2) + "\n")
    print(f"config dtype set to float16; {len(os.listdir(out_dir))} non-weight files copied")

    reports = {}
    for i, sh in enumerate(selected, 1):
        dst = out_dir / sh
        if dst.exists() and not args.overwrite:
            print(f"[{i}/{len(selected)}] {sh}: exists, skipped")
            continue
        names = [n for n, s in weight_map.items() if s == sh]
        local = [n for n in names if n in deltas]
        print(f"[{i}/{len(selected)}] {sh}: {len(names)} tensors, {len(local)} merged", flush=True)
        if args.dry_run:
            continue
        report = write_shard_streaming(
            dst, base_dir / sh, names, deltas, scale, metadata={"format": "pt"}
        )
        reports[sh] = report
        print(f"    wrote {dst.name} ({dst.stat().st_size/1e9:.2f} GB), {report['copied']} copied", flush=True)

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    touched_layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in touched if "layers." in n})
    report = {
        "base": str(base_dir),
        "adapter": str(adapter_dir),
        "lora_rank": r,
        "lora_alpha": alpha,
        "lora_scale": scale,
        "merged_tensors": len(deltas),
        "merged_layers": [touched_layers[0], touched_layers[-1]] if touched_layers else [],
        "output_dtype": "float16",
        "shards": reports,
    }
    # running the script once per shard must not lose the earlier shards
    report_path = out_dir / "merge_report.json"
    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text())
            shards = previous.get("shards", {})
            shards.update(report["shards"])
            previous.update({k: v for k, v in report.items() if k != "shards"})
            previous["shards"] = shards
            report = previous
        except (json.JSONDecodeError, AttributeError):
            pass
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"done: {len(report['shards'])} shard(s) merged, report in {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
