#!/usr/bin/env python3
"""Attribute a decode slowdown to the draft head or to the artifact size.

A decode rate is two numbers multiplied together:

    decode rate = tokens accepted per forward pass x forward passes per second

The first factor is the speculative decoder, the second is bandwidth bound on an
Apple Silicon machine because every pass reads all the weights. Ollama logs both
faces of the product (`speculative decode stats` gives iterations, drafts and
accepted tokens), so reading the log slice written during a measured run gives
the split without touching the engine.

With `--sweep` the same measurement is repeated over four kinds of content, which
is what separates "this model drafts shallower" from "this content happens to be
hard": deterministic prose, code, an enumerative answer, and creative writing at
temperature 1.

Models are unloaded before every run and the order alternates between
repetitions, so machine state cannot favour one of them.

    python decode_decompose.py --url http://host:11434 --ssh-host macmini \
        --models qwen3.8:27b-nvfp4 qwen3.8-pi:27b-nvfp4 --reps 2 --sweep
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import urllib.request

STATS = re.compile(
    r"iterations=(\d+) drafted=(\d+) accepted=(\d+) acceptance=([0-9.]+) "
    r"avg_draft=([0-9.]+) max_draft=(\d+)"
)

CONTENTS = [
    {
        "name": "prose",
        "temperature": 0.0,
        "seed": 21,
        "prompt": (
            "Write about 500 words of plain prose explaining how a heat pump moves heat "
            "against a temperature gradient, covering the compressor, the expansion valve "
            "and the two heat exchangers. No lists."
        ),
    },
    {
        "name": "code",
        "temperature": 0.0,
        "seed": 22,
        "prompt": (
            "Write a Python module that parses a safetensors header from bytes, returns the "
            "tensor names with their dtype and shape, and merges two shard indexes. Include "
            "docstrings and about 15 lines of pytest tests. Output only the code."
        ),
    },
    {
        "name": "enumeration",
        "temperature": 0.0,
        "seed": 23,
        "prompt": (
            "List twenty concrete reasons why a sharded model checkpoint fails to load, one "
            "per line, each with a short explanation of how to fix it."
        ),
    },
    {
        "name": "creative",
        "temperature": 1.0,
        "seed": 24,
        "prompt": (
            "Write a 400 word short story about a lighthouse keeper who receives a letter "
            "addressed to someone who does not exist."
        ),
    },
]


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def log_lines(host: str, path: str) -> int:
    out = subprocess.run(
        ["ssh", host, f"wc -l < {path}"], capture_output=True, text=True, check=True
    ).stdout
    return int(out.strip())


def log_slice(host: str, path: str, start: int) -> list[str]:
    out = subprocess.run(
        ["ssh", host, f"sed -n '{start + 1},$p' {path} | grep -a 'speculative decode stats'"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def artifact_sizes(url: str) -> dict[str, int]:
    with urllib.request.urlopen(f"{url}/api/tags", timeout=60) as response:
        tags = json.load(response)
    return {m["name"]: m["size"] for m in tags["models"]}


def measure(url: str, host: str, path: str, model: str, content: dict,
            num_predict: int) -> dict:
    post(f"{url}/api/generate", {"model": model, "prompt": "", "keep_alive": 0})
    time.sleep(3)
    before = log_lines(host, path)
    result = post(
        f"{url}/api/generate",
        {
            "model": model, "prompt": content["prompt"], "stream": False, "think": False,
            "options": {
                "temperature": content["temperature"],
                "seed": content["seed"],
                "num_predict": num_predict,
            },
        },
    )
    time.sleep(2)
    stats = [m for m in (STATS.search(line) for line in log_slice(host, path, before)) if m]

    iterations = sum(int(m.group(1)) for m in stats)
    accepted = sum(int(m.group(3)) for m in stats)
    drafted = sum(int(m.group(2)) for m in stats)
    decode_s = result["eval_duration"] / 1e9
    return {
        "model": model,
        "content": content["name"],
        "tokens": result["eval_count"],
        "decode_s": decode_s,
        "tok_s": result["eval_count"] / decode_s if decode_s else 0.0,
        "iterations": iterations,
        "tokens_per_iter": (1 + accepted / iterations) if iterations else 0.0,
        "s_per_iter": decode_s / iterations if iterations else 0.0,
        "acceptance": accepted / drafted if drafted else 0.0,
        "stats_lines": len(stats),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--ssh-host", default="macmini", help="host that owns the Ollama log")
    ap.add_argument("--log", default="/tmp/ollama.err.log")
    ap.add_argument("--models", nargs=2, required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--num-predict", type=int, default=400)
    ap.add_argument("--sweep", action="store_true", help="all four content types")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    sizes = artifact_sizes(args.url)
    contents = CONTENTS if args.sweep else CONTENTS[:1]
    runs: dict[tuple[str, str], list[dict]] = {
        (m, c["name"]): [] for m in args.models for c in contents
    }

    for rep in range(args.reps):
        for content in contents:
            order = args.models if (rep + contents.index(content)) % 2 == 0 \
                else list(reversed(args.models))
            for model in order:
                r = measure(args.url, args.ssh_host, args.log, model, content,
                            args.num_predict)
                runs[(model, content["name"])].append(r)
                print(f"  rep {rep + 1} {content['name']:12s} {model:22s} "
                      f"{r['tok_s']:5.1f} tok/s  {r['tokens_per_iter']:.2f} tok/pass  "
                      f"{r['s_per_iter'] * 1000:5.1f} ms/pass  acc {r['acceptance']:.2f}",
                      flush=True)
                if args.json_out:
                    with open(args.json_out, "w") as fh:
                        json.dump({f"{m}|{c}": v for (m, c), v in runs.items()}, fh, indent=2)

    print("\n| content | model | decode | tokens/pass | ms/pass | acceptance |")
    print("|---|---|---|---|---|---|")
    for content in contents:
        for model in args.models:
            rows = runs[(model, content["name"])]
            if not rows:
                continue
            med = lambda k: statistics.median(r[k] for r in rows)  # noqa: E731
            print(f"| {content['name']} | `{model}` | {med('tok_s'):.1f} tok/s | "
                  f"{med('tokens_per_iter'):.2f} | {med('s_per_iter') * 1000:.1f} | "
                  f"{med('acceptance'):.2f} |")

    print("\n| content | decode ratio | tokens/pass ratio | ms/pass ratio |")
    print("|---|---|---|---|")
    for content in contents:
        a = runs[(args.models[0], content["name"])]
        b = runs[(args.models[1], content["name"])]
        med = lambda rows, k: statistics.median(r[k] for r in rows)  # noqa: E731
        print(f"| {content['name']} | {med(b, 'tok_s') / med(a, 'tok_s'):.2f} | "
              f"{med(b, 'tokens_per_iter') / med(a, 'tokens_per_iter'):.2f} | "
              f"{med(b, 's_per_iter') / med(a, 's_per_iter'):.2f} |")

    print(f"\nartifact sizes: " + ", ".join(f"{m} {s / 1e9:.2f} GB"
                                            for m, s in sizes.items()
                                            if m in args.models))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({f"{m}|{c}": v for (m, c), v in runs.items()}, fh, indent=2)
        print(f"raw runs: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
