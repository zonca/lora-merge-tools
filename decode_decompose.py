#!/usr/bin/env python3
"""Decompose the decode rate of two Ollama models into tokens per forward pass.

decode rate = (tokens accepted per forward pass) * (forward passes per second)

The first factor is the speculative decoder: the server logs
`speculative decode stats` with the number of iterations, the drafts and the
accepted tokens, so tokens per pass is 1 + accepted/iterations. The second
factor is bandwidth bound on this hardware (every pass reads all the weights),
so seconds per pass should track the artifact size. Measuring both separates
"the adapter made the model accept fewer drafts" from "the artifact got bigger".

Models are unloaded before every measured run and alternate between
repetitions, so a change in machine state cannot favour one of them. Each run's
statistics are read from the slice of the log written during that run.

    python decode_decompose.py --url http://host:11434 \
        --models qwen3.8:27b-nvfp4 qwen3.8-pi:27b-nvfp4
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
PROMPT = (
    "Write about 500 words of plain prose explaining how a heat pump moves heat "
    "against a temperature gradient, covering the compressor, the expansion valve "
    "and the two heat exchangers. No lists."
)


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


def sizes(url: str) -> dict[str, int]:
    with urllib.request.urlopen(f"{url}/api/tags", timeout=60) as response:
        tags = json.load(response)
    return {m["name"]: m["size"] for m in tags["models"]}


def measure(url: str, host: str, path: str, model: str) -> dict:
    post(f"{url}/api/generate", {"model": model, "prompt": "", "keep_alive": 0})
    time.sleep(3)
    before = log_lines(host, path)
    result = post(
        f"{url}/api/generate",
        {
            "model": model, "prompt": PROMPT, "stream": False, "think": False,
            "options": {"temperature": 0.0, "seed": 21, "num_predict": 512},
        },
    )
    time.sleep(2)
    stats = [m for m in (STATS.search(line) for line in log_slice(host, path, before)) if m]

    iterations = sum(int(m.group(1)) for m in stats)
    accepted = sum(int(m.group(3)) for m in stats)
    decode_s = result["eval_duration"] / 1e9
    tokens = result["eval_count"]
    tokens_per_iter = (1 + accepted / iterations) if iterations else 0.0
    return {
        "model": model,
        "tokens": tokens,
        "decode_s": decode_s,
        "tok_s": tokens / decode_s if decode_s else 0.0,
        "iterations": iterations,
        "accepted": accepted,
        "acceptance": accepted / sum(int(m.group(2)) for m in stats) if stats else 0.0,
        "tokens_per_iter": tokens_per_iter,
        "s_per_iter": decode_s / iterations if iterations else 0.0,
        "load_s": result["load_duration"] / 1e9,
        "prompt_tokens": result["prompt_eval_count"],
        "stats_lines": len(stats),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--ssh-host", default="macmini", help="host that owns the Ollama log")
    ap.add_argument("--log", default="/tmp/ollama.err.log")
    ap.add_argument("--models", nargs=2, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    size = sizes(args.url)
    runs: dict[str, list[dict]] = {m: [] for m in args.models}
    for rep in range(args.reps):
        order = args.models if rep % 2 == 0 else list(reversed(args.models))
        for model in order:
            r = measure(args.url, args.ssh_host, args.log, model)
            r["artifact_gb"] = size.get(model, 0) / 1e9
            runs[model].append(r)
            print(f"  rep {rep + 1} {model}: {r['tok_s']:.1f} tok/s, "
                  f"{r['tokens_per_iter']:.2f} tok/pass, {r['s_per_iter'] * 1000:.1f} ms/pass, "
                  f"acceptance {r['acceptance']:.2f}", flush=True)

    print("\n| model | decode | tokens/pass | ms/pass | acceptance | GB/pass (effective) |")
    print("|---|---|---|---|---|---|")
    summary = {}
    for model in args.models:
        rows = runs[model]
        med = lambda k: statistics.median(r[k] for r in rows)  # noqa: E731
        summary[model] = {k: med(k) for k in
                          ("tok_s", "tokens_per_iter", "s_per_iter", "acceptance", "artifact_gb")}
        bw = summary[model]["artifact_gb"] / summary[model]["s_per_iter"]
        print(f"| `{model}` | {med('tok_s'):.1f} tok/s | {med('tokens_per_iter'):.2f} | "
              f"{med('s_per_iter') * 1000:.1f} | {med('acceptance'):.2f} | {bw:.0f} GB/s |")

    a, b = (summary[m] for m in args.models)
    print(f"\nartifact ratio b/a:      {b['artifact_gb'] / a['artifact_gb']:.3f}")
    print(f"tokens per pass ratio:   {b['tokens_per_iter'] / a['tokens_per_iter']:.3f}")
    print(f"seconds per pass ratio:  {b['s_per_iter'] / a['s_per_iter']:.3f}")
    print(f"decode ratio:            {b['tok_s'] / a['tok_s']:.3f}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(runs, fh, indent=2)
        print(f"raw runs: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
