#!/usr/bin/env python3
"""Benchmark token generation and prefill on an Ollama server, model by model.

Three measurements per model, the ones that matter for an agent workload:

* **prefill** (prompt processing), because an agent turn re-reads a long
  conversation and prefill is what a long context costs;
* **decode** (token generation), because that is what a reply costs;
* **load** (cold start), which is a one-off but ignores nothing.

Requests go to `/api/generate` with `stream: false`, so the server reports its
own counters (`prompt_eval_count`, `eval_count`, `*_duration`) next to the wall
clock. Every prompt carries a nonce and a unique filler, so repeat runs are not
served from Ollama's prompt cache, and each model is unloaded (`keep_alive: 0`)
before its first request so the load time is real.

    python bench_ollama.py --url http://127.0.0.1:11434 \
        --models qwen3.8:27b-nvfp4 qwen3.8-pi:27b-nvfp4

The decode number is also the multi-token prediction canary: a model whose MTP
head survived decodes far faster than memory bandwidth alone allows, and one
that lost it drops to roughly the bandwidth bound. Same-machine comparison of
the two models is what makes that visible.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request


def post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def one_run(base_url: str, model: str, prompt: str, num_predict: int, keep_alive) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
            "seed": 42,
        },
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    t0 = time.monotonic()
    result = post(f"{base_url}/api/generate", payload)
    wall = time.monotonic() - t0
    return {
        "wall_s": wall,
        "load_s": result.get("load_duration", 0) / 1e9,
        "prompt_tokens": result.get("prompt_eval_count", 0),
        "prompt_s": result.get("prompt_eval_duration", 0) / 1e9,
        "tokens": result.get("eval_count", 0),
        "decode_s": result.get("eval_duration", 0) / 1e9,
        "total_s": result.get("total_duration", 0) / 1e9,
        "response": result.get("response", ""),
    }


def filler(tokens: int, seed: int) -> str:
    """Deterministic prose of roughly `tokens` tokens, different for each seed."""
    rng = random.Random(seed)
    words = (
        "shard tensor merge adapter delta weight matrix stride buffer pipeline kernel "
        "vector scalar gradient memory bandwidth latency cache queue thread socket "
        "packet digest header offset frame window segment block cluster node replica "
        "checksum manifest pointer region extent mapping anchor horizon lattice"
    ).split()
    return " ".join(rng.choice(words) for _ in range(tokens))


def unload(base_url: str, model: str) -> None:
    try:
        post(f"{base_url}/api/generate", {"model": model, "prompt": "", "keep_alive": 0}, timeout=120)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        pass


def bench_model(base_url: str, model: str, runs: int, decode_tokens: int,
                prefill_lengths: list[int]) -> dict:
    print(f"\n== {model}")
    unload(base_url, model)

    cold = one_run(base_url, model, "Reply with the single word: ready", 16, None)
    print(f"  load: {cold['load_s']:.1f} s (cold start, {cold['tokens']} tokens generated)")

    decode = []
    for i in range(runs):
        prompt = (
            f"Conversation nonce {i}-{random.randrange(10**9)}. Explain in detail how a "
            "heat pump moves heat against a temperature gradient, covering the compressor, "
            "the expansion valve and the two heat exchangers."
        )
        r = one_run(base_url, model, prompt, decode_tokens, "30m")
        rate = r["tokens"] / r["decode_s"] if r["decode_s"] else 0.0
        decode.append(rate)
        print(f"  decode run {i + 1}: {r['tokens']} tokens in {r['decode_s']:.2f} s "
              f"= {rate:.1f} tok/s (wall {r['wall_s']:.2f} s)")

    prefill = {}
    for length in prefill_lengths:
        rates, counts = [], []
        for i in range(2):
            prompt = (
                f"Do not repeat the text. Nonce {i}-{random.randrange(10**9)}.\n"
                f"{filler(length, seed=i * 1000 + length)}\n"
                "Answer with one word: how many words were above?"
            )
            r = one_run(base_url, model, prompt, 1, "30m")
            if r["prompt_s"]:
                rates.append(r["prompt_tokens"] / r["prompt_s"])
                counts.append(r["prompt_tokens"])
        if rates:
            prefill[length] = {
                "tokens": int(statistics.mean(counts)),
                "tok_per_s": statistics.mean(rates),
            }
            print(f"  prefill ~{prefill[length]['tokens']} tokens: "
                  f"{prefill[length]['tok_per_s']:.0f} tok/s")

    return {
        "model": model,
        "load_s": cold["load_s"],
        "decode_tok_per_s": statistics.median(decode) if decode else 0.0,
        "decode_runs": decode,
        "prefill": prefill,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:11434", help="Ollama base URL")
    ap.add_argument("--models", nargs="+", required=True, help="model tags to compare")
    ap.add_argument("--runs", type=int, default=3, help="decode runs per model (median reported)")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--prefill-lengths", type=int, nargs="+", default=[2000, 8000],
                    help="approximate prompt lengths in tokens")
    ap.add_argument("--json-out", help="write the raw results here")
    args = ap.parse_args()

    results = [
        bench_model(args.url, m, args.runs, args.decode_tokens, args.prefill_lengths)
        for m in args.models
    ]

    print("\n| model | load (s) | decode (tok/s) | "
          + " | ".join(f"prefill {n} tok" for n in args.prefill_lengths) + " |")
    print("|---|---|---|" + "---|" * len(args.prefill_lengths))
    for r in results:
        cells = []
        for n in args.prefill_lengths:
            p = r["prefill"].get(n)
            cells.append(f"{p['tok_per_s']:.0f} tok/s" if p else "n/a")
        print(f"| `{r['model']}` | {r['load_s']:.1f} | {r['decode_tok_per_s']:.1f} | "
              + " | ".join(cells) + " |")

    if len(results) > 1:
        base, other = results[0]["decode_tok_per_s"], results[-1]["decode_tok_per_s"]
        if base and other:
            print(f"\ndecode ratio {results[-1]['model']} / {results[0]['model']}: "
                  f"{other / base:.2f}x")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"raw results: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
