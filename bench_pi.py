#!/usr/bin/env python3
"""Run the same real tasks through pi on two models and compare the cost.

Each task is a prompt plus a tool allowlist. The runner gives every task a
fresh working directory, runs `pi -p --mode json` on it, and reads the event
stream for the numbers that describe an agent workload:

* **turns**, the number of model calls, which is what a tool-efficiency gain
  shows up as;
* **prompt tokens processed** (sum of `usage.input`), because every turn
  re-reads the whole conversation and that is the prefill cost;
* **generated tokens** (sum of `usage.output`);
* **wall clock** for the whole task.

The tasks are read-only by construction: the agent is told not to send, move,
comment or create anything, and the tool allowlist keeps `write` and `edit`
away from the Trello and email tasks.

    python bench_pi.py --model-a macmini/qwen3.8:27b-nvfp4 \
                       --model-b macmini/qwen3.8-pi:27b-nvfp4 \
                       --tasks tasks.json --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_pi(model: str, prompt: str, workdir: Path, tools: list[str] | None,
           timeout: int) -> dict:
    cmd = ["pi", "-p", "--no-session", "--mode", "json", "--model", model]
    if tools:
        cmd += ["--tools", ",".join(tools)]
    cmd.append(prompt)

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
    wall = time.monotonic() - t0

    turns = 0
    tool_calls: list[str] = []
    input_tokens = output_tokens = 0
    final_text = ""
    error = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "turn_end":
            turns += 1
            for result in event.get("toolResults") or []:
                tool_calls.append(result.get("toolName") or result.get("name") or "?")
        elif etype == "message_end":
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            if message.get("role") == "assistant":
                input_tokens += usage.get("input", 0)
                output_tokens += usage.get("output", 0)
                for block in message.get("content") or []:
                    if block.get("type") == "text":
                        final_text = block.get("text", "")
        elif etype == "error":
            error = str(event.get("error") or event)[:400]

    return {
        "wall_s": round(wall, 1),
        "turns": turns,
        "tool_calls": len(tool_calls),
        "tool_names": sorted(set(tool_calls)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_text": final_text[:4000],
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-600:] if proc.returncode else "",
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True, help="baseline model, e.g. macmini/qwen3.8:27b-nvfp4")
    ap.add_argument("--model-b", help="candidate model; omit to measure only the baseline")
    ap.add_argument("--tasks", required=True, help="JSON task file")
    ap.add_argument("--out", required=True, help="where to write the raw results JSON")
    ap.add_argument("--repeat", type=int, default=1, help="runs per task per model")
    ap.add_argument("--baseline-results",
                    help="reuse the model-a results from an earlier --out file instead of "
                         "running model-a again")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per task run")
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    models = [m for m in (args.model_a, args.model_b) if m]
    results: dict[str, dict] = {}

    reused = {}
    if args.baseline_results:
        reused = json.loads(Path(args.baseline_results).read_text())
        if args.model_a not in reused:
            raise SystemExit(f"{args.baseline_results} has no results for {args.model_a}")

    for model in models:
        if model in reused:
            print(f"reusing {model} from {args.baseline_results}")
            results[model] = reused[model]
            continue
        results[model] = {}
        for task in tasks:
            runs = []
            for rep in range(args.repeat):
                workdir = Path(tempfile.mkdtemp(prefix=f"pibench-{task['id']}-"))
                if task.get("setup"):
                    subprocess.run(task["setup"], shell=True, cwd=workdir, check=False)
                print(f"\n== {model} / {task['id']} (run {rep + 1})", flush=True)
                try:
                    r = run_pi(model, task["prompt"], workdir, task.get("tools"),
                               args.timeout)
                except subprocess.TimeoutExpired:
                    r = {"wall_s": args.timeout, "turns": 0, "tool_calls": 0,
                         "tool_names": [], "input_tokens": 0, "output_tokens": 0,
                         "final_text": "", "returncode": -1, "stderr_tail": "",
                         "error": f"timeout after {args.timeout}s"}
                print(f"   {r['wall_s']:.0f} s, {r['turns']} turns, "
                      f"{r['tool_calls']} tool calls, "
                      f"{r['input_tokens']} in / {r['output_tokens']} out", flush=True)
                if r["error"]:
                    print(f"   error: {r['error']}", flush=True)
                runs.append(r)
            results[model][task["id"]] = runs

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")

    print("\n| task | model | wall (s) | turns | tool calls | prompt tok | gen tok |")
    print("|---|---|---|---|---|---|---|")
    for task in tasks:
        for model in models:
            runs = results[model][task["id"]]
            med = lambda key: statistics.median(r[key] for r in runs)  # noqa: E731
            print(f"| {task['id']} | `{model}` | {med('wall_s'):.0f} | {med('turns'):.0f} | "
                  f"{med('tool_calls'):.0f} | {med('input_tokens'):.0f} | "
                  f"{med('output_tokens'):.0f} |")

    if len(models) == 2:
        print("\n| task | wall ratio B/A | turn ratio B/A | gen-token ratio B/A |")
        print("|---|---|---|---|")
        for task in tasks:
            a, b = results[models[0]][task["id"]], results[models[1]][task["id"]]
            am = lambda key: statistics.median(r[key] for r in a)  # noqa: E731
            bm = lambda key: statistics.median(r[key] for r in b)  # noqa: E731
            ratio = lambda x, y: (y / x) if x else 0.0  # noqa: E731
            print(f"| {task['id']} | {ratio(am('wall_s'), bm('wall_s')):.2f}x | "
                  f"{ratio(am('turns'), bm('turns')):.2f}x | "
                  f"{ratio(am('output_tokens'), bm('output_tokens')):.2f}x |")

    print(f"\nraw results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
