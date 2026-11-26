#!/usr/bin/env python3
"""Run the same real tasks through pi on two models and compare the cost.

Each task is a prompt, a tool allowlist, an optional setup command and an
optional verify command. The runner gives every task a fresh working directory,
runs `pi -p --mode json` on it, then runs the verify command and records whether
the result actually works, which is the only way to tell a fast answer from a
correct one.

What it records, per run:

* **turns**, the number of model calls;
* **prompt tokens processed** (sum of `usage.input`), because every turn
  re-reads the whole conversation and that is the prefill cost;
* **generated tokens** (sum of `usage.output`);
* **wall clock**;
* **verify**, exit status of the task's own check (tests, file unchanged).

The two models are interleaved inside each repetition, so a change in machine
state cannot favour one of them, and repetitions are the outer loop so the
spread of a single task is visible.

The Trello and email tasks run with only `bash` and `read`, and the prompt says
plainly not to change anything, so the agent cannot move a card or send mail
while being measured.

    python bench_pi.py --model-a macmini/qwen3.8:27b-nvfp4 \
                       --model-b macmini/qwen3.8-pi:27b-nvfp4 \
                       --tasks bench_tasks.json --repeat 2 --out results.json
"""

from __future__ import annotations

import argparse
import json
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


def run_verify(command: str, workdir: Path) -> dict:
    proc = subprocess.run(command, shell=True, cwd=workdir, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    return {"verify_rc": proc.returncode, "verify_tail": " | ".join(tail)[:300]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True, help="baseline model, e.g. macmini/qwen3.8:27b-nvfp4")
    ap.add_argument("--model-b", help="candidate model; omit to measure only the baseline")
    ap.add_argument("--tasks", required=True, help="JSON task file")
    ap.add_argument("--out", required=True, help="where to write the raw results JSON")
    ap.add_argument("--repeat", type=int, default=1, help="repetitions of the whole task set")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per task run")
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    models = [m for m in (args.model_a, args.model_b) if m]
    results: dict[str, dict[str, list[dict]]] = {m: {t["id"]: [] for t in tasks} for m in models}

    for rep in range(args.repeat):
        for index, task in enumerate(tasks):
            order = models if (rep + index) % 2 == 0 else list(reversed(models))
            for model in order:
                workdir = Path(tempfile.mkdtemp(prefix=f"pibench-{task['id']}-"))
                if task.get("setup"):
                    subprocess.run(task["setup"], shell=True, cwd=workdir, check=False)
                print(f"\n== {model} / {task['id']} (rep {rep + 1})", flush=True)
                try:
                    r = run_pi(model, task["prompt"], workdir, task.get("tools"), args.timeout)
                except subprocess.TimeoutExpired:
                    r = {"wall_s": args.timeout, "turns": 0, "tool_calls": 0, "tool_names": [],
                         "input_tokens": 0, "output_tokens": 0, "final_text": "",
                         "returncode": -1, "stderr_tail": "",
                         "error": f"timeout after {args.timeout}s"}
                if task.get("verify"):
                    r.update(run_verify(task["verify"], workdir))
                else:
                    r.update({"verify_rc": None, "verify_tail": ""})
                print(f"   {r['wall_s']:.0f} s, {r['turns']} turns, {r['tool_calls']} tool calls, "
                      f"{r['input_tokens']} in / {r['output_tokens']} out, "
                      f"verify {'n/a' if r['verify_rc'] is None else r['verify_rc']}", flush=True)
                if r["error"]:
                    print(f"   error: {r['error']}", flush=True)
                r["workdir"] = str(workdir)
                results[model][task["id"]].append(r)
                # incremental save: a ten-repetition campaign runs for hours
                Path(args.out).write_text(json.dumps(results, indent=2) + "\n")

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")

    def med(rows, key):
        return statistics.median(r[key] for r in rows)

    print("\n| task | model | wall (s) | turns | tool calls | prompt tok | gen tok | verify |")
    print("|---|---|---|---|---|---|---|---|")
    for task in tasks:
        for model in models:
            rows = results[model][task["id"]]
            if not rows:
                continue
            checks = [r["verify_rc"] for r in rows if r["verify_rc"] is not None]
            verified = ("n/a" if not checks else
                        f"{sum(1 for c in checks if c == 0)}/{len(checks)}")
            print(f"| {task['id']} | `{model}` | {med(rows, 'wall_s'):.0f} | "
                  f"{med(rows, 'turns'):.0f} | {med(rows, 'tool_calls'):.0f} | "
                  f"{med(rows, 'input_tokens'):.0f} | {med(rows, 'output_tokens'):.0f} | "
                  f"{verified} |")

    if len(models) == 2:
        print("\n| task | wall ratio B/A | turn ratio B/A | gen-token ratio B/A |")
        print("|---|---|---|---|")
        walls, gens = [], []
        for task in tasks:
            a, b = results[models[0]][task["id"]], results[models[1]][task["id"]]
            if not a or not b:
                continue
            ratio = lambda x, y: (y / x) if x else 0.0  # noqa: E731
            walls.append(ratio(med(a, "wall_s"), med(b, "wall_s")))
            gens.append(ratio(med(a, "output_tokens"), med(b, "output_tokens")))
            print(f"| {task['id']} | {walls[-1]:.2f}x | "
                  f"{ratio(med(a, 'turns'), med(b, 'turns')):.2f}x | {gens[-1]:.2f}x |")
        if walls:
            print(f"\nmedian over tasks: wall {statistics.median(walls):.2f}x, "
                  f"generated tokens {statistics.median(gens):.2f}x")

    print(f"\nraw results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
