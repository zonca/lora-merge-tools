#!/usr/bin/env python3
"""Audit a published sharded checkpoint: every shard present and byte-identical.

Compares the sha256 recorded locally at build time with the `x-linked-etag` that
the Hub serves for each file, checks that the shard index on the Hub covers
exactly the shards that were built, and re-reads each remote safetensors header
with a range request (no download) to confirm the tensor names and the storage
dtype.

    python audit_repo.py zonca/Qwen3.8-27B-pi-adapted-merged \
        --sums sha256sums.txt --expect-dtype F16

Exit status is non-zero if anything does not match.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TOKEN_PATH = Path("~/.cache/huggingface/token").expanduser()


def curl(*args: str, binary: bool = False):
    cmd = ["curl", "-sL", *args]
    token = TOKEN_PATH.expanduser()
    if token.exists():
        cmd += ["-H", f"Authorization: Bearer {token.read_text().strip()}"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return out if binary else out.decode()


def head_etag(url: str) -> tuple[int, str]:
    out = subprocess.run(
        ["curl", "-sIL", url], capture_output=True, text=True, check=True
    ).stdout
    size, etag = 0, ""
    for line in out.splitlines():
        low = line.lower()
        if low.startswith("x-linked-size:"):
            size = int(line.split(":", 1)[1].strip())
        elif low.startswith("x-linked-etag:"):
            etag = line.split(":", 1)[1].strip().strip('"')
    return size, etag


def remote_header(repo: str, name: str) -> dict:
    """Read a safetensors header from the Hub with a range request."""
    buf = curl("-r", "0-4000000", f"https://huggingface.co/{repo}/resolve/main/{name}",
               binary=True)
    n = int.from_bytes(buf[:8], "little")
    header = json.loads(buf[8 : 8 + n])
    header.pop("__metadata__", None)
    return header


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="Hub repo id, e.g. user/model")
    ap.add_argument("--sums", default="sha256sums.txt",
                    help="file written by run_build.sh: '<sha256>  <shard>' per line")
    ap.add_argument("--expect-dtype", default="F16",
                    help="safetensors dtype every tensor must have (default F16)")
    args = ap.parse_args()

    sums = Path(args.sums)
    if not sums.exists():
        raise SystemExit(f"{sums} not found: run the build first, or pass --sums")

    expected = {}
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        expected[name] = digest

    meta = json.loads(curl(f"https://huggingface.co/api/models/{args.repo}?blobs=true"))
    remote = {s["rfilename"]: s for s in meta["siblings"]}
    problems: list[str] = []
    total = 0
    for name, digest in sorted(expected.items()):
        if name not in remote:
            problems.append(f"{name}: missing on the Hub")
            continue
        size, etag = head_etag(f"https://huggingface.co/{args.repo}/resolve/main/{name}")
        if size != remote[name]["size"]:
            problems.append(f"{name}: size {size} != {remote[name]['size']}")
        if etag != digest:
            problems.append(f"{name}: etag {etag[:16]} != local {digest[:16]}")
        total += size
        print(f"{name}: {size/1e9:.2f} GB {'ok' if etag == digest else 'MISMATCH'}")

    index = json.loads(
        curl(f"https://huggingface.co/{args.repo}/raw/main/model.safetensors.index.json")
    )
    weight_map = index["weight_map"]
    shards = set(weight_map.values())
    if shards != set(expected):
        problems.append(f"index lists {sorted(shards - set(expected))} that were not built")
    print(f"\nshards: {len(expected)} in the sums file, {len(shards)} in the index")
    print(f"tensors in the index: {len(weight_map)}")

    for name in sorted(expected):
        want = {n for n, s in weight_map.items() if s == name}
        got = remote_header(args.repo, name)
        if set(got) != want:
            problems.append(
                f"{name}: header has {len(got)} tensors, index expects {len(want)} "
                f"(missing {sorted(want - set(got))[:2]}, extra {sorted(set(got) - want)[:2]})"
            )
        bad_dtype = [k for k, v in got.items() if v["dtype"] != args.expect_dtype]
        if bad_dtype:
            examples = [(k, got[k]["dtype"]) for k in bad_dtype[:2]]
            problems.append(
                f"{name}: {len(bad_dtype)} tensors not {args.expect_dtype}, e.g. {examples}"
            )
    print(f"headers: names and {args.expect_dtype} dtype checked for every shard")
    print(f"total size: {total/1e9:.2f} GB")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nall shards on the Hub match the local build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
