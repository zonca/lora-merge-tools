#!/usr/bin/env python3
"""Download one file from a Hugging Face repository in parallel byte ranges.

The Hugging Face CDN answered full-file downloads at 1 to 2 MB/s for this
repository while 100 MB range requests came back at 25 to 40 MB/s each, so a
handful of parallel ranges turns a nine hour download into minutes. Stalled
ranges are retried and resumed, and the result is checked against the sha256 the
Hub reports.

    python hf_get.py --repo Qwen/Qwen3.8-27B \
                     --file model-00001-of-00018.safetensors \
                     --out /data/base/model-00001-of-00018.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

TOKEN_PATH = Path("~/.cache/huggingface/token").expanduser()


def read_token() -> str:
    try:
        return TOKEN_PATH.read_text().strip()
    except FileNotFoundError:
        return ""


def file_info(repo: str, filename: str, token: str) -> tuple[int, str]:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.load(r)
    for sib in meta["siblings"]:
        if sib["rfilename"] == filename:
            if sib.get("size") is None:
                break
            return int(sib["size"]), (sib.get("lfs") or {}).get("oid") or sib.get("oid") or ""
    raise SystemExit(f"{filename} not found in {repo} (size unknown)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(64 << 20), b""):
            h.update(block)
    return h.hexdigest()


def curl_cfg(token: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="hfget-"))
    cfg = d / "curlrc"
    cfg.write_text(
        (f'header = "Authorization: Bearer {token}"\n' if token else 'user-agent = "hf_get.py"\n')
    )
    cfg.chmod(0o600)
    return cfg


def fetch_range(url: str, cfg: Path, part: Path, start: int, end: int) -> None:
    """Download [start, end] into `part`, resuming whatever is already there."""
    want = end - start + 1
    for _ in range(8):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink()
            have = 0
        target = part if have == 0 else Path(str(part) + ".next")
        if target.exists():
            target.unlink()
        subprocess.run(
            [
                "curl", "-sL", "-K", str(cfg),
                "-r", f"{start + have}-{end}",
                "-o", str(target),
                "--speed-limit", "200000", "--speed-time", "25",
                "--retry", "3", "--retry-delay", "5", "--retry-all-errors",
                url,
            ],
            check=False,
        )
        if have:
            with open(part, "ab") as dst, open(target, "rb") as src:
                while True:
                    block = src.read(64 << 20)
                    if not block:
                        break
                    dst.write(block)
            target.unlink()
    if not part.exists() or part.stat().st_size != want:
        raise RuntimeError(f"range {start}-{end} incomplete")


def download(repo: str, filename: str, out: Path, threads: int, chunk_mb: int) -> None:
    token = read_token()
    size, expected_sha = file_info(repo, filename, token)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size == size:
        if not expected_sha or sha256_file(out) == expected_sha:
            print(f"{filename}: already complete")
            return

    cfg = curl_cfg(token)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

    chunk_size = chunk_mb << 20
    spans: list[tuple[int, int, Path]] = []
    start, i = 0, 0
    while start < size:
        end = min(start + chunk_size - 1, size - 1)
        spans.append((start, end, Path(f"{out}.{i}.part")))
        start = end + 1
        i += 1

    import queue as queue_mod

    lock = threading.Lock()
    work_queue: queue_mod.Queue = queue_mod.Queue()
    for span in spans:
        work_queue.put(span)

    errors: list[str] = []
    state = {"done": 0, "bytes": 0}
    t0 = time.time()

    def worker():
        while True:
            try:
                chunk = work_queue.get_nowait()
            except queue_mod.Empty:
                return
            start, end, part = chunk
            try:
                fetch_range(url, cfg, part, start, end)
            except RuntimeError as exc:
                with lock:
                    errors.append(str(exc))
                return
            with lock:
                state["done"] += 1
                state["bytes"] += end - start + 1
                rate = state["bytes"] / 1e6 / max(time.time() - t0, 1e-9)
                print(
                    f"  {state['done']}/{len(spans)} ranges, {rate:.1f} MB/s",
                    flush=True,
                )

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    if errors:
        raise SystemExit("download failed: " + "; ".join(errors[:3]))

    with open(out, "wb") as dst:
        for _, _, part in spans:
            with open(part, "rb") as src:
                while True:
                    block = src.read(64 << 20)
                    if not block:
                        break
                    dst.write(block)
            part.unlink()

    actual = sha256_file(out)
    if expected_sha and actual != expected_sha:
        out.unlink()
        raise SystemExit(f"sha256 mismatch: got {actual}, expected {expected_sha}")
    print(f"{filename}: {size/1e9:.2f} GB, sha256 ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=96)
    args = ap.parse_args()
    download(args.repo, args.file, Path(args.out), args.threads, args.chunk_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
