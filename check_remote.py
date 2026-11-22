#!/usr/bin/env python3
"""Fail unless the Hub reports the expected sha256 for a published file.

`hf upload` succeeding does not prove the bytes are the ones that were built,
so the pipeline asks the Hub for each shard's LFS sha256 and compares it with
the local hash:

    python check_remote.py <repo> <filename> <local-sha256>
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

SHA256 = re.compile(r"[0-9a-f]{64}")


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_remote.py <repo> <filename> <local-sha256>")
    repo, name, local = sys.argv[1:4]

    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        meta = json.load(response)
    sibling = next((s for s in meta["siblings"] if s["rfilename"] == name), None)
    if sibling is None:
        raise SystemExit(f"{name}: not on the Hub")

    remote = (sibling.get("lfs") or {}).get("sha256") or sibling.get("oid") or ""
    print(f"{name}: remote {remote[:16]} local {local[:16]}")
    if not SHA256.fullmatch(remote):
        raise SystemExit(
            f"{name}: the Hub did not report a sha256 (is the file above the 10 MB LFS limit?)"
        )
    if remote != local:
        raise SystemExit(f"{name}: remote sha256 {remote} != local {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
