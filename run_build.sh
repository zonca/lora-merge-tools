#!/bin/bash
# Merge a PEFT LoRA into a sharded Hugging Face checkpoint in float16 and
# publish the result, one shard at a time.
#
# For every shard: download (parallel byte ranges), merge, record the sha256,
# verify against the base and against an independent recomputation, upload,
# compare the sha256 the Hub reports, delete the local copy. It is resumable:
# a shard with an "OK <shard>" line in progress.log is skipped, so re-running
# after any failure continues where it stopped.
#
# Configuration (environment variables, all optional):
#   BASE_REPO     Hub repo id or local dir of the base checkpoint (Qwen/Qwen3.8-27B)
#   ADAPTER_REPO  Hub repo id or local dir of the PEFT adapter     (required)
#   OUTPUT_REPO   Hub repo to publish to                            (required)
#   WORKDIR       scratch space, ~2.5 shards of free disk           (./work)
#   PYTHON        interpreter with torch + safetensors              (python3)
#   THREADS       parallel ranges per download                      (8)
#   CHUNK_MB      range size in MB                                  (64)
#
# Example:
#   ADAPTER_REPO=harness-hub/Qwen3.8-27B-pi-adapted \
#   OUTPUT_REPO=zonca/Qwen3.8-27B-pi-adapted-merged ./run_build.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BASE_REPO=${BASE_REPO:-Qwen/Qwen3.8-27B}
ADAPTER_REPO=${ADAPTER_REPO:?set ADAPTER_REPO to the adapter repo id or a local directory}
OUTPUT_REPO=${OUTPUT_REPO:?set OUTPUT_REPO to the Hub repo to publish to}
W=${WORKDIR:-$HERE/work}
PY=${PYTHON:-python3}
THREADS=${THREADS:-8}
CHUNK_MB=${CHUNK_MB:-64}

LOG=$W/progress.log
SUMS=$W/sha256sums.txt
mkdir -p "$W/base" "$W/out" "$W/adapter"

fail() { echo "FAIL $*" | tee -a "$LOG"; exit 1; }

# --- metadata and adapter -----------------------------------------------------
if [ -d "$BASE_REPO" ]; then
  cp -n "$BASE_REPO"/*.json "$BASE_REPO"/*.txt "$W/base/" 2>/dev/null
else
  hf download "$BASE_REPO" --local-dir "$W/base" \
    --include "*.json" "*.txt" --exclude "*.safetensors*" >/dev/null || fail "metadata $BASE_REPO"
fi

if [ -d "$ADAPTER_REPO" ]; then
  cp -n "$ADAPTER_REPO"/adapter_model.safetensors "$ADAPTER_REPO"/adapter_config.json "$W/adapter/" \
    || fail "adapter copy"
else
  hf download "$ADAPTER_REPO" --local-dir "$W/adapter" \
    --include "adapter_model.safetensors" "adapter_config.json" >/dev/null || fail "adapter $ADAPTER_REPO"
fi

SHARDS=$("$PY" -c "
import json, sys
idx = json.load(open(sys.argv[1]))
print('\n'.join(sorted(set(idx['weight_map'].values()))))" "$W/base/model.safetensors.index.json")
[ -n "$SHARDS" ] || fail "no shards in $W/base/model.safetensors.index.json"

# --- shard loop ---------------------------------------------------------------
for sh in $SHARDS; do
  if grep -q "^OK $sh" "$LOG" 2>/dev/null; then
    echo "SKIP $sh (done)"
    continue
  fi
  echo "=== $sh $(date -Is) ===" | tee -a "$LOG"

  if [ ! -d "$BASE_REPO" ]; then
    "$PY" "$HERE/hf_get.py" --repo "$BASE_REPO" --file "$sh" --out "$W/base/$sh" \
      --threads "$THREADS" --chunk-mb "$CHUNK_MB" >> "$LOG" 2>&1 || fail "download $sh"
  fi

  "$PY" "$HERE/merge_lora.py" --base "$W/base" --adapter "$W/adapter" \
    --out "$W/out" --shards "$sh" >> "$LOG" 2>&1 || fail "merge $sh"

  LOCAL_SHA=$(sha256sum "$W/out/$sh" | cut -d' ' -f1)
  echo "$LOCAL_SHA  $sh" >> "$SUMS"

  "$PY" "$HERE/verify_merged.py" --base "$W/base" --adapter "$W/adapter" \
    --merged "$W/out" --shards "$sh" >> "$LOG" 2>&1 || fail "verify $sh"

  hf upload "$OUTPUT_REPO" "$W/out/$sh" "$sh" >> "$LOG" 2>&1 || fail "upload $sh"

  "$PY" "$HERE/check_remote.py" "$OUTPUT_REPO" "$sh" "$LOCAL_SHA" >> "$LOG" 2>&1 \
    || fail "remote sha $sh"

  echo "OK $sh" | tee -a "$LOG"
  rm -f "$W/base/$sh" "$W/out/$sh"
done
echo "ALL DONE $(date -Is)" | tee -a "$LOG"
