# lora-merge-tools

Merge a PEFT LoRA adapter into a sharded Hugging Face checkpoint and publish the
result, one shard at a time, in **float16**. Built to produce
[zonca/Qwen3.8-27B-pi-adapted-merged](https://huggingface.co/zonca/Qwen3.8-27B-pi-adapted-merged)
on a machine with 11 GB of RAM, but nothing in here is specific to that model:
any sharded `safetensors` checkpoint and any PEFT LoRA work.

## Two traps this exists to avoid

**1. Saving the merge in bfloat16 throws most of the adapter away.** A LoRA delta
is tiny: for the adapter this was written for, the RMS of the delta is about 1e-5
against weights whose absolute maximum is between 0.1 and 0.9, so each weight moves
by roughly 1e-4. bfloat16 has 8 explicit mantissa bits, which at a magnitude of 0.3
is a resolution of about 2e-3: an order of magnitude coarser than the change being
written. An independent merge of the same adapter measured 65 to 82% of the delta
surviving in bfloat16, against 99.9 to 100% in float16. So the delta is computed in
float32 and stored as float16.

Tensors the adapter does not touch are upcast bfloat16 to float16, which is exact:
bfloat16 has the wider exponent range but the shorter mantissa, so every normal
bfloat16 value is representable in float16.

**2. `PeftModel.merge_and_unload()` needs the whole model in RAM.** For a 27B model
that is 55 GB. The merge here opens one shard, converts one tensor at a time, and
writes the output shard with a streaming safetensors writer, so peak memory is one
tensor plus its float32 delta (well under 1 GB) and peak disk is two shards. It
runs on a laptop.

A third, smaller trap: on some networks the Hugging Face CDN answers a full-file
download at 1 to 2 MB/s while 64 to 256 MB byte ranges come back at 25 to 40 MB/s
each, so `hf_get.py` fetches in parallel ranges and turns nine hours into minutes.

## Scripts

| file | what it does |
|---|---|
| `merge_lora.py` | streaming float16 merge of a PEFT LoRA into a sharded checkpoint; writes one output shard per invocation, accumulates `merge_report.json` |
| `verify_merged.py` | per shard: every base tensor present, float16, untouched tensors bit-identical to the bf16 to fp16 upcast, merged tensors bit-identical to an independent `einsum` recomputation of `W + (alpha/r) * B @ A` |
| `hf_get.py` | parallel-range downloader with per-range resume and a sha256 check against the Hub |
| `check_remote.py` | fails unless the Hub reports the expected sha256 for an uploaded file |
| `audit_repo.py` | audit a published repo: local sha256 versus the Hub's `x-linked-etag` for every shard, plus a range-request read of each remote safetensors header to confirm names and dtype |
| `run_build.sh` | the whole pipeline per shard: download, merge, hash, verify, upload, remote sha check, delete. Resumable |
| `bench_ollama.py` | prefill, decode and cold-load benchmark for Ollama models, with a prompt nonce so repeats are not served from the prompt cache |
| `bench_pi.py` | runs the same real tasks through the [pi](https://github.com/earendil-works/pi) agent on two models and reports turns, prompt tokens, generated tokens and wall clock |
| `bench_tasks.json` | the task set `bench_pi.py` uses: a Trello card to triage, an email thread to summarize and answer, and a small coding task |

## How the merge maps the adapter onto the checkpoint

PEFT stores a LoRA for module `model.language_model.layers.0.mlp.up_proj` under
the key `base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight`,
and the checkpoint holds `.weight` of the same module. `merge_lora.py` strips the
`base_model.model.` prefix, pairs `lora_A` with `lora_B`, and **refuses to run if
any adapter module has no matching base tensor**: a silent mismatch produces a
model that loads, generates, and is simply not the one that was trained.

The scale is `lora_alpha / r` from the adapter config (2.0 for alpha 64, r 32),
PEFT's convention. rsLoRA adapters need `alpha / sqrt(r)` and are rejected rather
than silently mis-scaled.

## Quick start

```bash
git clone https://github.com/zonca/lora-merge-tools
cd lora-merge-tools
python3 -m venv venv && ./venv/bin/pip install torch safetensors   # CPU torch is enough
export PYTHON=$PWD/venv/bin/python

# merge one shard into a local directory, keeping the base alongside
./venv/bin/python merge_lora.py --base ./Qwen3.8-27B --adapter ./adapter \
    --out ./merged --shards model-00005-of-00018.safetensors
./venv/bin/python verify_merged.py --base ./Qwen3.8-27B --adapter ./adapter \
    --merged ./merged --shards model-00005-of-00018.safetensors
```

The whole build and publication, resumable, deleting each shard after it is
verified on the Hub:

```bash
ADAPTER_REPO=harness-hub/Qwen3.8-27B-pi-adapted \
OUTPUT_REPO=zonca/Qwen3.8-27B-pi-adapted-merged \
WORKDIR=/data/build ./run_build.sh
```

## Configuration of `run_build.sh`

| variable | default | meaning |
|---|---|---|
| `BASE_REPO` | `Qwen/Qwen3.8-27B` | Hub repo id, or a local directory |
| `ADAPTER_REPO` | (required) | Hub repo id, or a local directory with `adapter_model.safetensors` and `adapter_config.json` |
| `OUTPUT_REPO` | (required) | Hub repo to publish to, created with `hf repos create` |
| `WORKDIR` | `./work` | scratch space; needs about 2.5 base shards free |
| `PYTHON` | `python3` | interpreter with `torch` and `safetensors` |
| `THREADS` | `8` | parallel ranges per download |
| `CHUNK_MB` | `64` | range size in MB |

`WORKDIR/progress.log` is the resume state (`OK <shard>` lines) and
`WORKDIR/sha256sums.txt` is what `audit_repo.py` compares against.

## Benchmarking on Ollama

```bash
python bench_ollama.py --url http://127.0.0.1:11434 \
    --models qwen3.8:27b-nvfp4 qwen3.8-pi:27b-nvfp4 --runs 3
```

Reports cold load, median decode rate at 256 tokens, and prefill rate at two
prompt sizes. On an Apple Silicon machine the decode number doubles as a
multi-token-prediction canary: a model whose MTP head survived decodes well above
the memory-bandwidth bound, and one that lost it drops to roughly that bound. The
comparison only means something with both models on the same machine.

## Benchmarking an agent workload with pi

A token benchmark says what the model costs per token; an agent benchmark says what a
task costs, and on a local model the second is the one that matters, because every turn
re-reads the whole conversation. `bench_pi.py` runs the same prompts through
[pi](https://github.com/earendil-works/pi) against two model ids and reads pi's JSON
event stream for turns, prompt tokens, generated tokens and wall clock:

```bash
python bench_pi.py --model-a macmini/qwen3.8:27b-nvfp4 \
                   --model-b macmini/qwen3.8-pi:27b-nvfp4 \
                   --tasks bench_tasks.json --out results.json
```

The tasks in `bench_tasks.json` are a real Trello card to triage, a real email thread
to summarize and answer, and a small coding task; the first two run with `bash` and
`read` only, so the agent cannot change anything.

## What the verification actually guarantees

For every shard: names and shapes identical to the base, dtype float16 everywhere,
untouched tensors bit-identical to the base upcast, merged tensors bit-identical to
a recomputation that uses `einsum` instead of `matmul`. It checks the arithmetic,
not the model: whether the resulting model is *good* is an evaluation question, and
merging a LoRA cannot answer it.

## Licence

MIT for the code. The merged weights inherit the licence of the base model and the
adapter.
