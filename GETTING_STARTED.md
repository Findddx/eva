# Run eva

eva runs a persistent model worker behind a streaming HTTP API. Once the model is loaded, requests reuse the weight caches and receive output as it is generated.

## Install

The initial backend targets Linux, Python 3.12 and NVIDIA SM120 GPUs. Install a compatible NVIDIA driver, CUDA toolkit and C++ compiler first. The canonical kernels currently use PyTorch 2.14.1 with CUDA 13.0 and ExLlamaV3 1.6.3.

```bash
git clone https://github.com/Findddx/eva.git
cd eva
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation exllamav3==1.6.3
```

## Prepare model data

Supply a DeepSeek-V4.1-Flash-1017 EXL3 checkpoint and its complete RANMA MXFP8 Engram tables. Keep model data on storage separate from the source checkout. Preparing data is a one-time operation; existing compatible packs can be reused.

```bash
export EVA_MODEL=/models/deepseek-exl3
export EVA_DATA=/data/eva

DS41_MODEL="$EVA_MODEL" DS41_PACK="$EVA_DATA/experts" python tools/pack_exl3.py
python tools/repack_expert_alignment.py \
  "$EVA_DATA/experts/experts.bin" "$EVA_DATA/experts/experts-aligned.bin"
python tools/pack_dense_cache.py --model "$EVA_MODEL" --destination "$EVA_DATA/dense"
python tools/prepare_engram.py \
  --index /models/engram/engram-00001-of-00007.gguf \
  --output "$EVA_DATA/engram.json"
```

Replace the example Engram filename with the actual first piece. All seven pieces must be present. The tools read supplied files; they do not download model weights. Use new output paths when repacking.

## Start the service

```bash
python serve.py \
  --model "$EVA_MODEL" \
  --pack "$EVA_DATA/experts/experts-aligned.bin" \
  --dense-cache "$EVA_DATA/dense" \
  --engram-manifest "$EVA_DATA/engram.json" \
  --ram-gib 64 --context 131072 --port 8000
```

`--ram-gib` sets the host expert-cache budget, including its allocator reserve. Leave room for model loading and the host; the worker checks available memory before allocating. Context capacity can be selected from 8K through 128K. An optional `--counts /path/to/routes.npy` supplies an aggregate expert-routing prior; otherwise the initial prior is uniform.

The first startup compiles CUDA extensions and loads the model. `GET /health` reports progress and becomes `ready` when the worker can accept requests. The API binds to loopback by default. For access from another machine, forward the port over SSH or configure a bearer key with `EVA_API_KEY` before using a non-loopback address.

## Ask a question or evaluate a workload

```bash
# Interactive input; /paste accepts multiple lines, /quit exits.
python client.py

# Stream one response, then display first-text latency, decode rate and total time.
python client.py --prompt "Write a Python function that merges two sorted lists."

# Use your own input, or run the supplied suite and save actual outputs and timings.
python client.py --prompt-file task.txt --mode lookup
python client.py --suite examples/prompts.json --record outputs/results.jsonl
```

Use `--url` when the API is on another address. `--max-tokens` bounds output. The default `greedy` mode generates directly; `lookup` adds history and prompt-derived proposals verified by the target model.

Each request starts with independent model state while retaining weights and expert caches. Requests are serialized on the GPU; a bounded queue absorbs additional requests. Disconnecting the client requests cancellation at a safe inference boundary.

## API and tests

See [API.md](API.md) for SSE requests and timing fields, and [BENCHMARKS.md](BENCHMARKS.md) for workload measurement. The prompt suite covers a short function, ordinary programming, scientific reasoning and a code edit.

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

These tests exercise the HTTP protocol, streaming parser, request queue, cancellation and state-reset contracts without loading model weights.
