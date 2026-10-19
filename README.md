# eva

**Personal inference, built around your workstation.** · [简体中文](README.zh-CN.md)

eva is a local inference framework for running large mixture-of-experts models across GPU memory, RAM and NVMe. Its first backend focuses on DeepSeek with quantized EXL3 experts and complete Engram support.

A personal workstation repeatedly serves the same person, projects and tools. Those recurring workloads offer useful signals about which experts and context are worth keeping close. Our goal is to turn those signals into better memory placement and less data movement, so local AI becomes more responsive as it becomes familiar with your work.

## The idea

Keep useful experts close, promote and retire them as workloads change, and spend less time moving weights. Combine that with reusable context and finer expert storage to make better use of a personal machine.

- **Workload-aware residency:** learn from local routing samples and coordinate long-term preferences with the current request.
- **Persistent context:** reuse compatible KV and prefix state across requests, with clear ownership, eviction and invalidation.
- **Finer expert access:** explore smaller quantized expert shards and routing-aware scheduling while preserving the model's numerical contract.

Tiered execution and route-profile prototypes provide the starting point. Persistent cross-request context and finer expert sharding are the next steps; long-term gains will be measured on real sequences of work.

## What is inside

- GPU hot and dynamic expert caches, pinned host memory, aligned storage records and demand-driven transfers.
- History and prompt-derived proposals checked by the target model, with explicit acceptance, rollback and state ownership.
- Cooperative quantized MoE kernels, canonical batch operations, CUDA Graphs and shared workspaces.
- A persistent model worker, streaming HTTP API and interactive evaluation client with live output and request timings.

The current numerical path uses BF16 residuals and FP32 normalization around quantized execution, with explicit state and memory ownership.

## Measurements in context

| Workload | Recorded decode rate | Scope |
| --- | ---: | --- |
| Ordinary coding and scientific coding examples | about **24–25 tokens/s** | Earlier Half target with small-window drafting |
| Narrow edit to a supplied code listing | about **110 tokens/s** | BF16 residual target; substantial prompt reuse |

Task-specific decode results; loading, prefill and validation are separate. The two paths use different numerical configurations. [Measurement details](BENCHMARKS.md).

## Try it

Start the [model service](GETTING_STARTED.md), then stream a question:

```bash
python client.py --url http://127.0.0.1:8000 --prompt "Write a Python function that merges two sorted lists."
```

Run `python client.py` for interactive input, or use the [sample prompt suite](examples/prompts.json) to compare your own workloads.

## Get involved

Start with the [running guide](GETTING_STARTED.md), then explore the [architecture](ARCHITECTURE.md) and [contribution guide](CONTRIBUTING.md). Model data is supplied separately; the repository does not distribute weights.

We welcome kernel and systems engineers, model researchers, application builders and people with real coding or scientific workloads to evaluate. Useful contributions range from a small correctness fix to better scheduling, context reuse or a new backend.

We also welcome hardware testing, compute partnerships and financial sponsorship. Open an [issue](https://github.com/Findddx/eva/issues) to discuss a contribution or collaboration.

MIT. Built with ideas and code from the wider inference community; see [third-party notices](THIRD_PARTY_NOTICES.md).
