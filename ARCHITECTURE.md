# Architecture

eva separates request handling, model state, memory placement and execution so each part can be improved and evaluated independently.

## From a request to an answer

```text
Application / evaluation client
              |
       Request and token stream
              |
    Proposal -> target verification -> commit
              |
       Model backend and state
              |
 GPU hot set + dynamic slots <-> pinned RAM <-> NVMe
              |
     Quantized kernels and transfers
```

The serving interface is responsible for request lifecycle and delivery. The model backend owns tokenization, inference state and generation. An evaluation client measures the request as the application experiences it and checks the resulting task separately.

Transport streaming, model residency and cross-request context reuse have distinct lifetimes and interfaces.

## The first model backend

The DeepSeek backend retains the complete backbone, routed experts and Engram. Experts remain quantized in EXL3 representation. The BF16 residual path uses FP32 normalization before returning the normalized branches to Half execution.

History and prompt-derived drafting propose work to the target model. The target determines acceptance; rejected work must leave a valid committed state. Batch geometry is an execution choice, not a promise about the number of useful tokens accepted.

Precision, expert codec, router behavior and state layout together define a numerical configuration. A cache policy should preserve that configuration. Changes to quantization or model routing need their own accuracy evaluation.

## Tiered memory

- **GPU hot set:** selected experts remain resident to avoid repeated transfers.
- **GPU dynamic slots:** temporary residency follows the current request's actual routes.
- **Pinned host cache:** holds reusable expert records under a host-memory budget.
- **NVMe:** supplies records that are absent from the nearer tiers; Engram rows are read on demand.

Aligned records and direct reads reduce unnecessary copying. Demand scheduling stages available host records while managing the reads needed for misses. A transfer being enqueued does not establish useful overlap; end-to-end timing and byte accounting must show the effect.

Buffers have explicit owners, generations and readers. A GPU slot becomes reusable only after its last reader retires; a pinned source becomes writable only after its DMA obligations retire. Error handling keeps uncertain work owned until it can be drained safely.

## Adapting to personal work

A useful placement policy combines two time scales: a long-term prior from past work and a short-term working set from the current request. Neither should silently erase the other when memory is tight.

The existing route-profile prototype aggregates expert use and prepares placement candidates for a later launch. The next step is to coordinate promotion and retirement within byte budgets, then evaluate whether yesterday's observations improve tomorrow's requests. Placement learning changes where weights live; it does not update model knowledge.

Evaluation must use chronological workloads and preserve a held-out future window. Repeating one prompt many times can demonstrate a cache mechanism, but cannot establish sustained benefit for a person's changing work.

## Persistent context

Within a request, the backend already maintains KV and the state needed for speculative commit and rollback. Cross-request reuse adds a distinct cache contract: model and tokenizer compatibility, exact token prefixes, positional state, memory budgets and invalidation.

The intended direction is to retain useful, compatible context without forcing applications to keep the entire execution process or every intermediate allocation alive. Prefix reuse must preserve Engram and other persistent model state as well as attention KV.

## Finer expert storage and scheduling

Today's aligned expert records provide a clear storage unit. Finer shards could let the engine fetch or stage smaller useful portions, improving the fit between a routed operation and the available memory.

Shard boundaries must respect the quantized layout, scale groups and the consumers' lifetimes. Physical file splitting alone does not reduce the bytes a computation needs. Changing bit widths or selecting different experts is a separate model configuration, not a storage optimization in disguise.

The practical path is to identify a safe execution boundary, preserve the original accumulation order, and measure the complete request cost before expanding the design.

## How changes are evaluated

Correctness, useful task output and speed answer different questions. Tests cover output values, valid state transitions, commit boundaries and resource retirement; workload evaluation covers task behavior. Performance reports separate loading, prefill, first delivered token, decode and total request time.

The aim is a framework with clear interfaces and reproducible behavior. Current optimizations focus on a single-GPU backend; additional models, devices and parallel execution can be added behind explicit backend contracts.
