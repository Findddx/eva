# Contributing

eva welcomes people who care about making capable local models practical: kernel and systems engineers, model researchers, application builders and users with real workloads.

A useful contribution can be small. A reproducible failure, a simpler ownership rule or an honest negative result often matters more than another isolated throughput number.

## Where to help

- **Expert placement:** combine recent routing with longer-term preferences, under explicit GPU and host-memory budgets.
- **Persistent context:** define compatible KV and prefix reuse across requests, including invalidation and memory pressure.
- **Finer expert access:** investigate quantized shard boundaries and routing-aware transfer schedules.
- **Serving and evaluation:** improve streaming, cancellation, request accounting and realistic coding or scientific task suites.
- **Execution backends:** improve kernels and scheduling, or add a model/device adapter with a clear numerical configuration.

Open an [issue](https://github.com/Findddx/eva/issues) before a broad change so we can agree on the problem, interface and evaluation. Small fixes can go directly to a pull request.

## A good change explains its effect

Describe the concrete trigger and resulting behavior. Keep the patch focused and identify any change to precision, quantization, routing, state layout or default configuration.

For scheduling changes, distinguish the intended reduction in work or data movement from measured overlap. For numerical changes, state the reference and expected accuracy behavior. A component improvement is a reason to test the full request, not a substitute for doing so.

Preserve upstream attribution and licenses. Keep model weights, generated caches, private prompts and credentials outside the repository.

## Test the behavior that can break

Start with meaningful CPU tests for boundaries, state transitions and failure cleanup. Use GPU component tests for kernel behavior and complete requests for the resulting model path. Documentation fixes do not need unrelated large test matrices.

Important cases include invalid shapes or routes, empty work, stream ownership, buffer reuse, partial failures, cancellation and shutdown. New optional optimizations should preserve the established default behavior and expose their effective configuration.

When sharing results, include:

- The model/backend, numerical configuration and relevant execution settings.
- A public or synthetic input and an independent task check.
- Output-token counts, decode duration and total request cost.
- Loading, prefill and first delivered-token latency as separate stages.
- Memory budgets, relevant transferred bytes and the comparison order.

Use chronological held-out tasks for personal-placement research. Future routes, acceptance results or later requests must not choose today's placement or proposal. Keep private usage local; aggregate statistics should be reviewed before sharing.

## Reporting a problem

Include the expected behavior, actual behavior and a minimal shareable example. State whether it occurred during loading, prefill, generation, verification or shutdown. A concise error and configuration are more helpful than a large machine-specific dump.

An out-of-memory error, nonfinite output, invalid state transition or slower result is worth reporting. Keep the observed outcome intact so others can understand it.

## Collaboration and sponsorship

We welcome hardware testing, compute, research partnerships and financial sponsorship. Please use [Issues](https://github.com/Findddx/eva/issues) to discuss the scope, data permissions and a useful deliverable.

For the project direction, see [Architecture](ARCHITECTURE.md). For measurement scope, see [Benchmarks](BENCHMARKS.md).
