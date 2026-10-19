# Performance and evaluation

eva focuses on useful output per byte moved across the memory hierarchy.

| Workload | Observed decode rate | Execution path |
| --- | ---: | --- |
| Coding and scientific coding examples | **24–25 tokens/s** | Earlier FP16 path with draft-assisted generation |
| Narrow edit to a supplied code listing | **110 tokens/s** | BF16 residual path with prompt-guided proposals |

The editing workload reuses much of the supplied source. These task-specific figures use different decoding configurations; loading, prefill and validation are separate stages.

## Measure your workload

```bash
python client.py --suite examples/prompts.json --record outputs/results.jsonl
```

The client streams the actual answer and reports first-text latency, decode rate, total time and output-token count. JSONL records retain the input, output and complete timing fields for comparison. See [API.md](API.md) for their definitions.

The suite includes a short function, ordinary programming, scientific reasoning and an edit. Replace or extend it with the tasks you run regularly. Assess the returned code or reasoning alongside speed.

## Compare changes

For kernel and transfer work, measure the affected operation and the complete request. For placement policies, keep memory budgets and starting cache state comparable, and account for transferred bytes and migration costs.

For personal adaptation, use tasks in chronological order and evaluate later requests against earlier routing observations. This tests whether the placement policy follows a changing workflow rather than simply benefiting from repeated inputs.

Share the workload, configuration, output and timing summary when reporting a result. Small, well-explained comparisons help the community decide which improvements to build on.
