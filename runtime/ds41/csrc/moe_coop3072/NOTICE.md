# Third-party source notice

The `vendor/` headers are from [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
commit `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a`, copyright (c) 2025 Turboderp,
distributed under the MIT license reproduced in `LICENSE.exllamav3`.

`provenance.json` records the upstream and vendored SHA256 of every header.
All headers except `quant/exl3_moe_coop_kernel.cuh` retain their original bytes.
The coop512 ancestor changed only its private namespace, MAX_SLOTS256 to512,
and the original packed sort-key shifts8 to9. The coop2048 parent changed its
namespace and bound to2048 and guarded the old rotation-local integer shared
arrays/build_runs call. This child changes ONLY that parent's namespace and
MAX_SLOTS2048 to3072; the inherited old integer helper remains unreachable.
`prepare_runs.cuh` is project integer code:512 threads in six waves, stable
expert/slot rank with12 slot bits, run8 starts,
and a simple O(N^2) exclusive prefix. It completes before original rotation/A/B.
The A/B floating math, all helpers and reduction bodies remain unchanged. CPU
fixtures reverse the declared edits to parent2048, parent512 and upstream SHA.

`moe_coop3072.cpp` and `.cu` are the independent binding/launcher: they directly
instantiates only K2/K3 MUL1 narrow A/B templates, using one k-split, the
original rotation arithmetic, original workspace/counter layout and CUDA current
stream. It links to PyTorch/CUDA, never the upstream extension or DevCtx.
Headers may declare unused upstream host selectors; this frontend does not
call them. No upstream `.so`, venv or CUDA source is overwritten.
