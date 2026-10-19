# Third-party source notice

The `vendor/` headers are from [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
commit `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a`, copyright (c) 2025 Turboderp,
distributed under the MIT license reproduced in `LICENSE.exllamav3`.

`provenance.json` records the upstream and vendored SHA256 of every header.
All headers except `quant/exl3_moe_coop_kernel.cuh` retain their original bytes.
That header changes only its private namespace, MAX_SLOTS 256 to 512, and the
two packed sort-key shifts from 8 to 9. The A/B math and reduction body is kept.
The local CPU fixture reverses those edits and checks the original SHA256.

`binding.cu` is the independent project launcher/frontend: it directly
instantiates only K2/K3 MUL1 narrow A/B templates, using one k-split, the
original rotation kernel, original workspace/counter layout and CUDA current
stream. It links to PyTorch/CUDA, never the upstream extension or DevCtx.
Headers may declare unused upstream host selectors; this frontend does not
call them. No upstream `.so`, venv or CUDA source is overwritten.
