# Third-party source notice

The `vendor/` headers are from [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
commit `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a`, copyright (c) 2025 Turboderp,
distributed under the MIT license reproduced in `LICENSE.exllamav3`.

`provenance.json` records the upstream and vendored SHA256 of every header.
All headers except `quant/exl3_moe_coop_kernel.cuh` retain their original bytes.
The parent coop512 candidate changed only its private namespace, MAX_SLOTS256
to512, and the two original packed sort-key shifts8 to9. This isolated child
changes the namespace again, raises the bound to2048, and adds macro guards to
disable the parent's rotation-local integer shared arrays/build_runs call.
The old integer helper remains unreachable. `prepare_runs.cuh` is new project
integer code:512 threads, stable expert/slot rank with11 slot bits, run8 starts,
and a simple O(N^2) exclusive prefix. It completes before original rotation/A/B.
The A/B floating math, all helpers and reduction bodies remain unchanged. CPU
fixtures reverse the declared edits to both parent and original upstream SHA.

`binding.cu` is the independent project launcher/frontend: it directly
instantiates only K2/K3 MUL1 narrow A/B templates, using one k-split, the
original rotation arithmetic, original workspace/counter layout and CUDA current
stream. It links to PyTorch/CUDA, never the upstream extension or DevCtx.
Headers may declare unused upstream host selectors; this frontend does not
call them. No upstream `.so`, venv or CUDA source is overwritten.
