#pragma once

// Project-owned integer metadata preparation. All EXL3 floating kernels remain
// in the licensed vendor header, byte-identical after the declared provenance
// transformations. One block handles up to1536 actual slots in three waves.
namespace ds41_moe_coop2048_ns {

__global__ __launch_bounds__(512)
void ds41_prepare_runs2048(const MoeCoopParams p) {
    static_assert(THREADS == 512 && ROWS == 8, "fixed original floating geometry");
    __shared__ int16_t local[2048];
    __shared__ int16_t sorted[2048];
    __shared__ int n_active;
    __shared__ int n_runs;
    const int slots = p.bsz * p.topk;
    int* run_start = p.runs + 2;
    int* order = p.runs + 2 + p.slots_max + 1;
    if (threadIdx.x == 0) { n_active = 0; n_runs = 0; }
    for (int s = threadIdx.x; s < slots; s += 512) {
        const SlotInfo si = slot_info(p, s);
        local[s] = si.active ? (int16_t)si.local : (int16_t)-1;
    }
    __syncthreads();

    // Stable unique rank by (local expert, ORIGINAL slot);11 slot bits include
    // slots512..1535 without the original8/9-bit key collisions.
    for (int s = threadIdx.x; s < slots; s += 512) {
        const int e = local[s];
        if (e < 0) continue;
        const int mine = (e << 11) | s;
        int rank = 0;
        for (int u = 0; u < slots; ++u) {
            const int eu = local[u];
            rank += eu >= 0 && ((eu << 11) | u) < mine;
        }
        sorted[rank] = (int16_t)s;
        atomicMax(&n_active, rank + 1);
    }
    __syncthreads();

    // order is temporary flags until the FINAL barrier. No floating kernel or
    // other block may read p.runs until this entire prepare kernel completes.
    for (int pos = threadIdx.x; pos < n_active; pos += 512) {
        const int e = local[sorted[pos]];
        int first = 0;
        for (int u = 0; u < slots; ++u)
            first += local[u] >= 0 && local[u] < e;
        order[pos] = (pos == 0 || local[sorted[pos - 1]] != e || (pos - first) % 8 == 0);
    }
    __syncthreads();  // all block threads see the flags in global memory
    for (int pos = threadIdx.x; pos < n_active; pos += 512) {
        if (!order[pos]) continue;
        int run = 0;
        for (int u = 0; u < pos; ++u) run += order[u];
        run_start[run] = pos;
        atomicMax(&n_runs, run + 1);
    }
    __syncthreads();  // every prefix read finishes BEFORE any flag is replaced
    for (int pos = threadIdx.x; pos < n_active; pos += 512)
        order[pos] = sorted[pos];
    if (threadIdx.x == 0) {
        run_start[n_runs] = n_active;
        p.runs[0] = n_runs;
        p.runs[1] = 0;
    }
}

}  // namespace ds41_moe_coop2048_ns
