# Diagnosis Playbook — Pattern → Cause → Fix

For each observed rocprof signal, what does it typically mean, and what's the first fix to try? This synthesizes the CDNA3 / CDNA4 programming principles (the companion `cdna3-cdna4-hip-programming.md` at the repo root) with the profiling signals.

Read this after you've gathered the metrics (via [`05-analysis-dimensions.md`](05-analysis-dimensions.md)) — here you translate metrics into diagnoses and fix directions.

---

## How to use this doc

For each *observation* below, read:

- **Signals** — what specific counter values flag this pattern.
- **Why** — the underlying cause.
- **First-line fix** — the cheapest change to try.
- **Deeper fixes** — when first-line isn't enough.
- **Exceptions** — kernel types where this pattern is actually *expected* and should be left alone.

Most kernels will match 2-4 patterns simultaneously. **Rank them by magnitude** using rocprof-compute's Speed-of-Light gap-to-peak (from section 2.1.1) and the wait-cycle-percentage breakdown. Fix the biggest one first.

> **CU vs SM:** mental model — one CU is the CDNA equivalent of an SM. MI300X has 304 CUs (8 XCDs × 38 CUs, organized over 4 IODs). MI355X has 256 CUs (8 XCDs × 32 CUs, organized over 2 IODs). The patterns below are described in those units.

---

## Pattern A — Small grid / CU idle

**Signals:**
- `Workgroups_Launched < CU_count × workgroups_per_CU` (e.g., 64 workgroups on a 304-CU MI300X)
- `SQ_BUSY_CYCLES / GRBM_GUI_ACTIVE < 0.5` averaged over all CUs
- rocprof-compute section 2.1.0 reports "waves per CU < 1"
- rocprof-compute SoL: "compute throughput much less than peak; HBM throughput much less than peak"

**Why:** each workgroup occupies at most one CU; with fewer workgroups than CUs, some CUs are completely idle throughout the kernel. On MI300X this is amplified by the XCD partition — workgroups round-robin across 8 XCDs, so a 32-workgroup launch fills only 4 CUs per XCD.

**First-line fix:** increase grid size. Look for a dimension the kernel currently doesn't parallelize:
- Add a split along `K` (split-K for reductions / attention).
- Split across heads / channels if grouped.
- Use grid-stride loops so one workgroup does multiple work units — but only if work units are cheap.

**Deeper fixes:**
- **Persistent kernel**: launch one workgroup per CU, each workgroup dequeues work items from an atomic counter. Good for dynamic-shape cases.
- **Fuse with adjacent kernels** so more work fits in one launch.
- **CPX mode**: if your problem is naturally 8-way independent, MI300X CPX exposes 8 smaller GPUs (38 CU each) with their own HBM stack — a small grid can actually fill one of these.

**Exceptions:**
- LLM decode (batch=1, query_len=1) is fundamentally small. Split-K over KV length is the standard mitigation.
- Final reduction stages of a multi-level reduction are naturally small; fuse them into the producing kernel.

**Cross-ref:** CDNA principle 1 (the companion `cdna3-cdna4-hip-programming.md`).

---

## Pattern B — Tail effect (variable-length inputs)

**Signals:**
- Multi-workload: `max_seq_len / avg_seq_len > 3` in input distribution.
- Per-CU active cycles span 5-100× between slowest and fastest CU (rocprof-compute section 2.1.23, or per-CU SQ_BUSY_CYCLES).
- PMC timeline shape: long gradual tail at the end (visible via `plot_timeline.py`).
- **Per-XCD divergence** on MI300X SPX: SQ_WAVES per XCD shows one XCD running 30%+ longer than the others.
- `Workgroups_Launched / (CU_count × waves_per_CU) > 1.05` with partial last wave.

**Why:** each workgroup iterates some variable-size inner loop. When sequences have vastly different lengths, a few long-sequence workgroups keep running after everyone else finished. On MI300X, this is worse when the slow workgroups happen to be assigned to one XCD.

**First-line fix (cheap):**
- **Packed batching / sorting**: sort inputs by length (at the application level) so workgroups running concurrently do roughly equal work.
- **Split long sequences across more workgroups**: add a `split_factor` grid dimension; each workgroup handles `ceil(seq_len / split_factor)` tokens, and a small post-reduction combines partials.

**Deeper fixes:**
- **Chunkwise kernel**: break each sequence into fixed-size chunks, process chunks in parallel, then stitch with a small recurrence. This is the approach of flash-linear-attention's `chunk_delta_rule_fwd` for Mamba/GLA-style recurrences.
- **Classify-and-dispatch**: short sequences go through the simple path (one workgroup per seq), long sequences through the chunked path.
- **XCD-aware scheduling on MI300X**: if you can group related work onto the same XCD (better Infinity Cache reuse), do so via the order of workgroup IDs.

**Exceptions:**
- Short kernels (< 20 µs) where partial-wave cost is absolute-small.
- Workloads where you already pre-sort / pre-pack.

**Cross-ref:** CDNA principle 11.

---

## Pattern C — Uncoalesced global loads

**Signals:**
- rocprof-compute section 2.1.11: "Bytes per wavefront" for global_load_* is much less than the peak 256 B (e.g. 60 B).
- `TCP_TCC_READ_REQ_sum / TCP_TOTAL_READ_REQ > 0.7` (most vL1 accesses miss to L2).
- PC-sampling: primary `Wait_Reason` on the offending load line is `WAIT_INST_VMEM` or `WAIT_VMCNT`.
- ISA shows `global_load_dword` / `global_load_dwordx2` instead of `global_load_dwordx4`.

**Why:** lanes in a wave access non-contiguous addresses; hardware fetches extra cache lines that only a few lanes use.

**First-line fix:** rework the thread ↔ data mapping:
- If current pattern is `x[lane * K + i]` (stride K), flip to `x[lane + i * 64]` (contiguous across the 64-lane wave).
- Check AoS layouts: `struct { float a, b; } arr[N]` → `struct { float a[N], b[N]; }` so each field is a separate coalesced stream.

**Deeper fixes:**
- Use LDS as a transposer: coalesced-load to LDS, then arbitrary-access from LDS.
- Vectorize: replace scalar `global_load_dword` with `global_load_dwordx2` / `dwordx4` (use `int2` / `float4` / built-in `__hip_bfloat162` types). The compiler will emit the 128-bit variant when alignment + width allow.
- **MI355X**: prefer `global_load_lds_dwordx4` (CDNA4 widened it to 128-bit/lane) to avoid round-tripping through VGPR.

**Exceptions:**
- Gather/scatter by random index (sparse matmul, embedding lookup) — fundamentally uncoalesced. Sort the indices for locality if possible.
- Graph / tree traversal.

**Cross-ref:** CDNA principles 2, 13.

---

## Pattern D — Sparse / under-vectorized writes

**Signals:**
- rocprof-compute "Bytes per wavefront" for global_store_* < 128 B (out of 256 B peak).
- vL1 → L2 write request count high relative to bytes stored (write coalescing failing).
- Code contains patterns like `if (lane_id < K) { output[...] = ... }` with K < 32.

**Why:** only a subset of wave lanes write, so the L1 store buffer flushes half-empty cache lines.

**First-line fix:** pack the write. Have the wave collectively produce `K` values first (via `__shfl_*` or LDS reduction), then have exactly `K` contiguous lanes perform `K` consecutive writes.

If `K ≥ 64`: all lanes can write; make sure the per-lane index is contiguous.

If `K < 16`: consider batching multiple iterations' results into a vectorized write (e.g., 4 iterations' output packed into a single `global_store_dwordx4`).

**Deeper fixes:**
- Write into LDS first, then do a coalesced global store at the end of the workgroup.

**Exceptions:**
- Histogram / scatter (inherently sparse) — different optimization path, see Pattern G.

---

## Pattern E — Latency-bound (VMEM-wait-dominated)

**Signals:**
- `SQ_WAIT_INST_VMEM / SQ_BUSY_CYCLES > 0.40` or PC-sampling shows `WAIT_INST_VMEM > 40%` of samples.
- `(TCC_EA0_RDREQ_32B_sum × 32) / dur / peak_HBM_BW < 0.1` (→ not HBM-BW-bound).
- Hotspot lines are global loads (check `stall_hotspots_<tag>.txt`).
- ISA at the hotspot shows `global_load_*` followed by an `s_waitcnt vmcnt(0)` close behind.

**Why:** waves issue a load, then stall waiting for it to return before the next dependent op. Usually combined with low occupancy or insufficient ILP.

**First-line fix:** increase in-flight memory requests:
- **Unroll the load loop** so 4-8 loads are issued before any value is used. Compiler + hardware reorders.
- **Add more independent waves** — raise occupancy (Pattern J).
- **`global_load_lds`**: bulk-load directly from HBM to LDS without going through VGPR. CDNA4 widens it to `dwordx4` (128 b/lane).
- **Prefetch with `s_load_dword` + scratch** for the next tile while computing on the current.

**Deeper fixes:**
- Software pipelining: while tile N is being computed, pre-load tile N+1 into LDS.
- Move reused data to LDS so subsequent loads hit LDS.

**Exceptions:**
- Pointer chasing / graph traversal — data dep chain is fundamental.

**Cross-ref:** CDNA principles 7, 15.

---

## Pattern F — Compute-bound but not on Matrix Cores

**Signals:**
- `SQ_INSTS_VALU / SQ_BUSY_CYCLES` high (waves are issuing VALU constantly).
- `SQ_INSTS_MFMA = 0` (or negligible).
- Workload is matmul-ish (GEMM, attention, conv).

**Why:** kernel uses scalar / packed-FMA via the VALU pipe instead of Matrix Cores. On MI300X, MFMA can do many × the FMA throughput of scalar VALU for BF16→FP32. On MI355X with FP4/FP6/MXFP it's even larger.

**First-line fix:** use **MFMA intrinsics** (`__builtin_amdgcn_mfma_f32_32x32x8bf16_1k`, etc.) or move to **Composable Kernel** / **hipBLASLt**. Hand-rolling tcgen05-equivalent ISA is harder on AMD because there is no TMA/TMEM — you keep tiles in LDS and read with `ds_read_b128`. CK is the canonical path.

**Deeper fixes:**
- Restructure data layout to meet MFMA tile-shape constraints (e.g., `32x32x8` for BF16 on CDNA3, `mfma_scale_*_16x16x128_f4` on CDNA4).
- On CDNA4, switch the heavy MFMA path to FP6 or MXFP4 if numerics allow — same hardware throughput as larger formats.

**Exceptions:**
- Non-matrix workloads (reduction, sort, element-wise) — Matrix Cores don't help.
- Small matrices (M, N, K < 32) — MFMA tiles are too coarse; fall back to packed-FMA VALU.

**Cross-ref:** CDNA principle 10; the CDNA companion doc's section on MFMA / Matrix Cores has examples.

---

## Pattern G — Atomic contention

**Signals:**
- Wait time concentrates on `ATOMIC_*` ISA mnemonics.
- L2 atomic traffic is large relative to read/write traffic — check the `TCC_ATOMIC*_sum` family on your install (exact suffixes vary by ROCm release; `rocprofv3 -L | grep -i atomic` shows what's exposed).
- L2 throughput is high but compute throughput is low.

**Why:** many threads atomically updating few locations → serialization.

**First-line fix:** hierarchical reduction.
- Within-wave: `__shfl_down` / `__shfl_xor` (HIP keeps the `_sync` variants for source compatibility but on CDNA gfx9 the wave is intrinsically lockstep; `warpSize = 64`) — no atomic.
- Within-workgroup: LDS reduction (no atomic).
- Between workgroups: single global atomic at the end per workgroup.

**Deeper fixes:**
- LDS histogram that flushes to global in one coalesced pass.
- Bucketing: thread writes to `output[tid % N_buckets]`, followed by a merge kernel.
- Use the HW `global_atomic_add_f32` / `global_atomic_add_f64` (CDNA3+) when `-munsafe-fp-atomics` is set — faster than CAS loop.

**Exceptions:**
- RCCL-style communication — atomics are fundamental there.

**Cross-ref:** CDNA principle 12.

---

## Pattern H — LDS bank conflicts

**Signals:**
- `SQ_LDS_BANK_CONFLICT > 0` (and substantial vs `SQ_INSTS_LDS`).
- `WAIT_INST_LDS` waits concentrated on LDS load lines.
- Access pattern has regular strides that align to bank boundaries.

**Why:** LDS has 32 banks (4 B each on gfx9 — both CDNA3 and CDNA4 keep this layout; CDNA4 just raises the total LDS size per CU); same-bank accesses serialize.

**First-line fix:** padding. `__shared__ float tile[64][33]` instead of `[64][32]` breaks regular bank alignment. If your LDS budget allows it, padding is a one-line win.

**Deeper fixes:**
- Swizzle: XOR-scramble indices so accesses spread across banks.
- Restructure data layout so wave lanes access different banks.
- Use `ds_read_b128` (loads 16 B per lane in one instruction — 4 banks per access × 32 banks = good distribution by construction).

**Exceptions:**
- Broadcast reads (all lanes read same address) are conflict-free.
- Low LDS access volume — don't bother.

**Cross-ref:** CDNA principle 4.

---

## Pattern I — Synchronization overhead

**Signals:**
- PC sampling: `WAIT_BARRIER > 20%` of samples.
- Source hotspot line is an `s_barrier` (or `__syncthreads()` in source).

**Why:** `__syncthreads()` waits for the slowest wave. Combined with any per-wave work imbalance, this amplifies.

**First-line fix:**
- Replace workgroup-level syncs with wave-level primitives (`__shfl`, `__ballot`, etc.) where only wave-scoped synchronization is needed. On CDNA gfx9 a wave is 64 lanes that execute in lockstep, so `__syncwarp` is effectively a no-op (HIP keeps it for source compatibility with CUDA); the actual lane-coordination work is done by the shuffle/ballot intrinsic itself.
- Reduce total sync count — consolidate multiple synchronized phases.

**Deeper fixes:**
- Producer / consumer waves with explicit LDS-based ready flags or `s_sendmsg` (where available) instead of `s_barrier`.
- Avoid grouping disparate workload sizes inside the same workgroup (the slowest wave dictates the rest).

**Cross-ref:** CDNA principle 16.

---

## Pattern J — Low achieved vs theoretical occupancy

**Signals:**
- `Theoretical_Occupancy_pct > 50` but `Achieved_Occupancy_pct << 50`.
- rocprof-compute section 2.1.2 reports a notable gap with the bottleneck named (`VGPR` / `LDS` / `wkg-size`).

**Why:** Theoretical occupancy is the max waves that *could* be resident. Achieved is how many are *actually* running. Gap is caused by: stalls (leaves slots empty), imbalance (some CUs empty), short kernel (warmup dominates).

**Reading:** if the gap is large AND Pattern B (tail effect) is present, fixing imbalance will close the gap. If no imbalance, look at stall reasons (Pattern E, H, I).

**First-line fix:** look for the stall reason causing the gap and address that pattern.

---

## Pattern K — Register / AGPR spill (scratch traffic)

**Signals:**
- `Scratch_Per_Workitem > 0` in launch info.
- rocprof-compute section 2.1.22 reports non-zero scratch usage.
- VGPR + AGPR allocations near per-SIMD limits.
- (Counter-side scratch read/write counters exist on most ROCm versions but their exact names
  vary — `rocprofv3 -L | grep -i scratch` on your install if you want to track the cycles
  directly.)

**Why:** compiler couldn't fit all live variables in registers (VGPR + AGPR), spilled some to scratch (which is HBM-backed via the scratch buffer — extremely slow).

**First-line fix:** `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` on the kernel. This tells the compiler to stay within a register budget. HIP-equivalent of CUDA's same flag.

**Deeper fixes:**
- Reduce the number of live values: recompute values instead of caching, split the kernel into two.
- Move per-thread arrays to LDS with explicit indexing.
- On MFMA-heavy kernels, watch AGPR pressure specifically — accumulators land in AGPR and can spill independently of VGPR.

**Exceptions:**
- Large fused kernels (FlashAttention-style) accept some spill in exchange for larger savings upstream.

**Cross-ref:** CDNA principle 6.

---

## Pattern L — FP64 used unintentionally

**Signals:**
- An FP64-specific VALU counter (e.g. `SQ_INSTS_VALU_FP64`, name varies by ROCm version —
  check `rocprofv3 -L | grep FP64`) reports > 0 in a kernel that "should" be FP32.
- ISA shows `v_*_f64` / `v_fma_f64` instructions on hot lines.
- Worse on CDNA4: FP64 throughput halved relative to CDNA3 — same source code is now 2× slower if anything slipped to double.

**Why:** C/C++ floating-point literals (`1.0`, `0.5`, `3.14`) default to `double`. A `float x = a + 1.0 * b;` promotes `a + 1.0*b` to double.

**First-line fix:** add `f` suffix to all literals: `1.0f`, `0.5f`, `3.14f`. Use `__expf` / `__logf` / `__sinf` (HIP provides the same names, mapped to AMDGPU `__ocml_*` builtins).

**Cross-ref:** CDNA principle 8.

---

## Pattern M — Pipeline bubbles (no compute/memory overlap)

**Signals:**
- PMC timeline of SQ_INSTS_VALU and TCC_EA0_RDREQ shows a sawtooth (high compute ↔ high HBM alternating).
- `SQ_WAIT_INST_VMEM` is high *and* HBM throughput is high.

**Why:** kernel loads a tile, computes on it, loads next tile — single-buffered.

**First-line fix:** double-buffer. Use two LDS tiles; while computing on tile A, load tile B with `global_load_lds_dwordx4` (CDNA3+ async-ish: the load issues, compute proceeds, `s_waitcnt vmcnt(0)` enforces ordering before switching).

**Deeper fixes:**
- Multi-stage pipeline (3-4 stages possible with CDNA3 256+256 register budget). Use `global_load_lds` plus an explicit ring buffer of LDS tiles.
- On CDNA4, LDS remains 64 KB/CU; deeper pipelines have to come from the wider `global_load_lds` per-lane variants and the more flexible VGPR/AGPR repartitioning, not from extra LDS.

**Cross-ref:** CDNA principle 15.

---

## Pattern N — Wave divergence

**Signals:**
- rocprof-compute section 2.1.13 reports "wave occupancy under control flow" much less than 64.
- Divergent branches cluster on specific source lines.
- ISA shows `s_cbranch_*` followed by `s_andn2_b64 exec, exec, ...` (mask manipulation).

**Why:** lanes in a wave (64 lanes on CDNA) take different paths at a branch; hardware serializes by mask-toggling the exec register.

**First-line fix:**
- Rearrange so all lanes in a wave take the same branch. Sort / partition data if possible.
- Convert `if (cond) a else b` to branchless `mask * a + (1-mask) * b` — cheap if both sides are cheap.

**Exceptions:**
- Tree reductions in waves (last few steps have half / quarter / ... active). Use `__shfl_down_sync` to handle cleanly.
- Boundary handling (a few waves at tensor edge) — not worth fighting.

**Cross-ref:** CDNA principle 5.

---

## Ranking template for the final report

When you hand back an optimization plan, rank by `(expected speedup) × (effort ratio)`. rocprof-compute's Speed-of-Light gap-to-peak (section 2.1.1) is your best estimator — a kernel at 14% of peak HBM has 6× of upside on the memory axis (assuming the workload genuinely is memory-bound).

```
Priority 1: <pattern> — <concrete fix>
  Evidence: <counter values + source-line citation>
  Expected speedup: <X% from rocprof-compute SoL gap, or a per-pattern estimate>
  Effort: <low / medium / high>
  Why now: <reason this is the highest-leverage fix>

Priority 2: ...
```

A good rule of thumb: at most 3-5 priorities in the plan. More than that dilutes the signal, and priorities > 5 usually contribute < 5% speedup each.
