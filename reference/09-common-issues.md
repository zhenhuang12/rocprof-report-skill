# Common Issues & Gotchas (AMD ROCm profiling stack)

Collected solutions for the recurring frustrations of profiling HIP / Triton-AMD kernels with rocprofv3, rocprof-compute, and ATT/PC-sampling.

---

## rocprofv3 / rocprof-compute permissions

### `HSA_STATUS_ERROR: Profiling permissions not granted`

You need access to `/dev/kfd` and `/dev/dri/renderD*` and (usually) membership in the `render` (and on some distros also `video`) group.

```bash
ls -la /dev/kfd /dev/dri/renderD*       # check ownership
id                                       # confirm you're in render / video
getfacl /dev/kfd                         # any extra ACLs?
```

Fixes:

**A) Add yourself to the right groups:**
```bash
sudo usermod -aG render,video $USER
# log out and back in (or run `newgrp render`)
```

**B) In a container:** run with `--device=/dev/kfd --device=/dev/dri --group-add render --group-add video`. Many ROCm Docker images set this up automatically; check `docker inspect <container> | grep Devices`.

**C) ATT / PC sampling need additional privileges on some kernels.** Try `setcap cap_perfmon=ep $(which rocprofv3)` or, as a last resort, `sudo`. Unlike NVIDIA, you usually do NOT need root for plain PMC collection — that's a sign your `render` group is misconfigured.

### `Failed to load HSA tools library` / `libhsa-runtime64.so not found`

ROCm install path isn't on `LD_LIBRARY_PATH`:
```bash
source /opt/rocm/bin/env.sh                # or wherever your ROCm install lives
# or manually:
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export PATH=$ROCM_PATH/bin:$PATH
```

### `rocprofv3: command not found` but `rocprof` works

You're on ROCm 6.0/6.1 which only ships `rocprof` (the legacy tool). Upgrade to ROCm 6.2+ for `rocprofv3` and ROCm 6.3+ for `rocprof-compute` (formerly Omniperf). MI355X (gfx950) **requires ROCm 7+**.

---

## `--kernel-include-regex` (rocprofv3) / `-k`/`--kernel` (rocprof-compute) matches nothing

1. **Use the demangled name.** Templates produce something like `my_kernel<8, 256>(...)`. Check:
   ```bash
   # Modern (ROCm 6.4+)
   llvm-objdump --offloading --arch-name=amdgcn-amd-amdhsa--gfx942 \
                ./harness -o /tmp/co.o
   llvm-objdump --syms --demangle /tmp/co.o | grep -iE 'FUNC|KERNEL'

   # Legacy
   roc-obj-ls ./harness
   ```
   Match against the demangled form. Itanium-ABI mangling: `_Z9my_kernelILi8ELi256EEvPK...` demangles to `void my_kernel<8, 256>(...)`.

2. **Escape regex metacharacters carefully.** Parentheses, `<`, `>` may need escaping depending on the regex engine. rocprofv3 uses POSIX ERE by default.

3. **Kernel never launched.** Run the harness without rocprof and confirm the kernel actually runs. Add `AMD_LOG_LEVEL=4` to see kernel launches in the runtime log.

4. **Wrong `--offload-arch`.** If you compiled for `gfx906` and the host has `gfx942`, `hipErrorInvalidDeviceFunction` fires at launch and no kernel runs. Check `rocminfo | grep gfx` for the actual device arch.

---

## Source view is empty / `Source` column blank in PC-sampling CSV / ATT

The binary was compiled without `-gline-tables-only` (or `-g`). Add it:
```bash
hipcc -O3 -std=c++17 -gline-tables-only --offload-arch=gfx942 \
      -DHARNESS_FILLED_IN=1 \
      kernel.hip -o harness   # drop -DHARNESS_FILLED_IN=1 if kernel.hip is not from the template
```

For JIT / framework-integrated builds:

- **PyTorch `torch.utils.cpp_extension.load` on ROCm**: pass `extra_cuda_cflags=["-gline-tables-only"]` (the kwarg name is `extra_cuda_cflags` even on ROCm — it's just passed to `hipcc`).
- **Triton (Triton-MLIR for AMD)**: the MLIR pipeline owns the final HSACO; user hipcc flags are ignored. Easiest fix: build a standalone harness from the dumped HIP code. Set `TRITON_KERNEL_DUMP=1` or `MLIR_ENABLE_DUMP=1` to see the lowering.
- **Composable Kernel JIT**: edit the `ck.json` build config to inject `-gline-tables-only`, or rebuild the targeted kernel as a standalone.
- **rocBLAS / hipBLASLt**: most prebuilt kernels are stripped of debug info. To profile a specific GEMM, build a tiny harness that calls `hipblasltMatmul` with the same problem size — the source-line view will at least cover your host call site; for ISA-level you'll need ATT without source attribution.

---

## rocprofv3 rejects an argument as "unrecognized"

If `rocprofv3` errors with `unrecognized arguments: --list-avail` (or `--input-file`, `--list-counters`, etc.), you're using a flag from the legacy `rocprof` v1 CLI.

- The v3 long form for listing counters is `--list-avail` (short form: `-L`). The legacy `--list-counters` / `--list-metrics` / `--list-basic` / `--list-derived` are all gone.
- The v3 input-file flag is `-i <file>` (a YAML PMC spec); the legacy `--input-file <txt>` syntax is gone.
- Run `rocprofv3 --help` for the full v3 surface — many v1 flag names were renamed wholesale, not just shortened. Don't paste v1 recipes from old wikis without translating first.

---

## PC sampling silently no-ops

1. **Beta flag missing.** PC sampling is gated behind a beta flag in ROCm 6.4+. The CLI needs `--pc-sampling-beta-enabled` (or set `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1` in the environment).
2. **Method not supported on your hardware.** Try `--pc-sampling-method host_trap` first (note the underscore, not `host-trap`) — it works on MI200+ and is the most portable. `stochastic` is lower-overhead but requires MI300+ and a recent ROCm build with the kernel-mode feature compiled in.
3. **Sampling interval too coarse / wrong unit.** With `host_trap` you MUST use `--pc-sampling-unit time`; the interval is in **microseconds** with a runtime-enforced minimum (ROCm 6.4+ enforces a floor in the low-µs range and intervals below it return `HSA_STATUS_ERROR_INVALID_ARGUMENT`). 1000 (= 1 ms) is a sensible default; for sub-ms kernels, 100 µs is usually the smallest practical value before host_trap overhead distorts the profile. Passing `cycles` or `instructions` with `host_trap` is rejected at runtime as "PC sampling configuration is not supported"; those units are stochastic-only.
4. **Kernel too short.** Kernels under ~50 µs may not produce useful sample counts. Increase work in the harness (run the kernel in a small loop — but be aware rocprofv3 replays each PMC group, so this can blow up wall time).
5. **Permission issue.** Some ROCm builds require `CAP_PERFMON` for PC sampling. Try `sudo` to confirm it's a perms issue.

---

## ATT (Advanced Thread Trace / SQTT) gotchas

1. **Default captures 1 CU per SE.** That's intentional — ATT generates ~10s-100s of MB per CU per millisecond. To cover more, increase `--att-shader-engine-mask 0xF` (first 4 SEs) and pick the CU index that matches your tail-effect probe.
2. **Buffer overflow / truncated trace.** Bump `--att-buffer-size 0x40000000` (1 GB) and reduce the captured kernel duration. The trace is per-SE; total memory is buffer_size × num_SEs.
3. **`att_<tag>/*.json` empty or `Source` column blank.** Rebuild with `-gline-tables-only`. ATT also needs symbols to attribute to source — `strip` will silently break it.
4. **One JSON per CU/SE, not one for the run.** Glob `att_<tag>/**/*.json` and merge in Python. Use `att_tool` (ships with rocprofv3) for binary-format ATT.
5. **Captures the wrong kernel iteration.** ATT triggers on the *first* matching launch by default. To capture iteration N, combine with `--kernel-iteration-range "[N-N]"` (rocprofv3 uses a hyphen, 1-indexed, inclusive on both ends; the Python-slice form `[N:N+1]` is not accepted).

---

## rocprof-compute takes forever to finish

1. **PMC group replays.** `rocprof-compute profile` runs ~15-30 passes (one per PMC counter group), and rocprofv3 **replays the entire binary** each pass — not just the kernel. If your kernel takes 10 ms but init takes 5 s, each pass is 5.01 s and a full profile is ~150 s. Move host-side init out of the profile window, or shrink the harness.
2. **Roofline benchmarks add ~30 s on first run.** Roofline is ON by default (there is NO `--roofline` flag — invoking it crashes). Pass `--no-roof` to `rocprof-compute profile` to skip the roofline pass once you've cached it.
3. **Don't profile with debug builds.** `-O0 -g` is ~10× slower than `-O3 -gline-tables-only` and the codegen doesn't represent prod.

---

## `rocprof-compute analyze` output is mostly N/A

The PMC groups that back each section weren't collected. Causes:

- You passed `--pmc <small list>` instead of running `rocprof-compute profile`, so only those counters exist. Run the full `rocprof-compute profile` for SoL / section analysis.
- The IP block is disabled on this hardware partition (e.g., CPX mode reduces visible CUs and TCC channels).
- ROCm version too old for some sections (CDNA4-specific MFMA shapes need ROCm 7+).

Try `rocprof-compute analyze -p rpc_<tag> -b <ID>` to render a single section and see what's missing.

---

## Kernel crashes / produces NaN only under rocprof

1. **`AMD_SERIALIZE_KERNEL=3` exposes latent races.** Run with it set both with and without rocprof to confirm the bug is racy, not profiler-induced.
2. **`-munsafe-fp-atomics`** enables HBM atomic instructions that some old drivers handle incorrectly. If the kernel only NaN's with this flag, file a driver bug and fall back to `-mno-unsafe-fp-atomics` (uses CAS loops, slower but correct).
3. **GPU clock changes between replays.** Lock with `sudo rocm-smi --setperflevel high` and rerun.

---

## CSV column missing from `pmc_perf.csv`

1. **Wrong counter name for this gfx.** See [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md). gfx906/908/90a name differently than gfx942/950.
2. **Counter not in this PMC group.** rocprofv3 logs the group assignment — search the log for the counter name. If you need a specific counter, list it explicitly via `--pmc` or a YAML job file.
3. **Counter is conditional.** Some counters (MFMA per-shape) only emit non-zero values when the kernel actually used that instruction shape. The column exists; the value is 0.

Always wrap CSV reads in a safe accessor:
```python
def safe_col(df, name, default=None):
    return df[name] if name in df.columns else default
```

---

## `rocpd` import fails (ROCm 7+ Python helper)

```bash
python3 -c "import rocpd; print(rocpd.__file__)"
# If ImportError:
find /opt/rocm* -name "rocpd*" -type d 2>/dev/null
# e.g. /opt/rocm-7.0.0/share/rocprofiler-sdk/python
export PYTHONPATH=$PYTHONPATH:/opt/rocm-7.0.0/share/rocprofiler-sdk/python
python3 -c "import rocpd; print('OK')"
```

If still broken, fall back to plain `sqlite3` — the `.db` file uses the open `rocpd` schema and you don't need the helper:
```python
import sqlite3, pandas as pd
from pathlib import Path
# rocprofv3 names this `<pid>_results.db` — glob since PID varies per run
db_path = next(Path("trace_<tag>").glob("*_results.db"))
con = sqlite3.connect(db_path)
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con))
```

---

## Triton-AMD specific

### "I can't profile my Triton kernel"

- Set `MLIR_ENABLE_DUMP=1` (or `TRITON_KERNEL_DUMP=1` depending on Triton version) to dump the lowered HIP source.
- Locate the cached HSACO under `~/.triton/cache/<hash>/`.
- Build a standalone harness from the dumped source — that's the only reliable way to get `-gline-tables-only` source attribution.
- For Triton-emitted kernel names, demangling sometimes reveals long auto-generated symbols. `rocprofv3 --kernel-trace` (no regex) is the easiest way to see the actual symbol used.

### "Triton kernel name changes between runs"

Triton recompiles when input shapes / dtypes change. Pin one launch with fixed shapes, or compile once with `triton.compile(...)` and reuse.

---

## PyTorch + ROCm specific

### Profiling a PyTorch op's underlying kernel

1. Identify the kernel: `torch.profiler` with `activities=[torch.profiler.ProfilerActivity.CUDA]` (yes, `CUDA` — PyTorch's ROCm build uses the same enum) names it.
2. Many torch kernels on ROCm are MIOpen / hipBLASLt / Composable Kernel JIT outputs. The kernel symbol is auto-generated and not worth chasing — build a harness with the same problem size and profile that.
3. For `torch.compile`-generated Triton-on-ROCm, same notes as Triton above.

### Profiling HIP Graph-captured kernels

rocprofv3 handles HIP Graph launches — each captured kernel shows up as a separate dispatch in `kernel_trace.csv`. Use `--kernel-include-regex` + `--kernel-iteration-range` to target.

---

## Reproducibility

### Results jitter between runs

1. **Lock GPU clocks:**
   ```bash
   sudo rocm-smi --setperflevel high       # max sustained clocks
   # profile
   sudo rocm-smi --resetclocks              # restore
   ```
   For more control: `sudo rocm-smi --setsclk <level>` (SCLK = shader clock) and `--setmclk <level>` (memory).
2. **Avoid thermal throttling.** Check `rocm-smi --showtemp --showthrottle` during the run. MI300A APU variants share thermal headroom with CPU cores — pin CPU to perf governor too.
3. **Lock GPU partitioning.** `rocm-smi --showcomputepartition --showmemorypartition` to confirm SPX/NPS1 (or whichever your prod uses); changes require a reboot to take effect.

### Reports don't match colleague's results

- Check ROCm version (`rocm-smi --showversion`, `rocprofv3 --version`, `rocprof-compute --version`). Counter names and block IDs occasionally shift.
- Check GPU partition state at profile time (recorded in `sysinfo.csv` — wide single-row format on current rocprof-compute).
- Check exact hipcc invocation — a stray `-O0`, missing `-gline-tables-only`, or wrong `--offload-arch` makes a big difference.
- Check whether roofline ran (default-on; suppress with `--no-roof`). SoL percentages are normalized against the cached roofline benchmark.

---

## Output interpretation

### "Compute SoL = X%, is that good?"

It depends on the kernel type:
- GEMM / MFMA-heavy: should be 60%+ on MI300X. Below 30% is bad.
- Element-wise / reduction: usually 5-15%, because they're HBM-BW-bound — check HBM SoL instead.
- Attention / softmax: varies wildly; compare against a reference (e.g., FlashAttention-3 on AMD).

Always check both SoL gaps:
- If `HBM SoL` is high (>70%) and `Compute SoL` is low, the kernel is correctly HBM-bound — focus on reducing traffic, not raising compute.
- If both are low, the kernel is latency-bound — look at the stochastic-PC-sampling `Stall_Reason` breakdown (the only granular wait classification on gfx942/gfx950; `SQ_WAIT_INST_VMEM` is **not** a PMC on these gens — only `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS` exist as PMCs; the `host_trap` PC-sampling mode does NOT populate `Stall_Reason`).

### "rocprof-compute says `Wavefront occupancy = X / 8` — is that bad?"

Achieved occupancy < theoretical means waves can't all be resident — usually VGPR/AGPR or LDS pressure. Check the wavefront-launch block (`-b 7` in `rocprof-compute analyze`):
- High `Arch_VGPR` (>128/work-item) → register pressure
- High `LDS_Per_Workgroup` relative to CU LDS budget (**64 KB/CU on gfx942 (MI300X); 160 KB/CU on gfx950 (MI355X)** — CDNA4 enlarged LDS 2.5× and doubled read BW to 256 B/cycle) → LDS pressure limits resident workgroups
- Non-zero `Scratch_Per_Workitem` → spill (kills perf)
- High `Accum_VGPR` (AGPR pool, CDNA3+) on MFMA-heavy kernels → MFMA accumulator pressure

Fix: shrink VGPR with `__launch_bounds__`, refactor to use AGPR for MFMA accumulators (CDNA3+), pad/cut LDS allocation, hoist loop-invariants out of the kernel.

### "All my MFMA counters are 0"

Either:
1. The compiler didn't emit MFMA — check the ISA: `llvm-objdump --disassemble --arch=gfx942 harness | grep -i mfma`.
2. The PMC group containing the MFMA counters wasn't collected. Rerun rocprof-compute or add `--pmc SQ_INSTS_MFMA` (aggregate) or the relevant `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` counter for the per-dtype breakdown.
3. You're using Composable Kernel / hipBLASLt but profiling a wrapper that calls them indirectly — make sure the regex matches the actual MFMA-emitting kernel, not the launcher.

---

## ROCm version & hardware compatibility quick reference

| Feature | Min ROCm | Notes |
|---|---|---|
| `rocprofv3` | 6.2 | Default profiling tool from 6.2 forward |
| `rocprof-compute` | 6.3 | Formerly Omniperf (Omnitrace is a separate, system-wide tracer — different tool) |
| ATT / SQTT | 6.0 (rocprof) / 6.2 (rocprofv3) | Capture per SE/CU |
| PC sampling (`host_trap`) | 6.2 | MI200+; ROCm 6.4+ also requires `--pc-sampling-beta-enabled` |
| PC sampling (`stochastic`) | 6.3 | MI300+; same beta-flag requirement as above |
| Rocpd `.db` default | 7.0 | Single SQLite vs many CSVs |
| MI300X (gfx942) discrete | 6.0 | Full support |
| MI300A APU (gfx942 APU) | 6.0 | Shares counters with discrete; xGMI traffic visible |
| MI355X (gfx950) | **7.0** | 6.x will refuse `--offload-arch=gfx950` |
| MFMA `_F6F4` counter | 7.0 + gfx950 | CDNA4 only — no separate `_F4` / `_F6` / `_MXFP4` / `_MXFP6` / `_MXFP8` PMC suffixes |
| MFMA `_XF32` counter | 6.2 + gfx942/gfx950 | XF32 (TF32-equivalent) MFMA exists on BOTH CDNA3 and CDNA4 — not gfx950-only |
| ROCprof Compute Viewer | 6.3 | GUI for rocprof-compute output |
| RGP (Radeon GPU Profiler) | n/a | **Does NOT support CDNA / Instinct.** Use ROCprof Compute Viewer instead. |
