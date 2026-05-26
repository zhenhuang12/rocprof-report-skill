// ============================================================================
// harness_template.hip — starting point for an AMD profiling harness.
//
// Copy this file into your run directory, e.g.:
//     profile/<run_name>/harness/my_kernel_harness.hip
// and customize the sections marked with TODO(you).
//
// Compile with (MI300X / gfx942):
//     hipcc -O3 -std=c++17 -gline-tables-only \
//           --offload-arch=gfx942 \
//           -munsafe-fp-atomics \
//           my_kernel_harness.hip -o my_kernel_harness
//
// Compile with (MI355X / gfx950, ROCm 7+ required):
//     hipcc -O3 -std=c++17 -gline-tables-only \
//           --offload-arch=gfx950 \
//           -munsafe-fp-atomics \
//           my_kernel_harness.hip -o my_kernel_harness
//
// Fat binary (both):
//     hipcc -O3 -std=c++17 -gline-tables-only \
//           --offload-arch=gfx942 --offload-arch=gfx950 \
//           -munsafe-fp-atomics \
//           my_kernel_harness.hip -o my_kernel_harness
//
// Usage modes this template supports:
//     ./harness --workload <path.safetensors>        # real tensor values
//     ./harness <shape_arg1> <shape_arg2> ...        # synthetic at specific shape
//
// Why it's structured this way: see ../reference/02-harness-guide.md
// ============================================================================

// This template intentionally will not compile until you have filled in the
// kernel-specific sections (allocation shapes, kernel launch, validation). Once
// you've replaced every TODO(you) below, define HARNESS_FILLED_IN (either by
// `#define HARNESS_FILLED_IN 1` at the top of your copy, or by passing
// `-DHARNESS_FILLED_IN=1` to hipcc). The guard prevents accidentally profiling
// the unmodified skeleton.
#ifndef HARNESS_FILLED_IN
#error "harness_template.hip: replace all TODO(you) sections (lines 99, 126, 172, 180, 187, 193, 202), then define HARNESS_FILLED_IN (e.g. -DHARNESS_FILLED_IN=1)."
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Keeps the safetensors_loader header next to this file for easy relocation.
#include "safetensors_loader.h"

// ---------------------------------------------------------------------------
// HIP error helpers
// ---------------------------------------------------------------------------
#define HIP_CHECK(expr)                                                                 \
    do {                                                                                \
        hipError_t err = (expr);                                                        \
        if (err != hipSuccess) {                                                        \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,   \
                    hipGetErrorString(err));                                            \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

template <typename T>
static T* alloc_device(size_t n) {
    T* p = nullptr;
    HIP_CHECK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

// ---------------------------------------------------------------------------
// Synthetic fill helpers — replace scale with something sensible for your kernel
// ---------------------------------------------------------------------------
static void fill_bf16_random(std::vector<__hip_bfloat16>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = __float2bfloat16(d(rng));
}
static void fill_f16_random(std::vector<__half>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = __float2half(d(rng));
}
static void fill_f32_random(std::vector<float>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = d(rng);
}

// ============================================================================
// TODO(you): paste / include the kernel source here
//
//   - Include any __device__ helpers, intrinsics, and constants.
//   - HIP supports `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`
//     to budget VGPR — use it rather than searching for `--maxregcount` (no equivalent).
//   - Wave size on CDNA is 64 (compile-time `warpSize`).
//   - Add explicit template instantiations for every variant you want to be
//     able to target with `rocprofv3 --kernel-include-regex "..."`.
//
// Example:
//     template <int TILE_M, int TILE_N>
//     __global__ __launch_bounds__(256, 2)
//     void my_kernel(const __hip_bfloat16* A, const __hip_bfloat16* B,
//                    float* C, int M, int N, int K) {
//         // ... kernel body ...
//     }
//     // Explicit instantiations — without these, unused variants get stripped
//     // and rocprofv3's regex won't find them.
//     template __global__ void my_kernel<64, 64>(
//         const __hip_bfloat16*, const __hip_bfloat16*, float*, int, int, int);
//     template __global__ void my_kernel<128, 64>(
//         const __hip_bfloat16*, const __hip_bfloat16*, float*, int, int, int);
// ============================================================================

// ... your kernel goes here ...

// ============================================================================
// TODO(you): launch helper
//
// Take the input pointers and shape parameters, run the kernel with the
// chosen template parameters / grid / workgroup. Keep this simple — a single
// `hipLaunchKernelGGL(...)` or chevron-syntax call.
// ============================================================================
static void launch_kernel(/* input pointers, shape params, hipStream_t */) {
    // dim3 grid(...);
    // dim3 block(...);
    // hipLaunchKernelGGL((my_kernel<TILE_M, TILE_N>),
    //                    grid, block, /*sharedMem*/ 0, /*stream*/ 0,
    //                    A, B, C, M, N, K);
    // // Or with chevron syntax (HIP supports it):
    // // my_kernel<TILE_M, TILE_N><<<grid, block>>>(A, B, C, M, N, K);
    // HIP_CHECK(hipDeviceSynchronize());
    // HIP_CHECK(hipGetLastError());
}

// ============================================================================
// CLI parsing + input prep. Shape-match mode fills synthetic values at an
// exact shape (e.g., from a user-specified (M, N, K) tuple). --workload mode
// loads a real safetensors file.
// ============================================================================

static void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --workload <safetensors_path>\n"
            "      Load real tensor values from a workload file.\n"
            "  %s <shape_arg1> <shape_arg2> ...\n"
            "      Shape-match mode (synthetic at given shape).\n",
            argv0, argv0);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    bool use_workload = false;
    std::string workload_path;

    // --- Parse arguments -----------------------------------------------------
    if (std::string(argv[1]) == "--workload") {
        if (argc < 3) { usage(argv[0]); return 2; }
        use_workload = true;
        workload_path = argv[2];
    } else {
        // TODO(you): parse your shape args from argv here.
        // E.g., int M = std::atoi(argv[1]); int N = std::atoi(argv[2]); ...
    }

    // --- Load real workload if requested ------------------------------------
    SafetensorsFile st;
    if (use_workload) {
        st = SafetensorsFile::load(workload_path);
        // TODO(you): read shapes from the safetensors header.
        //   const auto& a_entry = st.entry("A");
        //   int M = (int)a_entry.shape[0];
        //   int K = (int)a_entry.shape[1];
        fprintf(stderr, "[harness] loaded workload: %s\n", workload_path.c_str());
    }

    // --- TODO(you): allocate inputs / outputs on host & device --------------
    // std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.f);
    // float *d_A = alloc_device<float>(h_A.size());
    // float *d_B = alloc_device<float>(h_B.size());
    // float *d_C = alloc_device<float>(h_C.size());

    // --- TODO(you): fill inputs ---------------------------------------------
    // if (use_workload) {
    //     std::memcpy(h_A.data(), st.tensor_bytes("A"), h_A.size() * sizeof(float));
    //     std::memcpy(h_B.data(), st.tensor_bytes("B"), h_B.size() * sizeof(float));
    // } else {
    //     fill_f32_random(h_A, 0xA0A0ULL);
    //     fill_f32_random(h_B, 0xB0B0ULL);
    // }

    // --- TODO(you): copy inputs to device ----------------------------------
    // HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), hipMemcpyHostToDevice));

    // --- Launch --------------------------------------------------------------
    fprintf(stderr, "[harness] launching kernel...\n");
    launch_kernel(/* d_A, d_B, d_C, M, N, K */);
    fprintf(stderr, "[harness] done.\n");

    // --- Cleanup -------------------------------------------------------------
    // HIP_CHECK(hipFree(d_A));
    // HIP_CHECK(hipFree(d_B));
    // HIP_CHECK(hipFree(d_C));

    return 0;
}
