#include "common.cuh"
#include "fwht.cuh"

#include <cstdlib>

template <typename T>
__device__ __forceinline__ float fwht_load(const T value) {
    return value;
}

template <>
__device__ __forceinline__ float fwht_load<half>(const half value) {
    return __half2float(value);
}

template <int N, typename T, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = fwht_load(src[i * warp_size + lane]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Large-N path. The register kernel above keeps N/warp_size floats per thread, so it stops being
// viable well before the arithmetic does: N=4096 would need 128 registers per thread and spill,
// which is why the switch below used to end at 2048 and simply decline anything larger (the whole
// op then fell back to CPU). Stage the row in shared memory instead and run all log2(N) butterfly
// stages there. One row per block; every thread handles several butterflies per stage. Slower per
// row than the register path, so it is used only where that path cannot go.
#define FWHT_SMEM_THREADS 256

template <int N, typename T, bool has_signs>
__launch_bounds__(FWHT_SMEM_THREADS, 1)
__global__ void fwht_cuda_smem(const T * src, float * dst, const int64_t n_rows, const float scale,
                               const float * signs, const int n_blk) {
    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        float v = fwht_load(src[i]) * scale;
        if (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    __syncthreads();

    // Same butterfly and the same sign convention as the register path: the low element of a pair
    // takes x + y, the high one x - y.
#pragma unroll 1
    for (int h = 1; h < N; h *= 2) {
        for (int idx = threadIdx.x; idx < N / 2; idx += FWHT_SMEM_THREADS) {
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        dst[i] = s[i];
    }
}


// Wide rows at small row counts (decode): one row per block instead of per warp.
// The warp kernel serialises every stage on one warp, and the shared-memory kernel above synchronises on each stage.
// Both leave most of the GPU idle at these shapes.
#define FWHT_BLOCK_THREADS 256

// q8: 0 = float output only, 1 = also q8_1 with int sums (quantize_row_q8_1_isum_cuda), 2 = same with perm16.
// norm: src is the input of RMS_NORM(eps) * norm_w over rows of norm_len values, applied while loading.
template <int N, int NT, typename T, bool has_signs, int q8 = 0, bool norm = false>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs, const int n_blk, void * q8_dst,
                                const float * norm_w, const float eps, const int norm_len) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();

    // each block reduces the whole normalized row again: norm_len values, cheap next to the transform
    float norm_scale = 1.0f;
    if constexpr (norm) {
        __shared__ float warp_ss[NT / warp_size];
        const T * x_row = src + (r * N / norm_len) * norm_len;
        float ss = 0.0f;
        for (int col = tid; col < norm_len; col += NT) {
            const float xi = fwht_load(x_row[col]);
            ss += xi * xi;
        }
        ss = warp_reduce_sum<warp_size>(ss);
        if (lane == 0) {
            warp_ss[tid / warp_size] = ss;
        }
        __syncthreads();
        ss = 0.0f;
#pragma unroll
        for (int w = 0; w < NT / warp_size; ++w) {
            ss += warp_ss[w];
        }
        norm_scale = rsqrtf(ss / norm_len + eps);
        norm_w += (r * N) % norm_len;
    }

    src += r * N;
    dst += r * N;

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        if constexpr (norm) {
            // same order as RMS_NORM + MUL, then the signed transform
            reg[i] = (norm_scale * fwht_load(src[i * NT + tid])) * norm_w[i * NT + tid];
            reg[i] *= scale;
        } else {
            reg[i] = fwht_load(src[i * NT + tid]) * scale;
        }
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    if constexpr (q8 != 0) {
        // each warp holds 32 consecutive values of every register, one q8_1 block
        block_q8_1 * y = (block_q8_1 *) q8_dst + r * (N / QK8_1);
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            const float amax = warp_reduce_max<QK8_1>(fabsf(reg[i]));
            const float d    = amax / 127.0f;
            const int   q    = amax == 0.0f ? 0 : (int) roundf(reg[i] / d);
            const int   sumq = warp_reduce_sum<QK8_1>(q);
            const int   ib   = (i * NT + tid) / QK8_1;
            const int   iqs  = lane % QK8_1;
            y[ib].qs[q8 == 2 ? (iqs & 16) | ((iqs & 3) << 2) | ((iqs >> 2) & 3) : iqs] = (int8_t) q;
            if (iqs == 0) {
                y[ib].ds = make_half2(d, __ushort_as_half((unsigned short) sumq));
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

static const bool fwht_legacy = getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;

template <typename T>
static bool fwht_launch(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    cudaStream_t stream = ctx.stream();
    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n) {
#define FWHT_CASE(NN) \
        case NN: \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        default:
            break;
    }
    // From 512 up, one block of FWHT_BLOCK_THREADS per row (fwht_cuda_block).
    // The older kernels were the largest single kernel of a decode step at these widths.
    // GGML_CUDA_FWHT_LEGACY=1 restores them for A/B.
#define FWHT_SMEM_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_SMEM_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true; \
        }
#define FWHT_BLOCK_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk, nullptr, nullptr, 0.0f, 0); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1, nullptr, nullptr, 0.0f, 0); \
            } \
            return true; \
        }
    if (fwht_legacy) {
        switch (n) {
            FWHT_CASE(512)
            FWHT_CASE(1024)
            FWHT_CASE(2048)
            FWHT_SMEM_CASE(4096)
            FWHT_SMEM_CASE(8192)
            default:
                return false;
        }
    }
    switch (n) {
        FWHT_BLOCK_CASE(512)
        FWHT_BLOCK_CASE(1024)
        FWHT_BLOCK_CASE(2048)
        FWHT_BLOCK_CASE(4096)
        FWHT_BLOCK_CASE(8192)
#undef FWHT_CASE
#undef FWHT_SMEM_CASE
#undef FWHT_BLOCK_CASE
        default:
            return false;
    }
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    float * dst_d = (float *) dst->data;
    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        return fwht_launch<float>(ctx, (const float *) src->data, dst_d, n, rows, scale, signs, n_blk);
    }
    return fwht_launch<half>(ctx, (const half *) src->data, dst_d, n, rows, scale, signs, n_blk);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}

bool ggml_cuda_op_fwht_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * norm_w_t,
                             const float eps, const ggml_tensor * signs_t, ggml_tensor * dst, void * q8, const bool perm16) {
#if defined(GGML_USE_HIP)
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst) || dst->type != GGML_TYPE_F32 ||
            src->type != GGML_TYPE_F32 || fwht_legacy) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    const float * norm_w   = nullptr;
    int           norm_len = 0;
    if (norm_w_t) {
        norm_len = src->ne[0];
        if (!signs || norm_w_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(norm_w_t) ||
                norm_w_t->ne[0] != norm_len || ggml_nrows(norm_w_t) != 1 || norm_len % n != 0) {
            return false;
        }
        norm_w = (const float *) norm_w_t->data;
    }

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;
    const float   scale = 1 / sqrtf(n);
    const int     mode  = q8 ? (perm16 ? 2 : 1) : 0;
    const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, ctx.stream());

#define FWHT_FUSED_LAUNCH(NN, SIGNS, MODE, NORM) \
    ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, float, SIGNS, MODE, NORM>, lp, src_d, dst_d, rows, scale, \
        signs, n_blk, q8, norm_w, eps, norm_len)
#define FWHT_FUSED_CASE(NN) \
        case NN: \
            if (norm_w) { \
                if (mode == 2)      { FWHT_FUSED_LAUNCH(NN, true, 2, true); } \
                else if (mode == 1) { FWHT_FUSED_LAUNCH(NN, true, 1, true); } \
                else                { FWHT_FUSED_LAUNCH(NN, true, 0, true); } \
            } else if (mode == 0) { \
                return false; \
            } else if (signs) { \
                if (mode == 2) { FWHT_FUSED_LAUNCH(NN, true, 2, false); } else { FWHT_FUSED_LAUNCH(NN, true, 1, false); } \
            } else { \
                if (mode == 2) { FWHT_FUSED_LAUNCH(NN, false, 2, false); } else { FWHT_FUSED_LAUNCH(NN, false, 1, false); } \
            } \
            return true;
    switch (n) {
        FWHT_FUSED_CASE(512)
        FWHT_FUSED_CASE(1024)
        FWHT_FUSED_CASE(2048)
        default:
            return false;
    }
#undef FWHT_FUSED_CASE
#undef FWHT_FUSED_LAUNCH
#else
    GGML_UNUSED_VARS(ctx, src, norm_w_t, eps, signs_t, dst, q8, perm16);
    return false;
#endif // defined(GGML_USE_HIP)
}
