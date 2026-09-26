#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// HIP: signed FWHT (signs may be null) with optional fused parts. norm_w: src is the input of RMS_NORM(eps) * norm_w
// (needs signs). q8: also write the result as q8_1 with int sums (perm16: 4x4 transposed per 16 values), as
// quantize_row_q8_1_isum_cuda does; dst rows must be whole q8_1 rows. Returns false if it does not apply.
bool ggml_cuda_op_fwht_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * norm_w,
                             float eps, const ggml_tensor * signs, ggml_tensor * dst, void * q8, bool perm16);

// Gated head norm in front of the signed FWHT: dst = FWHT(signs * perm(silu(z) * RMS_NORM(x) * norm_w)), where the norm
// runs over head rows of hd values and perm is an optional permuted view of the gated tensor (pne/pnb in floats, the
// identity when not permuted). HIP only, with the same q8 output option as ggml_cuda_op_fwht_fused.
struct ggml_cuda_fwht_gated_norm_args {
    const float * x;      // RMS_NORM input, rows of hd values
    int           x_s1, x_s2, x_s3;
    int           hd;
    int           H, T;   // x->ne[1], x->ne[2]
    const float * norm_w; // hd values
    float         eps;
    const float * z;      // gate, row stride z_s1
    int           z_s1;
    int           pne[4];
    int           pnb[4];
};
bool ggml_cuda_op_fwht_gated_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_fwht_gated_norm_args & args,
                                  const ggml_tensor * signs, ggml_tensor * dst, void * q8, bool perm16);
