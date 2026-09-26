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
