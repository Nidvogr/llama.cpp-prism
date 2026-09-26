#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// Like ggml_cuda_op_fwht_signed (signs may be null), but also writes the result as q8_1 with int sums (perm16:
// transposed 4x4 per 16 values), as quantize_row_q8_1_isum_cuda does. dst rows must be whole q8_1 rows.
bool ggml_cuda_op_fwht_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * signs,
                          ggml_tensor * dst, void * q8, bool perm16);
