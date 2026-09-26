#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// PTQ1_0 / PQ2_0 mat-vec on RDNA3/RDNA4 with the q8_1 (int sum) activations in src1_q8_1, which it fills first if quantize.
// Several mat-muls on the same src1 can share one src1_q8_1 of ggml_cuda_mmvq_lowbit_rdna_q8_size bytes.
bool   ggml_cuda_mmvq_lowbit_rdna_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
size_t ggml_cuda_mmvq_lowbit_rdna_q8_size(const ggml_tensor * src1);
void   ggml_cuda_mul_mat_vec_q_lowbit_rdna(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_device & fusion, void * src1_q8_1, bool quantize);
