# Handoff: ROCm/HIP optimization for Ternary Bonsai-2 27B on the RX 9070 XT

Status for the next agent. Read AGENTS.md first (repo rules: ASCII only, short simple comments, reuse existing
infrastructure, no new files under `tests/`).

## Goal

Make the Ternary Bonsai-2 27B GGUFs from
`gaston-parravicini/Ternary-Bonsai-2-27B-Uncensored-Gaston-GGUF` (PTQ1_0 and PQ2_0, each with and without MTP) run
faster on the ROCm/HIP backend, tuned for the RX 9070 XT.

- The card is gfx1201 (RDNA4): wave32, 64 KB LDS, about 640 GB/s.
- The model is Qwen3.5 hybrid: GDN (gated delta net) linear attention layers plus full attention layers.
- Its projections are Hadamard rotated: a sign MUL, then a FWHT, then the low-bit mat-mul.

Guiding rules, carried over from the notes of the previous agent (NInfer on a 4090):

- Measure on hardware.
- Correctness first.
- Greedy output must stay lossless apart from float summation order.

## Status: NOTHING HAS RUN ON A GPU YET

This container has no AMD GPU. Everything is:

- compiled for gfx1201 (some kernels also for gfx1100, gfx1030, gfx906 and gfx942);
- checked at the ISA level (instruction counts, VGPRs, no scratch);
- checked with host-side C++ mirrors of the bit tricks;
- covered by new test-backend-ops cases, which have never been run against a real ROCm device.

The first step for anyone with the card is `scripts/hip/bench-bonsai-rdna.sh` (below). Then set defaults from the
numbers.

## Branches and commits

- Work branch: `claude/youthful-franklin-nbkkas`. Base: `adfffbe`, which is also where the default branch `prism` was.
- The branch has also been fast-forwarded onto `prism` (the user asked for "main"; the repo has no `main`, and
  `prism` is its default branch).

Commits, oldest first:

| commit | what |
|---|---|
| 0e22812 | PTQ1_0 mat-vec kernel for RDNA3/RDNA4 (decode) |
| a8b770e | PTQ1_0 prefill through MMQ on RDNA3/RDNA4 (WMMA) and CDNA (MFMA) |
| 0530a88 | PQ2_0 in the same RDNA low-bit mat-vec kernel |
| 045c63c | quantize a shared activation once for several low-bit mat-vecs |
| 7e300c7 | write the q8_1 activation inside the FWHT kernel (no separate quantize launch) |
| 915aec4 | review fixes (duplicated tests, DISABLE_FUSION, RDNA gating, dedup) |
| 8065877 | A/B knobs + `scripts/hip/bench-bonsai-rdna.sh` + perf/eval test shapes |
| 0428e0f | fold RMS_NORM + MUL into the signed FWHT (+q8) |
| 301de4c | opt-in PTQ1_0 prefill via expand-to-Q8_0 then Q8_0 MMQ tiles |
| 4280f81 | one launch for GDN output norm + gate + head regroup + FWHT (+q8) |
| 106eeff | bench script: SPEC / LONGCTX / output head notes |
| 3121474 | GDN state read in place (rows mode) on ROCm, no get_rows gather |
| 9200394 | opt-in F16 GDN recurrent state (`LLAMA_RS_STATE_F16=1`) |
| (last) | bench script entries for the state knobs + this file |

`git log --format='%h %s%n%b' adfffbe..HEAD` has the detailed reasoning of each commit, with numbers.

## Format background (needed to read the kernels)

- **PTQ1_0** is a 128-weight block of 28 bytes: `qs[24]` at 5 trits per byte, `qh[2]` at 4 trits per byte, and an fp16
  `d` at bytes 26-27.
  - Element order: `qs` word w < 4 holds elements `16t + 4w`; words 4 and 5 hold `80 + 8t + 4(w-4)`; `qh` holds
    `120 + 2t + h`.
  - Digit walk: `w = v*3; d = w >> 8; v = w & 0xFF`.
  - Prefix identity: `H_t = floor(3^t * b / 256)`, `d_t = H_(t+1) - 3*H_t`. dot4 is linear in the unsigned operand,
    so you can dot the prefixes and difference them.
- **PQ2_0** is a 34-byte block: fp16 `d` first, then `qs[32]`. A 2-bit code c means value c - 1. Element j is in byte
  j/4 at bits 2*(j%4).
- **q8_1 "isum" variant** (`quantize_q8_1<isum, perm16>` in quantize.cu):
  - `ds.y` holds the exact int16 sum of the quants (bit-cast), so the -1 offset of the trits/codes folds in exactly.
  - `perm16` transposes each group of 16 values as 4x4, so the PQ2_0 mask trick `(w >> 2k) & 0x03030303` lines up.
- **AMD intrinsics:**
  - The dot instruction is `v_dot4_i32_iu8`, reached through `__builtin_amdgcn_sudot4(a_signed, a, b_signed, b, c, clamp)`.
  - `__builtin_amdgcn_perm(a, b, sel)`: selectors 0-3 pick bytes of the SECOND argument and 4-7 bytes of the first.
    This is the reverse of CUDA `__byte_perm`, and it is a classic bug source.

## What was changed (by file)

- `ggml/src/ggml-cuda/mmvq.cu/.cuh`: the new RDNA3/RDNA4 low-bit mat-vec (HIP only).
  - `mul_mat_vec_lowbit_rdna<type, ncols, lanes_per_row (8|16), has_fusion, has_gate>` with the block dots
    `ptq1_0_rdna_block` (prefix form for 1-2 columns, digit walk for 3-8) and `pq2_0_rdna_block`.
  - Public entry points: `ggml_cuda_mmvq_lowbit_rdna_supported`, `ggml_cuda_mmvq_lowbit_rdna_q8_size` and
    `ggml_cuda_mul_mat_vec_q_lowbit_rdna(ctx, src0, src1, dst, fusion, q8, quantize)`.
  - `ggml_cuda_mul_mat_vec_q` routes to it when `!ids` and supported. It reuses a shared q8 through
    `ctx.find_lowbit_q8`.
  - Constraints: K % 1024 == 0, no MUL_MAT_ID, no broadcast.
  - ISA: about 200 VALU ops per PTQ1_0 block (was about 340), about 100 for PQ2_0, 94 VGPRs, occupancy 16, no scratch.
- `quantize.cu/.cuh`: `quantize_q8_1<isum, perm16>`, the shared launcher `quantize_row_q8_1_launch`, and
  `quantize_row_q8_1_isum_cuda(...)`.
- `mmq.cu`, `mmq.cuh`, `mmq-load-tiles.cuh` and `mmq-config-{cdna,rdna3,rdna3-5,rdna4}.cuh`:
  - PTQ1_0 MMQ is enabled on HIP (supported when `turing_mma || amd_wmma || amd_mfma`), with a HIP branch in the
    tile loader.
  - The PTQ1_0 tile configs are copied from PQ2_0.
  - `GGML_CUDA_MMQ_MAX_J` caps the J in `mul_mat_q_switch_J`.
  - `template-instances/mmq-instance-ptq1_0.cu` and `generate_cu_files.py` lost their CUDA-only guard.
- `fwht.cu/.cuh`:
  - `fwht_cuda_block<N, NT, T, has_signs, q8, norm>` takes an optional RMS_NORM * w prologue and an optional q8_1
    output.
  - `fwht_gated_norm_block` computes `FWHT(signs * perm(silu(z) * rms(x) * w))` in one launch.
  - Shared helpers: `fwht_block_stages` and `fwht_block_store_q8`.
  - Host entry points: `ggml_cuda_op_fwht_fused` and `ggml_cuda_op_fwht_gated_norm`. HIP only, N in {512, 1024, 2048}.
- `ggml-cuda.cu` (graph loop):
  - `rdna_shared_q8_enabled` (RDNA3/4 and HIP, `GGML_HIP_RDNA_SHARED_Q8` != 0) and the `cuda_ctx->lowbit_q8`
    registry, cleared per evaluation.
  - Matchers `ggml_cuda_try_fwht_gated_norm` and `ggml_cuda_try_fwht_fused` (plain, norm, signed). They run before
    `ggml_cuda_try_fuse` and honor `GGML_CUDA_DISABLE_FUSION`.
  - The via-Q8 prefill path is in `ggml_cuda_mul_mat`.
  - GDN `supports_op`: rows mode and the F16 state.
  - `ggml_cuda_try_gdn_cache_fusion` accepts an F16 cache.
- `common.cuh`: `lowbit_q8_entry`, `lowbit_q8` and `find_lowbit_q8()` on the backend context.
- `convert.cu/.cuh`: `convert_ptq1_0_to_q8_0_cuda` (for via-Q8).
- `gated_delta_net.cu/.cuh`:
  - Rows mode (`state_rows`, `state_row_stride`).
  - `COLS` per warp (`GGML_HIP_GDN_COLS_PER_WARP`).
  - Typed F16 state in and out, for S_v=128, !KDA and !G_PRECOMPUTED only.
  - The fused cache write struct is `{data, slot_stride, type}`.
- `ggml/src/ggml.c`: `ggml_gated_delta_net_rows` accepts F16 states.
- `ggml/src/ggml-cpu/ops.cpp`: the CPU GDN rows mode reads F16 states.
- `src/models/qwen35.cpp`, `delta-net-base.cpp`, `models.h`:
  - Rows mode is used on ROCm for single-sequence ubatches (with MTP snapshots, or when the fused GDN op runs).
  - Metal rows mode is limited to an F32 cache.
- `src/llama-model.cpp`: `LLAMA_RS_STATE_F16=1` makes the recurrent state F16 for QWEN3NEXT, QWEN35 and QWEN35MOE.
- `src/llama-graph.cpp`: state zeroing uses `fill` for non-F32 states (F32 keeps the original `scale` by 0).
- Tests:
  - `tests/test-ptq1_0-cuda-dot.cpp`: host mirrors of the RDNA PTQ1_0 block (both variants), the HIP MMQ loader and
    PQ2_0. Exact integer checks plus exhaustive byte sweeps.
  - `tests/test-backend-ops.cpp`:
    - fused-gate PTQ1_0/PQ2_0 cases at K=1024/2048;
    - `MUL_MAT_SHARED_SRC1`;
    - `test_fwht_signed(type_w, n_mats, norm)` and `test_fwht_gated_norm`, both whole graph;
    - GDN at the Bonsai shape (16 k heads, 128 dim, v_repeat 3), rows mode with raw gates, and an F16 `type_state`;
    - PTQ1_0/PQ2_0 MMQ shapes (n = 9, 67, 130);
    - perf cases for the prefill shapes and the GDN.
- `scripts/hip/bench-bonsai-rdna.sh`: the whole measurement plan (below).

## Runtime knobs

All defaults are what was judged best without hardware. Every knob exists for A/B runs.

| env | default | effect |
|---|---|---|
| `GGML_HIP_RDNA_LOWBIT_MMVQ=0` | on | back to the generic MMVQ for PTQ1_0/PQ2_0 |
| `GGML_HIP_RDNA_SHARED_Q8=0` | on | no shared q8, no FWHT+q8, no norm/gated-norm FWHT fusions |
| `GGML_CUDA_DISABLE_FUSION=1` | off | disables all fusions, including the new FWHT ones |
| `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0` | 1<<30 | PTQ1_0 prefill through fp16 dequant + hipBLAS (old path) |
| `GGML_CUDA_MMQ_MAX_J=<8..128>` | 128 | cap on the MMQ tile width |
| `GGML_HIP_PTQ1_0_MMQ_VIA_Q8=<n>` | 0 (off) | PTQ1_0 mat-muls with >= n columns: expand to Q8_0, run Q8_0 tiles |
| `GGML_HIP_GDN_COLS_PER_WARP=1\|2\|4` | 1 | GDN state columns per warp (S_v=128, not KDA; forced to 1 for F16 state) |
| `GGML_GDN_STATE_GATHER=1` | off | old get_rows state gather instead of rows mode |
| `LLAMA_RS_STATE_F16=1` | off | F16 GDN state cache (half the state traffic; check perplexity) |
| `GGML_CUDA_FWHT_LEGACY` | (existing) | old FWHT path |

"Old behavior" for comparisons is
`GGML_HIP_RDNA_LOWBIT_MMVQ=0 GGML_HIP_RDNA_SHARED_Q8=0 GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0` (plus `GGML_GDN_STATE_GATHER=1`
for the GDN change).

## How to test on the card

```
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
SPEC=1 PROFILE=1 PPL_FILE=wiki.test.raw scripts/hip/bench-bonsai-rdna.sh build \
    Ternary-Bonsai-2-27B-Uncensored-Gaston-PTQ1_0-MTP.gguf \
    Ternary-Bonsai-2-27B-Uncensored-Gaston-PQ2_0-MTP.gguf
```

The script writes `bench-rdna-<date>/summary.md` with these sections:

1. test-backend-ops vs CPU for every knob;
2. kernel perf;
3. llama-bench, old vs new and per knob;
4. greedy output diff, old vs new;
5. perplexity: old, new and F16 state;
6. MTP / ngram speculative decoding;
7. `LONGCTX=1`: 8k depth with f16 or q8_0 KV;
8. `PROFILE=1`: rocprofv3 trace.

`QUICK=1` skips the per-knob sweep.

Quick manual checks:

```
build/bin/test-backend-ops -b ROCm0 -o MUL_MAT -p "ptq1_0|pq2_0"
build/bin/test-backend-ops -b ROCm0 -o MUL_MAT_HADAMARD
build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET
build/bin/test-backend-ops -b ROCm0 -o MUL_MAT_SHARED_SRC1
```

Host-only bit-trick test (no GPU needed; all pass):

```
g++ -O2 -std=c++17 tests/test-ptq1_0-cuda-dot.cpp -o /tmp/tdot && /tmp/tdot
```

## Dev environment used here (container without a GPU)

- **ROCm:** the TheRock nightly tarball, unpacked to `/opt/rocm-therock`:
  `https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx120X-all-7.14.0a20260612.tar.gz`
- **Build dir:** `/tmp/claude-0/build-hip`, configured with:
  - `-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm-therock -DLLAMA_CURL=OFF`
  - `-DCMAKE_BUILD_TYPE=Release`
  - `-DCMAKE_HIP_COMPILER=/opt/rocm-therock/llvm/bin/clang++`
- **Build:** `ninja -j4 ggml-hip llama test-backend-ops`. A full ggml-hip build takes a long time on 4 cores.
- **Link fails on executables:** `librocroller.so.1` was deleted from the tarball for disk space and is needed by
  `libhipblaslt`. Relink the executable by hand, with the flag before the libs:
  ```
  cd /tmp/claude-0/build-hip
  eval "$(ninja -t commands test-backend-ops | tail -1 | sed 's/&& :$//; s|-o bin/test-backend-ops|-o bin/test-backend-ops -Wl,--unresolved-symbols=ignore-in-shared-libs|')"
  ```
- **ISA inspection:** compile one .cu to assembly with `/opt/rocm-therock/llvm/bin/clang++` plus the flags from
  `compile_commands.json` / `ninja -t commands`, adding `-S --offload-arch=gfx1201 --cuda-device-only`. Then count
  VALU ops, and check `.vgpr_count` and `scratch` in the metadata.
- **Stub run of the bench script:** a stub model and binaries in `/tmp/claude-0/stub` were used to dry-run the bench
  script. That only proves the script's plumbing.

## Design decisions and things deliberately NOT done

- **No new PTQ1_0 GEMM for prefill.** The ISA shows MMQ PTQ1_0 at about 1580 instructions per K step vs about 1050
  for Q8_0, with the same 32 WMMA ops. The cost is the base-3 unpack in the tile loader, so a new GEMM would re-derive
  MMQ. The opt-in via-Q8 path is the experiment instead: it expands once and costs about 1 extra byte per weight.
  Measure it.
- **Residual ADD not folded into the norm+FWHT.** The ADD runs in place, and blocks writing their part of the sum
  would race with blocks still reading the row for the norm.
- **Rows mode on ROCm only for single-sequence ubatches.** Multi-sequence batches keep the gather, because of the
  relocation ordering hazard noted in `build_rs_cache_view`. CUDA keeps the gather (unchanged).
- **The F16 state is opt-in.** It rounds the state after every ubatch. The perplexity and greedy impact are unknown
  until measured.
- **RDNA2 PTQ1_0 MMQ configs dropped.** MMQ is not enabled for PTQ1_0 on RDNA2 (no WMMA), so it keeps hipBLAS.
- **Weight repacking (task 10) not done, on purpose.**
  - The PQ2_0 file already is the 2-bit repack: same trits, trivially unpacked.
  - An offline repack of PTQ1_0 only pays if a trace shows decode is ALU-bound rather than bandwidth-bound.
  - Compare PTQ1_0 vs PQ2_0 tok/s in the bench:
    - PQ2_0 at about 2 bits/weight faster than PTQ1_0 at 1.75 means PTQ1_0 is compute-bound, and a repack (or
      loading PTQ1_0 as PQ2_0) is worth doing.
    - Otherwise it is not.
- **Output head.** It is 248k x 5120 Q6_K, about 1 GB per token read. That is a large share of decode. A smaller head
  type is an offline, quality-affecting choice, documented in the bench script header:
  `llama-quantize --allow-requantize --output-tensor-type q5_k ...` (dry run first, then compare perplexity).
- **Speculative decoding (MTP / ngram)** already exists upstream. The bench only measures which settings win on the
  card.

## Suggested next steps (in order)

1. Run the bench script on the 9070 XT. Fix any test-backend-ops failure first.
   - The riskiest pieces are the perm selectors in the HIP MMQ loader and the gated-norm matcher.
   - Also risky: the F16 GDN state kernels and the shared q8 registry lifetime inside HIP graphs (the entries point
     into pool memory released at the end of the evaluation).
2. From `summary.md`, set the defaults: `GGML_HIP_GDN_COLS_PER_WARP`, `GGML_CUDA_MMQ_MAX_J` for RDNA4, the via-Q8
   threshold, and whether the F16 state is acceptable.
3. Read the rocprofv3 trace (`PROFILE=1`) for the remaining per-token launch count and the biggest kernels. Likely
   next targets:
   - the attention layers' small ops (a fusion like the GDN one);
   - the output head;
   - MUL_MAT_ID if an MoE variant matters.
4. Decide on the repack using the PTQ1_0 vs PQ2_0 numbers (see above).
5. If a GPU shows the RDNA kernel loses at some shape, `ggml_cuda_mmvq_lowbit_rdna_supported` and
   `mmvq_lowbit_rdna_lanes_per_row` are the places to restrict it.
