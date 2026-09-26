// Verifies vec_dot_ptq1_0_q8_1 numerically on the host by stubbing the CUDA intrinsics.
// nvcc is unavailable here, so this cannot prove the kernel compiles, but it does prove
// the dot-product math and the element mapping agree with the CPU codec on real data.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>

// ---- CUDA intrinsic stubs -------------------------------------------------
static inline int ggml_cuda_dp4a(int a, int b, int c) {          // __dp4a
    const int8_t* pa = (const int8_t*)&a; const int8_t* pb = (const int8_t*)&b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
}
struct half2_stub { float lo, hi; };
static inline float __low2float(half2_stub h) { return h.lo; }

#define QK8_1 32
struct block_q8_1 { half2_stub ds; int8_t qs[QK8_1]; };
static inline int get_int_b4(const int8_t* qs, int j) {           // 4 bytes as an int
    int v; memcpy(&v, qs + j*4, 4); return v;
}

#define QK_PTQ1_0 128
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; float d; };

// ---- transcribed from ggml-cuda/dequantize.cuh ---------------------------
static inline int ptq1_0_trit(const block_ptq1_0 * x, const int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];              n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }
    uint32_t v = b;
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1;
}

// ---- transcribed from ggml-cuda/vecdotq.cuh ------------------------------
static inline float vec_dot_ptq1_0_q8_1(const void* vbq, const block_q8_1* bq8_1,
                                        const int& kbx, const int& iqs) {
    const block_ptq1_0 * bq  = (const block_ptq1_0 *) vbq + kbx;
    const block_q8_1   * bq8 = bq8_1 + iqs;
    const int base = iqs * 32;
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        const int t0 = ptq1_0_trit(bq, base + j*4 + 0);
        const int t1 = ptq1_0_trit(bq, base + j*4 + 1);
        const int t2 = ptq1_0_trit(bq, base + j*4 + 2);
        const int t3 = ptq1_0_trit(bq, base + j*4 + 3);
        const int qx = (t0 & 0xFF) | ((t1 & 0xFF) << 8) | ((t2 & 0xFF) << 16) | ((t3 & 0xFF) << 24);
        const int u  = get_int_b4(bq8->qs, j);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
    }
    return (float) bq->d * __low2float(bq8->ds) * sumi;
}

// ---- HIP vector idiom (transcribed from vecdotq.cuh HIP branch) -------------
// Device-proven rule: each output byte uses the 3 low bits of its selector
// BYTE (not nibble); sel bit2=0 picks the SECOND arg (inverted vs CUDA).
static inline uint32_t hip_perm(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t sel = (s >> (8*i)) & 7;
        uint32_t byte = ((sel & 4) == 0) ? (b >> (8*(sel & 3))) : (a >> (8*(sel & 3)));
        r |= (byte & 0xFFu) << (8*i);
    }
    return r;
}

static inline float vec_dot_ptq1_0_q8_1_vec(const void* vbq, const block_q8_1* bq8_1,
                                            const int& kbx, const int& iqs, int * diff) {
    const block_ptq1_0 * bq = (const block_ptq1_0 *) vbq + kbx;
    int sumi[4] = { 0, 0, 0, 0 };
    int sumu[4] = { 0, 0, 0, 0 };
    const uint32_t * qs32 = (const uint32_t *) bq->qs;
    for (int w = 0; w < 4; ++w) {
        uint32_t v_lo = hip_perm(0, qs32[w], 0x0C010C00);
        uint32_t v_hi = hip_perm(0, qs32[w], 0x0C030C02);
        for (int t = 0; t < 5; ++t) {
            uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
            v_lo = w_lo & 0x00FF00FFu; v_hi = w_hi & 0x00FF00FFu;
            const int e = t * 16 + 4 * w;
            const int u = get_int_b4(bq8_1[iqs + (e >> 5)].qs, (e & 31) >> 2);
            const int q = (int) hip_perm(w_hi, w_lo, 0x07050301);
            sumi[e >> 5] += ggml_cuda_dp4a(q, u, 0);
            sumu[e >> 5] += ggml_cuda_dp4a(0x01010101, u, 0);
        }
    }
    for (int w = 0; w < 2; ++w) {
        uint32_t v_lo = hip_perm(0, qs32[4 + w], 0x0C010C00);
        uint32_t v_hi = hip_perm(0, qs32[4 + w], 0x0C030C02);
        for (int t = 0; t < 5; ++t) {
            uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
            v_lo = w_lo & 0x00FF00FFu; v_hi = w_hi & 0x00FF00FFu;
            const int e = 80 + t * 8 + 4 * w;
            const int u = get_int_b4(bq8_1[iqs + (e >> 5)].qs, (e & 31) >> 2);
            const int q = (int) hip_perm(w_hi, w_lo, 0x07050301);
            sumi[e >> 5] += ggml_cuda_dp4a(q, u, 0);
            sumu[e >> 5] += ggml_cuda_dp4a(0x01010101, u, 0);
        }
    }
    for (int h = 0; h < 2; ++h) {
        uint32_t v = bq->qh[h];
        for (int t = 0; t < 4; ++t) {
            const uint32_t w = v * 3;
            const int      q = (int) (w >> 8);
            v                = w & 0xFF;
            const int e      = 120 + t * 2 + h;
            const int a      = (int) bq8_1[iqs + (e >> 5)].qs[e & 31];
            sumi[e >> 5] += q * a;
            sumu[e >> 5] += a;
        }
    }
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
        if (diff) diff[k] = sumi[k] - sumu[k];
        acc += __low2float(bq8_1[iqs + k].ds) * (float) (sumi[k] - sumu[k]);
    }
    return (float) bq->d * acc;
}

// ---- RDNA3/RDNA4 block dot (transcribed from ptq1_0_rdna_block in mmvq.cu) ----
// Returns the per-chunk integer sums sum((trit - 1)*q); isum[c] stands in for the int16 sum that quantize_row_q8_1_isum_cuda stores in ds.y.
static inline int hip_udot(uint32_t u, int s, int c) {               // __builtin_amdgcn_sudot4(false, u, true, s, c, false)
    const int8_t * ps = (const int8_t *) &s;
    for (int i = 0; i < 4; ++i) c += (int) ((u >> (8*i)) & 0xFF) * ps[i];
    return c;
}
static inline uint32_t hip_umul24(uint32_t a, uint32_t b) { return (a & 0xFFFFFFu) * (b & 0xFFFFFFu); }
static inline uint32_t hip_hi8(uint32_t lo, uint32_t hi) { return hip_perm(hi, lo, 0x07050301); }

static void vec_dot_ptq1_0_rdna(const block_ptq1_0 * bq, const block_q8_1 * y, const bool prefix, int * out) {
    int q[7];
    memcpy(q, bq->qs, 24);
    q[6] = (int) ((uint32_t) bq->qh[0] | ((uint32_t) bq->qh[1] << 8));
    uint32_t h[32], hp[32];
    for (int w = 0; w < 6; ++w) {
        uint32_t lo = hip_perm(0, q[w], 0x0C010C00);
        uint32_t hi = hip_perm(0, q[w], 0x0C030C02);
        uint32_t m  = 1;
        for (int t = 0; t < 5; ++t) {
            const int i = w < 4 ? 4*t + w : 16 + 2*t + w;
            if (prefix) {
                hp[i] = t == 0 ? 0 : h[i - (w < 4 ? 4 : 2)];
                m *= 3;
                h[i] = hip_hi8(hip_umul24(lo, m), hip_umul24(hi, m));
            } else {
                const uint32_t wl = hip_umul24(lo, 3), wh = hip_umul24(hi, 3);
                h[i] = hip_hi8(wl, wh);
                lo = wl & 0x00FF00FF; hi = wh & 0x00FF00FF;
            }
        }
    }
    const uint32_t vh = hip_perm(0, q[6], 0x0C010C00);
    if (prefix) {
        const uint32_t p1 = hip_umul24(vh, 3), p2 = hip_umul24(vh, 9), p3 = hip_umul24(vh, 27), p4 = hip_umul24(vh, 81);
        h[30] = hip_hi8(p1, p2); h[31] = hip_hi8(p3, p4);
        hp[30] = hip_perm(p1, 0, 0x07050000); hp[31] = hip_hi8(p2, p3);
    } else {
        uint32_t v = vh;
        for (int i = 30; i < 32; ++i) {
            const uint32_t w0 = hip_umul24(v, 3); v = w0 & 0x00FF00FF;
            const uint32_t w1 = hip_umul24(v, 3); v = w1 & 0x00FF00FF;
            h[i] = hip_hi8(w0, w1);
        }
    }
    for (int c = 0; c < 4; ++c) {
        int s1 = 0, s3 = 0, isum = 0;
        for (int k = 0; k < 32; ++k) isum += y[c].qs[k];
        for (int k = 0; k < 8; ++k) {
            const int i = 8*c + k;
            const int a = get_int_b4(y[c].qs, k);
            s1 = hip_udot(h[i], a, s1);
            if (prefix && i >= 4 && i != 20 && i != 21) s3 = hip_udot(hp[i], a, s3);
        }
        out[c] = s1 - 3*s3 - isum;
    }
}

// ---- MMQ tile loader, HIP branch (transcribed from ggml_cuda_mmq_load_tiles_ptq1_0) ----
// Fills row[32] with the signed trits of one block in element order, 4 per int.
static void load_tile_ptq1_0_hip(const block_ptq1_0 * bxi, int * row) {
    int words[7];
    memcpy(words, bxi->qs, 24);
    words[6] = (int) ((uint32_t) bxi->qh[0] | ((uint32_t) bxi->qh[1] << 8));
    for (int lane = 0; lane < 8; ++lane) {
        const uint32_t packed = (uint32_t) words[lane < 7 ? lane : 6];
        uint32_t v_lo = hip_perm(0, packed, 0x04010400);
        uint32_t v_hi = hip_perm(0, packed, 0x04030402);
        const bool full_lane = lane < 6;
        v_hi = full_lane ? v_hi : v_lo;
        const int dst_base   = lane < 4 ? lane : 16 + lane;
        const int dst_stride = lane < 4 ? 4 : 2;
        int q[5];
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3, w_hi = v_hi * 3;
            v_lo = w_lo & 0x00FF00FF; v_hi = w_hi & 0x00FF00FF;
            q[t] = (int) ((hip_perm(w_hi, w_lo, 0x07050301) + 0x7F7F7F7Fu) ^ 0x80808080u);
            if (full_lane) row[dst_base + t * dst_stride] = q[t];
        }
        if (lane == 6) {
            row[30] = (int) hip_perm(q[1], q[0], 0x05040100);
            row[31] = (int) hip_perm(q[3], q[2], 0x05040100);
        }
    }
}

// ---- PQ2_0 RDNA3/RDNA4 block dot (transcribed from pq2_0_rdna_block in mmvq.cu) ----
struct block_pq2_0 { uint8_t qs[32]; };
// quantize_q8_1<isum, perm16>: value r of a 32-chunk is stored at this position
static inline int perm16_pos(int r) { return (r & 16) | ((r & 3) << 2) | ((r >> 2) & 3); }

static void vec_dot_pq2_0_rdna(const block_pq2_0 * x, const block_q8_1 * y_perm, int * out) {
    uint32_t qs[8];
    memcpy(qs, x->qs, sizeof(qs));
    uint32_t m[32];
    for (int w = 0; w < 8; ++w) for (int k = 0; k < 4; ++k) m[4*w + k] = (qs[w] >> (2*k)) & 0x03030303;
    for (int c = 0; c < 4; ++c) {
        int s1 = 0, isum = 0;
        for (int k = 0; k < 32; ++k) isum += y_perm[c].qs[k];
        for (int k = 0; k < 8; ++k) s1 = hip_udot(m[8*c + k], get_int_b4(y_perm[c].qs, k), s1);
        out[c] = s1 - isum;
    }
}

// Exact per-chunk integer check against dequantize_row_pq2_0 (00=-1, 01=0, 10=+1, 11=+2), all byte values at every position.
static long check_pq2_0_rdna(void) {
    unsigned seed = 7;
    auto rnd = [&]() { seed = seed*1103515245u + 12345u; return (seed >> 16) & 0xFFFF; };
    long bad = 0;
    for (int trial = 0; trial < 5000 + 32*256; ++trial) {
        block_pq2_0 x;
        for (int i = 0; i < 32; ++i) x.qs[i] = rnd() & 0xFF;
        if (trial >= 5000) { const int t = trial - 5000; x.qs[t / 256] = (uint8_t) (t % 256); }
        block_q8_1 y[4], yp[4];
        for (int c = 0; c < 4; ++c) {
            for (int i = 0; i < 32; ++i) y[c].qs[i] = (int8_t) ((int) (rnd() % 255) - 127);
            for (int i = 0; i < 32; ++i) yp[c].qs[perm16_pos(i)] = y[c].qs[i];
        }
        int got[4];
        vec_dot_pq2_0_rdna(&x, yp, got);
        for (int c = 0; c < 4; ++c) {
            int ref = 0;
            for (int i = 0; i < 32; ++i) {
                const int j = 32*c + i;
                ref += ((int) ((x.qs[j/4] >> (2*(j % 4))) & 3) - 1) * y[c].qs[i];
            }
            if (ref != got[c]) { if (++bad < 4) printf("  PQ2_0 RDNA MISMATCH trial %d chunk %d: ref %d got %d\n", trial, c, ref, got[c]); }
        }
    }
    return bad;
}

// ---- reference: dequantize the block, dequantize q8_1, dot in float ------
static void ref_dequant(const block_ptq1_0* x, float* out) {
    const uint8_t pow3[6]={1,3,9,27,81,243}; const size_t st[3]={32,16,8};
    int o=0; size_t j=0;
    for (size_t s=0;s<3;++s){ const size_t c=st[s];
        for(; j+c<=sizeof(x->qs); j+=c)
            for(size_t n=0;n<5;++n) for(size_t m=0;m<c;++m){
                uint8_t q=x->qs[j+m]*pow3[n]; out[o++]=(float)((int)(((uint16_t)q*3)>>8)-1)*x->d; }
    }
    for(size_t n=0;n<4;++n) for(size_t h=0;h<2;++h){
        uint8_t q=x->qh[h]*pow3[n]; out[o++]=(float)((int)(((uint16_t)q*3)>>8)-1)*x->d; }
}

int main(void) {
    unsigned seed=99;
    auto rnd=[&](){ seed=seed*1103515245u+12345u; return (seed>>16)&0xFFFF; };
    double worst_rel = 0.0; long checks = 0; long int_bad = 0, int_checks = 0; double worst_scaled = 0.0;

    for (int trial=0; trial<5000; ++trial) {
        block_ptq1_0 w;
        for (int i=0;i<24;++i) w.qs[i]=rnd()&0xFF;
        for (int i=0;i<2;++i)  w.qh[i]=rnd()&0xFF;
        w.d = 0.01f + (rnd()%1000)/50000.0f;

        block_q8_1 y[4];
        for (int b=0;b<4;++b) {
            y[b].ds.lo = 0.005f + (rnd()%1000)/80000.0f; y[b].ds.hi = 0.f;
            for (int i=0;i<32;++i) y[b].qs[i]=(int8_t)((int)(rnd()%255)-127);
        }

        float wf[QK_PTQ1_0]; ref_dequant(&w, wf);

        // EXACT test: the kernel's integer accumulator per chunk must equal the
        // reference integer sum of trit*q8. This isolates logic from float rounding.
        int ref_sumis[4];
        for (int c = 0; c < 4; ++c) {
            int ref_sumi = 0;
            for (int i = 0; i < 32; ++i) {
                const int trit = (int) llround((double) wf[c*32+i] / (double) w.d);
                ref_sumi += trit * (int) y[c].qs[i];
            }
            ref_sumis[c] = ref_sumi;
            // recompute the kernel's sumi by dividing its float result back out
            const float got = vec_dot_ptq1_0_q8_1(&w, y, 0, c);
            const int got_sumi = (int) llround((double) got / ((double) w.d * (double) y[c].ds.lo));
            if (ref_sumi != got_sumi) { ++int_bad; if (int_bad < 4)
                printf("  INT MISMATCH trial %d chunk %d: ref %d got %d\n", trial, c, ref_sumi, got_sumi); }
            ++int_checks;
        }

        // reference dot over all four chunks, in float
        double ref_total = 0.0;
        for (int c=0;c<4;++c)
            for (int i=0;i<32;++i)
                ref_total += (double)wf[c*32+i] * ((double)y[c].qs[i] * (double)y[c].ds.lo);

        // kernel dot, chunk by chunk as MMVQ calls it
        double got_total = 0.0;
        for (int c=0;c<4;++c) got_total += vec_dot_ptq1_0_q8_1(&w, y, 0, c);

        // scale by the sum of magnitudes so cancellation in ref_total cannot inflate it
        double mag = 0.0;
        for (int c=0;c<4;++c) for (int i=0;i<32;++i)
            mag += fabs((double)wf[c*32+i] * (double)y[c].qs[i] * (double)y[c].ds.lo);
        const double denom = fabs(ref_total) > 1e-9 ? fabs(ref_total) : 1.0;
        const double rel = fabs(got_total - ref_total) / denom;
        if (rel > worst_rel) worst_rel = rel;
        if (mag > 0) { const double sc = fabs(got_total - ref_total)/mag; if (sc > worst_scaled) worst_scaled = sc; }
        ++checks;

        // HIP vector idiom must agree with the scalar transcription. Integer
        // diffs are exact; float totals may differ by 1 ulp (association order).
        int vec_diff[4];
        const float got_vec = vec_dot_ptq1_0_q8_1_vec(&w, y, 0, 0, vec_diff);
        for (int c = 0; c < 4; ++c) {
            if (vec_diff[c] != ref_sumis[c]) { ++int_bad;
                if (int_bad < 8) printf("  VEC MISMATCH trial %d chunk %d: ref %d vec %d\n",
                    trial, c, ref_sumis[c], vec_diff[c]); }
        }
        for (int pf = 0; pf < 2; ++pf) {
            int rdna[4];
            vec_dot_ptq1_0_rdna(&w, y, pf != 0, rdna);
            for (int c = 0; c < 4; ++c) {
                if (rdna[c] != ref_sumis[c]) { ++int_bad;
                    if (int_bad < 8) printf("  RDNA MISMATCH trial %d prefix %d chunk %d: ref %d rdna %d\n",
                        trial, pf, c, ref_sumis[c], rdna[c]); }
            }
        }
        {
            int row[32];
            load_tile_ptq1_0_hip(&w, row);
            for (int e = 0; e < QK_PTQ1_0; ++e) {
                const int trit = (int) llround((double) wf[e] / (double) w.d);
                if (((const int8_t *) row)[e] != trit) { ++int_bad;
                    if (int_bad < 8) printf("  MMQ HIP LOADER MISMATCH trial %d elem %d: ref %d got %d\n",
                        trial, e, trit, ((const int8_t *) row)[e]); break; }
            }
        }
        if (fabs(got_vec - (float) got_total) > 1e-6f * fmaxf(1.0f, fabsf((float) got_total))) { ++int_bad;
            if (int_bad < 12) printf("  VEC FLOAT trial %d: scalar %.9g vec %.9g\n",
                trial, got_total, (double) got_vec); }
    }
    // Exhaustive sweep: every packed-byte value at every word position, qh too.
    // Exact per-chunk integer diffs (like the trial loop); any lane-packing
    // or selector-order regression breaks equality here without a GPU.
    long exh_checks = 0, exh_bad = 0;
    for (int pos = 0; pos < 8; ++pos) {
        for (int v = 0; v < 256; ++v) {
            block_ptq1_0 w;
            memset(&w, 0xA5, sizeof(w));
            if (pos < 6) ((uint32_t *) w.qs)[pos] = v * 0x01010101u;
            else         w.qh[pos - 6] = (uint8_t) v;
            w.d = 0.03f;
            block_q8_1 y[4];
            for (int b = 0; b < 4; ++b) {
                y[b].ds.lo = 0.02f; y[b].ds.hi = 0.f;
                for (int i = 0; i < 32; ++i) y[b].qs[i] = (int8_t) ((i * 7 + b * 13 + v) % 251 - 125);
            }
            float wf[QK_PTQ1_0]; ref_dequant(&w, wf);
            int vec_diff[4], rdna_seq[4], rdna_pfx[4];
            vec_dot_ptq1_0_q8_1_vec(&w, y, 0, 0, vec_diff);
            vec_dot_ptq1_0_rdna(&w, y, false, rdna_seq);
            vec_dot_ptq1_0_rdna(&w, y, true,  rdna_pfx);
            for (int c = 0; c < 4; ++c) {
                int ref_sumi = 0;
                for (int i = 0; i < 32; ++i) {
                    ref_sumi += (int) llround((double) wf[c*32+i] / (double) w.d) * (int) y[c].qs[i];
                }
                int tile_sumi = 0;
                {
                    int row[32];
                    load_tile_ptq1_0_hip(&w, row);
                    for (int i = 0; i < 32; ++i) tile_sumi += ((const int8_t *) row)[c*32 + i] * (int) y[c].qs[i];
                }
                if (vec_diff[c] != ref_sumi || rdna_seq[c] != ref_sumi || rdna_pfx[c] != ref_sumi || tile_sumi != ref_sumi) { ++exh_bad;
                    if (exh_bad < 8) printf("  EXH MISMATCH pos %d val %d chunk %d: ref %d vec %d rdna %d/%d\n",
                        pos, v, c, ref_sumi, vec_diff[c], rdna_seq[c], rdna_pfx[c]); }
                ++exh_checks;
            }
        }
    }
    const long pq2_bad = check_pq2_0_rdna();
    printf("  PQ2_0 RDNA mismatches : %ld\n", pq2_bad);
    exh_bad += pq2_bad;
    printf("  dot products compared : %ld (4 chunks each)\n", checks);
    printf("  worst relative error  : %.3e\n", worst_rel);
    printf("  worst err / sum|terms| : %.3e   (immune to cancellation)\n", worst_scaled);
    printf("  exact integer checks  : %ld, mismatches %ld\n", int_checks, int_bad);
    printf("  exhaustive vec==scalar: %ld, mismatches %ld\n", exh_checks, exh_bad);
    if (int_bad == 0) printf("  LOGIC EXACT: integer accumulator matches reference on every chunk\n");
    else              printf("  LOGIC BUG in the kernel\n");
    return (int_bad != 0) || (exh_bad != 0);
}
