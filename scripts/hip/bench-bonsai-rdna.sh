#!/usr/bin/env bash
# Correctness and speed checks for the RDNA3/RDNA4 PTQ1_0 / PQ2_0 paths (Bonsai-2 27B on e.g. a RX 9070 XT).
#
# usage: scripts/hip/bench-bonsai-rdna.sh <build dir> <model.gguf> [more models ...]
#
# env:
#   OUT=<dir>            results directory (default: bench-rdna-<date>)
#   BACKEND=ROCm0        test-backend-ops backend name
#   QUICK=1              skip the per-knob llama-bench sweep, only old vs new
#   PPL_FILE=<file>      also run llama-perplexity (e.g. wikitext-2 test file) old vs new
#   SPEC=1               also run MTP speculative decoding (needs an -MTP model)
#   PROFILE=1            also record a rocprofv3 kernel trace of a short decode run
#
# Knobs (all default on / unchanged, set to compare):
#   GGML_HIP_RDNA_LOWBIT_MMVQ=0      old generic mat-vec for PTQ1_0 / PQ2_0
#   GGML_HIP_RDNA_SHARED_Q8=0        no shared activation quantization (incl. FWHT + quantize)
#   GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0 PTQ1_0 prefill through fp16 dequantize + hipBLAS
#   GGML_HIP_GDN_COLS_PER_WARP=1|2|4 gated delta net columns per warp (default 1)
#   GGML_CUDA_MMQ_MAX_J=<8..128>     cap the MMQ tile width (prefill tile shape)

set -u

if [ $# -lt 2 ]; then
    sed -n '2,22p' "$0"
    exit 1
fi

BUILD=$1
shift
MODELS=("$@")
BIN=$BUILD/bin
OUT=${OUT:-bench-rdna-$(date +%Y%m%d-%H%M%S)}
BACKEND=${BACKEND:-ROCm0}
mkdir -p "$OUT"
SUMMARY=$OUT/summary.md

OLD_ENV="GGML_HIP_RDNA_LOWBIT_MMVQ=0 GGML_HIP_RDNA_SHARED_Q8=0 GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0"
PROMPT="Write a C function that parses a comma separated list of integers and returns their sum. Explain each step."

log() { echo "$@" | tee -a "$SUMMARY"; }

run() { # run <name> <env> <cmd...>
    local name=$1 envs=$2
    shift 2
    echo "+ $envs $*" > "$OUT/$name.log"
    env $envs "$@" >> "$OUT/$name.log" 2>&1 < /dev/null
    local rc=$?
    echo "rc=$rc" >> "$OUT/$name.log"
    return $rc
}

log "# RDNA low-bit checks $(date)"
log ""
log "build: $BUILD, models: ${MODELS[*]}"
command -v rocminfo >/dev/null && log "gpu: $(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*')"
log ""

# 1. correctness against the CPU backend
log "## test-backend-ops (vs CPU)"
log ""
log "| test | knobs | result |"
log "|---|---|---|"
tbo() { # tbo <name> <env> <args...>
    local name=$1 envs=$2
    shift 2
    run "tbo-$name" "$envs" "$BIN/test-backend-ops" -b "$BACKEND" "$@"
    local res
    res=$(grep -E "tests passed|FAIL" "$OUT/tbo-$name.log" | tail -1)
    log "| $name | ${envs:-default} | ${res:-see tbo-$name.log} |"
}
tbo mul_mat_lowbit       ""                                   -o MUL_MAT -p "ptq1_0|pq2_0"
tbo mul_mat_lowbit_old   "GGML_HIP_RDNA_LOWBIT_MMVQ=0"        -o MUL_MAT -p "ptq1_0|pq2_0"
tbo mul_mat_vec_fusion   ""                                   -o MUL_MAT_VEC_FUSION
tbo mul_mat_shared_src1  ""                                   -o MUL_MAT_SHARED_SRC1
tbo mul_mat_hadamard     ""                                   -o MUL_MAT_HADAMARD
tbo mul_mat_hadamard_old "GGML_HIP_RDNA_SHARED_Q8=0"          -o MUL_MAT_HADAMARD
for c in 1 2 4; do
    tbo "gated_delta_net_cols$c" "GGML_HIP_GDN_COLS_PER_WARP=$c" -o GATED_DELTA_NET
done
for j in 64 128; do
    tbo "mul_mat_mmq_maxj$j" "GGML_CUDA_MMQ_MAX_J=$j" -o MUL_MAT -p "ptq1_0|pq2_0"
done
log ""

# 2. kernel timings
log "## kernel timings (test-backend-ops perf), see perf-*.log"
log ""
perf() { # perf <name> <env> <args...>
    local name=$1 envs=$2
    shift 2
    run "perf-$name" "$envs" "$BIN/test-backend-ops" perf -b "$BACKEND" "$@"
    log "### $name (${envs:-default})"
    log '```'
    grep -E "^ *(MUL_MAT|GATED_DELTA_NET)[A-Z_]*\(" "$OUT/perf-$name.log" | sed 's/^ *//' | tee -a "$SUMMARY" > /dev/null
    log '```'
}
perf mul_mat_new      ""                                     -o MUL_MAT -p "type_a=(ptq1_0|pq2_0)"
perf mul_mat_old      "$OLD_ENV"                             -o MUL_MAT -p "type_a=(ptq1_0|pq2_0)"
perf mul_mat_maxj64   "GGML_CUDA_MMQ_MAX_J=64"               -o MUL_MAT -p "type_a=(ptq1_0|pq2_0),.*n=(64|512)"
for c in 1 2 4; do
    perf "gdn_cols$c" "GGML_HIP_GDN_COLS_PER_WARP=$c"        -o GATED_DELTA_NET -p "head_count=16"
done
log ""

# 3. end to end
log "## llama-bench"
log ""
bench() { # bench <name> <env> <model>
    local name=$1 envs=$2 model=$3
    run "bench-$name" "$envs" "$BIN/llama-bench" -m "$model" -ngl 99 -fa 1 -p 512 -n 128 -r 3 -o md
    log "### $name (${envs:-default})"
    grep -E "^\|" "$OUT/bench-$name.log" | tee -a "$SUMMARY" > /dev/null
    log ""
}
for m in "${MODELS[@]}"; do
    base=$(basename "$m" .gguf)
    bench "$base-old" "$OLD_ENV" "$m"
    bench "$base-new" ""         "$m"
    if [ "${QUICK:-0}" != "1" ]; then
        bench "$base-no-lowbit-mmvq" "GGML_HIP_RDNA_LOWBIT_MMVQ=0"        "$m"
        bench "$base-no-shared-q8"   "GGML_HIP_RDNA_SHARED_Q8=0"          "$m"
        bench "$base-no-ptq1-mmq"    "GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0"   "$m"
        bench "$base-gdn-cols2"      "GGML_HIP_GDN_COLS_PER_WARP=2"       "$m"
        bench "$base-gdn-cols4"      "GGML_HIP_GDN_COLS_PER_WARP=4"       "$m"
        bench "$base-mmq-maxj64"     "GGML_CUDA_MMQ_MAX_J=64"             "$m"
    fi
done

# 4. greedy output, old vs new: identical up to float summation order, so a late divergence can be a tie
log "## greedy output (old vs new, 256 tokens, temp 0)"
log ""
for m in "${MODELS[@]}"; do
    base=$(basename "$m" .gguf)
    for v in old new; do
        envs=""
        [ $v = old ] && envs=$OLD_ENV
        # generated text goes to stdout, logs to stderr
        env $envs "$BIN/llama-completion" -m "$m" -ngl 99 -fa 1 -no-cnv --temp 0 --seed 1 -n 256 \
            --no-display-prompt -p "$PROMPT" > "$OUT/greedy-$base-$v.txt" 2> "$OUT/greedy-$base-$v.log" < /dev/null
    done
    if cmp -s "$OUT/greedy-$base-old.txt" "$OUT/greedy-$base-new.txt"; then
        log "- $base: identical"
    else
        log "- $base: DIFFERENT, first differing line:"
        log '```'
        diff "$OUT/greedy-$base-old.txt" "$OUT/greedy-$base-new.txt" | head -6 | tee -a "$SUMMARY" > /dev/null
        log '```'
    fi
done
log ""

# 5. perplexity
if [ -n "${PPL_FILE:-}" ]; then
    log "## perplexity ($PPL_FILE, ctx 512, 40 chunks)"
    log ""
    for m in "${MODELS[@]}"; do
        base=$(basename "$m" .gguf)
        for v in old new; do
            envs=""
            [ $v = old ] && envs=$OLD_ENV
            run "ppl-$base-$v" "$envs" "$BIN/llama-perplexity" -m "$m" -ngl 99 -fa 1 -f "$PPL_FILE" -c 512 --chunks 40
            log "- $base $v: $(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$OUT/ppl-$base-$v.log")"
        done
    done
    log ""
fi

# 6. MTP speculative decoding
if [ "${SPEC:-0}" = "1" ]; then
    log "## MTP speculative decoding (llama-speculative-simple, 256 tokens)"
    log ""
    for m in "${MODELS[@]}"; do
        base=$(basename "$m" .gguf)
        for v in old new; do
            envs=""
            [ $v = old ] && envs=$OLD_ENV
            for d in 0 3 4; do
                if [ $d = 0 ]; then
                    run "spec-$base-$v-d$d" "$envs" "$BIN/llama-completion" -m "$m" -ngl 99 -fa 1 -no-cnv --temp 0 -n 256 -p "$PROMPT"
                    log "- $base $v no draft: $(grep -o 'eval time.*tokens per second)' "$OUT/spec-$base-$v-d$d.log" | tail -1)"
                else
                    run "spec-$base-$v-d$d" "$envs" "$BIN/llama-speculative-simple" -m "$m" -ngl 99 -fa 1 --temp 0 -n 256 \
                        --spec-type draft-mtp --spec-draft-n-max $d -p "$PROMPT"
                    log "- $base $v draft $d: $(grep -o 'decoded.*t/s' "$OUT/spec-$base-$v-d$d.log" | tail -1), $(grep -o 'accept    = .*' "$OUT/spec-$base-$v-d$d.log" | tail -1)"
                fi
            done
        done
    done
    log ""
fi

# 7. kernel trace
if [ "${PROFILE:-0}" = "1" ] && command -v rocprofv3 >/dev/null; then
    log "## rocprofv3 kernel trace (decode, 64 tokens), see prof-*/"
    for m in "${MODELS[@]}"; do
        base=$(basename "$m" .gguf)
        rocprofv3 --kernel-trace --stats -d "$OUT/prof-$base" -o run -- \
            "$BIN/llama-bench" -m "$m" -ngl 99 -fa 1 -p 0 -n 64 -r 1 > "$OUT/prof-$base.log" 2>&1
        stats=$(find "$OUT/prof-$base" -name "*kernel_stats.csv" | head -1)
        if [ -n "$stats" ]; then
            log '```'
            head -25 "$stats" | tee -a "$SUMMARY" > /dev/null
            log '```'
        fi
    done
fi

echo
echo "done: $SUMMARY"
