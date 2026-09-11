#!/usr/bin/env bash
# Compare FlyDSL policy spaces against hipBLASLt on the mapped Kimi-K3 shapes.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT_DIR="${OUT_DIR:-/host_out/flydsl_policy_dev}"
MP="${MP:-8}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-31}"
TIMEOUT="${TIMEOUT:-3600}"
FLYDSL_SPACE="${FLYDSL_SPACE:-extended}"
FLYDSL_HIP_MAP="${FLYDSL_HIP_MAP:-aiter/configs/bf16_hipblaslt_mapped_dev_kernels.csv}"

mkdir -p "${OUT_DIR}"

extra=()
if [[ "${FLYDSL_SPACE}" == "subspace" ]]; then
  extra+=(--flydsl-hip-map "${FLYDSL_HIP_MAP}")
fi

# Do not use --shape_grouped: one LLVM abort or 3600s timeout then
# discards the whole shape (thousands of FlyDSL kernels on one GPU).
python3 csrc/gemm_a16w16/gemm_tuner.py \
  --input_file aiter/configs/bf16_hipblaslt_mapped_dev.csv \
  --tuned_file "${OUT_DIR}/bf16_hipblaslt_mapped_dev_tuned.csv" \
  --profile_file "${OUT_DIR}/bf16_hipblaslt_mapped_dev_profile.csv" \
  --libtype flydsl,hipblaslt \
  --with-hipblaslt \
  --flydsl-space "${FLYDSL_SPACE}" \
  "${extra[@]}" \
  --mp "${MP}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --timeout "${TIMEOUT}" \
  --errRatio 0.005 \
  --all \
  --verbose \
  2>&1 | tee "${OUT_DIR}/bf16_hipblaslt_mapped_dev.log"
