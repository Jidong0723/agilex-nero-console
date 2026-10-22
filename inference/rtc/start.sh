#!/usr/bin/env bash
set -euo pipefail
rtc_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd /root/openpi
export PATH="$HOME/.local/bin:$PATH"
export BOTO_CONFIG="$HOME/.boto"
export PYTHONPATH="/root/openpi/src:/root/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
checkpoint='/root/autodl-tmp/nero/checkpoints/pi05_nero_chunk7_full_h20/nero_h20_from11649_pro6000_b32/11649'
rtc_args=(--checkpoint "$checkpoint" --port "${RTC_PORT:-8002}"
  --steps "${RTC_NUM_STEPS:-10}" --normal-steps "${RTC_NORMAL_STEPS:-10}"
  --guidance-cap "${RTC_GUIDANCE_CAP:-5}"
  --prefix-attention-schedule "${RTC_PREFIX_ATTENTION_SCHEDULE:-exp}"
  --prefix-mask-mode "${RTC_PREFIX_MASK_MODE:-timestamp_frozen}"
  --warmup-inference-delay "${RTC_WARMUP_INFERENCE_DELAY:-6}"
  --warmup-execute-horizon "${RTC_WARMUP_EXECUTE_HORIZON:-6}"
  --terminal-noise-time "${RTC_TERMINAL_NOISE_TIME:-0}"
  --terminal-refinement-steps "${RTC_TERMINAL_REFINEMENT_STEPS:-1}"
  --compute-dtype "${RTC_COMPUTE_DTYPE:-bfloat16}" --parameter-dtype "${RTC_PARAMETER_DTYPE:-bfloat16}")
if [[ "${RTC_COMPACT_PREFIX:-0}" == "1" ]]; then rtc_args+=(--compact-prefix); fi
if [[ "${RTC_VALIDATE:-1}" == "1" ]]; then rtc_args+=(--validate); fi
exec uv run --frozen --no-sync python -u "$rtc_directory/diagnostic_server.py" "${rtc_args[@]}"
