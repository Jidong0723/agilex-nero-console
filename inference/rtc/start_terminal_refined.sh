#!/usr/bin/env bash
# Opt-in numerical integrator, not upstream's uniform Euler grid.
# Explicit comparison entrypoint: legacy ones profile uses D=6/E=6.
# Production chooses exp and its validated timing in start_production.sh.
# Ordinary inference remains on 10 uniform flow steps.
set -euo pipefail
rtc_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RTC_NUM_STEPS="${RTC_NUM_STEPS:-10}" RTC_NORMAL_STEPS=10 RTC_GUIDANCE_CAP=1000
export RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-ones}" RTC_TERMINAL_NOISE_TIME=0.001
export RTC_COMPACT_PREFIX=1 RTC_COMPUTE_DTYPE=bfloat16 RTC_PARAMETER_DTYPE=bfloat16
# The legacy validation matrix intentionally sweeps D/E outside this profile.
# Profile-specific real-sample/SSH checks are saved in the 20261021 report;
# start.sh still warms up all prompt buckets before publishing readiness.
export RTC_VALIDATE="${RTC_VALIDATE:-0}"
exec bash "$rtc_directory/start.sh"
