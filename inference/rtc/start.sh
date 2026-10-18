#!/usr/bin/env bash
set -euo pipefail
cd /root/openpi
export PATH="$HOME/.local/bin:$PATH"
export BOTO_CONFIG="$HOME/.boto"
export PYTHONPATH="/root/openpi/src:/root/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
rtc_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint='/root/autodl-tmp/nero/checkpoints/pi05_nero_chunk7_full_h20/nero_h20_from11649_pro6000_b32/11649'
exec uv run --frozen --no-sync python "$rtc_directory/serve.py" --checkpoint "$checkpoint" --port "${RTC_PORT:-8002}" --validate
