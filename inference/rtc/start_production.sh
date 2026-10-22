#!/usr/bin/env bash
# The only supported cloud production launch configuration.
set -euo pipefail
if [[ $# != 0 ]]; then
  echo 'Production arguments are pinned; do not pass experimental overrides.' >&2
  exit 2
fi
rtc_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>/root/autodl-tmp/nero_rtc_production.lock
if ! flock -n 9; then
  echo 'RTC production is already starting/running; do not launch another copy.'
  exit 0
fi
if /root/openpi/.venv/bin/python - <<'PY'
import errno
import socket
import sys
with socket.socket() as probe:
    try:
        probe.bind(('127.0.0.1', 8002))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            sys.exit(10)
        raise
PY
then
  : # Free port; no TCP connection or fake WebSocket handshake was created.
else
  check_status=$?
  if [[ "$check_status" == 10 ]]; then
    echo 'Port 8002 is occupied. No new model was started; check console metadata.'
    exit 0
  fi
  echo 'Port check failed; refusing to load another model.' >&2
  exit "$check_status"
fi
cd /root/openpi
export PATH="$HOME/.local/bin:$PATH"
export BOTO_CONFIG="$HOME/.boto"
export PYTHONPATH="/root/openpi/src:/root/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --frozen --no-sync python -u "$rtc_directory/diagnostic_server.py" \
  --checkpoint /root/autodl-tmp/nero/checkpoints/pi05_nero_chunk7_full_h20/nero_h20_from11649_pro6000_b32/11649 \
  --host 127.0.0.1 --port 8002 \
  --steps 20 --normal-steps 10 --guidance-cap 1000 \
  --prefix-attention-schedule exp --prefix-mask-mode timestamp_frozen \
  --warmup-inference-delay 6 --warmup-execute-horizon 6 \
  --terminal-noise-time 0.001 --terminal-refinement-steps 10 \
  --compute-dtype bfloat16 --parameter-dtype bfloat16 --compact-prefix
