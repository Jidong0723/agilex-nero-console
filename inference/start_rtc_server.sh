#!/usr/bin/env bash
set -euo pipefail
NERO_PACKAGE=/root/autodl-tmp/nero_h10_9999_20261005
cd "$NERO_PACKAGE"
export NERO_RUN_ROOT="$NERO_PACKAGE"
export OPENPI_DATA_HOME="$NERO_PACKAGE/cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$NERO_PACKAGE:${PYTHONPATH:-}"
exec "$NERO_PACKAGE/openpi/.venv/bin/python" "$NERO_PACKAGE/rtc_20261006/serve_nero_rtc.py" \
  --checkpoint "$NERO_PACKAGE/checkpoint" --port 8000 \
  --audit-dir "$NERO_PACKAGE/rtc_20261006/audit" "$@"
