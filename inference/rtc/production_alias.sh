#!/usr/bin/env bash
# Cloud compatibility alias: old launcher names cannot select old profiles.
set -euo pipefail
rtc_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$rtc_directory/start_production.sh" "$@"
