#!/usr/bin/env bash
set -euo pipefail

# This runtime has no MuJoCo sidecar. start_all.sh is kept as an alias of
# start_realtime.sh, for consistency with sim/.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_realtime.sh"
