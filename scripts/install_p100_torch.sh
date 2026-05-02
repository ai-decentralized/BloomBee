#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper for Pascal/P100 users. The generic installer now
# probes the local GPUs and driver, so this script simply preserves the old
# command name documented in earlier BloomBee setup notes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/install_compatible_torch.py" "$@"
