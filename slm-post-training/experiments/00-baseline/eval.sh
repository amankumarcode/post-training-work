#!/usr/bin/env bash
# Evaluate the untouched base model.
# Extra arguments pass through, e.g. `bash experiments/00-baseline/eval.sh --limit 10` for a smoke test.
set -euo pipefail

cd "$(dirname "$0")/../.."
python -m slmpt.eval.run_eval --config experiments/00-baseline/config.yaml "$@"
