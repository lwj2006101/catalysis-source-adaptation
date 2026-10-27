#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python -c 'import sys; sys.path.insert(0,sys.argv[1]); from server_guard import require_server; require_server()' "$task_root/code"
export TMPDIR="$task_root/runs/tmp"
export MPLCONFIGDIR="$task_root/runs/matplotlib_cache"
export PIP_CACHE_DIR="$task_root/runs/pip_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$task_root/runs/logs"
cd "$task_root"
python code/verify_math.py 2>&1 | tee runs/logs/verify_math.log
python code/run_history.py --run-name smoke_v1 --seeds 42 --folds 0 --workers 1 2>&1 | tee runs/logs/smoke_v1.log
python code/audit_results.py --run-name smoke_v1 2>&1 | tee runs/logs/smoke_audit.log
python code/run_history.py --run-name history_v1 --workers "${STEP8_WORKERS:-2}" 2>&1 | tee runs/logs/history_v1.log
python code/audit_results.py --run-name history_v1 2>&1 | tee runs/logs/history_audit.log
python code/aggregate_results.py --run-name history_v1 2>&1 | tee runs/logs/history_summary.log
python code/freeze_full_history.py 2>&1 | tee runs/logs/full_history_freeze.log
echo '历史开发与最终冻结已运行；新的真实数据验证仍须完成数据来源核查。'
