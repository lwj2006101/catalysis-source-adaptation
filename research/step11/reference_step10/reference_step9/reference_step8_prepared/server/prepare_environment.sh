#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "$task_root/" in /home/HDD/upc/lwj/*/) ;; *) echo '解压到 /home/HDD/upc/lwj 内。' >&2; exit 2 ;; esac
[[ "$(id -un)" == 'upc' ]] || { echo '需要 upc 用户。' >&2; exit 2; }
command -v conda >/dev/null || { echo '请加载服务器已有 Conda；环境创建也应位于本项目。' >&2; exit 2; }
export CONDA_PKGS_DIRS="$task_root/runs/conda_packages"
export PIP_CACHE_DIR="$task_root/runs/pip_cache"
export TMPDIR="$task_root/runs/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
task_env="$task_root/runs/conda_env"
if [[ ! -x "$task_env/bin/python" ]]; then conda create -y -p "$task_env" python=3.12.14 pip; fi
conda run -p "$task_env" python -m pip install -r "$task_root/requirements.txt"
echo "环境准备完成：conda run --no-capture-output -p $task_env bash $task_root/server/run_history.sh"
