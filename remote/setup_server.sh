#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
mkdir -p remote/logs
exec > >(tee -a remote/logs/setup.log) 2>&1
trap 'echo "Setup failed at line $LINENO. See remote/logs/setup.log."' ERR

echo "Checking uploaded file contents..."
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('remote/bundle-manifest.json').read_text())
for name, expected in manifest.items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Upload hash mismatch: {name}')
print(f'Verified {len(manifest)} files.')
PY

echo "Server resources:"
hostname
uname -m
python3 --version
nproc
free -h
df -h "$project_dir"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "nvidia-smi is unavailable. This project's solvers use CPU."
fi
command -v tmux >/dev/null

# Keep the Python runtime and Graphviz under the account's home directory.
# Micromamba supplies prebuilt binaries; the actual Python packages live in a venv.
tools_dir="$HOME/.local/share/minimum-edit-dfa"
runtime_dir="$tools_dir/runtime-python311"
mkdir -p "$tools_dir"
if [[ ! -x "$runtime_dir/bin/python3.11" || ! -x "$runtime_dir/bin/dot" ]]; then
    case "$(uname -m)" in
        x86_64) platform=linux-64 ;;
        aarch64) platform=linux-aarch64 ;;
        *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
    esac
    if [[ ! -x "$tools_dir/bin/micromamba" ]]; then
        python3 - "$platform" "$tools_dir/micromamba.tar.bz2" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(
    f'https://micro.mamba.pm/api/micromamba/{sys.argv[1]}/latest', sys.argv[2])
PY
        tar -xjf "$tools_dir/micromamba.tar.bz2" -C "$tools_dir" bin/micromamba
    fi
    "$tools_dir/bin/micromamba" create --yes --no-rc \
        --root-prefix "$tools_dir/mamba" --prefix "$runtime_dir" \
        --override-channels --channel conda-forge --strict-channel-priority \
        'python=3.11' graphviz
fi

release=$(basename "$project_dir")
venv_dir="$HOME/venvs/minimum-edit-dfa-$release"
if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$runtime_dir/bin/python3.11" -m venv "$venv_dir"
fi
export PATH="$venv_dir/bin:$runtime_dir/bin:$PATH"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), sys.version
print('Research Python:', sys.version)
PY
dot -V
# Check PNG rendering before installing or starting any long experiment.
printf 'digraph { a -> b; }\n' | dot -Tpng -o remote/logs/graphviz-check.png
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: --require-hashes -r remote/requirements-lock.txt
python -m pip install 'poetry-core==2.4.0'
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -m unittest discover -s tests -p test_dot_io.py -v
python -m pip freeze > remote/logs/installed-packages.txt
"$tools_dir/bin/micromamba" list --prefix "$runtime_dir" --explicit \
    > remote/logs/runtime-packages.txt

# Bash-escaped paths allow reconnecting without manual activation.
printf 'export PATH=%q:%q:"$PATH"\nexport MPLBACKEND=Agg\nexport PYTHONUNBUFFERED=1\nexport CUDA_VISIBLE_DEVICES=""\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n' \
    "$venv_dir/bin" "$runtime_dir/bin" > remote/environment.sh
echo "Environment ready: $venv_dir"
echo "Run a short example with: bash remote/run_server.sh smoke"
