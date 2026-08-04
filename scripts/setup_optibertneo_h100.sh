#!/usr/bin/env bash
set -euo pipefail

# Build a fresh, self-contained SM90 environment. Run this on a build node
# after loading the site's CUDA toolkit (CUDA 12.6 for the current checkout).
# The script never modifies or removes an existing conda environment.

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
complex_root=$(cd "$project_root/.." && pwd)
pytorch_repo=$complex_root/pytorch
environment_prefix=${OPTIBERT_ENV_PREFIX:-$complex_root/.venv/optibertneo-h100}
build_root=${OPTIBERT_BUILD_ROOT:-/tmp/optibertneo-h100-build}
max_jobs=${MAX_JOBS:-16}
conda_bin=${CONDA_EXE:-conda}

if [[ ! "$environment_prefix" = /* ]]; then
    echo "OPTIBERT_ENV_PREFIX must be an absolute path: $environment_prefix" >&2
    exit 2
fi
if [[ -e "$environment_prefix" ]]; then
    echo "Refusing to modify existing environment: $environment_prefix" >&2
    echo "Choose a new OPTIBERT_ENV_PREFIX." >&2
    exit 2
fi
if [[ ! -d "$pytorch_repo/.git" ]]; then
    echo "Custom PyTorch checkout not found: $pytorch_repo" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is unavailable; load the CUDA toolkit module first." >&2
    exit 1
fi
if ! command -v "$conda_bin" >/dev/null 2>&1; then
    echo "conda is unavailable: $conda_bin" >&2
    exit 1
fi
if [[ ! "$max_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer." >&2
    exit 2
fi

mkdir -p "$(dirname "$environment_prefix")" "$build_root/wheels"
"$conda_bin" create --prefix "$environment_prefix" python=3.11 -y
python_bin=$environment_prefix/bin/python
export PYTHONNOUSERSITE=1

"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install \
    "setuptools>=77,<82" \
    "cmake>=3.27" \
    ninja \
    packaging \
    pyyaml \
    requests \
    six \
    "typing-extensions>=4.15" \
    wheel \
    filelock \
    fsspec \
    jinja2 \
    "networkx>=2.5.1" \
    "numpy==1.26.4" \
    "optree>=0.13" \
    psutil \
    "sympy>=1.13.3"

torch_commit=$(git -C "$pytorch_repo" rev-parse HEAD)
torch_source=$build_root/pytorch-sm90
if [[ ! -e "$torch_source/.git" ]]; then
    git -C "$pytorch_repo" worktree add --detach "$torch_source" "$torch_commit"
    git -C "$torch_source" submodule update --init --recursive
fi
if [[ "$(git -C "$torch_source" rev-parse HEAD)" != "$torch_commit" ]]; then
    echo "$torch_source exists at a different PyTorch commit." >&2
    exit 1
fi

export MAX_JOBS=$max_jobs
export CMAKE_BUILD_PARALLEL_LEVEL=$max_jobs
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST=9.0

echo "Building custom PyTorch $torch_commit for CUDA arch $TORCH_CUDA_ARCH_LIST"
"$python_bin" -m pip wheel \
    "$torch_source" \
    --wheel-dir "$build_root/wheels" \
    --no-build-isolation \
    --no-deps \
    -v

torch_wheel=$(
    find "$build_root/wheels" -maxdepth 1 -type f -name 'torch-*.whl' \
        -printf '%T@ %p\n' |
        sort -nr |
        sed -n '1s/^[^ ]* //p'
)
if [[ -z "$torch_wheel" ]]; then
    echo "PyTorch build completed without producing a wheel." >&2
    exit 1
fi
"$python_bin" -m pip install "$torch_wheel"

triton_version=$(<"$torch_source/.ci/docker/triton_version.txt")
triton_commit=$(<"$torch_source/.ci/docker/ci_commit_pins/triton.txt")
triton_short_commit=${triton_commit:0:8}
"$python_bin" -m pip install --no-deps \
    --index-url https://download.pytorch.org/whl/nightly/ \
    "triton==${triton_version}+git${triton_short_commit}"

"$python_bin" -m pip install -r "$project_root/requirements-optibertneo-h100.txt"
"$python_bin" -m pip install -e "$complex_root" --no-deps
"$python_bin" -m pip install -e "$project_root" --no-deps

{
    echo "pytorch_commit=$torch_commit"
    echo "triton_version=$triton_version"
    echo "triton_commit=$triton_commit"
    echo "cuda_arch_list=$TORCH_CUDA_ARCH_LIST"
    echo "cuda_toolkit=$(nvcc --version | sed -n 's/^.*release \\([^,]*\\).*$/\\1/p')"
    echo "neobert_commit=$(git -C "$project_root" rev-parse HEAD)"
    echo "complex_attention_commit=$(git -C "$complex_root" rev-parse HEAD)"
    "$python_bin" -m pip freeze
} >"$environment_prefix/optibertneo-build-manifest.txt"

"$python_bin" -c 'import torch; arches = torch.cuda.get_arch_list(); assert any(arch in {"sm_90", "compute_90"} for arch in arches), f"SM90 missing from PyTorch build: {arches}"; assert torch.tensor([1.0]).numpy().tolist() == [1.0], "PyTorch was built without NumPy support"; print(f"PyTorch SM90/NumPy build checks passed: {arches}")'
"$python_bin" "$project_root/scripts/pretraining/preflight_optibertneo.py" \
    --skip-slurm

echo
echo "Environment ready: $environment_prefix"
echo "Use: export OPTIBERT_PYTHON=$python_bin"
echo "Run the GPU/H100 preflight inside an allocation before training."
