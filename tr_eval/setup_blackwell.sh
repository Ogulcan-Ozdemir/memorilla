#!/usr/bin/env bash
# Set up a virtual environment that runs on Blackwell GPUs (sm_120, e.g. Colab's RTX PRO 6000) with the pinned stack.
#
# The pinned torch 2.7.0 from PyPI is the CUDA 12.6 build, and the vLLM 0.9.1 wheel has no sm_120 attention kernels, so
# both fail with "no kernel image is available". This script installs torch/torchvision/torchaudio 2.7.0 from the
# CUDA 12.8 index and vLLM 0.9.1 compiled for sm_120. The vLLM wheel is taken from $WHEEL_DIR when it is there;
# otherwise it is built from source (about 16 minutes on 48 cores, with the CUDA 12.8 toolkit installed next to the
# system one) and saved to $WHEEL_DIR for the next session.
#
# Usage (from the repository root): bash tr_eval/setup_blackwell.sh [venv] [wheel_dir]
# Defaults: /content/venv128 and $DRIVE/wheels. Then run every script with <venv>/bin/python.
# Do not run `uv pip install -e .` into the venv afterwards: it would replace the sm_120 vLLM with the PyPI wheel.
set -euo pipefail

VENV=${1:-/content/venv128}
WHEEL_DIR=${2:-${DRIVE:?set DRIVE or pass a wheel directory}/wheels}
PY=$VENV/bin/python
CUDA128=/usr/local/cuda-12.8
VLLM_VERSION=0.9.1
TORCH_INDEX=https://download.pytorch.org/whl/cu128

[ -x "$PY" ] || uv venv -q "$VENV" --python 3.11
uv pip install --python "$PY" -q -e ".[tr]"
uv pip install --python "$PY" -q --reinstall-package torch --reinstall-package torchvision \
    --reinstall-package torchaudio "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128" \
    --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

wheel=$(ls "$WHEEL_DIR"/vllm-*.whl 2>/dev/null | head -n 1 || true)  # any vLLM wheel there is an sm_120 build
if [ -z "$wheel" ]; then
    if [ ! -x "$CUDA128/bin/nvcc" ]; then
        . /etc/os-release
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb" \
            -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -q
        DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends cuda-minimal-build-12-8 \
            cuda-libraries-dev-12-8 cuda-nvtx-12-8 cuda-nvrtc-dev-12-8 cuda-profiler-api-12-8
    fi
    src=$(mktemp -d)/vllm
    git clone -q --depth 1 -b "v$VLLM_VERSION" https://github.com/vllm-project/vllm "$src"
    (
        cd "$src"
        "$PY" use_existing_torch.py
        uv pip install --python "$PY" -q pip -r requirements/build.txt
        export CUDA_HOME=$CUDA128 PATH=$CUDA128/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0" VLLM_TARGET_DEVICE=cuda
        export MAX_JOBS=${MAX_JOBS:-32} NVCC_THREADS=1 SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION+sm120"
        "$PY" -m pip wheel --no-build-isolation --no-deps . -w "$src/dist"
    )
    mkdir -p "$WHEEL_DIR"
    wheel=$(ls "$src"/dist/vllm-*.whl | head -n 1)
    cp "$wheel" "$WHEEL_DIR/"
fi
uv pip install --python "$PY" -q --no-deps --reinstall "$wheel"
"$PY" -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__, '|', torch.cuda.get_device_name())"
