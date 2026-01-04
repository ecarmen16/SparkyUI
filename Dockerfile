# CUDA 13.0 for Blackwell GB10 (sm_121 / compute_121)
# CUDA 12.8 only supports up to sm_120, but GB10 is sm_121.
# "devel" includes nvcc so we can compile CUDA extensions like SageAttention.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REF=master
ARG SAGEATTN_REF=main

# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    build-essential ninja-build cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create venv (keeps python deps isolated inside container)
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Upgrade packaging tools
RUN pip install -U pip setuptools wheel

# ---- PyTorch (ARM64 + CUDA 13.0) ----
# PyTorch cu130 wheels work with CUDA 13.0.x runtime.
# ARM64 wheels available: torch-2.9.1+cu130, torchvision-0.24.1
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision

# ---- ComfyUI ----
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    cd /opt/ComfyUI && \
    git checkout ${COMFYUI_REF} || true

RUN pip install -r /opt/ComfyUI/requirements.txt

# ---- ComfyUI-Manager ----
# Handled at runtime by entrypoint.sh (clones if missing in mounted volume)
# This ensures latest version on each container start

# ---- SageAttention ----
# GB10 is compute capability 12.1 (sm_121).
# CUDA 13.0 NVCC supports sm_121, so we compile directly for it.
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV CUDA_HOME=/usr/local/cuda

# Build/install SageAttention from repo with sm_121 support
RUN pip install --no-build-isolation "git+https://github.com/thu-ml/SageAttention@${SAGEATTN_REF}" || true

# Expose ComfyUI
EXPOSE 8188

# Entry script handles runtime updates / flags
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
