# SparkyUI

**ComfyUI + SageAttention for NVIDIA DGX Spark (Blackwell GB10)**

A Docker-based ComfyUI setup specifically engineered for the DGX Spark's unique ARM64 + Blackwell architecture.

## Why This Exists

The NVIDIA DGX Spark uses the **GB10 GPU** with compute capability **12.1 (sm_121)** - Blackwell architecture. This creates challenges:

| CUDA Version | Max Compute Capability | Can compile for GB10? |
|--------------|------------------------|----------------------|
| CUDA 12.8 | sm_120 | **No** |
| CUDA 13.0+ | sm_121 | **Yes** |

Standard ComfyUI containers and PyTorch wheels don't support sm_121. SparkyUI solves this by:

1. Using **CUDA 13.0.2** base image (supports sm_121)
2. Installing **PyTorch cu130** ARM64 wheels
3. Compiling **SageAttention** with `TORCH_CUDA_ARCH_LIST="12.1"`
4. Disabling **Triton/torch.compile** (doesn't support sm_121 yet)

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/SparkyUI.git
cd SparkyUI

# Configure paths
cp .env.example .env
# Edit .env with your paths

# Build (compiles SageAttention for sm_121 - takes ~10 min)
docker compose build

# Start
docker compose up -d

# View logs
docker compose logs -f
```

**Access:** http://localhost:8188 (or your DGX Spark's IP on LAN)

## Requirements

- **NVIDIA DGX Spark** (or other GB10-based system)
- **Docker** with NVIDIA Container Toolkit
- **NVIDIA Driver** 560+ (tested with 580.95)
- **~15GB** disk for Docker image
- **Models** from existing ComfyUI install (mounted read-only)

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
# Path to your existing ComfyUI models (mounted read-only)
COMFYUI_HOST_PATH=/path/to/your/ComfyUI

# Path for SparkyUI data (custom_nodes, outputs, inputs)
SPARKYUI_DATA_PATH=/path/to/SparkyUI

# Optional: pin to specific versions
COMFYUI_REF=master
SAGEATTN_REF=main
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DGX Spark Host                          │
│  Ubuntu 24.04 (DGX OS 7) / Driver 580.x                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Docker Container (sparkyui:cu130)         │   │
│  │                                                      │   │
│  │  CUDA 13.0.2 + PyTorch 2.9.1+cu130                  │   │
│  │  SageAttention 2.2.0 (compiled for sm_121)          │   │
│  │  ComfyUI 0.7.x + ComfyUI-Manager                    │   │
│  │                                                      │   │
│  │  Key env vars:                                       │   │
│  │    TORCH_CUDA_ARCH_LIST="12.1"                      │   │
│  │    TORCHDYNAMO_DISABLE="1"                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                    Port 8188 (LAN)                          │
└─────────────────────────────────────────────────────────────┘
```

## Version Compatibility

Tested combinations:

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA Base | 13.0.2 | Required for sm_121 |
| PyTorch | 2.9.1+cu130 | ARM64 wheel from PyTorch index |
| torchvision | 0.24.1+cu130 | ARM64 wheel |
| SageAttention | 2.2.0 | Compiled with sm_121 |
| ComfyUI | 0.7.0 | master branch |
| Driver | 580.95 | DGX OS 7 default |

## Known Limitations

1. **PyTorch Warning**: You'll see a warning about compute capability 12.1 being "outside supported range (8.0-12.0)". This is harmless - PyTorch works, and SageAttention's custom kernels are compiled natively.

2. **torch.compile Disabled**: Triton doesn't support sm_121 yet. `torch.compile()` is disabled via environment variables. Some nodes may run slower than on supported architectures.

3. **No GitHub Actions CI**: Can't build for ARM64 + sm_121 in GitHub's hosted runners. Must build locally on DGX Spark.

## Troubleshooting

### "no kernel image is available for execution on the device"
Your SageAttention wasn't compiled for sm_121. Rebuild:
```bash
docker compose build --no-cache
```

### PyTorch can't find CUDA
Ensure NVIDIA Container Toolkit is installed:
```bash
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

### ComfyUI-Manager missing
The entrypoint auto-clones it. Check logs:
```bash
docker compose logs | grep -i manager
```

## Future

When these land, SparkyUI can be simplified:
- [ ] PyTorch native sm_121 support → remove explicit `TORCH_CUDA_ARCH_LIST`
- [ ] Triton sm_121 support → remove `TORCHDYNAMO_DISABLE`
- [ ] SageAttention prebuilt ARM64 wheels → remove source build

## License

MIT
