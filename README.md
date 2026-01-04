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
5. **Optimized for Grace-Blackwell unified memory architecture**

## Unified Memory Architecture

The DGX Spark's Grace-Blackwell architecture uses **unified memory** - a coherent memory fabric shared between CPU and GPU. This is fundamentally different from discrete GPUs and requires different optimization strategies.

**Key insight: Don't fight the fabric.** Forcing everything GPU-side (`--gpu-only`, `--cache-none`) actually hurts performance.

**Optimized flags (default in SparkyUI):**
```bash
--disable-pinned-memory   # Reduces overhead on unified fabric
--force-fp16              # Enables SageAttention optimization
--fp16-unet --fp16-vae --fp16-text-enc  # FP16 precision throughout
--dont-upcast-attention   # Keeps attention in FP16 for speed
```

**What NOT to use:**
- `--gpu-only` - fights the unified memory fabric, hurts performance
- `--cache-none` - disables natural caching, slows model loading
- `--disable-mmap` - prevents memory-mapped model loading

**CUDA environment variables** are also tuned for unified memory:
- `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` - prefer GPU allocation
- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` - let fabric manage memory
- `OMP_NUM_THREADS=20` - utilize all 20 ARM cores

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

## Host-Level GPU Optimizations (Optional)

For maximum performance, apply these optimizations on the **host** (not in Docker):

```bash
# Lock GPU clocks to maximum (3003 MHz) - prevents throttling
sudo nvidia-smi -lgc 3003,3003

# Enable core clock boost (GPU core > memory clock for compute)
sudo nvidia-smi boost-slider --vboost 1

# Enable persistence mode (reduces driver load latency)
sudo nvidia-smi -pm 1

# Verify settings
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,persistence_mode --format=csv
```

**Note:** GPU clock settings don't persist across reboots due to GB10 firmware behavior. Re-apply after each boot.

## SageAttention Notes

SageAttention PR #297 added sm_121 support but was merged then reverted due to stability issues. Our approach:

- Build SageAttention from main branch with `TORCH_CUDA_ARCH_LIST="12.1"`
- Disable Triton via `TORCHDYNAMO_DISABLE=1` (Triton doesn't support sm_121a)
- This gives working SageAttention without the unstable PR #297 changes

For full Triton support (more complex), see [HurbaLurba's DGX-SPARK-COMFYUI-DOCKER](https://github.com/HurbaLurba/DGX-SPARK-COMFYUI-DOCKER) which builds custom Triton from source.

## Future

When these land, SparkyUI can be simplified:
- [ ] PyTorch native sm_121 support → remove explicit `TORCH_CUDA_ARCH_LIST`
- [ ] Triton sm_121 support → remove `TORCHDYNAMO_DISABLE`
- [ ] SageAttention prebuilt ARM64 wheels → remove source build

## Credits

- Unified memory architecture insights from [HurbaLurba's DGX-SPARK-COMFYUI-DOCKER](https://github.com/HurbaLurba/DGX-SPARK-COMFYUI-DOCKER)
- SageAttention by [thu-ml](https://github.com/thu-ml/SageAttention)
- ComfyUI by [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## License

MIT
