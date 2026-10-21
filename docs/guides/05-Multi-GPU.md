# Multi-GPU

HartsyInference supports two ways to use more than one GPU:

- **Separate jobs:** one backend per GPU. Different generations run on different cards at the same time.
- **One job across cards:** one backend uses a second GPU for parts of the same generation.

## Separate jobs

1. Add a HartsyInference backend for each GPU.
2. Give each backend its own `GPU_ID`: `0`, then `1`, and so on on CUDA. On Vulkan, see the note below.
3. Leave the other GPU fields empty.

On CUDA, each number is a different card. SwarmUI sends each generation to a free backend. If you are not sure which card a number refers to, start one generation on each backend and watch `nvidia-smi` to see which card is busy. CUDA lists the fastest card first, so the order can differ from `nvidia-smi`.

On Vulkan, `0` means the engine picks the best GPU, and that card may be one another backend uses. When you run several backends on Vulkan, give each one a non-zero number, and check each backend's log to confirm which GPU it bound to.

## One job across two cards

Set these fields on a single backend. They use CUDA, so they need two NVIDIA GPUs that work with CUDA. On a backend that uses Vulkan, setting any of them stops the backend with an error.

| Field | Use it when |
| --- | --- |
| `TextEncoderGpuId` | Text encoders take a large share of VRAM, as with video models and FLUX. |
| `VaeGpuId` | Decoding large images or video needs room on the main card. |
| `DitShardGpuId` | The diffusion model is too large for one card. |
| `LmShardGpuId` | A large language or audio language model needs more memory. |
| `CfgParallelGpuId` | The model fits on both cards and you want faster steps. |

Rules:

- `DitShardGpuId` and `CfgParallelGpuId` cannot be set together. The backend fails to start if both are set.
- `DitShardGpuId` pools memory, so it can let a model run that does not fit on one card. It does not make each step faster.
- `CfgParallelGpuId` adds speed only when the model fits on both cards. If the second card cannot hold a full copy, the generation still succeeds but runs one pass at a time. The engine log shows `[CfgParallel]` with `active` or `fell-back` to tell you which path ran.

### Models verified with DitShardGpuId

- Krea 2
- Qwen-Image 20B
- MiniMax H3 (fp8 build)
- FLUX.1, for plain generations. ControlNet, Kontext, inpainting, and regional prompts run unsharded.

Other families may work through the same code path, but these are the ones verified end to end.

## Check that both cards are used

Start a generation and watch `nvidia-smi` on both cards. A sharded model shows memory on both GPUs. A single-card generation shows memory on one GPU only.

## More detail

The [two-GPU setup notes](../15-Two-GPU-Setups.md) include worked examples and their limits.
