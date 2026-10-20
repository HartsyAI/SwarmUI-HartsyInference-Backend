# Backend Settings

Every setting on the HartsyInference backend card, with its default and what it does. Most people only change `GPU_ID` and `LowVram`, and only when they have a reason to.

## Core settings

| Setting | Default | What it does |
| --- | --- | --- |
| `ComputeBackend` | `auto` | Picks the compute path. `auto` tries CUDA first, then Vulkan, then CPU. Only options this machine can use are listed. |
| `GPU_ID` | `0` | The GPU this backend uses. Ignored on the CPU backend. |
| `LowVram` | `Auto (recommended)` | How much the engine streams weights to save VRAM. See the table below. |
| `OverQueue` | `1` | How many extra requests can wait behind the active generation. |
| `Previews` | on | Shows progress previews while a generation runs. Turn it off to skip preview encoding. |
| `AutoUpdate` | `false` | Controls engine updates. See below. |

### ComputeBackend

- **CUDA** appears only when an NVIDIA GPU is present and a test computation on it succeeds.
- **Vulkan** appears when a real Vulkan GPU is present, such as an AMD or Intel card. Software renderers do not count as a GPU.
- The list is built when SwarmUI starts. After you install a driver or plug in a GPU, restart SwarmUI.

### GPU_ID

Use a single number, such as `0` or `1`.

- Under CUDA, the number is CUDA's device order, which can differ from `nvidia-smi`. CUDA lists the fastest card first. To confirm which card a backend uses, watch `nvidia-smi` memory while a generation runs.
- Under Vulkan, `0` means the engine picks the best GPU. Other numbers use the Vulkan loader's raw device index. The backend's startup log names the device it bound to.

To use several GPUs for separate jobs, add one backend per GPU with a different `GPU_ID`. See the [multi-GPU guide](05-Multi-GPU.md).

### LowVram

| Option | What it does |
| --- | --- |
| `Auto (recommended)` | Measures free VRAM before each phase and streams only when the model would not otherwise fit. Cards with room keep the full-speed path. |
| `Performance (never stream, fail instead)` | Never streams and never evicts. A model that does not fit fails with an out-of-memory error. |
| `Balanced (unload between phases)` | Releases each phase's weights when the phase ends, so the next phase has room. Streams only when measurements show it must. |
| `Aggressive (always stream, half-precision caches)` | Always streams, halves the precision of cross-step caches, and uses smaller chunks. Useful when another program shares the GPU. |
| `Maximum (every lever, may spill to a 2nd GPU)` | Turns on every memory lever, including offloading activations, quantized compute, and freeing after every generation. It can spill onto a second GPU when one is configured. Slowest, and some levers change numerics. |

Streaming is usually several times slower than running a model fully on the card. It is what lets large models run on smaller cards at all.

You can override these levers for a single generation under the **VRAM & Memory** parameter group.

### OverQueue

- `-1`: the backend only shows status and cannot generate.
- `0`: one generation at a time, with nothing waiting. Extra requests go to other backends right away.
- `1` (default): one generation runs and one more request waits.

Generations always run one at a time on a single backend. The queue only controls how many requests wait.

### AutoUpdate

- `false` (default): never checks for a newer engine.
- `true`: on start, checks NuGet for a newer engine. If one exists, it downloads it and rebuilds the extension.
- `aggressive`: the same as `true`, but it also clears the NuGet caches first and restarts SwarmUI to load the new build.

## Multi-GPU settings

These fields move parts of one generation to a second GPU. Leave them empty for a single-GPU setup.

| Setting | What it moves | Notes |
| --- | --- | --- |
| `TextEncoderGpuId` | Text encoders such as CLIP and T5. | Outputs can differ slightly on a different GPU architecture. |
| `VaeGpuId` | Image or video encoding and decoding. | The latents cross between cards, so the denoising is unchanged. |
| `DitShardGpuId` | The diffusion model, split across both cards. | Adds memory, not speed. Verified for Krea 2, Qwen-Image 20B, MiniMax H3 (fp8), and FLUX.1 plain generations. |
| `LmShardGpuId` | Large language and audio language models, split across both cards. | Does not change image models. |
| `CfgParallelGpuId` | The negative-prompt guidance pass, run next to the positive pass. | Adds speed when the model fits on both cards. Cannot be combined with `DitShardGpuId`. |

Each GPU field uses the same numbering as `GPU_ID`. The [multi-GPU guide](05-Multi-GPU.md) explains which combinations work.

## Next step

Read [04: Model folders](04-Model-Folders.md) to place your checkpoints.
