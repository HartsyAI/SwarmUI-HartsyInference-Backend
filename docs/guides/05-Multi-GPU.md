# Multi-GPU

HartsyInference can use more than one GPU in two ways:

- **Separate jobs:** one backend per GPU. Different generations run on different cards at the same time.
- **One job across cards:** one backend puts parts of the same generation on a second GPU.

## Separate jobs

1. Add a HartsyInference backend for each GPU.
2. On each backend, pick a different GPU in **GPU**. The list names each card.
3. Leave the **Multi-GPU** section on **None**.
4. Optional: under **Advanced**, set **Queue Size** to `0` on each backend. A second generation then goes to the idle GPU right away instead of waiting behind the first.

SwarmUI sends each generation to a free backend. The extension also checks whether the model fits each card's memory, so a large model is not sent to a small card just because that card was free.

Under Vulkan, the first entry in **GPU** is **Auto**, which lets the engine pick the best GPU. That card may be one another backend uses. When you run several backends on Vulkan, choose a specific GPU for each one, and do not use Auto on any of them. When a backend starts, its log names the GPU it uses.

## One job across two cards

These settings are in the card's **Multi-GPU** section. They need two NVIDIA GPUs that work with CUDA. When this machine does not have them, the section is greyed out and says why.

Each list names your CUDA GPUs. The main GPU is marked `(main GPU)` and cannot be picked, because placing a part on the main GPU does nothing.

| Setting | Saved as | Use it when |
| --- | --- | --- |
| Text Encoder GPU | `TextEncoderGpuId` | The text encoder takes a large share of memory, as with big video models. |
| VAE GPU | `VaeGpuId` | Decoding large images or long videos needs room on the main GPU. |
| Model Split GPU | `DitShardGpuId` | The model is too large for one GPU. |
| CFG Parallel GPU | `CfgParallelGpuId` | The model fits on both GPUs and you want faster steps. |

### Text Encoder GPU

Runs the text encoder on the second GPU, which frees its memory on the main GPU. A text encoder on its own GPU stays loaded between generations. A different GPU model can change results slightly, but they stay the same from run to run.

When it takes effect, the log says `Text encoder placed off the primary backend`.

### VAE GPU

Runs image and video encoding and decoding on the second GPU. The denoising itself does not change. A different GPU model can change a few pixel values slightly.

When it takes effect, the log says `VAE encode/decode placed off the primary backend`.

### Model Split GPU

Splits the diffusion model across the main GPU and this one, so their memory adds up. It adds memory, not speed: steps are no faster than on one card, and can be a little slower.

Verified models:

- Krea 2
- Qwen-Image 20B
- MiniMax H3 (fp8 build)
- FLUX.1, for plain generations. ControlNet, Kontext, inpainting, and regional prompts run unsplit.

Other families may work through the same code path, but these are the ones verified end to end.

### CFG Parallel GPU

Runs the negative prompt pass on this GPU at the same time as the positive pass on the main GPU. It makes steps faster, but the model must fit on both GPUs. It only helps models that run a real negative prompt pass, such as Wan video, or FLUX with a negative prompt and CFG above 1.

If the second GPU cannot hold the whole model, the generation still works but runs one pass at a time. The engine log shows `[CfgParallel]` with `active` or `fell-back`, so you can tell which happened.

### Combinations that do not work

- **Model Split GPU** and **CFG Parallel GPU** cannot be used together, and the backend does not start with both set. The card blocks picking both and explains why.
- A Multi-GPU setting on the main GPU does nothing. The card marks the main GPU in each list.

## Check that both cards are used

Start a generation and watch `nvidia-smi`. A split model shows memory on both GPUs. A single-card generation shows memory on one GPU only. CUDA numbers GPUs fastest first, which can differ from the order `nvidia-smi` uses, so match the cards by name.

## More detail

The [two-GPU setup notes](../15-Two-GPU-Setups.md) include worked examples and their limits.
