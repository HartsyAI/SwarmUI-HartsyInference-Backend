# Troubleshooting

Start with the logs. Open **Server** > **Logs**, set the filter to `HartsyInference`, and read the newest lines.

## The HartsyInference backend type is missing

- Turn on **Show Advanced** under **Server** > **Backends**.
- If it is still missing, check **Server** > **Logs** for a build error and restart SwarmUI.

## The extension failed to build

Search the logs for:

- `Build of extension project`
- `Failed to build extension in folder`

Fix the error the build reports, then restart SwarmUI.

## The extension shows as installed but the backend does not start

- Confirm `ComputeBackend` is `auto`.
- Confirm `GPU_ID` names a GPU this machine has. A value that is not a number, or is negative, is ignored, and the log warns that the engine is choosing a device. A number for a GPU that does not exist stops the backend with an error.
- Update your GPU driver, then restart SwarmUI.

## CUDA is not offered in ComputeBackend

The list is built when SwarmUI starts. Fix the driver or the CUDA setup, then restart SwarmUI. CUDA only appears when a real test computation on the card succeeds.

## A model does not appear

- Confirm the file is inside your model root. See [04: Model folders](04-Model-Folders.md).
- Put it in the correct folder for its type.
- Refresh the model list.
- Make sure you selected the main checkpoint, not a VAE or text encoder.

## A control does not appear

Select the model first. SwarmUI shows controls based on the model you select.

## The model runs out of memory

- Keep `LowVram` on `Auto (recommended)`.
- Close other programs that use the GPU.
- Try a smaller image size, fewer video frames, or a smaller model.
- If you have a second GPU, see [05: Multi-GPU](05-Multi-GPU.md).

`Performance (never stream, fail instead)` never streams, so an out-of-memory error is expected with that setting on a model that does not fit. Switch back to `Auto (recommended)` to let the engine stream.

## A generation is much slower than expected

Streaming is much slower than running a model fully on the card. Check the `LowVram` setting, and check that the model fits your card. The [benchmarks in the README](../../README.md#benchmarks) show typical timings on an RTX 4090.

## Collect the details for a bug report

Open an issue in the [extension repository](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend/issues) and include:

- Your operating system and GPU.
- The model name and file type.
- The generation settings: size, steps, sampler, and `LowVram`.
- The HartsyInference lines from **Server** > **Logs**.

Do not post private prompts, images, access tokens, or API keys.
