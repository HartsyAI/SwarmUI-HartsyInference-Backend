# HartsyInference for SwarmUI

Local image, video, and music generation for [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI), powered by a pure C# inference engine. There is no Python environment, virtual environment, or second server to manage.

![A Dany fantasy portrait generated locally with Krea 2 and HartsyInference](Assets/readme/dany-hero.png)

![License: MIT](https://img.shields.io/badge/license-MIT-blue) ![Status: beta](https://img.shields.io/badge/status-beta-orange) ![SwarmUI extension](https://img.shields.io/badge/SwarmUI-extension-7c3aed)

## Why HartsyInference

| Feature | What it means |
| --- | --- |
| **One application** | The engine loads inside SwarmUI. You keep the model picker, prompt box, history, and previews you already use. |
| **No Python** | The engine ships as a NuGet package that this extension references. You do not install it separately. |
| **Images, video, and music** | One backend covers the image, video, and music model families listed below. |
| **Memory handling** | `LowVram` streams weights only when a model would not otherwise fit. Cards with room keep the full-speed path. |
| **Multiple GPUs** | Run separate jobs on separate cards, or split one supported model across two cards. |

## Quick start

1. [Install the extension](docs/guides/01-Install.md) and restart SwarmUI.
2. In **Server** > **Backends**, turn on **Show Advanced**, then add **HartsyInference (Pure C# Inference)**.
3. Put a supported checkpoint in its model folder, refresh the model list, and select **Generate**.

The [first generation guide](docs/guides/02-First-Generation.md) walks through each step with screenshots.

## Requirements

| Item | Details |
| --- | --- |
| SwarmUI | A current install on Windows or Linux. |
| GPU | An NVIDIA GPU with a recent driver uses CUDA. AMD and Intel GPUs can use Vulkan. `ComputeBackend` set to `auto` tries CUDA, then Vulkan, then CPU. |
| VRAM | Model size decides what fits. Smaller or quantized checkpoints fit on smaller cards. When a model does not fit, streaming makes it run, but the engine notes streaming is typically 5 to 8 times slower than a fully resident model. |
| Disk | Model downloads range from a few GB to dozens of GB. |
| Network | Needed for the first build and for model downloads. |

Model licenses differ from this extension's license. Check each model's page before you download it or use it commercially. Fibo 2.0 (Bria) is CC-BY-NC-4.0, so non-commercial use only.

## Supported models

Support depends on the model family and, for some controls, on the specific checkpoint. SwarmUI only shows controls that the selected model can use.

| Type | Families |
| --- | --- |
| **Images** | Stable Diffusion 1.5, SDXL, SD3 and SD3.5, FLUX.1, FLUX.2 (dev and Klein), FLUX.3 Klein, Qwen-Image and Qwen-Image 2.1 (including the Turbo build), Z-Image, Krea 2, Chroma (including Radiance), AuraFlow, F-Lite, Ideogram 4, Boogu, ERNIE-Image, Lumina 2, HunyuanImage 2.1, OmniGen 2, Anima, HiDream-I1, Kandinsky 5, Lance 3B, Lens, MOSS-Image, Mage-Flow, Fibo 2.0 |
| **Video** | Wan 2.1 and 2.2 (text-to-video, VACE, Animate, and speech-to-video), HunyuanVideo, LTX-Video and LTX-Video 2, Kandinsky 5 video (Lite and Pro), Lance 3B video, MiniMax H3 |
| **Music** | ACE-Step 1.5 and ACE-Step v1, MiniMax Music 3 |

Editing and restoration tools appear when the selected model supports them:

- **Image to image:** start from an existing image and control how much it changes.
- **Inpainting:** paint a mask over an area to replace or repair it, with grow, shrink, blur, and crop-to-mask controls.
- **Refiner:** refine a result with a second supported image model.
- **Upscale:** enlarge a finished image with Real-ESRGAN or SeedVR2 when no refiner model is set.
- **Remove Background:** cut the subject out as a transparent PNG.
- **Seamless tiling:** make images that repeat cleanly for textures and backgrounds.
- **Guidance:** ControlNet, IP-Adapter, and LoRA, each with its own strength, when the model supports them.

## Benchmarks

These numbers come from the engine's [image scoreboard](https://github.com/HartsyAI/HartsyInference/blob/main/benchmarks/scoreboards/IMAGE.md). Each row is measured on an RTX 4090, with the same request sent through the SwarmUI API to HartsyInference and to ComfyUI. Each value is the warm median of three runs after one cold run. The faster time is in bold.

| Model | Steps | HartsyInference | ComfyUI |
| --- | ---: | ---: | ---: |
| Krea 2 Turbo | 8 | **4.5 s** | 6.5 s |
| FLUX.1 Dev | 20 | **9.5 s** | 12.5 s |
| Qwen-Image | 20 | **40.6 s** | 58.2 s |
| Z-Image Turbo | 8 | 4.2 s | **3.1 s** |
| SDXL | 20 | 10.9 s | **7.5 s** |

HartsyInference is not faster on every model. The scoreboard lists every measured family, including the ones where ComfyUI is faster. Results depend on hardware, checkpoint format, drivers, and generation settings.

## Backend settings

Most people only need the defaults. The card groups its settings into **Basics**, **Multi-GPU**, and **Advanced**, and each `?` links to the guide.

| Setting | Default | What it does |
| --- | --- | --- |
| Compute Backend | Auto | Picks CUDA, Vulkan, or CPU. Only options the machine can actually use are listed. |
| GPU | `0` | The GPU this backend uses, listed by name. Run one backend per GPU. |
| VRAM Mode | Auto | How hard the engine works to fit models into GPU memory. Auto only saves memory when a model needs it. |
| Previews | On | Shows a preview image while a generation runs. |
| Queue Size (Advanced) | `1` | How many more generations may wait on this backend while it is busy. |
| Engine Updates (Advanced) | Don't update | Keeps engine updates manual. |

Every setting is explained in the [backend settings guide](docs/guides/03-Backend-Settings.md).

## Use more than one GPU

The simplest setup is one backend per GPU. Pick a different **GPU** on each backend, and SwarmUI sends different generations to different cards.

One backend can also use a second NVIDIA GPU for the same generation, from the card's **Multi-GPU** section:

| Setting | What it moves to the second GPU |
| --- | --- |
| Text Encoder GPU | Text encoders (CLIP, T5, and similar). |
| VAE GPU | Image or video encoding and decoding. |
| Model Split GPU | The diffusion model itself, split across both cards to pool their memory. Verified for Krea 2, Qwen-Image 20B, MiniMax H3 (fp8), and FLUX.1 without ControlNet or inpaint. |
| CFG Parallel GPU | The negative-prompt guidance pass, run alongside the positive pass. This speeds up generation when the model fits on both cards. |

Model Split GPU and CFG Parallel GPU cannot be set together. Sharding mainly adds memory, and does not always make generation faster. See the [multi-GPU guide](docs/guides/05-Multi-GPU.md) and the [two-GPU setup notes](docs/15-Two-GPU-Setups.md).

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| The HartsyInference backend type is missing | Turn on **Show Advanced** under **Server** > **Backends**. If it is still missing, check **Server** > **Logs** for build errors and restart SwarmUI. |
| The extension does not load | Check **Server** > **Logs** for `Build of extension project` or `Failed to build extension`. Fix the reported error, then restart. |
| The model is not listed | Confirm the file is inside your model root and in the right folder. Select the main checkpoint, not a VAE or text encoder. Refresh the model list. |
| A control does not appear | Select the model first. SwarmUI shows controls based on the selected model. |
| The model runs out of memory | Keep **VRAM Mode** on Auto. Close other GPU programs, or try a smaller size or a smaller model. |
| **Compute Backend** does not offer CUDA | The list is built when SwarmUI starts, and the log says why CUDA was left out. Fix the driver, then restart SwarmUI. NVIDIA cards older than the engine's CUDA kernels run on Vulkan instead. |

More detail is in the [troubleshooting guide](docs/guides/06-Troubleshooting.md). When you report a problem, include your OS, GPU, model name, generation settings, and the HartsyInference lines from **Server** > **Logs**. Do not post private prompts, images, or API keys.

## Documentation

**Guides for users**

- [01: Install](docs/guides/01-Install.md)
- [02: First generation](docs/guides/02-First-Generation.md)
- [03: Backend settings](docs/guides/03-Backend-Settings.md)
- [04: Model folders](docs/guides/04-Model-Folders.md)
- [05: Multi-GPU](docs/guides/05-Multi-GPU.md)
- [06: Troubleshooting](docs/guides/06-Troubleshooting.md)

**Reference for developers**

- [Overview](docs/00-Overview.md) and [architecture](docs/01-Architecture.md)
- [Backend lifecycle](docs/06-Backend-Lifecycle.md)
- [Parameters and feature flags](docs/07-Parameters-And-Feature-Flags.md)
- [Web API routes](docs/08-Web-API-Routes.md)
- [Known issues](docs/14-Known-Issues-And-TODO.md)

## License

This extension is licensed under the [MIT License](LICENSE). Model files keep their own licenses.
