# Backend Settings

The HartsyInference backend card has three sections:

- **Basics**: the settings most people check. The defaults suit most setups.
- **Multi-GPU**: puts parts of one generation on a second NVIDIA GPU. It stays folded unless you use it.
- **Advanced**: queueing, engine updates, and the kernel folder. It stays folded unless you change something there.

Each `?` on the card opens a short explanation with a link back to this guide. The explanation also shows the setting's saved name, such as `GPU_ID`, which is the name the logs use.

## Basics

| Setting | Saved as | Default | What it does |
| --- | --- | --- | --- |
| Compute Backend | `ComputeBackend` | Auto | Picks the hardware: CUDA, Vulkan, or CPU. |
| GPU | `GPU_ID` | `0` | The GPU this backend runs on. |
| VRAM Mode | `LowVram` | Auto (recommended) | How hard the engine works to fit models into GPU memory. |
| Previews | `Previews` | On | Shows a preview image while a generation runs. |

### Compute Backend

- **Auto** uses CUDA when it works, and Vulkan otherwise. Its entry names the one it uses on your machine, such as `Auto: Vulkan (recommended)`.
- **CUDA** appears only when an NVIDIA GPU is present and a test computation on it succeeds. If CUDA is missing on an NVIDIA card, the log says why. A card older than the engine's CUDA kernels support still runs on Vulkan.
- **Vulkan** appears when a real GPU supports it, including AMD and Intel cards. Software renderers do not count.
- **CPU** is very slow and cannot run fp8 or bf16 models. Use it only for testing.
- The list is built when SwarmUI starts. After you change a GPU or a driver, restart SwarmUI.

If Auto finds no usable GPU, its entry reads `Auto (no usable GPU found)` and the backend does not start. It never falls back to CPU on its own, because the CPU backend cannot run most models. Pick CPU yourself if you want it.

### GPU

The list names each GPU and its memory. It changes to match Compute Backend.

- Under CUDA, each entry is a CUDA device, such as `GPU 1: NVIDIA GeForce RTX 3090 (24 GB)`. CUDA puts the fastest card first, so its numbers can differ from `nvidia-smi`. Go by the names on the card.
- Under Vulkan, the first entry is **Auto**, which lets the engine pick the best GPU and names the one it picks. The other entries are your other GPUs. Software renderers are not listed.
- A GPU that fails the engine's CUDA test is left out of the CUDA list, and the log says why.
- On CPU the field is greyed out, because the CPU backend has no GPU to pick.

A saved value the list does not offer, such as a GPU you removed, shows as `(saved value)` and stays until you pick something else.

To run generations on several GPUs at once, add one backend per GPU and pick a different GPU on each. See the [multi-GPU guide](05-Multi-GPU.md).

### VRAM Mode

| Option | What it does |
| --- | --- |
| Auto (recommended) | Measures free memory, and only saves memory when a model would not fit otherwise. |
| Performance (never stream) | Keeps the whole model on the GPU. A model that does not fit fails with an out-of-memory error. |
| Balanced (unload between stages) | Frees each stage's weights when the stage ends, so the next stage has room. Streams only when it must. |
| Aggressive (always stream) | Always streams, and works in smaller pieces. Useful when another program shares the GPU. |
| Maximum (save the most memory) | Asks for every memory saver the engine has. The slowest option. |

Streaming loads a model in pieces while it runs. It is several times slower than keeping the whole model on the GPU, but it lets large models run on smaller cards. Streaming needs the CUDA compute backend. On Vulkan and CPU, models load fully.

Not every model uses every saver that Aggressive and Maximum ask for yet. The log names any a generation skipped.

You can change the savers for a single generation under the **VRAM & Memory** parameter group.

Older saved values still work: `on` acts as Aggressive, and `off` acts as Performance. They show as `(saved value)` until you pick an option.

### Previews

Shows a preview image while a generation runs. Some models only move the progress bar. Turn it off to save a little time.

## Multi-GPU

These settings put parts of one generation on a second NVIDIA GPU. They need two NVIDIA GPUs that work with CUDA. When this machine does not have them, the section is greyed out and says why. Leave them on **None** for a single-GPU setup.

| Setting | Saved as | What it moves |
| --- | --- | --- |
| Text Encoder GPU | `TextEncoderGpuId` | The text encoder. |
| VAE GPU | `VaeGpuId` | Image and video encoding and decoding. |
| Model Split GPU | `DitShardGpuId` | Part of a model that is too big for one GPU. |
| CFG Parallel GPU | `CfgParallelGpuId` | The negative prompt pass, run at the same time as the positive pass. |

The [multi-GPU guide](05-Multi-GPU.md) explains each one and which combinations work.

## Advanced

| Setting | Saved as | Default | What it does |
| --- | --- | --- | --- |
| Queue Size | `OverQueue` | 1 | How many more generations can wait on this backend while it is busy. |
| Engine Updates | `AutoUpdate` | Don't update | Whether to look for a newer engine when the backend starts. |
| Kernel Folder | `KernelDirectory` | Empty | Where the engine's compiled GPU kernels are. Empty finds them automatically. |

### Queue Size

- `0`: nothing waits on this backend. Extra generations go to another backend right away. This suits one backend per GPU.
- `1` (default): one more generation can wait here.
- `2` to `5`: more generations can wait here.

A backend always runs one generation at a time. Queue Size only decides where the waiting ones wait.

An older saved value of `-1` makes the backend take no generations at all, and shows as `(saved value)`. To pause a backend, turn it off with the switch on its card instead.

### Engine Updates

| Option | Saved as | What it does |
| --- | --- | --- |
| Don't update (default) | `false` | Never looks for a newer engine. |
| Update, then wait for you to restart | `true` | When the backend starts, it checks NuGet for a newer engine. If one exists and the extension builds against it, the update is staged, and the backend stays off until you restart SwarmUI. |
| Update and restart SwarmUI automatically | `aggressive` | The same check, then SwarmUI restarts itself to load the new engine. If an update still has not applied after a restart, it does not restart again. |

The engine runs inside SwarmUI, so a new engine only loads when SwarmUI restarts. The older value `force` acts like `aggressive`.

### Kernel Folder

Leave it empty. The engine's compiled GPU kernels, in its `Ptx` and `Spirv` folders, ship with the extension and are found automatically. Set this only if you moved the kernels or built your own. When the backend starts, its log lists the kernel folders it uses.

The folder applies to every HartsyInference backend. If you clear it, restart SwarmUI for that to take effect.

## Next step

Read [04: Model folders](04-Model-Folders.md) to place your checkpoints.
