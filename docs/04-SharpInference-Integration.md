# 04 — SharpInference Integration

The contract between this extension and the upstream SharpInference library.
Everything we use, everything we wish existed, and where to find it in the
SharpInference repo (`/home/kalebbroo/Desktop/Projects/SharpInference/`).

## Distribution model

**Phase 1 — git submodule + ProjectReference.** Until SharpInference ships on
NuGet, vendor it under `Vendor/SharpInference/` as a submodule and reference the
relevant `.csproj` files directly:

```bash
cd src/Extensions/SwarmUI-SharpInference
git submodule add <SharpInference-repo-url> Vendor/SharpInference
git submodule update --init --recursive
```

Then in `SwarmUI-SharpInference.csproj`:

```xml
<ItemGroup>
  <ProjectReference Include="Vendor/SharpInference/src/SharpInference.Core/SharpInference.Core.csproj" />
  <ProjectReference Include="Vendor/SharpInference/src/SharpInference.Diffusion/SharpInference.Diffusion.csproj" />
  <ProjectReference Include="Vendor/SharpInference/src/SharpInference.ModelHandler/SharpInference.ModelHandler.csproj" />
  <ProjectReference Include="Vendor/SharpInference/src/SharpInference.Tokenizers/SharpInference.Tokenizers.csproj" />
  <ProjectReference Include="Vendor/SharpInference/src/SharpInference.Cpu/SharpInference.Cpu.csproj" />
  <ProjectReference Include="Vendor/SharpInference/src/SharpInference.Cuda/SharpInference.Cuda.csproj" />
  <ProjectReference Include="Vendor/SharpInference/src/SharpInference.Vulkan/SharpInference.Vulkan.csproj" />
</ItemGroup>
```

**Phase 2 — NuGet PackageReferences.** Once SharpInference publishes packages, swap
the `<ProjectReference>` block for `<PackageReference Include="SharpInference.Diffusion" Version="..." />`
and a few siblings, and remove the submodule.

## What we consume from SharpInference

### From `SharpInference.Core`

| Type | Path | Why |
|------|------|-----|
| `IBackend` | `src/SharpInference.Core/Backends/IBackend.cs:6` | Required by every pipeline; we instantiate one of `CpuBackend` / `CudaBackend` / `VulkanBackend` and pass it to pipelines |
| `Tensor` | `src/SharpInference.Core/Tensors/Tensor.cs:8` | We don't typically construct these — the pipeline does — but we touch them for LoRA application |
| `DeviceKind` | `src/SharpInference.Core/Backends/DeviceKind.cs:4` | Identifies the device of a tensor |

### From `SharpInference.Cpu` / `SharpInference.Cuda` / `SharpInference.Vulkan`

| Type | Why |
|------|-----|
| `CpuBackend` | Default fallback; constructor takes no args |
| `CudaBackend(int deviceOrdinal, string ptxDir)` | Primary GPU path on NVIDIA |
| `VulkanBackend()` (constructor signature TBC) | Cross-vendor GPU path |

We instantiate exactly one of these per Swarm-configured backend instance, based on
the user's setting.

### From `SharpInference.ModelHandler`

| Type | Path | Why |
|------|------|-----|
| `SafeTensorsLoader` | `src/SharpInference.ModelHandler/SafeTensors/` | Load `.safetensors` checkpoints. Pattern: `using var loader = new SafeTensorsLoader(); loader.Load(path); var tensors = loader.GetAllTensors();` |
| `LoraStack` | `src/SharpInference.ModelHandler/Lora/LoraStack.cs:9` | Multi-LoRA application |
| `LoraFile` | `src/SharpInference.ModelHandler/Lora/LoraFile.cs` | Loaded LoRA file representation |
| `LoraTarget` enum | `src/SharpInference.ModelHandler/Lora/LoraTarget.cs:7` | UNet / Transformer / ClipL / ClipG |
| `ModelRegistry` | `src/SharpInference.ModelHandler/Registry/` | Optional — in-memory model cache. We may use our own `PipelineCache` instead |

We do **not** use `HuggingFace/` (download utilities) or `Gguf/` in v1. Models come
from Swarm's `Models/` folder; the user is responsible for putting files there.

### From `SharpInference.Tokenizers`

| Type | Why |
|------|-----|
| `ClipTokenizer(string vocabPath, string mergesPath)` | Tokenize prompt + negative prompt for SD/SDXL/Flux CLIP-L paths |
| `T5Tokenizer` | T5 path for Flux / SD3 |

### From `SharpInference.Diffusion`

This is the bulk of our consumption surface.

#### Pipelines (`src/SharpInference.Diffusion/Pipelines/`)

Each pipeline is **a sealed concrete class with a unique constructor**. There is
no `IPipeline` interface (despite an unused skeleton at
`src/SharpInference.Core/Pipelines/IDiffusionPipeline.cs`). Every pipeline class
exposes:

- A constructor taking the `IBackend`, the loaded model components, and (for some)
  a config record
- Either a `Generate(TextToImageRequest req, Action<GenerationProgress>?)` method
  (returns `(byte[] rgbData, int w, int h, int seed)`) or a
  `GenerateFromTokens(int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest, ...)`
  variant

**Pipelines we'll wire:**

| Pipeline | Constructor signature (current) | Notes |
|----------|---------------------------------|-------|
| `StableDiffusion15Pipeline` | `(IBackend, ClipTextEncoder, UNet, VaeDecoder)` | SD1.5 |
| `SdxlPipeline` | `(IBackend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet, VaeDecoder)` | SDXL |
| `SdxlInpaintPipeline` | inpaint variant — takes pre-encoded source | Blocked by missing VaeEncoder |
| `Sd3Pipeline` | TBC | SD3 |
| `FluxPipeline` | `(IBackend, ClipTextEncoder clipL, T5TextEncoder t5, FluxTransformer, VaeDecoder, FluxConfig)` | Flux dev/schnell |
| `Flux2Pipeline` | TBC | Flux 2 |
| `AuraFlowPipeline` | TBC | AuraFlow |
| `ChromaPipeline` | TBC | Chroma |
| `ZImagePipeline` | TBC | Z-Image |
| `HunyuanImagePipeline` | TBC | HunyuanImage |

`TBC` = constructor signature to be confirmed by reading the upstream source. The
pipeline-translation layer (`Generation/ModelSupport.cs`) holds one
`IPipelineHandler` per architecture, so the dispatch is closed over the differences.

#### Models (`src/SharpInference.Diffusion/Models/`)

| Subdir | What we use |
|--------|-------------|
| `Denoisers/` | `UNet`, `UNetConfig.Sd15`, `UNetConfig.SdxlBase`, `FluxTransformer`, `FluxConfig`, `Sd3Transformer`, `Flux2Transformer`, etc. |
| `TextEncoders/` | `ClipTextEncoder`, `ClipTextEncoderConfig.{Sd15,SdxlClipL,SdxlClipG}`, `T5TextEncoder`, `LlamaStyleEncoder` |
| `Vae/` | `VaeDecoder`, `VaeTiledDecoder`, `VaeConfig.{Sd15,Sdxl,Sd3,Flux,Flux2,...}` |

Loading pattern (manual today, no façade):

```csharp
// 1. Tokenizer
var tokenizer = new ClipTokenizer(vocabPath, mergesPath);

// 2. Text encoder
var clipConfig = ClipTextEncoderConfig.Sd15;
var textEncoder = new ClipTextEncoder(clipConfig);
using var clipLoader = new SafeTensorsLoader();
clipLoader.Load(textEncoderPath);
textEncoder.LoadWeights(clipLoader.GetAllTensors(), prefix: "text_model");

// 3. UNet
var unet = new UNet(UNetConfig.Sd15);
using var unetLoader = new SafeTensorsLoader();
unetLoader.Load(unetPath);
unet.LoadWeights(unetLoader.GetAllTensors());

// 4. VAE
var vae = new VaeDecoder(VaeConfig.Sd15);
using var vaeLoader = new SafeTensorsLoader();
vaeLoader.Load(vaePath);
vae.LoadWeights(vaeLoader.GetAllTensors());

// 5. Pipeline
using var pipeline = new StableDiffusion15Pipeline(backend, textEncoder, unet, vae);
```

For Swarm we don't have separate per-component files — Swarm checkpoints are
typically a single .safetensors with everything inside. Our model loader
(`Generation/ModelSupport.cs`) needs to **partition the all-in-one checkpoint by
prefix** (e.g., `text_model.*`, `model.diffusion_model.*`, `first_stage_model.*`)
and feed each component its slice.

#### Schedulers (`src/SharpInference.Diffusion/Schedulers/`)

`EulerDiscreteScheduler`, `DdimScheduler`, `DpmPlusPlus2MScheduler`, `LcmScheduler`,
`FlowMatchEulerDiscreteScheduler`. Pipelines select a scheduler internally based on
`TextToImageRequest.Scheduler` (a string). We map Swarm's sampler enum to that string.

#### Adapters (`src/SharpInference.Diffusion/Adapters/`)

`ControlNet`, `IpAdapter`. These exist as classes but **pipelines don't accept them
yet.** Wiring them in is upstream work scheduled for phase 5.

### From `SharpInference.Diffusion/Requests/`

`TextToImageRequest` is the only DTO. Properties:

```
Prompt          (string, required)
NegativePrompt  (string, "")
Steps           (int, 20)
CfgScale        (float, 7.5f)
Width           (int, 512)
Height          (int, 512)
Seed            (int?, null = random)
Scheduler       (string?, null = pipeline default)
```

We populate this from `T2IParamInput`. There is no separate Img2ImgRequest /
InpaintRequest yet — img2img/inpaint pipelines reuse `TextToImageRequest` plus
explicit tensor parameters on the method signature.

## Gaps we'll either route around or upstream

These are required for Comfy parity but don't exist in SharpInference today.

| # | Gap | Impact | Plan |
|---|-----|--------|------|
| 1 | **TFM mismatch (net10 vs net8)** | Cannot reference SharpInference at all | Upstream PR — multi-target net8.0/net10.0. **Phase 0 blocker.** |
| 2 | **No `VaeEncoder`** | Blocks all img2img / inpaint paths | Upstream PR. Phase 4 dependency. Substantial — VAE encode is a real model, not a wrapper. |
| 3 | **No `CancellationToken` on pipelines** | UI cancel button does nothing | Workaround in phase 3: stop-flag checked in progress callback (only stops at next-step boundary, with up-to-1-step latency). Long-term: upstream PR. |
| 4 | **Progress callback exposes only `(Step, Total, Elapsed)`** | Can't show preview images | Upstream: extend `GenerationProgress` to expose the in-flight latent (or a hook to extract it). Phase 3. |
| 5 | **No `IPipeline` interface — pipelines have non-uniform ctors** | Our dispatcher is a manual switch | We work around with our own `IPipelineHandler` abstraction in `Generation/ModelSupport.cs`. No upstream change strictly needed. |
| 6 | **No `PipelineFactory` / `ModelRegistry.LoadAsync`** | We have to manually partition checkpoints by tensor-prefix | We do this in our model loader. Optional upstream improvement; not blocking. |
| 7 | **`LoraStack.ApplyToWeights` is destructive** | Can't change LoRAs between gens cheaply | Workaround: keep an unmodified weights cache, deep-copy + apply per-generation, or PR a non-destructive variant upstream. Phase 2 design decision. |
| 8 | **Pipelines don't accept ControlNet in ctor** | No ControlNet parity | Upstream PR per pipeline class. Phase 5. |
| 9 | **No ControlNet preprocessors** | User has to provide pre-processed canny/depth maps | Upstream feature request — ports of canny (trivial), zoedepth (small NN), openpose (large), lineart. Phase 5. |
| 10 | **No textual-inversion embedding API** | Embeddings parameter is dead | Upstream feature request. Phase 5 (low priority). |
| 11 | **`SharpInference.Server` is a stub** | We can't lean on it for an out-of-process variant | Upstream finishes phase 7 of their own roadmap; meanwhile our backend is in-process only |

We open issues for #1 and #2 during phase 1 so they're in flight by the time we need them.

## How we package the dependency

Once SharpInference ships NuGets:

```xml
<ItemGroup>
  <PackageReference Include="SharpInference.Core" Version="x.y.z" />
  <PackageReference Include="SharpInference.Diffusion" Version="x.y.z" />
  <PackageReference Include="SharpInference.ModelHandler" Version="x.y.z" />
  <PackageReference Include="SharpInference.Tokenizers" Version="x.y.z" />
  <PackageReference Include="SharpInference.Cpu" Version="x.y.z" />
  <PackageReference Include="SharpInference.Cuda" Version="x.y.z" />
  <PackageReference Include="SharpInference.Vulkan" Version="x.y.z" />
</ItemGroup>
```

Then the user's Swarm install picks them up automatically as part of the extension's
NuGet restore step. **No manual git-clone of SharpInference required.**

## Native dependencies (CUDA / Vulkan)

- **CUDA** — SharpInference's CUDA backend ships PTX files (`Ptx/` folder, copied to
  output by SharpInference's build). User must have CUDA 12.x driver installed; no CUDA
  Toolkit needed (Driver API only).
- **Vulkan** — SharpInference ships SPIR-V shaders. User must have Vulkan 1.3 ICD.
- **CPU** — pure managed; no native deps.

The extension's csproj will need to ensure the `Ptx/` and SPIR-V folders are
copied to the Swarm output directory. SharpInference's build props should handle
this; if not, we add an `<ItemGroup>` with `<Content>` copy rules.

## Versioning and compatibility

SharpInference uses semantic versioning (planned). We pin to a specific minor in our
`<PackageReference>` and bump intentionally. A SharpInference major bump should
require an explicit extension review — the pipeline constructors are the contract,
and they could shift between majors.
