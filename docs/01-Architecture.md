# 01 — Architecture

## Component diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SwarmUI process (single .NET process — no subprocess)                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  SwarmUI core                                                       │  │
│  │   • T2IAPI (web API entry)                                          │  │
│  │   • BackendHandler (request router, queue)                          │  │
│  │   • T2IParamInput / T2IParamTypes (parameter system)                │  │
│  │   • T2IModel (model metadata, file path)                            │  │
│  │   • Image (output image type)                                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              │ Swarm picks our backend (IsAlive,         │
│                              │ IsValidForThisBackend, capacity check)    │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  HartsyInference SwarmUI extension (this repo)                       │  │
│  │                                                                     │  │
│  │   SwarmUIHartsyInference : Extension                                 │  │
│  │     OnInit() registers backend type, params, feature flags, perms   │  │
│  │                                                                     │  │
│  │   HartsyInferenceBackend : AbstractT2IBackend                        │  │
│  │     • Init()       — create IBackend (CPU/Vulkan/CUDA), warm caches │  │
│  │     • Generate()   — translate input → run → return Image[]         │  │
│  │     • GenerateLive — same, but pushes progress + previews           │  │
│  │     • LoadModel()  — load checkpoint into pipeline                  │  │
│  │     • Shutdown()   — dispose pipelines + backend                    │  │
│  │                                                                     │  │
│  │   InferencePipeline (the WorkflowGenerator equivalent)              │  │
│  │     • Step list keyed by priority                                   │  │
│  │     • Per-architecture loaders (SD15 / SDXL / Flux / SD3 / ...)     │  │
│  │     • Builds (pipeline_object, request_dto) tuple from T2IParamInput│  │
│  │                                                                     │  │
│  │   PipelineCache                                                     │  │
│  │     • Caches loaded weights per (model_path, dtype, device)         │  │
│  │     • Survives across generations until LoadModel changes target    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              │ direct method calls — no IPC, no JSON     │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  HartsyInference (NuGet packages, also in-process)                   │  │
│  │                                                                     │  │
│  │   • IBackend (CpuBackend / CudaBackend / VulkanBackend)             │  │
│  │   • Diffusion.Pipelines.{StableDiffusion15,Sdxl,Flux,Sd3,...}       │  │
│  │   • Diffusion.Models.{Denoisers,TextEncoders,Vae}                   │  │
│  │   • Diffusion.Schedulers.{Euler,DDIM,DPM++,LCM,FlowMatch}           │  │
│  │   • Diffusion.Adapters.{ControlNet,IpAdapter}                       │  │
│  │   • ModelHandler.{SafeTensors,Lora,Registry}                        │  │
│  │   • Tokenizers.{Clip,T5,Whisper,Qwen3}                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│                        GPU (CUDA / Vulkan) or CPU                        │
└──────────────────────────────────────────────────────────────────────────┘
```

## Layers, top-down

### 1. Extension class (`SwarmUIHartsyInference.cs`)

The entry point. Inherits from `SwarmUI.Core.Extension`. Responsibilities, by lifecycle hook:

| Hook | Responsibility |
|------|----------------|
| `OnPreInit()` | Register feature flags this extension's backend will advertise; register any custom `T2IModelClass` for unusual checkpoints; add CSS/JS assets to `OtherAssets` if any. |
| `OnInit()` | Register the backend type via `Program.Backends.RegisterBackendType<HartsyInferenceBackend>()`; register any new parameters via `T2IParamTypes.Register<>()`; register permissions; call `HartsyInferenceWebAPI.Register()`; subscribe to model-folder change events. |
| `OnPreLaunch()` | Final wiring (route maps, etc.). Probably nothing to do. |
| `OnShutdown()` | Best-effort: tell the backend handler we're shutting down (the handler itself does the per-instance shutdown). |

### 2. Backend class (`Backends/HartsyInferenceBackend.cs`)

Subclass of `SwarmUI.Backends.AbstractT2IBackend`. **One instance per user-configured
"backend" entry** in Swarm's Server → Backends UI. A user might configure two: one
on CUDA device 0 and one on the CPU, for example.

Required overrides (see [`06-Backend-Lifecycle.md`](./06-Backend-Lifecycle.md) for
detail):

- `Init()` — construct `IBackend`, set up shader caches, set `Status = IDLE`
- `Shutdown()` — dispose pipelines, dispose `IBackend`
- `Generate(input)` — return `Image[]`
- `LoadModel(model, input)` — load weights for a given checkpoint
- `SupportedFeatures` — yield feature-flag IDs

Optional but expected:

- `GenerateLive(input, batchId, takeOutput)` — push live progress JObjects + final images
- `IsValidForThisBackend(input)` — reject inputs we can't satisfy (wrong architecture, missing component)
- `FreeMemory(systemRam)` — call `IBackend.FreeWeights()` and clear pipeline cache

### 3. Inference pipeline (`Generation/InferencePipeline.cs`)

The architectural equivalent of `ComfyUIBackend.WorkflowGenerator`. Where Comfy's
generator builds a JSON DAG of nodes, ours builds a `PipelineExecution` —
a (concrete pipeline instance, request DTO, post-processing list) tuple ready to be
called. See [`05-Pipeline-Translation.md`](./05-Pipeline-Translation.md) for the
full design.

Key properties carried over from `WorkflowGenerator`:

- **Step priority list.** Steps run in priority order. Extension authors (or future
  internal code) can register `AddStep(action, priority)` to inject behaviour.
- **Per-architecture model loaders** registered via `AddModelGenStep`-equivalent —
  each handles one model class (SD15, SDXL, Flux, etc.) and assembles the right
  pipeline.
- **Mutable context object** (`PipelineContext`) threaded through steps. Holds the
  current pipeline being assembled, current weights, current adapters, etc.

### 4. Model + weight cache (`Generation/PipelineCache.cs`)

Keyed by `(model_path, dtype, IBackend instance)`. Holds:

- The loaded `Pipeline` object (e.g., `StableDiffusion15Pipeline`)
- The weight `Dictionary<string, Tensor>` for each component (text encoder, UNet, VAE)
- Last-used timestamps for LRU eviction

When the user runs a second generation with the same checkpoint, no reload happens.
When they switch checkpoints, the LRU evicts the least-recently-used pipeline and
calls `IBackend.FreeWeights()` on its tensor set.

### 5. Web API surface (`WebAPI/HartsyInferenceWebAPI.cs`)

A small set of HTTP routes the extension adds for diagnostics and admin tasks. See
[`08-Web-API-Routes.md`](./08-Web-API-Routes.md). At minimum:

- `HartsyInferenceProbeModel` — given a checkpoint path, return detected architecture and required components
- `HartsyInferenceListLoadedPipelines` — show the pipeline cache state
- `HartsyInferenceClearCache` — flush the pipeline cache (admin)
- `HartsyInferenceGetDeviceInfo` — list available CUDA / Vulkan / CPU devices

This is **much smaller** than `ComfyUIWebAPI.cs` — we don't have workflows to save,
no node types to enumerate, no LoRA extraction (yet).

## Data flow: a single generation

1. **Browser → Swarm WebAPI.** User clicks Generate. Swarm receives a `T2IParamInput`
   (prompt, negative prompt, model name, steps, seed, LoRAs, etc.).
2. **Swarm BackendHandler picks a backend.** Iterates registered backends. For each,
   checks `IsAlive()` and `IsValidForThisBackend(input)`. Picks one with capacity.
3. **`HartsyInferenceBackend.GenerateLive(input, batchId, takeOutput)`.**
4. **Pipeline assembly.** `InferencePipeline.Execute(input)`:
   - Determine model architecture from `input.Get(T2IParamTypes.Model)`'s
     `ModelClass.CompatClass` (SD15 / SDXL / Flux / SD3 / ...).
   - Look up the registered architecture handler.
   - Find or load the components from the cache:
     - Tokenizer(s)
     - Text encoder(s) — `ClipTextEncoder` for SD1.5/SDXL-CLIP-L, `+ ClipTextEncoder(SdxlClipG)` for SDXL, `+ T5TextEncoder` for Flux/SD3
     - Denoiser — `UNet` for SD/SDXL, `FluxTransformer` for Flux, `Sd3Transformer` for SD3
     - VAE decoder
   - If LoRAs requested: build a `LoraStack`, call `ApplyToWeights()` on the relevant
     weight dicts (UNet, ClipL, ClipG). Note: LoRA application **mutates the cached
     weights**, so the cache key must include applied-LoRA fingerprint or we must
     deep-clone before applying.
   - If ControlNet requested: load the ControlNet adapter and pass to the pipeline
     (when HartsyInference's pipeline ctors accept it — see
     [`04-HartsyInference-Integration.md`](./04-HartsyInference-Integration.md) gaps
     section).
   - Construct the pipeline (each pipeline class has a unique constructor).
   - Build a `TextToImageRequest` DTO from `T2IParamInput`.
5. **Run inference.** Call `pipeline.GenerateFromTokens(...)` (synchronous). Pass a
   `progress => takeOutput(...)` callback that converts `GenerationProgress` to the
   `JObject` format `GenerateLive` expects.
6. **Post-process.** HartsyInference returns raw RGB bytes. Wrap them as a
   `SwarmUI.Image`. If the user wants extra steps (Swarm's "after generation" hooks
   handled by core — we don't own those), they run after our return.
7. **Return.** `takeOutput(image)` for each image; method returns. Swarm forwards to
   the browser via WebSocket.

## Why this structure (rationales)

- **One pipeline cache per `IBackend`.** HartsyInference tensors are device-bound —
  a tensor allocated for a CUDA backend can't be used by a CPU backend. Two configured
  Swarm backends (CUDA + CPU) can't share a cache.
- **`InferencePipeline` rather than ad-hoc switch-on-architecture in `Generate()`.**
  The Comfy `WorkflowGenerator` design proved out a step-priority + per-architecture
  step pattern that lets extensions inject behaviour without modifying core. We get
  the same benefit for free, and future extensions to *this* extension (e.g., a
  side extension that adds support for some new model) can register their own steps.
- **No HTTP between Swarm and HartsyInference.** The whole point is in-process. We
  ignore HartsyInference.Server entirely (it's not implemented yet anyway).
- **One backend type, not two.** The Comfy extension has both `comfyui_api`
  (remote URL) and `comfyui_selfstart` (auto-launch). For us there's no "remote"
  notion in v1 — we always run in-process. Single backend type, with settings to
  pick CUDA / Vulkan / CPU and device ordinal.

## What we explicitly inherit from the Comfy extension's design

- **Feature flags.** Same string IDs ("controlnet", "ipadapter", "video", etc.) so
  Swarm's existing parameter-gating Just Works. Our backend advertises a subset based
  on what HartsyInference supports.
- **Step-priority pipeline assembly.** Mirroring `WorkflowGenerator.AddStep`.
- **Per-architecture model loaders.** Mirroring `WorkflowGeneratorModelSupport`.
- **Permission groups.** Following the same `PermInfoGroup` pattern.

## What we explicitly don't inherit

- **JSON workflow IR.** We don't build a JSON DAG. We call HartsyInference methods
  directly. The "workflow" is a `PipelineExecution` C# object.
- **HTTP/WebSocket bridge.** Comfy needs `/ComfyBackendDirect/{*Path}` and a
  WebSocket relay for live updates. We don't — `Action<GenerationProgress>` callbacks
  on the same thread are enough.
- **Custom node packs / Python install management.** `InstallableFeatures` is for
  Comfy nodes; irrelevant here.
