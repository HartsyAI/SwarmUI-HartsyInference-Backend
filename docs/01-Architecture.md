# 01 — Architecture

This extension is a thin mapper. It does not build a workflow graph, does not own a
model/weight cache, and does not construct pipelines. All of that — architecture
detection, recipe construction, pipeline caching, side-model download, and composition
(LoRA / ControlNet / IP-Adapter / refiner / img2img / inpaint / regional) — lives in the
`HartsyInference.Engine` NuGet package. The extension's job is narrower: translate a
SwarmUI request into an Engine request, call the Engine, and translate the result back.
`Backends/HartsyInferenceBackend.cs` says this in its own doc comment, and the rest of
the extension's source is consistent with it.

## Component diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│  SwarmUI process (single .NET process — no subprocess)                    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  SwarmUI core                                                       │  │
│  │   T2IAPI, BackendHandler, T2IParamInput/T2IParamTypes, T2IModel     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                        │ BackendHandler picks a backend:                  │
│                        │ IsAlive, IsValidForThisBackend, capacity         │
│                        ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  SwarmUI-HartsyInference-Backend extension (this repo)              │  │
│  │                                                                     │  │
│  │  SwarmUIHartsyInference : Extension        (entry point)            │  │
│  │   registers the "hartsyinference" backend type, params, perms,      │  │
│  │   web routes; cross-checks param feature flags against the          │  │
│  │   backend's DeclaredFeatures at startup                             │  │
│  │                                                                     │  │
│  │  HartsyInferenceBackend : AbstractT2IBackend   (the mapper)         │  │
│  │   Init/Shutdown/FreeMemory — own one InferenceEngine instance       │  │
│  │   LoadModel  — checks drivability only, does not load anything      │  │
│  │   GenerateLive — map request -> call Engine -> stream progress ->   │  │
│  │                  map result back to a SwarmUI Image                 │  │
│  │   IsValidForThisBackend — the honesty guard (refuses unserviceable  │  │
│  │                  params instead of silently dropping them)          │  │
│  │                                                                     │  │
│  │  ModelSupport (Generation/ModelSupport.cs)                          │  │
│  │   compat-class string -> Engine family id + Kind (Image/Video/Music)│  │
│  │                                                                     │  │
│  │  ControlNetPreprocessing + Canny/DepthAnything/OpenPose annotators  │  │
│  │   (+ AnnotatorDownloader for their weights) — host-side hint        │  │
│  │   preprocessing before the Engine ever sees an image                │  │
│  │                                                                     │  │
│  │  HartsyInferenceWebAPI  — a handful of read-mostly admin/           │  │
│  │   diagnostic HTTP routes, no workflow IR to expose                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                        │ direct in-process method calls —                 │
│                        │ no IPC, no JSON-over-HTTP                        │
│                        ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  HartsyInference.Engine (NuGet package, in-process)                 │  │
│  │                                                                     │  │
│  │  InferenceEngine — the facade this extension programs against.      │  │
│  │  Owns the compute backend(s), the loaded-pipeline caches, and       │  │
│  │  exposes per-modality services. This extension only calls:          │  │
│  │    .Images  — still-image generation                                │  │
│  │    .Video   — video generation                                      │  │
│  │    .Music   — audio/music generation                                │  │
│  │    .Restore — optional SeedVR2 restore/upscale pass                 │  │
│  │    .ComputeBackend — borrowed for host-side ControlNet annotators   │  │
│  │  (the Engine offers more services — Text, Speech, Mesh, World, …    │  │
│  │  — this extension doesn't call them)                                │  │
│  │                                                                     │  │
│  │  RecipeRegistry / VideoRecipeRegistry — per-family recipes that     │  │
│  │  actually build and run a pipeline; asked "do you support family X" │  │
│  │  at call time rather than the extension hard-coding that answer     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│                        ▼                                                  │
│                  GPU (CUDA / Vulkan) or CPU                               │
└───────────────────────────────────────────────────────────────────────────┘
```

Distribution: the Engine is a `<PackageReference Include="HartsyInference" Version="2.0.0-alpha.17">`
in `SwarmUI-HartsyInference.csproj` — not a git submodule, not a `Vendor/` folder, not a
`ProjectReference`. The private `HartsyInference.*` assemblies (and their own deps,
Google.Protobuf / Microsoft.ML.Tokenizers) are copied into this extension's own output
folder so Swarm's per-extension `AssemblyLoadContext` can resolve them — that is also why
the CUDA PTX / Vulkan SPIR-V kernel folders resolve next to the extension DLL rather than
in Swarm's main runtime directory. `-p:UseLocalHartsy=true -p:HartsyRepo=<path>` swaps the
package reference for prebuilt local DLL references, for engine development against
unpublished builds.

## Layers, top-down

### 1. Extension entry point (`SwarmUIHartsyInference.cs`)

Inherits `SwarmUI.Core.Extension`. `OnPreInit()` registers model classes core doesn't know
about (ACE-Step v1, Lance image/video, MusicGen, YuE, F-Lite) before model folders are
scanned. `OnInit()` registers the HartsyInference-specific `T2IParamGroup`s and params
(Wan-Animate driving, Ideogram 4 magic prompt, Video Restore, the Music group), registers
the backend type via `Program.Backends.RegisterBackendType<Backends.HartsyInferenceBackend>("hartsyinference", ...)`
— one type, no `_api`/`_selfstart` split, because there is no "remote" mode: everything
runs in-process — registers the web routes, and finally calls
`WarnOnUndeclaredFeatureFlags()`.

That last check is a startup self-test: it walks every param registered under this
extension's group and confirms its `FeatureFlag` is covered by
`HartsyInferenceBackend.DeclaredFeatures` (or `T2IEngine.DisregardedFeatureFlags`).
`T2IEngine` silently refuses any job whose required flags aren't advertised by the chosen
backend, and that refusal names no param — so a flag added to a param and forgotten on
the backend's declared list is otherwise invisible until a generation mysteriously stops
working. Logging the mismatch at startup catches it immediately instead.

### 2. The backend (`Backends/HartsyInferenceBackend.cs`)

Subclass of `SwarmUI.Backends.AbstractT2IBackend`. One instance per backend entry
configured in Swarm's Server → Backends UI — a user can run several, each pointed at a
different `GPU_ID`. Each instance owns exactly one `HartsyInference.Engine.InferenceEngine`.

- **`Init()`** parses the backend's settings (compute backend, `GPU_ID`, and the optional
  multi-GPU placement knobs — text-encoder/VAE/CFG-parallel/DiT-shard/LM-shard device IDs,
  detailed in `15-Two-GPU-Setups.md`), validates the chosen device eagerly with
  `BackendFactory.Validate` so a bad `GPU_ID` fails at startup with `Status = ERRORED`
  instead of reporting healthy and dying on the first request, refuses to silently fall
  back to CPU when CUDA is unavailable (the CPU kernels are F32-only and will corrupt
  memory on fp8/bf16 checkpoint weights), assembles a `PlacementConfig` if any split-device
  setting is present, and constructs the `InferenceEngine`. `MaxUsages = 1 + OverQueue`
  (the scheduler's routing threshold); `Status = RUNNING` (not `IDLE` — `RUNNING` is Swarm's
  "alive and ready" state, `IDLE` would make the router skip this backend).
- **`Shutdown()`** disposes the `InferenceEngine` and any borrowed/owned preprocessor
  device, and releases this instance's device-usage registration.
- **`FreeMemory(systemRam)`** calls `InferenceEngine.FreeMemory()` to evict cached
  pipelines/weights; used by the WebAPI's clear-cache route and by the backend itself as
  an out-of-VRAM retry step (see the data-flow section below).
- **`LoadModel(model, input)`** does **not** load anything. It resolves the model's compat
  class through `ModelSupport`, returns `false` (so Swarm can route to another backend) if
  the architecture isn't drivable, otherwise records `CurrentModelName` and returns `true`.
  The Engine loads and caches the actual pipeline lazily, inside the generate call, keyed
  on checkpoint path plus the request's LoRA/component identity — a separate load step here
  would only fight that cache, so there isn't one.
- **`Generate(input)`** is a thin wrapper over `GenerateLive`: it calls
  `GenerateLive(input, "single", collector)` and gathers whatever images the collector
  callback receives. `GenerateLive` is the real implementation, not a peer method.
- **`IsValidForThisBackend(input)`** is the honesty guard — see its own section below.
- **`SupportedFeatures` / `DeclaredFeatures`** — the static list of feature-flag strings
  this backend advertises to `T2IEngine`, documented further in `07-Parameters-And-Feature-Flags.md`.

Auto-update: setting `AutoUpdate` to `true`/`aggressive` makes `Init()` check NuGet for a
newer engine build and stage it into the extension's build output. Because the Engine is
an in-process library, not a subprocess Swarm can relaunch, a staged update cannot hot-swap
— the backend errors itself out with a "restart to load it" message until the user (or
`aggressive` mode itself) restarts SwarmUI. Full mechanics are in `06-Backend-Lifecycle.md`.

### 3. Request mapping (`ModelSupport.cs` + the private `Build*Request` methods)

`ModelSupport` is a small static translation table: `Dictionary<string, Family>` mapping a
SwarmUI `T2IModel.ModelClass.CompatClass.ID` (e.g. `"stable-diffusion-xl-v1"`, `"flux-1"`,
`"wan-22-5b"`) to an Engine family id (e.g. `"sdxl"`, `"flux1"`, `"wan-22-5b"`) plus a
`Kind` (`Image` / `Video` / `Music`). It never hard-codes *whether* a family is drivable —
`IsArchitectureSupported` asks the Engine's `RecipeRegistry` / `VideoRecipeRegistry` at
call time, so the mapping can't drift out of sync with what the Engine actually ships.

The backend's `Build*Request` methods (`BuildImageRequest`, `BuildVideoRequest`,
`BuildMusicRequest`, plus their helpers `BuildLoras`, `BuildControlNets`, `BuildIpAdapter`,
`BuildRefiner`, `BuildRegional`, `BuildComponents`, `BuildReferenceImages/Videos/Audios`,
`BuildImageExtra`) are the **only** place in the extension that reads
`input.Get(T2IParamTypes.*)`. Each produces one Engine-native request record
(`ImageRequest` / `VideoRequest` / `MusicRequest`); anything the Engine's flat contract
doesn't name as a first-class field rides in the request's `Extra` dictionary under a key
documented in the Engine's `Features.RequestExtras`. A param the Engine has no way to
express at all is refused in `IsValidForThisBackend` rather than silently dropped.

ControlNet hints are preprocessed host-side before the request is built:
`ControlNetPreprocessing.DetectMode` picks the annotator (Canny / Depth-Anything / OpenPose
/ …) from the selected ControlNet checkpoint, and `ControlNetPreprocessing.Preprocess` runs
it over the hint image using the backend's preprocessor device. This is deliberate — the
Engine's contract expects an already-preprocessed hint, and the annotators need to consume
SwarmUI `Image` objects and fetch their own weights through Swarm's model-download
machinery, which only exists on this side of the boundary.

### 4. Web API (`WebAPI/HartsyInferenceWebAPI.cs`)

A handful of read-mostly HTTP routes with no ComfyUI equivalent, because an in-process
backend can answer "what will you do with this model?" and "what's resident right now?"
directly instead of proxying to a separate server: `HartsyInferenceGetSupportedArchs`,
`HartsyInferenceProbeModel`, `HartsyInferenceListLoadedPipelines`, `HartsyInferenceGetDeviceInfo`,
and the one mutating route, `HartsyInferenceClearCache`. Full request/response shapes are
in `08-Web-API-Routes.md`.

## Data flow: one generation (`GenerateLive`)

1. Swarm's `BackendHandler` picks this backend (`IsAlive` + `IsValidForThisBackend` +
   capacity) and calls `HartsyInferenceBackend.GenerateLive(input, batchId, takeOutput)`.
2. `_genLock.WaitAsync()` serializes generations on this instance — a backend with
   `OverQueue > 0` can have extra jobs dispatched to it, but the Engine's caches and device
   are shared state, so they still run strictly one at a time; queued jobs simply wait here.
3. `_cancelCts` is created linked to `input.InterruptToken`, so the generation page's Stop
   button actually cancels the in-flight Engine call.
4. The model's compat class resolves through `ModelSupport.Resolve` to a `Family`
   (id + `Kind`); an unsupported architecture throws immediately with
   `ModelSupport.WhyNotSupported`'s explanation. `ModelSupport.BuildSpec` turns the `T2IModel`
   into an Engine `ModelSpec` (family id as the catalog id, resolved checkpoint path).
5. Dispatch on `family.Kind`: `GenerateImage` calls `_engine.Images.GenerateAsync`,
   `GenerateVideo` calls `_engine.Video.GenerateAsync`, `GenerateMusic` calls
   `_engine.Music.GenerateAsync`. Each is handed the request built by the mapping layer
   (section 3 above), an `IProgress<StepPreview>` bridge, and the cancellation token.
6. If the Engine throws `OutOfVramException`, the backend calls `FreeMemory(false)` (the
   usual cause is a bigger model still cached from an earlier request) and retries once. A
   second `OutOfVramException` becomes a `SwarmReadableErrorException` built by
   `DescribeVramFailure`, which quotes the Engine's own geometry/byte-requirement message
   and adds actionable advice (lower resolution/frames, or configure a shard/LowVram setting).
7. Optional SeedVR2 restore pass: if the request set a Video Restore model, the still or
   video path routes the decoded pixels/frames through `_engine.Restore.RestoreAsync`
   before returning. The video path calls `_engine.FreeMemory()` first — the resident
   video DiT and SeedVR2's VAE peak can't both fit in VRAM at once.
8. The Engine's result metadata (`result.Meta` — things like the checkpoint-sniffed
   architecture that Swarm has no way to derive itself) is copied onto
   `input.ExtraMeta["hartsy_*"]`, plus `hartsy_engine_seed`. Raw RGB/frame/audio bytes are
   wrapped as a SwarmUI `Image` (`RgbToImage`, `VideoOutputEncoder`, or a WAV `Image` for
   music) and handed to `takeOutput` as a `T2IEngine.ImageOutput`.
9. In `finally`: the lock is released (unblocking the next queued job) and `Status` is left
   at `RUNNING` regardless of success or failure — the backend itself is healthy either way;
   only `ERRORED` means the backend can't serve requests at all.

Progress ticks use a custom `InlineProgress<T> : IProgress<T>` rather than the framework's
`Progress<T>`, which posts through a captured `SynchronizationContext` — that would reorder
or drop ticks fired from the sampler thread. Every tick is throttled to a 5%-boundary log
line and forwarded to `takeOutput` as either a preview `JObject` (when `PreviewEncoder` has
pixels to encode) or a plain `{batch_index, overall_percent, current_percent}` object.

## Why this structure (rationale still true in the code)

- **Thin mapper, not a workflow builder.** All architecture-specific behavior — which
  components a family needs, how LoRA/ControlNet/IP-Adapter compose, sampling defaults —
  lives in the Engine's recipes. The extension only has to know a compat-class string maps
  to a family id; it never has to know *how* that family generates. This is what keeps
  `ModelSupport.cs` at ~250 lines instead of growing a per-architecture loader for every
  new model family the Engine adds.
- **The honesty guard.** `IsValidForThisBackend` checks every composition feature a request
  would need against what the Engine's own recipe for that family declares
  (`IArchitectureRecipe.Supports`, or `VideoFeatures` for video, both queried through
  `ModelSupport`) and refuses by name when something isn't backed. This can never claim
  support the Engine doesn't have, because it is driven by the Engine's own declaration
  rather than a separately maintained list that could drift out of sync.
- **Advertising `"comfyui"` alongside `"hartsyinference"`.** When both a Comfy and a
  HartsyInference backend are configured, a request can pick up Comfy-tagged params (e.g.
  Sampler) that SwarmUI can't split across two backends. Without also advertising
  `"comfyui"`, such a request would match neither backend's feature set and refuse
  outright. Advertising it is only safe because of two things downstream:
  `ValidateComfyOnlyParams` refuses any actually-set Comfy-only param this backend can't
  service (routing the user to a Comfy backend instead), and
  `WarnOnUndeclaredFeatureFlags()` (entry-class section above) keeps this backend's own
  params honest against the same declared-features list. All three pieces work together;
  removing any one breaks the loop.
- **Annotation stays host-side.** The Engine's request contract expects an
  already-preprocessed ControlNet hint, not a raw image plus a mode enum — because the
  annotators need SwarmUI's `Image` type and Swarm's model-download machinery to fetch
  their own weights, neither of which the Engine depends on.
- **The preprocessor device is borrowed, not owned, when an Engine exists.** A second CUDA
  context on the same GPU would collide with the Engine's per-context state, and disposing
  a borrowed device would evict the Engine's resident weights mid-lifecycle — so
  `PreprocessBackend()` reuses `InferenceEngine.ComputeBackend` whenever one exists, and
  only creates (and owns) a standalone device when there is no Engine to borrow from.
- **No IPC on the generation path.** Swarm calls the Engine's C# methods directly — no
  subprocess, no HTTP, no JSON workflow graph to serialize. (This is scoped to
  generation: the optional auto-update check queries nuget.org, and Ideogram 4's magic
  prompt calls a running LLM backend — the extension isn't network-isolated in general.)
- **One backend type.** Unlike the ComfyUI extension's `comfyui_api` (remote) vs.
  `comfyui_selfstart` (auto-launch) split, there is exactly one HartsyInference backend
  type, because there is no remote mode — it always runs in-process. Per-instance settings
  (compute backend, device ordinals, placement) differentiate configured instances instead.

## What's out of scope for this doc

- What the Engine itself does once called (recipe internals, pipeline caching, model
  download) — `04-HartsyInference-Integration.md`.
- Feature gaps versus ComfyUI — `11-Comfy-Parity-Punchlist.md`.
- Open bugs and TODOs — `14-Known-Issues-And-TODO.md`.
