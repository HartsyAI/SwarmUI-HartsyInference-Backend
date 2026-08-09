# 04 — HartsyInference Integration

The contract between this extension and the `HartsyInference` engine: how it's
distributed, what the extension actually calls, and what's genuinely still out
of reach.

## Distribution model

The engine ships as a single NuGet meta-package, `HartsyInference`. It pulls
in every backend (CPU/CUDA/Vulkan) and every modality assembly
(`HartsyInference.Diffusion`, `.Video`, `.Audio`, `.Vision`, `.LLM`, `.World`,
`.ThreeD`, …) plus private deps (`Microsoft.ML.Tokenizers`, `Google.Protobuf`).
`SwarmUI-HartsyInference.csproj` references it as a plain
`<PackageReference>`:

```xml
<PackageReference Include="HartsyInference" Version="2.0.0-alpha.17" />
```

There is no git submodule — the extension's `Vendor/` folder contains only a
`.gitkeep`. The version is pinned to an **exact** prerelease, not a range: the engine's
own type surface (constructors, request records, feature enums) is the
contract, and it does shift between alpha builds, so a floating range would
let an incompatible engine silently restore under this extension. `RestoreNoHttpCache`
is set in the csproj so a freshly-published alpha isn't hidden behind NuGet's
~30-minute flat-container cache.

A `UseLocalHartsy=true` MSBuild property switches the `<PackageReference>`
for direct `<Reference>`s with `HintPath`s into a locally built
`HartsyInference.Cli` bin folder (the engine's full runtime closure), for
engine development against unpublished DLLs. This must never be the default —
end users only have nuget.org — and the csproj enforces that by defaulting it
`false`.

## What the extension consumes

The extension does not construct pipelines, load tokenizers/text-encoders/UNets/VAEs,
or partition checkpoints by tensor prefix. All of that lives in
`HartsyInference.Engine`, and the extension programs against exactly one
entry point: `HartsyInference.Engine.InferenceEngine`.

`Backends/HartsyInferenceBackend.cs` constructs one `InferenceEngine` per
backend instance and calls into its lazily-constructed services:

| Engine surface | Used for |
|---|---|
| `InferenceEngine(selector, options)` / `InferenceEngine(selector, ordinal, options)` | Construct the engine: compute backend (`auto`/`cpu`/`cuda`/`vulkan`), an explicit device ordinal when one is configured, low-VRAM policy, and multi-GPU `PlacementConfig` (text-encoder / VAE / CFG-parallel / shard device split) |
| `InferenceEngine.Images.GenerateAsync(ModelSpec, ImageRequest, IProgress<StepPreview>, CancellationToken)` | Still images |
| `InferenceEngine.Video.GenerateAsync(ModelSpec, VideoRequest, …)` | Video clips, returns `VideoGenerationResult` (frames + an optional `AudioBuffer` for the models that generate a synced soundtrack) |
| `InferenceEngine.Music.GenerateAsync(ModelSpec, MusicRequest, …)` | Music/audio, returns an encoded `AudioResult` |
| `InferenceEngine.Restore.RestoreAsync(ModelSpec, RestoreRequest, …)` | Optional SeedVR2 restore/upscale pass over an image or a video's frames |
| `InferenceEngine.FreeMemory()` | Evict cached pipelines (used on OOM retry and on Swarm's "free memory" action) |
| `InferenceEngine.BackendDescription` | Human-readable device string for the backend's load-status log |
| `BackendFactory.Validate(selector)` / `.Resolve(selector)` / `.Kind(selector)` | Eager device validation at `Init()` so a bad `GPU_ID` fails at backend startup, not on the first generation |
| `Engine.Recipes.RecipeRegistry` / `Video.WanVideoRecipe` / `VideoRecipeRegistry` | Per-family `ImageFeatures`/`VideoFeatures` flags (`Generation/ModelSupport.cs` reads these to answer "is this feature drivable for this architecture") |

Every `GenerateAsync` overload above takes a `CancellationToken`; the backend
links it to Swarm's `T2IParamInput.InterruptToken` so the UI's Stop button
cancels an in-flight generation.

The request DTOs (`ImageRequest`, `VideoRequest`, `MusicRequest`) are flat
records under `HartsyInference.Engine.Requests`. `Backends/HartsyInferenceBackend.cs`
builds one per generation from `T2IParamInput` — this is the only place in
the extension that reads `T2IParamTypes.*`. Composition features (LoRA stack,
ControlNet conditioning list, IP-Adapter, refiner, img2img/inpaint, regional
prompting, variation seed) are fields on `ImageRequest` itself; the engine
applies them inside the pipeline, not the extension.

`Generation/ModelSupport.cs` is the only architecture-mapping layer left: a
dictionary from SwarmUI's `CompatClass.ID` to an engine family id plus a
`Kind` (`Image`/`Video`/`Music`). It does not construct anything — whether a
family is actually drivable is asked of `RecipeRegistry`/`VideoRecipeRegistry`
at call time, so the mapping can't drift from what the engine will really do.

## LoRA application

`HartsyInference.Engine.Features.LoraApplier.BuildAndApply` merges a LoRA
stack directly into the weight dictionaries it's given (matching keys are
replaced with newly-allocated merged tensors owned by the returned
`LoraStack`, which the caller disposes once the components built from those
dicts are done). That mutation is real, but it never poisons a shared cache:
each per-family recipe (`SdxlRecipe`, `Flux1Recipe`, `Sd15Recipe`,
`MiniMaxH3Recipe`) re-reads and converts the checkpoint from disk into fresh
owned tensors on every `Construct()` call before handing them to
`BuildAndApply`, and the engine's pipeline cache key includes the LoRA stack
identity — a different LoRA selection is a cache miss that triggers a fresh
`Construct()` rather than reusing (or re-merging into) an already-merged
dictionary.

## Native dependencies

The NuGet package's own build targets copy the CUDA PTX kernels (`Ptx/`) and
Vulkan SPIR-V shaders (`Spirv/`) into the extension's output folder
automatically — no manual `<Content>` copy rules needed in this extension's
csproj. CPU is pure managed with no native deps. Per the engine's own
`README.nuget.md`: CUDA needs compute capability 8.0+ and CUDA 12.x/13.x
userspace libraries (FP8 paths need 8.9+); Vulkan needs a 1.3+ runtime.

## Real remaining gaps

These are refused or left undone in code today, not hypothetical:

| Gap | Where |
|---|---|
| No two-stage "generate an image, then animate it with a separate video model" flow (Comfy's `ImageToVideoGenInfo` / Video Model param) | `IsValidForThisBackend` refuses any non-video model with a `VideoModel` param set — no engine equivalent |
| Comfy prompt-syntax tags `<object:…>`, `<clear:…>`, `<embed:…>`, `<break>` | `IsValidForThisBackend`'s `UnsupportedPromptSyntax` regex refuses these outright rather than feeding the raw tag into a text encoder |
| No LoRA on music models | `ValidateMusic` refuses any LoRA selection for `acestep`/`musicgen`/`yue` |
| HunyuanVideo image-to-video conditioning | `Generation/ModelSupport.cs`: the `hunyuan-video` family maps only the text-to-video recipe; I2V conditioning is a recipe TODO, not a mapping gap |
| MiniMax-H3 fl2va vs ref2va task detection is filename-based | `ModelSupport.MiniMaxH3TaskFeatures` — the two checkpoint variants are byte-identical in key set and tensor shape, so there's no header sniff; an unrecognized filename keeps the full feature union rather than guessing |

See `11-Comfy-Parity-Punchlist.md` for the broader Comfy-parity feature list
and `14-Known-Issues-And-TODO.md` for open work tracked outside this contract.
Architecture-level design (why the extension is a thin mapper at all) is in
`01-Architecture.md`.
