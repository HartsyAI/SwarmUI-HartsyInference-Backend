# 05 — Pipeline Translation

How we convert a Swarm `T2IParamInput` into a HartsyInference pipeline call. This is
the architectural equivalent of `ComfyUIBackend.WorkflowGenerator` —  but instead
of producing a JSON DAG, we produce a `PipelineExecution` C# object.

## The objects

```csharp
// Generation/PipelineContext.cs
public class PipelineContext
{
    public T2IParamInput UserInput { get; init; }
    public IBackend Backend { get; init; }
    public T2IModel Model { get; init; }
    public string CompatClass { get; init; } // "stable-diffusion-v1", "stable-diffusion-xl-v1-base", "Flux.1-dev", ...

    // Mutable state filled by steps
    public PipelineExecution Execution { get; set; } = new();
    public PipelineCacheEntry CacheEntry { get; set; }
    public List<Action> PostGenActions { get; } = new();

    // Filled by per-architecture model loader steps
    public object Pipeline { get; set; }                // e.g. StableDiffusion15Pipeline
    public Dictionary<string, Tensor> UnetWeights { get; set; }
    public Dictionary<string, Tensor> ClipLWeights { get; set; }
    public Dictionary<string, Tensor> ClipGWeights { get; set; }
    public Dictionary<string, Tensor> VaeWeights { get; set; }
    public ClipTokenizer TokenizerL { get; set; }
    public ClipTokenizer TokenizerG { get; set; }
    public T5Tokenizer TokenizerT5 { get; set; }
}

public class PipelineExecution
{
    public Func<Action<GenerationProgress>, Image[]> Run { get; set; }
    // Optional: partial-state hooks
    public Action OnCancel { get; set; }
}
```

## The step machine

Mirroring `WorkflowGenerator.AddStep(action, priority)`:

```csharp
// Generation/PipelineSteps.cs
public static class PipelineSteps
{
    private static readonly List<(double priority, Action<PipelineContext> step)> _steps = new();
    private static readonly List<(double priority, Action<PipelineContext> step)> _modelLoaders = new();
    private static readonly object _lock = new();

    public static void AddStep(Action<PipelineContext> step, double priority)
    {
        lock (_lock) _steps.Add((priority, step));
    }

    public static void AddModelLoader(Action<PipelineContext> step, double priority)
    {
        lock (_lock) _modelLoaders.Add((priority, step));
    }

    internal static IEnumerable<Action<PipelineContext>> AllSteps()
    {
        lock (_lock)
        {
            return _modelLoaders.Concat(_steps)
                .OrderBy(t => t.priority)
                .Select(t => t.step)
                .ToList();
        }
    }
}
```

## Step priorities (mirroring Comfy convention)

| Priority | Purpose |
|----------|---------|
| `< -100` | Pre-model-loading. Custom architectures register their loader at e.g. `-200` to run before built-in model detection. |
| `-100` to `-1` | Built-in per-architecture model loaders. Each one gates on `ctx.CompatClass == "..."`. Sets `ctx.Pipeline` and component fields. |
| `0` | Tokenization, prompt embedding prep. Runs after the pipeline + tokenizer are loaded. |
| `1–5` | Adapter application (LoRAs, ControlNet conditioning prep). |
| `5–9` | Final pipeline build — assembles `PipelineExecution.Run`. |
| `10` | Output / save hooks (we typically just return the image; Swarm core saves). |

Architecture-specific loader gates on `ctx.CompatClass`. Example for SD1.5:

```csharp
PipelineSteps.AddModelLoader(ctx =>
{
    if (ctx.CompatClass != "stable-diffusion-v1") return;

    var entry = HartsyInferenceBackend.Cache.Get(ctx.Backend, ctx.Model.Name);
    if (entry == null)
    {
        entry = LoadSd15Components(ctx);
        HartsyInferenceBackend.Cache.Put(ctx.Backend, ctx.Model.Name, entry);
    }

    ctx.CacheEntry = entry;
    ctx.Pipeline = entry.Pipeline;
    ctx.UnetWeights = entry.UnetWeights;
    ctx.ClipLWeights = entry.ClipLWeights;
    ctx.VaeWeights = entry.VaeWeights;
    ctx.TokenizerL = entry.TokenizerL;
}, -50);
```

## The architecture detection table

Maps Swarm's `T2IModelClass.CompatClass` (which is what Comfy already uses to
identify a checkpoint) to the HartsyInference pipeline. Defined in
`Generation/ModelSupport.cs`:

| `CompatClass` (from Swarm core) | Pipeline | Components |
|---------------------------------|----------|------------|
| `stable-diffusion-v1` | `StableDiffusion15Pipeline` | ClipTokenizer, ClipTextEncoder(Sd15), UNet(Sd15), VaeDecoder(Sd15) |
| `stable-diffusion-v1-inpainting` | `StableDiffusion15Pipeline` (inpaint mode) | + mask handling |
| `stable-diffusion-xl-v1-base` | `SdxlPipeline` | 2× ClipTokenizer, 2× ClipTextEncoder, UNet(SdxlBase), VaeDecoder(Sdxl) |
| `stable-diffusion-xl-v1-refiner` | `SdxlPipeline` w/ refiner UNet | Smaller UNet variant |
| `stable-diffusion-v3-medium` | `Sd3Pipeline` | ClipL + ClipG + T5, Sd3Transformer, VaeDecoder(Sd3) |
| `Flux.1-dev` / `Flux.1-schnell` | `FluxPipeline` | ClipL + T5, FluxTransformer, VaeDecoder(Flux) |
| `Flux.2-dev` | `Flux2Pipeline` | LlamaStyleEncoder, Flux2Transformer, VaeDecoder(Flux2) |
| `aura-flow-v1` | `AuraFlowPipeline` | T5, AuraFlowTransformer, VaeDecoder(AuraFlow) |
| `chroma-v1` | `ChromaPipeline` | T5, ChromaTransformer, VaeDecoder(Chroma) |
| `Z-image-v1` | `ZImagePipeline` | (TBC) |
| `hunyuan-image-v1` | `HunyuanImagePipeline` | (TBC) |

Each row is one `AddModelLoader` registration in `Generation/ModelSupport.cs`.

## All-in-one checkpoint partitioning

Swarm checkpoints typically bundle text encoder + UNet + VAE in one .safetensors.
HartsyInference's components want their tensors as separate dicts. Our loader
partitions by key prefix:

| Prefix | Component (SD1.5/SDXL) |
|--------|-----------------------|
| `cond_stage_model.transformer.text_model.*` | CLIP-L (Sd15 / SdxlClipL) |
| `conditioner.embedders.0.transformer.text_model.*` | CLIP-L (SDXL) |
| `conditioner.embedders.1.model.*` | CLIP-G (SDXL) |
| `model.diffusion_model.*` | UNet |
| `first_stage_model.*` | VAE |

For Flux / SD3 the prefixes differ (e.g. `text_encoders.clip_l.*`,
`text_encoders.t5xxl.*`, `model.transformer.*`, `vae.*`). Each architecture handler
owns its own partition function — we don't try to do it generically.

## Steps in execution order (typical SDXL t2i)

1. `[-50]` SDXL model loader — gates on CompatClass, fills cache entry.
2. `[0]` Tokenize prompt + negative prompt with both CLIP-L and CLIP-G tokenizers.
3. `[1]` LoRA application — if any LoRAs in `T2IParamInput.Loras`, build a `LoraStack`,
   call `ApplyToWeights(backend, unetWeights: ctx.UnetWeights, clipLWeights: ctx.ClipLWeights, clipGWeights: ctx.ClipGWeights)` on a working copy (see LoRA cache strategy below).
4. `[2]` ControlNet adapter prep (when wired upstream — phase 5). Loads the
   ControlNet weights, attaches to pipeline.
5. `[5]` Build `PipelineExecution.Run` — closures over `pipeline`, tokens, `TextToImageRequest`.
6. `[10]` Return.

User-installed extensions could `AddStep` at e.g. `4.5` to inject behaviour between
ControlNet prep and pipeline build, mirroring how third-party Comfy extensions
extend `WorkflowGenerator`.

## LoRA cache strategy (decision pending — phase 2)

`LoraStack.ApplyToWeights` mutates the weight dicts. If we apply LoRAs to the cached
UNet weights, the next generation that *doesn't* want those LoRAs would get them
applied anyway.

**Option A — Re-apply per generation, deep-copy first.**
Every generation deep-copies `ctx.CacheEntry.UnetWeights` into a working dict, applies
LoRAs there. Cost: large memory copy each gen (~3 GB for SDXL UNet in fp16). Bad.

**Option B — Keep two weight dicts in cache: pristine and last-applied.**
Cache holds `PristineWeights` (never touched) and `WorkingWeights` (mutated). On each
gen, compare requested LoRA fingerprint to last-applied fingerprint:
- Same fingerprint → no-op
- Different → reset `WorkingWeights` from `PristineWeights` (reload? or undo last LoRA?), then apply new stack
The reset is the painful part — undoing a destructive merge requires storing pristine.

**Option C — PR a non-destructive `ApplyOverride` upstream.**
Add a HartsyInference API that takes a base weight dict + LoRA stack and produces
runtime-merged tensors *during* the forward pass (or as a separate output dict).
Cleanest long-term, requires upstream work.

**Option D — Reload from disk on LoRA change.**
Simplest but slow on disk-bound systems.

**Recommendation:** start with **Option D** (correct, simple) in phase 2; switch to
**Option C** in parallel as an upstream PR. Option B is a possible interim if D is
unacceptably slow.

## Mapping `T2IParamInput` → `TextToImageRequest`

```
T2IParamInput.Get(T2IParamTypes.Prompt)         → TextToImageRequest.Prompt
T2IParamInput.Get(T2IParamTypes.NegativePrompt) → TextToImageRequest.NegativePrompt
T2IParamInput.Get(T2IParamTypes.Steps)          → TextToImageRequest.Steps
T2IParamInput.Get(T2IParamTypes.CFGScale)       → TextToImageRequest.CfgScale
T2IParamInput.Get(T2IParamTypes.Width)          → TextToImageRequest.Width
T2IParamInput.Get(T2IParamTypes.Height)         → TextToImageRequest.Height
T2IParamInput.Get(T2IParamTypes.Seed)           → TextToImageRequest.Seed (-1 → null)
T2IParamInput.Get(T2IParamTypes.Sampler)        → mapped via SamplerMap → TextToImageRequest.Scheduler
```

`SamplerMap` (defined in `Generation/SamplerMap.cs`):

| Swarm sampler name | HartsyInference scheduler string |
|--------------------|-------------------------------|
| `euler` | `"euler"` |
| `euler_ancestral` | `"euler_a"` (TBC — confirm HartsyInference name) |
| `dpmpp_2m` | `"dpm++_2m"` |
| `ddim` | `"ddim"` |
| `lcm` | `"lcm"` |
| `(Flux)` | `"flow_match_euler"` (auto-selected for Flux pipelines) |

Unmapped samplers fall back to `null` (= pipeline default) with a warning logged.

## How extension authors plug in

Following the Comfy convention:

```csharp
// In a third-party extension's OnInit:
PipelineSteps.AddStep(ctx =>
{
    if (ctx.UserInput.TryGet(MyCoolParam, out int blurAmount))
    {
        ctx.PostGenActions.Add(() => /* post-process the image */);
    }
}, 9.5);
```

We don't expose `WGNodeData` equivalents — there's no DAG to splice into. Instead,
extensions can:

- Mutate `ctx.UnetWeights` etc. before the pipeline runs (e.g., a custom adapter)
- Append a callback to `ctx.PostGenActions` to run after generation
- Replace `ctx.Pipeline` entirely with their own object that exposes a compatible
  `Generate` shape (advanced)

## Diff vs ComfyUI's WorkflowGenerator

| Aspect | Comfy `WorkflowGenerator` | This `InferencePipeline` |
|--------|--------------------------|-------------------------|
| Output | JSON DAG of node IDs | C# `PipelineExecution` closure |
| Step priority | Yes, sorted ascending | Yes, same convention |
| Per-architecture extension point | `AddModelGenStep` | `AddModelLoader` (just sugar over `AddStep` at low priority) |
| Parameter substitution | `${param_name}` template tags | Direct C# property access on `ctx.UserInput` |
| Cross-step state | `g.LoadingModel`, `g.FinalImageOut`, `g.CurrentVae` (node-id arrays) | `ctx.Pipeline`, `ctx.UnetWeights`, etc. (typed C# fields) |
| Caching | Per-workflow (Comfy server caches loaded models) | Our `PipelineCache` |
| Validation | Comfy server validates JSON, can reject | We type-check at compile time |
