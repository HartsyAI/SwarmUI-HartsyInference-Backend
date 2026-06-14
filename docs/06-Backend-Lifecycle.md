# 06 — Backend Lifecycle

How `HartsyInferenceBackend` (subclass of `AbstractT2IBackend`) handles its lifetime
and per-request flow.

## The backend contract (recap)

From `src/Backends/AbstractT2IBackend.cs` and `src/Backends/AbstractBackend.cs`:

| Member | Required? | Purpose |
|--------|-----------|---------|
| `Task Init()` | Yes | Construct backend, validate, set `Status` |
| `Task Shutdown()` | Yes | Tear down, free resources |
| `IEnumerable<string> SupportedFeatures` | Yes | Capability flags advertised to Swarm |
| `Task<Image[]> Generate(T2IParamInput input)` | Yes | One-shot generation |
| `Task GenerateLive(T2IParamInput, batchId, takeOutput)` | Optional but expected | Streaming generation w/ live updates |
| `Task<bool> LoadModel(T2IModel, T2IParamInput)` | Yes | Pre-load a checkpoint (used by Swarm to warm before generations) |
| `bool IsValidForThisBackend(T2IParamInput)` | Optional | Filter incompatible inputs early |
| `Task<bool> FreeMemory(bool systemRam)` | Optional | Release VRAM/RAM on demand |
| `volatile string CurrentModelName` | Yes | Tracks loaded model |
| `BackendStatus Status` | Inherited | DISABLED / ERRORED / WAITING / LOADING / IDLE / RUNNING |

## Settings record

Configured via Swarm's `AutoConfiguration` — saved to `Data/Backends.fds`. Defined as
a nested class on `HartsyInferenceBackend`:

```csharp
public class HartsyInferenceBackendSettings : AutoConfiguration
{
    [ConfigComment("Compute backend to use. 'auto' picks best available (CUDA > Vulkan > CPU).")]
    public string ComputeBackend = "auto"; // auto | cuda | vulkan | cpu

    [ConfigComment("Device ordinal (0 = first GPU). Ignored for CPU.")]
    public int DeviceOrdinal = 0;

    [ConfigComment("Default dtype for model weights. fp16 saves VRAM; fp32 is more accurate on CPU.")]
    public string DefaultDtype = "fp16"; // fp16 | bf16 | fp32

    [ConfigComment("Maximum number of pipelines to keep cached in memory.")]
    public int MaxCachedPipelines = 2;

    [ConfigComment("Tile VAE decode for outputs above this dimension (px). 0 = always tile.")]
    public int TiledVaeThreshold = 1024;

    [ConfigComment("Path to PTX kernel directory (CUDA only). Default = bundled.")]
    public string PtxDirectory = "";
}
```

## Init() — happy path

```csharp
public override async Task Init()
{
    AddLoadStatus("Resolving compute backend");
    string chosen = ResolveBackendChoice(Settings.ComputeBackend);
    AddLoadStatus($"Compute backend: {chosen}");

    AddLoadStatus("Constructing IBackend");
    _backend = chosen switch
    {
        "cuda" => new CudaBackend(Settings.DeviceOrdinal,
                                  string.IsNullOrEmpty(Settings.PtxDirectory)
                                    ? Path.Combine(AppContext.BaseDirectory, "Ptx")
                                    : Settings.PtxDirectory),
        "vulkan" => new VulkanBackend(/* ordinal TBC */),
        _ => new CpuBackend()
    };

    AddLoadStatus($"IBackend ready ({_backend.Capabilities.Name})");

    _cache = new PipelineCache(_backend, Settings.MaxCachedPipelines);

    Status = BackendStatus.IDLE;
    AddLoadStatus("Ready.");
}
```

`ResolveBackendChoice("auto")` probes available devices in order:
1. `try { new CudaBackend(0, ptxDir); return "cuda"; } catch { }`
2. `try { new VulkanBackend(); return "vulkan"; } catch { }`
3. `return "cpu";`

If `chosen` was an explicit setting and instantiation throws, we set `Status = ERRORED`,
record the exception in `LoadStatusReport`, and rethrow — the user sees it in
Server → Backends.

## Shutdown()

```csharp
public override async Task Shutdown()
{
    Status = BackendStatus.DISABLED;

    _cache?.DisposeAll();        // Disposes pipeline objects + frees weight tensors
    _cache = null;

    _backend?.Dispose();         // Releases CUDA context / Vulkan device
    _backend = null;

    CurrentModelName = null;
    await Task.CompletedTask;
}
```

The `using` blocks in HartsyInference's pipeline classes mean disposing the pipeline
also disposes the components. We rely on that.

## SupportedFeatures

```csharp
public override IEnumerable<string> SupportedFeatures
{
    get
    {
        yield return "hartsyinference";  // our own feature flag
        yield return "comfyui";          // claim Comfy parity for params marked with this flag
        yield return "refiners";
        yield return "endstepsearly";
        yield return "lora";

        // Phase-gated:
        if (PipelineSteps.HasArchitecture("Flux.1-dev")) yield return "flux";
        if (HasVaeEncoder()) yield return "img2img";   // returns false until phase 4
        if (HasControlNetWiring()) yield return "controlnet"; // false until phase 5
        // ... etc
    }
}
```

**Note on the `"comfyui"` flag.** Most Swarm parameters that we'd want to support are
registered with `FeatureFlag: "comfyui"` because Swarm's authors didn't anticipate
non-Comfy backends needing them. Advertising `"comfyui"` makes those params surface
for our backend too. This is a pragmatic choice — see
[`07-Parameters-And-Feature-Flags.md`](./07-Parameters-And-Feature-Flags.md) for the
nuance.

## LoadModel(model, input)

Called when Swarm wants to pre-warm a model — typically when the user picks a model
in the dropdown without yet hitting Generate.

```csharp
public override async Task<bool> LoadModel(T2IModel model, T2IParamInput input)
{
    if (CurrentModelName == model.Name) return true;

    Status = BackendStatus.LOADING;
    try
    {
        var ctx = new PipelineContext
        {
            UserInput = input ?? new T2IParamInput(/* defaults */),
            Backend = _backend,
            Model = model,
            CompatClass = model.ModelClass?.CompatClass ?? "unknown"
        };

        var loader = ResolveLoader(ctx.CompatClass);
        if (loader == null)
        {
            Logs.Warning($"[HartsyInference] Unknown architecture '{ctx.CompatClass}' for model {model.Name}");
            Status = BackendStatus.IDLE;
            return false;
        }

        await Task.Run(() => loader.Invoke(ctx));   // sync work on background thread

        CurrentModelName = model.Name;
        Status = BackendStatus.IDLE;
        return true;
    }
    catch (Exception ex)
    {
        Logs.Error($"[HartsyInference] LoadModel failed: {ex}");
        Status = BackendStatus.IDLE;
        return false;
    }
}
```

## Generate(input) and GenerateLive

`Generate` is just `GenerateLive` collecting the images:

```csharp
public override async Task<Image[]> Generate(T2IParamInput input)
{
    var images = new List<Image>();
    await GenerateLive(input, "single", obj =>
    {
        if (obj is Image img) images.Add(img);
    });
    return images.ToArray();
}
```

`GenerateLive` is where the action is:

```csharp
public override async Task GenerateLive(T2IParamInput input, string batchId, Action<object> takeOutput)
{
    Status = BackendStatus.RUNNING;
    try
    {
        // 1. Build context
        var model = input.Get(T2IParamTypes.Model);
        var ctx = new PipelineContext
        {
            UserInput = input,
            Backend = _backend,
            Model = model,
            CompatClass = model.ModelClass?.CompatClass ?? "unknown"
        };

        // 2. Run all registered steps
        foreach (var step in PipelineSteps.AllSteps())
        {
            step(ctx);
        }

        // 3. Drive the pipeline
        if (ctx.Execution?.Run == null)
            throw new InvalidOperationException("No pipeline step assembled an Execution.Run");

        // Convert HartsyInference progress callback → Swarm live updates
        long startMs = Environment.TickCount64;
        Action<GenerationProgress> progressBridge = p =>
        {
            // Cancellation check (phase 3 workaround)
            _cancelToken.ThrowIfCancellationRequested();

            takeOutput(new JObject
            {
                ["batch_index"] = batchId,
                ["overall_percent"] = (double)p.Step / p.TotalSteps,
                ["current_percent"] = (double)p.Step / p.TotalSteps,
                ["preview"] = null  // populated when in-flight latent decode is supported (phase 3)
            });
        };

        // 4. Run on a background thread (HartsyInference is sync)
        Image[] images = await Task.Run(() => ctx.Execution.Run(progressBridge));

        // 5. Yield results
        foreach (var img in images) takeOutput(img);

        // 6. Run any post-gen actions registered by extensions
        foreach (var action in ctx.PostGenActions) action();
    }
    finally
    {
        Status = BackendStatus.IDLE;
    }
}
```

## Cancellation

Until HartsyInference adds `CancellationToken` parameters to its pipeline methods
(see [`04-HartsyInference-Integration.md`](./04-HartsyInference-Integration.md) gap #3),
we use a stop-flag pattern:

- The backend holds a `CancellationTokenSource` per active generation.
- `progressBridge` calls `_cancelToken.ThrowIfCancellationRequested()` at every step.
- This terminates generation between denoising steps — granular to ~1 step.
- Swarm's standard cancel mechanism flows through `BackendData.Cancel()` → we expose
  this via overriding `BackendData.RequestCancel()` (need to confirm the exact hook
  name in Swarm core).

This isn't ideal — a 30-second step on Flux means up to 30s cancel latency. The fix
is upstream.

## IsValidForThisBackend(input)

Reject inputs we can't satisfy *before* Swarm's queue picks us:

```csharp
public override bool IsValidForThisBackend(T2IParamInput input)
{
    var model = input.Get(T2IParamTypes.Model);
    if (model == null)
    {
        input.RefusalReasons.Add("No model selected");
        return false;
    }

    string compat = model.ModelClass?.CompatClass;
    if (string.IsNullOrEmpty(compat) || ResolveLoader(compat) == null)
    {
        input.RefusalReasons.Add($"Architecture '{compat}' not supported by HartsyInference (yet)");
        return false;
    }

    // Img2img / inpaint requires VaeEncoder — phase 4
    if (input.TryGet(T2IParamTypes.InitImage, out _) && !HasVaeEncoder())
    {
        input.RefusalReasons.Add("img2img not yet supported in HartsyInference (VaeEncoder pending)");
        return false;
    }

    return true;
}
```

This makes Swarm fall back to a different backend (e.g. ComfyUI) automatically when
we can't service a request, instead of erroring at generation time.

## FreeMemory(systemRam)

```csharp
public override async Task<bool> FreeMemory(bool systemRam)
{
    bool freedAnything = false;

    if (_cache != null)
    {
        freedAnything |= _cache.EvictAll();
    }

    // FreeWeights is the IBackend method that reclaims tensor allocations
    _backend?.FreeWeights();

    if (systemRam)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    return freedAnything;
}
```

`PipelineCache.EvictAll()` returns true if any entries were dropped.

## Backend status state machine

> **Important — Swarm semantics, not process-scheduler semantics:**
> - `RUNNING` = **alive and ready to accept generations** (the resting healthy state).
> - `IDLE` = **alive but currently unavailable** (e.g. a remote API backend whose
>   endpoint is unreachable; it might recover). `BackendHandler` *excludes* IDLE
>   backends from request routing and feature aggregation.
> - Status does **not** track per-generation utilization. That's `BackendData.Usages`.
>   A backend with an active generation stays at `Status = RUNNING`; `Usages`
>   counts up to `MaxUsages`, then `CheckIsInUse` blocks new dispatches.

```
            ┌──────────────────┐
            │     WAITING      │  (initial state, before Init)
            └─────────┬────────┘
                      │ Init() called
                      ▼
            ┌──────────────────┐
            │     LOADING      │  (during heavy work — Init, big model load)
            └─────────┬────────┘
              success │ error
              ┌───────┴───────┐
              ▼               ▼
       ┌──────────┐     ┌──────────┐
       │ RUNNING  │     │ ERRORED  │  (init failed; not retryable from here)
       └────┬─────┘     └──────────┘
            │
            │ user toggles off               ┌──────────┐
            ├───────────────────────────────►│ DISABLED │
            │                                └──────────┘
            │
            │ recoverable failure mode       ┌──────────┐
            │ (e.g. remote API gone)         │   IDLE   │
            ├───────────────────────────────►│ — alive, │
            │                                │   not    │
            │ recovers                       │ usable   │
            │◄───────────────────────────────┤          │
            │                                └──────────┘
```

For our in-process HartsyInference backend, `IDLE` is essentially never the right
state — we don't have a remote endpoint that can drop. `LoadModel` enters `LOADING`
and returns to `RUNNING`; `GenerateLive` stays `RUNNING` throughout (utilization
flows through `Usages`); a failed generation leaves Status at `RUNNING` because the
*backend* is still healthy, only the request errored.
