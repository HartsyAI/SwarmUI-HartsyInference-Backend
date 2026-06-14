# 07 — Parameters and Feature Flags

How parameters surface in the Swarm UI for our backend, and what feature flags we
advertise.

## Two questions, two mechanisms

| Question | Mechanism |
|----------|-----------|
| "What parameters does the user see?" | Parameter registration (`T2IParamTypes.Register<T>`) plus feature-flag gating |
| "Which backend can service which generation?" | `SupportedFeatures` plus `IsValidForThisBackend` |

## Parameter ownership

There are three buckets of parameters:

### A. Core Swarm parameters (we don't own these)

Prompt, negative prompt, width, height, steps, CFG, seed, model, sampler. These are
registered in Swarm core with no `FeatureFlag` (or with the universal `"text2image"`
flag). They surface for any backend.

### B. ComfyUI-extension parameters that we want to support

The ComfyUI backend extension registers ~60 params with `FeatureFlag: "comfyui"`:
LoRAs, ControlNet inputs, IP-Adapter weights, refiner toggles, IP-Adapter weight type,
free-U values, etc. These were tagged `"comfyui"` because Swarm's authors didn't
foresee non-Comfy backends needing them.

**Initial recommendation was to advertise `"comfyui"` ourselves. After reading the
mechanism end-to-end ([T2IEngine.cs:100-108](../../../Text2Image/T2IEngine.cs),
[T2IParamInput.cs:664-678](../../../Text2Image/T2IParamInput.cs)), that was wrong.**

Here's why: a param's `FeatureFlag` becomes a hard requirement at backend-selection
time. If we claim `"comfyui"`, Swarm will route every comfyui-tagged param request
to us, including ones we cannot service (Sampler, Scheduler, RefinerUpscaleMethod,
custom workflow IR, FreeU, etc. — all tagged `"comfyui"`). We'd silently ignore
them at gen time. That's the worst kind of UX bug — the user thinks their setting
took effect, but the image came out the same.

The honest pattern is:

1. **Advertise only flags we genuinely service.** Mirror Comfy's
   [`NodeToFeatureMap`](../../../BuiltinExtensions/ComfyUIBackend/ComfyUIBackendExtension.cs)
   pattern: at Init, probe what HartsyInference can actually do, declare those
   specific flags.
2. **Accept that we lose params we can't support.** Without `"comfyui"`, the UI
   hides Sampler/Scheduler/etc. when HartsyInference is the only configured backend.
   That's the correct UX — those params don't have a home in our pipeline today.
3. **Coexist gracefully with Comfy.** When both backends are configured, the UI
   shows the union of features. Users who pick comfyui-only options route to Comfy
   automatically via `T2IEngine`'s flag check.
4. **As capabilities ship, add the specific flag.** `"lora"`, `"refiners"`,
   `"endstepsearly"`, `"img2img"`, `"inpaint"`, `"controlnet"`, `"ipadapter"` —
   each only when the code path genuinely works.

This matches the [`docs/Making Extensions.md`](../../../../docs/Making%20Extensions.md)
guidance and the working API Backends extension's pattern.

### C. Parameters specific to HartsyInference

Cases where HartsyInference exposes a setting that has no Comfy equivalent — e.g.,
quantization mode, FP8 scaling, scheduler-specific options.

We register these under our own group with `FeatureFlag: "hartsyinference"`:

```csharp
public static T2IRegisteredParam<string> HartsyInferenceDtype;
public static T2IRegisteredParam<int> HartsyInferenceTilesize;

public static T2IParamGroup HartsyInferenceParamGroup;

public override void OnInit()
{
    HartsyInferenceParamGroup = new("HartsyInference", Toggles: false, Open: false, IsAdvanced: true);

    HartsyInferenceDtype = T2IParamTypes.Register<string>(new(
        "HartsyInference Dtype",
        "Per-generation dtype override. fp16 (default), bf16, fp32.",
        "fp16",
        Toggleable: true,
        Group: HartsyInferenceParamGroup,
        FeatureFlag: "hartsyinference",
        GetValues: _ => new[] { "fp16", "bf16", "fp32" }));
}
```

## Feature flags advertised by `SupportedFeatures`

The full list (subject to phase progress — see
[`02-Comfy-Feature-Parity-Matrix.md`](./02-Comfy-Feature-Parity-Matrix.md)):

| Flag | What it gates | Phase we'll claim it |
|------|---------------|----------------------|
| `hartsyinference` | Our own settings group | 1 (active) |
| `text2image` | Universal text-to-image params | 1 (active) |
| `flux-dev` | FluxGuidanceScale param. In `DisregardedFeatureFlags` so advertising is informational — backend selection doesn't enforce it | 1 (active) |
| `lora` | LoRA list | 2 |
| `refiners` | Refiner workflow params | 3 |
| `endstepsearly` | End-step early-out | 3 |
| `img2img` / `init_image` | Init image / denoise strength | 4 (after VaeEncoder) |
| `inpaint` | Mask + inpaint | 4 |
| `controlnet` | ControlNet inputs | 5 (after upstream wiring) |
| `ipadapter` | IP-Adapter inputs | 5 |
| `variation_seed` | Second seed for blended noise | 5+ (needs upstream) |
| `freeu` | FreeU multipliers | 5+ |
| `seamless` | Seamless tiling | 5+ |
| `video` | Video models | — (out of scope for this extension) |
| `yolov8` | YOLO post-processing | — (separate extension) |
| `comfyui` | **Never.** This means "I can service any Comfy-extension-registered param" — including Sampler, Scheduler, Refiner, workflow IR. We can't. |

A feature flag is **only** advertised when we can actually service requests gated by
it. Lying ("we support controlnet!" then erroring at generation time) is worse than
being honest ("we don't yet, please use ComfyUI for this").

The `IsValidForThisBackend` method is the second line of defense — even if a flag
is advertised, a specific input might still be unservable (wrong architecture,
missing component) and we should reject early.

## Permissions

Mirror the `APIBackendsPermissions` / `ComfyUIBackendExtension` pattern. All
permissions live in the extension entry file:

```csharp
public static class HartsyInferencePermissions
{
    public static readonly PermInfoGroup Group = new(
        "HartsyInference",
        "Permissions for the pure-C# HartsyInference backend.");

    public static readonly PermInfo PermUseHartsyInference = Permissions.Register(new(
        "use_hartsyinference",
        "Use HartsyInference backend",
        "Allows generating images using the in-process HartsyInference backend.",
        PermissionDefault.POWERUSERS, Group));

    public static readonly PermInfo PermAdminHartsyInference = Permissions.Register(new(
        "admin_hartsyinference",
        "Administer HartsyInference",
        "Allows clearing the pipeline cache, probing models, and managing devices.",
        PermissionDefault.ADMINS, Group));
}
```

The first gates whether `HartsyInferenceBackend.IsValidForThisBackend` returns true at
all for a user. The second gates the admin-only HTTP routes (see
[`08-Web-API-Routes.md`](./08-Web-API-Routes.md)).

## Parameter-flag examples

### LoRAs (already registered by Comfy extension)

The `Loras` parameter is registered by the Comfy extension with `FeatureFlag: "comfyui"`.
By advertising `"comfyui"` in our `SupportedFeatures`, the LoRA selector surfaces for
us automatically. We pick it up in `Generation/PipelineSteps.cs`:

```csharp
PipelineSteps.AddStep(ctx =>
{
    if (!ctx.UserInput.TryGet(ComfyUIBackendExtension.LoraParam, out var loras))
        return;

    var stack = new LoraStack();
    foreach (var loraName in loras)
    {
        stack.AddFromPath(ResolveLoraPath(loraName), GetWeight(loraName));
    }

    ApplyLorasToContext(ctx, stack);
}, 1);
```

We **read** the parameter that the Comfy extension **registered**. We don't
re-register it. This is the same pattern the API-Backends extension uses (it reads
core params like `T2IParamTypes.Width` via the `=>` alias trick).

### Sharpinference-only: Dtype override

```csharp
HartsyInferenceDtype = T2IParamTypes.Register<string>(new(
    "HartsyInference Dtype",
    "Override the model's loaded dtype. fp16 = saves VRAM, fp32 = max accuracy on CPU.",
    "fp16", Toggleable: true,
    Group: HartsyInferenceParamGroup,
    FeatureFlag: "hartsyinference",
    GetValues: _ => new[] { "fp16", "bf16", "fp32" }));
```

Because the flag is `"hartsyinference"`, this control only appears when our backend is selected.

## What we explicitly don't surface

- Comfy workflow editor: it's a tab, not a param. No way to surface a workflow
  editor without a workflow IR. We don't have one.
- Stored Custom Workflows: same.
- TensorRT compile button: not applicable.

## Parameter validation

Validation that requires backend awareness (e.g., "FP8 requires Ada Lovelace+ GPU")
lives in our `Generation/PipelineSteps.cs` early-priority steps. They throw with
descriptive errors that Swarm surfaces to the user.

Parameter validation that's input-shape (e.g., "width must be multiple of 8") lives
where Swarm core puts it — at the `T2IParamType` level, with a `Validator`. We don't
override.
