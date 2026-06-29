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
LoRAs, ControlNet inputs, IP-Adapter weights, refiner toggles, the alternate-guidance
node family, custom-workflow IR, etc. These were tagged `"comfyui"` because Swarm's
authors didn't foresee non-Comfy backends needing them.

**We DO advertise `"comfyui"`. An earlier version of this doc argued against it — that
was wrong, and it actively broke generation when a Comfy backend was also configured.**

Why advertising is required: a param's `FeatureFlag` becomes a *hard requirement* on
the request at backend-selection time ([T2IEngine.cs:100-108](../../../Text2Image/T2IEngine.cs),
[T2IParamInput.cs:671-685](../../../Text2Image/T2IParamInput.cs)). The required flags
are the **union** across every param actually sent, and SwarmUI cannot split one
request across two backends. So when Comfy and HartsyInference are both installed, the
shared UI sends both families' params and a single request carries both `"comfyui"`
and `"hartsyinference"`. If we don't advertise `"comfyui"`, no single backend covers
the union → the generation is refused outright (Comfy lacks `hartsyinference`, we lack
`comfyui`). That deadlock is the bug this design fixes.

The honest pattern is therefore a **two-layer** one:

1. **Advertise `"comfyui"` so requests reach us.** We declare `"comfyui"` so comfyui-tagged
   requests are not pre-filtered away before our validator runs. This mirrors the built-in
   peer [`SwarmSwarmBackend`](../../../Backends/SwarmSwarmBackend.cs), which also advertises
   `"comfyui"`. It is execution-safe: core never routes us through Comfy's workflow
   builder on the basis of this flag — each backend runs its own `Generate`.
2. **Enforce honesty in `IsValidForThisBackend`.** A flag-driven guard there refuses —
   and cleanly routes to a Comfy backend — any comfyui-tagged param actually set that we
   can't service. The guard iterates the params present on the request, and refuses any
   `"comfyui"`-flagged one that isn't in a small allow-list of params we genuinely honor
   (Sampler, Scheduler, Refiner Sampler/Scheduler, Refiner Upscale Method — see
   `HonoredComfyParams`). Custom-workflow IR (`comfyworkflowraw` /
   `comfyuicustomworkflow`) is refused explicitly. So advertising `"comfyui"` does **not**
   mean "we silently serve everything Comfy-tagged."

This satisfies [`docs/Making Extensions.md`](../../../../docs/Making%20Extensions.md)
Standards #2 (non-breakage of core) and #3 (self-containment / "just works"), without
touching `DisregardedFeatureFlags` (which Standard #4 forbids an extension from doing).

### Proposed core cleanup: a `standard_sampling` split

The root cause is that core's `"comfyui"` flag is *overloaded*: it means both "the
backend-agnostic Sampler/Scheduler/Refiner-override param family" and "real ComfyUI
custom-workflow capability." A peer backend that only wants the former is forced to claim
the latter (the small white lie above).

The clean upstream fix would be to split a `"standard_sampling"` flag out of `"comfyui"`
(move the 5 standard sampler/refiner params onto it and add it to Comfy's
`FeaturesSupported`). A peer backend could then advertise `"standard_sampling"` to service
sampling params **without** claiming `"comfyui"`, so genuinely-Comfy-only requests route to
Comfy purely by the flag filter — no white lie, no guard reliance.

This is **not implemented** — it's a core change to raise with upstream. Our extension does
not depend on it: the `"comfyui"` + validator-guard approach above is fully self-contained
on stock SwarmUI. If/when core gains `"standard_sampling"`, advertise it in `SupportedFeatures`
to drop the lie.

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
| `comfyui` | Advertised so comfyui-tagged requests reach our validator instead of being pre-filtered out (the coexistence fix). We do **not** silently serve everything Comfy-tagged — the `IsValidForThisBackend` guard refuses comfyui-only params we can't run (custom workflow IR, alternate-guidance nodes, GLIGEN, SAM2, etc.) and routes them to Comfy. | 1 (active) |

With the exception of the broad `comfyui` routing flag (kept honest by the validator
guard), a capability flag is **only** advertised when we can actually service requests
gated by it. Lying ("we support controlnet!" then erroring at generation time) is worse
than being honest ("we don't yet, please use ComfyUI for this").

The `IsValidForThisBackend` method is the second line of defense — even if a flag
is advertised, a specific input might still be unservable (wrong architecture,
missing component) and we should reject early.

## Coexistence model (Comfy + HartsyInference both installed)

How a request routes when both backends exist:

| Request | Routes to |
|---------|-----------|
| Plain gen, model only HartsyInference has (e.g. nvfp4 Ideogram) | HartsyInference (model-availability filter) |
| Plain gen, model both backends have | Either — load-balanced. Pin one with the **Backend Type** or **Exact Backend ID** advanced params |
| Custom ComfyUI workflow, or any comfyui-only param we can't run | Comfy (our `IsValidForThisBackend` guard refuses and routes there) |
| Sampler / Scheduler / Refiner sampler set | Either — we honor these (`HonoredComfyParams` allow-list) |

Two invariants make this work:

- **The Ideogram Magic Prompt param is `Toggleable`** (opt-in). A non-toggleable flagged
  param is sent on *every* request, which would force `"hartsyinference"` onto unrelated
  generations and refuse the Comfy backend. Any HartsyInference param carrying a
  `FeatureFlag` must be `Toggleable` for this reason.
- **We never block.** If we can't service a request we add a clear `RefusalReason` and
  return false, which routes the request to a Comfy backend if one is configured, rather
  than failing the generation.

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
