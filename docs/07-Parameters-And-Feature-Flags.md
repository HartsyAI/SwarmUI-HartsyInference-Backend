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

Prompt, negative prompt, width, height, steps, CFG, seed, model, LoRAs. These are
registered in Swarm core with no `FeatureFlag` (or with the universal `"text2image"`
flag). They surface for any backend.

### B. ComfyUI-extension parameters that we want to support

The ComfyUI backend extension registers ~60 params with `FeatureFlag: "comfyui"`:
Sampler/Scheduler, ControlNet inputs, IP-Adapter weights, refiner toggles, the
alternate-guidance node family, custom-workflow IR, etc. These were tagged `"comfyui"`
because Swarm's authors didn't foresee non-Comfy backends needing them.

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
   peer [`SwarmSwarmBackend`](../../../Backends/SwarmSwarmBackend.cs), whose `SupportedFeatures`
   forwards its remote's flag set — `"comfyui"` included whenever the remote has a Comfy backend.
   It is execution-safe: core never routes us through Comfy's workflow builder on the basis of
   this flag — each backend runs its own `Generate`.
2. **Enforce honesty in `IsValidForThisBackend`.** A flag-driven guard there (`ValidateComfyOnlyParams`)
   refuses — and cleanly routes to a Comfy backend — any comfyui-tagged param actually set that we
   can't service. The guard iterates the params present on the request, and refuses any
   `"comfyui"`-flagged one that isn't in a small allow-list of params we genuinely honor
   (`HonoredComfyParams`): Sampler, Scheduler, Refiner Sampler/Scheduler, Refiner Upscale
   Method, the FLUX.1 Redux style-model strengths (merge/multiply/apply-start), and the
   IP-Adapter scheduling knobs (weight/start/end/weight-type) — mapped onto the engine's
   `redux.*` / `ipadapter.*` Extra keys respectively. Custom-workflow IR (`comfyworkflowraw` /
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

Cases where HartsyInference exposes a setting that has no Comfy equivalent. The pattern is
always the same: declare a `T2IParamGroup`, then `T2IParamTypes.Register<T>` each param into
it with `Toggleable: true` and `FeatureFlag: "hartsyinference"` (see the invariant below on
why `Toggleable` is not optional). A minimal example:

```csharp
public static T2IParamGroup HartsyInferenceParamGroup;
public static T2IRegisteredParam<string> SamplerParam;

public override void OnInit()
{
    HartsyInferenceParamGroup = new("HartsyInference", Toggles: false, Open: false, IsAdvanced: true);

    SamplerParam = T2IParamTypes.Register<string>(new(
        "HartsyInference Sampler",
        "Sampler for SD 1.5 / SDXL generations on the HartsyInference backend.",
        "euler",
        Toggleable: true,
        Group: HartsyInferenceParamGroup,
        FeatureFlag: "hartsyinference",
        GetValues: _ => new List<string> { "euler", "ddim", "dpm++2m", "lcm" }));
}
```

In practice `SwarmUIHartsyInference.OnInit` registers several dozen params across a handful
of groups, one per feature area rather than one flat list:

- **`HartsyInferenceParamGroup`** — the catch-all: `HartsyInference Sampler`, `HartsyInference
  Init Image Mode`, and the Wan-Animate conditioning inputs (reference image, auto-preprocess
  toggle, pose/face driving-video overrides, video audio reference).
- **`Ideogram4ParamGroup`** — the Ideogram 4 magic-prompt toggle and its optional LLM-model
  override (`Generation.Ideogram4MagicPrompt`).
- **`VideoRestoreParamGroup`** — SeedVR2 restore/upscale knobs (model, target width/height,
  clip frames, overlap, strength), applied to generated video frames or a still image.
- **`MusicParamGroup`** — ACE-Step edit modes (source audio, edit mode, repaint span, cover
  strength) plus the 5 Hz LM planner and advanced CFG/sampling knobs, and the YuE sampling
  knobs (temperature/top-k/top-p/repetition penalty). Every name carries a `Hartsy Music`
  prefix because AudioLab already registers plain names like "Source Audio" or "Top K", and a
  cleaned-name collision in `T2IParamTypes.Register` crashes SwarmUI at init.

One param breaks the "own group" pattern deliberately: `FaceIdV2WeightParam` (FaceID-PlusV2
shortcut strength) is registered into *Comfy's* `GroupImagePrompting`, flagged `"ipadapter"`
instead of `"hartsyinference"`, so it surfaces next to the rest of the IP-Adapter controls
exactly when an IP-Adapter is selected — not only when our backend happens to be picked.

(The old `HartsyInference Dtype` / `Tile VAE Threshold` params were removed: the engine
never read them — dtype is engine-policy per model (bf16 unsupported by the shared kernels) and VAE
tiling is decided from measured VRAM, not a pixel threshold.)

## Feature flags advertised by `SupportedFeatures`

`HartsyInferenceBackend.DeclaredFeatures` is the full, current list — `SupportedFeatures`
just returns it, so every flag below is advertised unconditionally. The "Enforcement" column
is where each one is actually checked, if it's checked at all beyond just being declared:

| Flag | What it gates | Enforcement |
|------|---------------|-------------|
| `hartsyinference` | Our own params (section C above) | none needed — the flag only reaches us via our own params |
| `comfyui` | Lets comfyui-tagged requests reach our validator; honesty enforced by `ValidateComfyOnlyParams` — see the two-layer design above | `ValidateComfyOnlyParams` |
| `text2image` | Universal text-to-image params | none — always serviceable |
| `flux-dev` | FluxGuidanceScale param | none — in `T2IEngine.DisregardedFeatureFlags` so it's informational only, doesn't gate backend selection |
| `lora` | LoRA list | per-family, `ValidateImageFeatures` against the recipe's `ImageFeatures.Lora` |
| `endstepsearly` | End-step early-out | none — `ImageRequest.EndStepsEarly` is unconditional |
| `refiners` | Refiner workflow params | per-family, `ImageFeatures.Refiner` |
| `img2img` | Init image / denoise strength | per-family, `ImageFeatures.Img2Img` |
| `inpaint` | Mask + inpaint | per-family, `ImageFeatures.Inpaint` |
| `controlnet` | ControlNet inputs | per-family, `ImageFeatures.ControlNet` |
| `ipadapter` | IP-Adapter inputs, plus `FaceIdV2WeightParam` | per-family, `ImageFeatures.IpAdapter` |
| `variation_seed` | Second seed for blended noise | per-family, `ImageFeatures.VariationSeed` |
| `video` | Video models: Wan (T2V/I2V/VACE/Animate/S2V), LTX-Video, LTX-2 (2.3 and 2.5), Lance Video, MiniMax-H3 | per-family, `ValidateVideo` against `VideoFeatures` (a separate enum from `ImageFeatures`) |

A family that doesn't support a per-family flag reports `ImageFeatures.None` for it
(`ModelSupport.SupportedFeatures`), and `IsValidForThisBackend`'s `ValidateImageFeatures`
step refuses the request early rather than letting it fail mid-generation. `freeu` and
`yolov8` are not advertised — the engine has no equivalent for the first, and YOLO
post-processing belongs to a separate extension. (`seamless` *is* advertised as of the
central `Conv2D` interception work; this list previously said otherwise.)

`ValidateVideo` also refuses an incomplete **LTX-2.5 bundle**. 2.5 ships split across four
files (DiT, Gemma-4 text encoder, conv video VAE, audio VAE) and the engine's recipe takes
one path, so the extension hands it the containing *directory*. If a companion is missing
the engine would silently fall back to LTX-**2.3**'s Gemma 3 and 2.3 VAEs and generate with
them — a plausible video from the wrong model — so the request is refused instead, naming
what to stage. See [13 — Video Models Plan](./13-Video-Models-Plan.md).

With the exception of the broad `comfyui` routing flag (kept honest by the validator
guard), a capability flag is **only** advertised when we can actually service requests
gated by it. Lying ("we support controlnet!" then erroring at generation time) is worse
than being honest ("we don't yet, please use ComfyUI for this").

## Coexistence model (Comfy + HartsyInference both installed)

How a request routes when both backends exist:

| Request | Routes to |
|---------|-----------|
| Plain gen, model only HartsyInference has (e.g. nvfp4 Ideogram) | HartsyInference (model-availability filter) |
| Plain gen, model both backends have | Either — load-balanced. Pin one with the **Backend Type** or **Exact Backend ID** advanced params |
| Custom ComfyUI workflow, or any comfyui-only param we can't run | Comfy (our `IsValidForThisBackend` guard refuses and routes there) |
| Sampler / Scheduler / Refiner sampler set, Redux style-model strengths, IP-Adapter scheduling knobs | Either — we honor these (`HonoredComfyParams` allow-list) |

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

### LoRAs (a core Swarm param, not a Comfy one)

`T2IParamTypes.Loras` (plus the parallel `LoraWeights` / `LoraTencWeights` /
`LoraSectionConfinement` lists) is registered by **Swarm core**, not the Comfy extension,
and carries no `FeatureFlag` at all — so it's bucket A, not bucket B, and it's sent to
whichever backend is selected regardless of advertised flags. `HartsyInferenceBackend.BuildLoras`
reads the four lists directly and resolves each name to an on-disk path through Swarm's LoRA
model set, throwing a clear error if a named LoRA isn't found:

```csharp
private static LoraStack BuildLoras(T2IParamInput input)
{
    if (!input.TryGet(T2IParamTypes.Loras, out List<string> names) || names is null || names.Count == 0)
        return null;
    // ... resolve each name through Program.T2IModelSets["LoRA"], pair with LoraWeights/
    // LoraTencWeights/LoraSectionConfinement by index, build a LoraStack.
}
```

Because the param itself is unflagged, the `"lora"` entry in `DeclaredFeatures` doesn't
gate anything at backend-selection time — it's enforced one layer down, in
`ValidateImageFeatures`: if LoRAs are set and the loaded model's recipe reports
`ImageFeatures.None` for `Lora`, the request is refused with a clear reason instead of
silently ignoring the selection.

## What we explicitly don't surface

- Comfy workflow editor: it's a tab, not a param. No way to surface a workflow
  editor without a workflow IR. We don't have one.
- Stored Custom Workflows: same.
- TensorRT compile button: not applicable.

## Parameter validation

Validation that requires backend awareness (e.g., architecture support, comfyui-only
params, per-family `ImageFeatures` checks) lives in `HartsyInferenceBackend.IsValidForThisBackend`
and its private `Validate*` helpers (`ValidateMusic`, `ValidateVideo`, `ValidateImageFeatures`,
`ValidateComfyOnlyParams` — see the "honesty guard" region of `Backends/HartsyInferenceBackend.cs`).
They add a `RefusalReason` and return false, which routes to Comfy rather than throwing.

Parameter validation that's input-shape (e.g., "width must be multiple of 8") lives
where Swarm core puts it — at the `T2IParamType` level, with a `Validator`. We don't
override.

## Engine performance profile

The engine's performance profile (fused attention, fp8 GEMM, resident weights, etc.) is
default-on inside the engine itself — the extension configures none of it. See the engine's
[`docs/PERFORMANCE.md`](https://github.com/HartsyAI/HartsyInference/blob/main/docs/PERFORMANCE.md),
which is the single source of truth for this.
