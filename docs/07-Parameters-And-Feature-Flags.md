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
public static T2IParamGroup AceStepParamGroup, AceStepGuidanceGroup;
public static T2IRegisteredParam<string> AceStepSolverParam;

public override void OnInit()
{
    ScriptFiles.Add("Assets/hartsy-params.js");   // the model-aware half; see below
    AceStepParamGroup = new("ACE-Step", Toggles: false, Open: false);
    AceStepGuidanceGroup = new("ACE-Step Guidance", Open: false, IsAdvanced: true, Parent: AceStepParamGroup);

    AceStepSolverParam = T2IParamTypes.Register<string>(new(
        "ACE-Step Solver",
        "ACE-Step diffusion solver.",
        "ode",
        Toggleable: true,
        Group: AceStepGuidanceGroup,
        // Comma is AND: our backend must be running AND an ACE-Step model selected.
        FeatureFlag: "hartsyinference,hartsy_acestep",
        GetValues: _ => new List<string> { "ode", "sde" }));
}
```

### Names carry the model family, never the vendor

A param is named for the **model family that honors it** (`ACE-Step Solver`, `YuE Stage-1 Top K`),
never for this extension. Two consequences worth knowing before adding one:

- A param's ID is its name with everything except `a-z` stripped
  (`T2IParamTypes.CleanTypeNameMatcher = AsciiMatcher.LowercaseLetters`), so digits and
  punctuation vanish — "Ideogram 4 Magic Prompt" is `ideogrammagicprompt`. Check a new name
  against every registered param before using it: `T2IParamTypes.Register` is backed by
  `Dictionary.Add`, so a cleaned-name collision crashes SwarmUI at init. AudioLab in particular
  owns the bare `Cover Strength` / `LM Top K` / `YuE Temperature` spellings.
- Renaming an existing param needs a `T2IParamTypes.ParameterRemaps` entry
  (`RegisterParameterRemaps`). That dictionary is applied inside `GetType`, so one entry covers
  both saved presets and "reuse parameters" from previously generated images.

`SwarmUIHartsyInference.OnInit` registers 35 params. **There is deliberately no group named
after this extension.** A group named for the backend can never be model-scoped, and that is
the structural reason every param used to be permanently visible. Each one either lives in the
matching *core* group or in a group named for the *model family* — which is safe precisely
because such a group is model-gated and therefore absent for other models.

Into core's own groups:

| Param | Core group |
|---|---|
| `CFG Rescale` | Alternate Guidance |
| `Init Image Mode` | Init Image |
| `Video Audio Reference` | Advanced Video (beside core's `Video Audio Input`) |
| `FaceID V2 Weight` | Image Prompting (flagged `"ipadapter"`, not `"hartsyinference"`, so it appears whenever an IP-Adapter is selected rather than only when our backend is picked) |

Into model-family groups:

- **`WanAnimateParamGroup`** ("Wan Animate") — reference image, auto-preprocess toggle, and the
  pose/face driving-video overrides.
- **`Ideogram4ParamGroup`** ("Ideogram 4") — the magic-prompt toggle and its optional LLM-model
  override (`Generation.Ideogram4MagicPrompt`).
- **`VideoRestoreParamGroup`** ("Restore / Upscale") — SeedVR2 knobs (model, target
  width/height, clip frames, frame overlap, strength). Not named "Video": it runs over a still
  image just as well. This one keeps the plain `"hartsyinference"` flag, because the pass really
  is backend-scoped rather than model-scoped.
- **`AceStepParamGroup`** ("ACE-Step") with three real nested children via `Parent:` —
  *Editing* (source audio, edit mode, repaint span, cover strength), *Planner* (the 5 Hz LM
  planner and its sampling knobs), *Guidance* (solver, ADG, CFG interval).
- **`YueParamGroup`** ("YuE") — the Stage-1 sampling knobs.

Nested groups are real: `T2IParamGroup` takes a `Parent`, which is how core nests Video Obscure
Options under Advanced Video. A group needs no flag of its own — `hideUnsupportableParams` hides
any group left with no visible params, so selecting MusicGen collapses the whole ACE-Step tree.

### Model gating lives in `Assets/hartsy-params.js`

`"hartsyinference"` answers "is our backend running", never "can this model use this". The
model half is a `featureSetChangers` entry (core's hook, also used by SwarmUI-AudioLab and
SwarmUI-API-Backends) that grants/removes a small set of `hartsy_*` flags from the selected
model's compat class: `hartsy_ideogram4`, `hartsy_acestep`, `hartsy_yue`, `hartsy_wan_animate`,
`hartsy_audio_ref`, `hartsy_refedit_choice`. Params then require `"hartsyinference,<flag>"` —
comma is **AND**. Every such flag must also be in `HartsyInferenceBackend.DeclaredFeatures`, or
the generation is refused with a message naming no param; `WarnOnUndeclaredFeatureFlags` checks
this at startup for every registered param carrying a `hartsy`-prefixed flag.

Two flags come free from core and need no JS: `sdxl` (used by `CFG Rescale`, which only SDXL
honors) is granted per model by core and sits in `T2IEngine.DisregardedFeatureFlags`, so it
gates visibility without ever gating backend selection.

The same script also hides *core's* params — LoRAs, ControlNet, Refiner, Seamless Tiling,
Variation Seed, inpainting — on families whose recipe doesn't declare the matching
`ImageFeatures`, using the per-architecture map that `HartsyInferenceGetSupportedArchs` now
returns. That half only applies when no ComfyUI backend is loaded, since Comfy can service
those on families our engine can't. The audio half (hiding Width/Height/Init Image on ACE-Step,
YuE and MusicGen checkpoints) applies regardless, because those controls are meaningless on a
music model whichever backend runs it.

⚠️ Three extensions now rewrite `param.feature_flag` on the same core params (ours, AudioLab's
and API-Backends'), each with its own save/restore key. Never capture a value starting with
`__` as the "original" — that is another extension's marker, and restoring it would hide a core
param on every model until the page reloads.

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
