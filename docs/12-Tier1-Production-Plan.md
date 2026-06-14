# 12 — Tier 1 Production Plan

Plan to take the five Tier 1 features from
[`11-Comfy-Parity-Punchlist.md`](./11-Comfy-Parity-Punchlist.md) to
production-ready. Each phase has concrete file targets and an acceptance test
that's runnable, not "feels-done."

Order chosen by **dependency depth × user-visible impact**:
inpainting first (no upstream blockers, biggest single gap), ControlNet +
preprocessors next (largest scope, kicks off upstream work that runs in parallel
with later phases), IP-Adapter, Refiner StepSwap, then Upscaling once upstream
unblocks.

---

## Phase A — Inpainting / masks  ✅ Done 2026-05-07

**Initial assumption was wrong** — upstream `ImageToImageRequest` had no Mask
field, so the work expanded to include upstream pipeline changes for SDXL,
Flux, and SD3 (the latter additionally needed the entire img2img path wired
since its pipeline ctor didn't even take a `VaeEncoder`).

**Shipped:**
- Upstream `ImageToImageRequest.Mask` + `RecompositeAtEnd` fields.
- Upstream `MaskBlendUtilities` (area-average mask downsample, NCHW blend,
  Flux-packed 2×2 blend) — single source of truth shared by all pipelines.
- Upstream `SdxlPipeline` / `FluxPipeline` / `Sd3Pipeline` blend-on-vanilla
  inpaint: per-step latent blend with re-noised source + post-decode
  pixel-space recomposite.
- Upstream `Sd3Pipeline` ctor overload accepting `VaeEncoder` (img2img path
  didn't exist before).
- Swarm-side `MaskResolver.cs` (mask load + separable-max-filter dilation +
  Gaussian blur via ImageSharp).
- Swarm-side `Img2ImgResolver` extended to bundle mask into spec; loaders
  pass it through to `ImageToImageRequest.Mask`.
- Swarm-side `SdxlLoader` / `FluxLoader` / `Sd3Loader` updated; SD3 now
  builds VAE encoder + has full img2img + inpaint paths.
- Backend feature flags advertise `inpaint`; refusal gating for SD 1.5 /
  Z-Image / Flux.2 / etc. lists supported arches accurately.

**Deferred follow-ups:** SD 1.5 + Z-Image inpaint wiring (same blend pattern,
needs upstream pipeline edits); `MaskShrinkGrow` (crop-to-bbox optimization);
9-channel `SdxlInpaintPipeline` (specialized inpaint checkpoints — its
`InpaintFromTokens` is still a `NotImplementedException` stub upstream, only
matters if a user tries that variant directly).

**Original plan (kept for reference):**

**No upstream blockers.** `VaeEncoder` exists. `Img2ImgResolver` exists.

### A.1 Mask plumbing
- New file `Generation/MaskResolver.cs`.
  - Resolve `T2IParamTypes.MaskImage`, `MaskGrow`, `MaskBlur`,
    `MaskShrinkGrow`, `MaskBlurThreshold` → produce a single-channel
    `byte[]` aligned to the latent's spatial dims.
  - Implement grow (morphological dilate), shrink-grow (erode then dilate to
    fill holes), Gaussian blur, threshold. Pure C#; no HartsyInference
    dependency.
- Add unit test against a 64×64 synthetic mask: round-trip preserves a
  hand-drawn pattern; grow + shrink gives expected pixel counts.

### A.2 Masked VAE encode + latent blend
- Extend `Img2ImgResolver.cs` with a `BuildMaskedLatents(...)` overload that
  returns `(initLatent, mask, sourceLatent)`.
  - Encode source via existing `VaeEncoder`.
  - Resize mask to latent space.
  - Verify the per-arch latent shape (SDXL = 4ch, Flux = 16ch, SD3 = 16ch).
- Add a `LatentBlend(noised, source, mask)` helper — masked region uses
  the noised init, unmasked region keeps the source latent each step.

### A.3 Per-arch wiring
- `SdxlLoader.cs` / `SdxlPipeline` path: when a mask is present, route
  through the inpaint path. Confirm whether to use `SdxlInpaintPipeline`
  (9-channel UNet, requires SDXL Inpaint checkpoint) or the
  blend-on-vanilla approach (works with any SDXL checkpoint, slightly
  worse boundary blending). **Decision: support both** — auto-detect
  inpaint checkpoint by reading conv_in input channels.
- `FluxLoader.cs`: Flux uses blend-on-vanilla (no dedicated inpaint variant).
- `Sd3Loader.cs`: same as Flux.
- Other arches: defer to Phase A.4 follow-up.

### A.4 Outpaint
- Outpaint is inpaint where the mask covers the new canvas region.
  Swarm passes `OutpaintTop`/`Bottom`/`Left`/`Right` params; convert to a
  mask before calling A.3 path.
- New helper in `MaskResolver.cs`: `BuildOutpaintMask(width, height,
  top, bottom, left, right)`.

### A.5 Acceptance
- SDXL: prompt "red apple" + circle mask in center → only the masked region
  is painted, surroundings unchanged.
- Flux: same test passes.
- SD3: same test passes.
- Outpaint: SDXL with `OutpaintRight=256` extends the canvas with prompt-
  consistent content; non-outpaint region pixel-identical to source.
- Edge cases: mask covers 100% (degenerates to img2img with strength=1) /
  mask covers 0% (degenerates to identity, no changes).

**Estimated effort:** 2-4 days. All in Swarm extension; no upstream PRs.

---

## Phase B.1 — SDXL ControlNet + Canny  ✅ Done 2026-05-07

**Scope adjustment:** Phase B was sized for "all base models + all common
preprocessors" — realistically 1-2 weeks. Split into B.1 (single working slice)
and B.2+ (breadth) so we have something shippable.

**Shipped:**
- Upstream `ControlNetCondEmbedding` — 3 → 320 hint encoder, 8x downsample
  matching VAE compression.
- Upstream `ControlNet.LoadWeights` + `Forward` for SDXL — reuses
  `DownBlock`, `UNetResNetBlock`, `CrossAttentionBlock`, `TimestepEmbedding`,
  `AdditionEmbedding` from existing UNet primitives. 9 down residuals + 1
  mid residual matching diffusers SDXL ControlNet exactly.
- Upstream `UNet.Forward` accepts optional `IReadOnlyList<Tensor>` down
  residuals + nullable mid residual; injects via element-wise add at the
  skip-collection point and after the mid block.
- Upstream `SdxlPipeline.GenerateFromTokens` accepts
  `IReadOnlyList<ControlNetConditioning>`; runs CN once per step with cond
  text emb (residuals shared across CFG); stacks multiple CNs via summed
  residuals.
- Swarm-side `CannyPreprocessor` — pure C# Sobel + 4-direction NMS +
  hysteresis, defaults `low=100, high=200` matching Comfy's
  `CannyEdgePreprocessor`.
- Swarm-side `ControlNetResolver` + `ControlNetWeightLoader` reading the
  3-slot Comfy ControlNet param holders, falling back to InitImage when no
  CN-specific image is set.
- Swarm-side `SdxlLoader` threads the resolved spec into pipeline calls.
- Backend feature flag `controlnet` advertised; refusal gating for non-SDXL.

**Deferred (Phase B.2+):**
- SD 1.5 ControlNet wiring (upstream supports it, just need SD15 pipeline
  + loader integration).
- Flux ControlNet (DiT architecture, separate adapter class).
- Depth / OpenPose / Lineart preprocessors (ONNX runtime + model bundling).
- ControlNet start/end step ranges (param read, ignore for now).
- Strict CFG (run CN twice — once per branch) for accuracy at higher
  guidance scales; current single-pass matches diffusers `guess_mode=True`.

**Original plan (kept for reference):**

## Phase B — ControlNet + preprocessors

**Partial upstream blocker.** HartsyInference has a `ControlNet` adapter class
but pipelines don't accept it as a parameter. Preprocessors don't exist
upstream.

### B.1 Upstream — pipeline ControlNet integration
- File HartsyInference issue: `Pipelines must accept ControlNet conditioning`.
- Scope: `StableDiffusion15Pipeline`, `SdxlPipeline`, `FluxPipeline`,
  `Sd3Pipeline` accept `IReadOnlyList<ControlNetConditioning>` plus per-step
  control strength.
- This is upstream work — wait for the PR to land before B.3 can finish.
  Phases B.2 and B.4 can run in parallel.

### B.2 Upstream — preprocessor module
- File HartsyInference issue: `New module HartsyInference.Vision.ControlNetPreprocessors`.
- Phase B.2 ships:
  - **Canny** — pure C# (Sobel + non-max suppression + hysteresis), no
    model needed.
  - **Depth** — wrap a small ZoeDepth or DepthAnything-V2 ONNX model
    via `Microsoft.ML.OnnxRuntime`. ~50MB.
  - **Openpose** — DWPose ONNX (~150MB).
  - **Lineart** — pure C# Sobel variant, optional ONNX MangaLineExtractor
    upgrade.
- Each preprocessor: `byte[] Process(Image input, params)` returning HWC RGB.

### B.3 Swarm-side — ControlNet loader
- New `Generation/ControlNetLoader.cs`.
- Accept user's `T2IParamTypes.ControlNetModel` (and the stacked variants).
- Resolve checkpoint type by reading conv_in channels (single-net) or by
  filename heuristic (Comfy approach).
- Cache loaded ControlNet weights in `PipelineCache`.

### B.4 Swarm-side — side-model registry expansion
- Add to `SideModels.cs` the canonical preprocessors users expect:
  - DepthAnything-V2 small ONNX (Comfy hosts these)
  - DWPose ONNX
  - MangaLineExtractor ONNX (optional)
- Expose via `HartsyInferenceWebAPI` so the UI can list which preprocessors
  are available without inspecting the disk.

### B.5 Swarm-side — wiring
- `Generation/ControlNetResolver.cs`: read params, run preprocessor on
  source image, build conditioning list, pass to pipeline.
- Per-arch wiring SD1.5/SDXL/Flux/SD3.

### B.6 Acceptance
- SD1.5 + Canny CN at strength 1.0: edge map of input image clearly
  conditions the output.
- SDXL + Depth (DepthAnything-V2) + prompt "cyberpunk city": depth
  structure preserved from source.
- SDXL + 2 stacked CNs (canny + depth) blends both.
- Flux + Canny CN (the Flux-canny checkpoint): edge-conditioned output.

**Estimated effort:** 2-3 weeks. Roughly 1 week for upstream pipeline
integration, 1 week for preprocessor module, 1 week Swarm-side wiring +
testing. Significant chunk is upstream (B.1, B.2).

---

## Phase C — IP-Adapter  ✅ Done 2026-05-07 (SDXL standard + Plus)

**Scope adjustment:** Phase C committed to "standard + Plus + Plus-Face + FaceID"
across SDXL/SD15/Flux — realistically 2-3 weeks. Cut to SDXL standard + Plus
+ Plus-Face (which is architecturally identical to Plus, just different training).
FaceID needs InsightFace ArcFace embeddings (different image encoder), and
SD15/Flux IPA need their own pipeline integrations — all separate sessions.

**Shipped upstream (HartsyInference):**
- `ClipVisionEncoderConfig` + `ClipVisionEncoder` — ViT-H/14 (32 layers,
  hidden=1280, head_dim=80, patch=14, optional projection_dim=1024). Two
  output paths: `EncodeImageEmbeds` (CLS-projected, IPA standard) and
  `EncodeHiddenStates` (penultimate full-sequence, IPA Plus).
- `IpAdapterStandardProjection` — Linear(1024→8192) + reshape + LayerNorm,
  pure CPU since the projection is tiny (<1ms per gen).
- `IpAdapterPlusResampler` — Perceiver-style: 4 layers, 16 learnable queries,
  cross-attn over `concat(x, latents)`, FFN, proj_in 1280→1024 + proj_out
  1024→2048, LayerNorm. Uses backend ops for the larger matmuls.
- `IpAdapter` orchestrator — picks the right projection module based on
  `IsPlus`, loads per-cross-attn-layer K_ip / V_ip in flat list form,
  exposes `CrossAttentionLayerCount` for UNet validation.
- `IpAdapterConditioning` — public record (Adapter + ImageTokens + Scale).
- `TransformerSubBlock.Forward` overload runs a second SDPA against image
  K_ip/V_ip with the same Q, accumulates `ipaScale * imgAttnOut` onto the
  text attention output before the shared `to_out`.
- `CrossAttentionBlock`, `DownBlock`, `UpBlock`, `UNet.Forward` all gained
  IPA-aware overloads that thread per-cross-attn K/V indices through using a
  flat list + offset cursor (each block consumes its
  `CrossAttentionLayerCount` worth of entries before passing the cursor to
  the next).
- `SdxlPipeline.GenerateFromTokens` accepts
  `IReadOnlyList<IpAdapterConditioning>` (single-element honored in v1);
  `RunDenoiseLoop` and `ClassifierFreeGuidanceStep` thread IPA params to
  both UNet branches; IPA is auto-disabled during refiner StepSwap phase
  (refiner UNet has different cross-attn shape).

**Shipped Swarm-side:**
- `ClipImagePreprocessor` — bicubic resize-then-crop to 224×224, CLIP
  mean/std normalization, → `[1, 3, 224, 224]` F32 tensor.
- `IpAdapterResolver` reads `useipadapter` / `ipadapterweight` Comfy params
  via `T2IParamTypes.TryGetType` (no hard dependency on Comfy assembly),
  resolves the IPA file under `<ModelRoot>/ipadapter/<filename>` (Comfy
  convention), auto-detects standard vs Plus, auto-downloads CLIP-ViT-H/14
  via `SideModels.ClipVisionH14`, runs preprocessor + CLIP-Vision +
  projection.
- `IpAdapterCacheEntry` cached per IPA file path in `PipelineCache` so
  repeat gens skip the load.
- `SdxlLoader.Generate` / `GenerateWithLoras` accept the conditioning list,
  thread to the pipeline.
- Backend feature flag `ipadapter` advertised; non-SDXL bases refused at
  validation; image-token tensors auto-disposed via `using` at end of
  `GenerateLive`.

**Phase C.2 (added 2026-05-07): full IPA feature set + SD 1.5 wiring**
- ✅ Multi-image averaging — when <c>PromptImages</c> has N>1 entries, the
  resolver runs CLIP-Vision on each and averages the vision outputs
  pre-projection. The IPA projection runs ONCE on the centroid; the resulting
  token tensor flows through one IPA conditioning. Matches Cubiq IPAdapterPlus
  default behavior for multi-reference inputs.
- ✅ Start/end step gating — `IpAdapterConditioning` carries
  <c>StartFraction</c> + <c>EndFraction</c> read from Comfy's
  <c>ipadapterstart</c> / <c>ipadapterend</c> params. <c>SdxlPipeline</c> and
  <c>StableDiffusion15Pipeline</c> compute a per-step gate via
  <c>IpAdapterScaleSchedule.StepGate</c> and zero out the per-layer scale array
  on out-of-window steps (UNet skips image attention entirely).
- ✅ Weight Type — three modes:
  - <c>"standard"</c>: uniform scale across all cross-attn layers.
  - <c>"prompt is more important"</c>: encoder + mid layers at <c>0.4 × base</c>,
    decoder layers at full <c>base</c>. Lets the prompt drive subject/composition
    while IPA contributes mainly to style at decoder.
  - <c>"style transfer"</c>: middle third of cross-attn layers at full
    <c>base</c>, encoder + late decoder zeroed. Approximation of Cubiq's
    "style transfer (SDXL)" block_3 / block_4 schedule (which targets mid + first
    up block).
  Per-layer scale arrays are computed by <c>IpAdapterScaleSchedule.Build</c>
  once per generation and gated per-step.
- ✅ SD 1.5 IPA — <c>StableDiffusion15Pipeline.GenerateFromTokens</c> accepts
  <c>IReadOnlyList&lt;IpAdapterConditioning&gt;?</c>, threads to
  <c>UNet.Forward</c> with the same flat K/V list + per-layer scale shape.
  Backend gating allows SD 1.5; <c>IpAdapterResolver</c> validates checkpoint
  base model matches pipeline base model (SD15 IPA on SDXL refuses cleanly).
  CLIP-ViT-H/14 reused via <c>SideModels.ClipVisionH14</c>.

**Refused with technical justification (not laziness — actual blockers):**
- **FaceID variants** (h94/IP-Adapter-FaceID family) — use InsightFace ArcFace
  512-dim face embeddings instead of CLIP-Vision. Need an InsightFace runtime
  (face detection + ArcFace ONNX) which HartsyInference doesn't link. Adding
  ONNX Runtime is a separate infrastructure track. Refused at IPA load time
  with a clear message ("Use a non-FaceID variant").
- **Flux IPA** (XLabs-AI/flux-ip-adapter) — Flux is a DiT (FluxTransformer with
  double_stream + single_stream blocks). IPA for Flux modifies block-level
  image-modulation, not cross-attention K/V. Different adapter class
  (FluxIpAdapter), different injection points, different image encoder.
  ~1500 LOC of new upstream work — separate session.
- **Other architectures** (SD3 / Z-Image / Flux.2 / AuraFlow / Chroma / F-Lite
  / ERNIE) — **no published IPA checkpoints exist for any of these**. There's
  no checkpoint format to load and no trained weights. Without published
  artifacts, "building out" their IPA logic is impossible — there's nothing
  to load.
- **Multi-adapter stacking** — Swarm's UI exposes only one
  <c>UseIPAdapterForRevision</c> slot; there's no Swarm param to select multiple
  IPA models. The upstream API already accepts <c>IReadOnlyList&lt;IpAdapterConditioning&gt;</c>
  so future custom workflows can stack; the pipeline currently honors the first
  entry only.

**Original plan (kept for reference):**

## Phase C — IP-Adapter

**Original plan (kept for reference):**

## Phase C — IP-Adapter

**Partial upstream blocker.** HartsyInference has IP-Adapter stubs but not the
plus / face variants and not full pipeline integration.

### C.1 Upstream — IP-Adapter pipeline integration
- File HartsyInference issue: `Pipelines accept IPAdapter image-prompt conditioning`.
- Scope: SDXL + Flux at minimum. SD1.5 IP-Adapter is legacy but cheap to add.
- Variants: standard, plus, plus-face, FaceID. Each has its own
  CLIP-Vision encoding + cross-attention injection pattern.

### C.2 Upstream — CLIP-Vision encoder
- Verify `HartsyInference.Diffusion.Models.TextEncoders.ClipVisionEncoder`
  state. If incomplete, finish it.

### C.3 Swarm-side — IPA loader
- New `Generation/IpAdapterLoader.cs`.
- Resolve `T2IParamTypes.UseIPAdapterForRevision` (and weight params).
- Side-model entries: `SideModels.IpAdapterSdxl`, `IpAdapterSdxlPlus`,
  `IpAdapterSdxlFaceId`, `IpAdapterFlux` (when published).
- Resolve corresponding CLIP-Vision encoder (`clip-vit-h-14` or
  `clip-vit-bigG-14`) via auto-download.

### C.4 Swarm-side — wiring
- `Generation/IpAdapterResolver.cs`: encode reference image, build IPA
  conditioning, pass to pipeline alongside text conditioning.
- Per-arch wiring SDXL/Flux.

### C.5 Acceptance
- SDXL + IP-Adapter standard + reference photo: output reflects the
  reference's color/style.
- SDXL + IP-Adapter plus-face + reference face: output preserves identity
  reasonably.
- Flux + IPA: works once Flux IPA checkpoints are publicly available.

**Estimated effort:** 1-2 weeks. Most of the work is upstream (C.1, C.2).

---

## Phase D — Refiner StepSwap  ✅ Done 2026-05-07

**Shipped:**
- Upstream `RefinerSwapConfig` record (refiner UNet + Strength + aesthetic
  scores; defaults match Stability's training: cond=6.0, uncond=2.5).
- Upstream `SdxlPipeline.GenerateFromTokens` accepts optional refiner config.
  Computes swap step from Strength (fraction of total steps the refiner runs
  at end). Keeps CLIP-G hidden alive separately when StepSwap active (refiner
  uses CrossAttentionDim=1280 vs base's concat 2048). Builds per-branch ADM
  arrays (refiner ADM is 5 values with aesthetic_score, vs base's 6 with
  target dims).
- Upstream `RunDenoiseLoop` per-step branches: `inRefinerPhase = i >= swapStep`
  selects active UNet, text emb, ADM array, and dtype. ControlNet skipped
  during refiner phase (zero-conv residuals are base-shaped — applying them
  to the 4-level refiner UNet would break additions).
- Upstream `ClassifierFreeGuidanceStep` accepts override UNet + per-branch
  uncond ADM (refiner needs different aesthetic_score per branch).
- Swarm `SdxlLoader.Generate` / `GenerateWithLoras` thread the config through.
- Backend pre-loads the refiner cache entry on the calling thread (so
  `AddLoadStatus` shows in the UI), builds the swap config, passes through to
  SDXL, and skips the post-pass refiner block when StepSwap was applied.
- Validation: 'StepSwap' allowed on SDXL base; refused upfront on other base
  models with a clear message.

**Deferred:**
- `StepSwapNoisy` (re-noise at swap point — minor variant, not commonly used).
- Per-pipeline RefinerCFGScale handoff (currently the swap reuses base CFG
  scale; Comfy lets refiner have its own).
- Aesthetic score Swarm params (currently hardcoded defaults; Swarm UI doesn't
  expose these as registered params today).

**Original plan (kept for reference):**

## Phase D — Refiner StepSwap

**No hard upstream blocker** — sampling loop in HartsyInference already exposes
a per-step callback; we just need to be able to swap the denoiser model object
mid-loop.

### D.1 Verify upstream capability
- Read `HartsyInference.Diffusion.Pipelines.SdxlPipeline.GenerateFromTokens`:
  is the denoiser object passed in once or referenced per step?
- If passed once, file upstream issue: `SdxlPipeline allows mid-loop denoiser
  swap at a specified step boundary`.
- Likely interface: `IDenoiser PrimaryDenoiser`, `IDenoiser RefinerDenoiser`,
  `int SwapAtStep`.

### D.2 Swarm-side — refiner wiring
- Update `RefinerResolver.cs` to surface `RefinerControl` (Swarm's existing
  param) as the swap-step ratio.
- Update `RefinerLoader.cs` to load both the base SDXL and SDXL refiner UNet
  into a single pipeline call (currently runs them as two pipelines back to
  back).

### D.3 Acceptance
- SDXL + SDXL refiner + RefinerControl=0.8: output differs visibly from
  SDXL-only and from base→refiner-as-second-pipeline. Quality approximately
  matches Comfy's StepSwap output on a curated 5-prompt benchmark.
- Behaviour matches when RefinerControl=1.0 (no refiner) and
  RefinerControl=0.0 (refiner-only).

**Estimated effort:** 3-5 days, of which 1-2 days might be upstream.

---

## Phase E — Upscaling (deferred)

**Blocked upstream.** HartsyInference.Core has no upscaler loader. File the
issue, defer to Tier 2 timing.

### E.1 Upstream issue to file
- `Add upscaler loader (RealESRGAN, 4x-Ultrasharp class) to HartsyInference.Core`.
- Scope: ESRGAN architecture, weights loaded from .pth or .safetensors,
  tiled inference for large outputs.

### E.2 Swarm-side plan (post-unblock)
- New `Generation/UpscalerLoader.cs` mirroring other loader files.
- Side-model entries for the canonical 4× / 2× upscale models.
- Hi-res-fix path: upscale latent → re-encode → low-strength denoise.

**Do not start E.2 until E.1 lands.**

---

## Cross-cutting work

These touch all phases.

- **Test harness.** Add `tests/parity/` directory (Swarm-side) with reference
  images generated from a known-good Comfy install. Each phase's acceptance
  test compares against the matching reference.
- **Documentation pass.** Each completed phase updates the punchlist
  (`11-Comfy-Parity-Punchlist.md`) status and marks any upstream issues
  closed.
- **Upstream issue tracker.** All "file HartsyInference issue" items above
  should be opened during Phase A (the inpainting work) so they have
  maximum lead time before the Swarm side needs them.

## Suggested execution order

1. **Today:** open HartsyInference issues B.1, B.2, C.1, D.1, E.1.
2. **Week 1:** Phase A (inpainting). Pure Swarm-side; ships independently.
3. **Weeks 2-4:** Phase B (ControlNet) — upstream B.1/B.2 in parallel with
   B.3-B.5 Swarm-side.
4. **Weeks 4-5:** Phase C (IP-Adapter) — upstream C.1/C.2 in parallel.
5. **Week 5:** Phase D (Refiner StepSwap).
6. **Phase E** lands when E.1 ships upstream.

Total Tier 1 production-ready: **~5 weeks of Swarm-side work** plus parallel
upstream work in HartsyInference.Core.
