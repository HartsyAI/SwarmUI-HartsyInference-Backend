# 11 — Comfy Parity Punchlist

Canonical "what's left to ship" list.

Last refresh: 2026-08-09. Excludes Comfy-only features that don't apply
(workflow editor, custom-node packs, WebSocket passthrough, subprocess
self-start, ComfyUser session management).

Support is resolved from the Engine at request time, not hard-coded in this
doc: `Generation/ModelSupport.cs` maps a SwarmUI compat class to an Engine
family id, and `RecipeRegistry` / `VideoRecipeRegistry` decide whether that
family has a recipe and which `ImageFeatures` / `VideoFeatures` it declares.
`ModelSupport.SupportedArchitectures` / `PendingArchitectures` are the live
answer to "what can this backend drive today" — check those instead of
expecting a per-architecture table here to stay current.

Per-arch facts below (`Supports`/`ImageFeatures`/`VideoFeatures` flags) come
from the HartsyInference.Core engine repo. The extension consumes a *pinned*
engine package (`HartsyInference` NuGet version in
`SwarmUI-HartsyInference.csproj`, alpha.17 as of this refresh) — this repo has
a documented history of the extension silently dropping out when its source
runs ahead of its pin. If the pin trails engine HEAD, re-check before
assuming a HEAD-only feature is live in a shipped build. Facts sourced from
the extension's own code (param registration, request-building, validation)
aren't affected by this.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked
upstream (waiting on HartsyInference.Core).

---

## Production push (active order)

- [x] **P1 — Sampler + clip skip.** `SamplerParam` ("HartsyInference Sampler":
  euler/ddim/dpm++2m/lcm) routes through `SchedulerFactory` for SD 1.5 / SDXL;
  flow-match architectures (Flux, SD3, Z-Image, …) ignore it by design. Clip
  skip reads Swarm's `T2IParamTypes.ClipStopAtLayer` straight into
  `ImageRequest.ClipSkip` — no separate upstream param was needed. Scheduler
  *type* selection (karras / exponential / …) was never in scope: the request
  builder sends `Scheduler = null` with the comment "the Engine resolves the
  family's canonical schedule; Comfy's Scheduler param has no analogue" — this
  stays a real, permanent gap, not a TODO. Batch size > 1 is the same kind of
  gap: `Batch = 1` always, with "Swarm drives batching itself: one Generate
  call per image" — there is no latent-batched generation path.
- [ ] **P2 — Hires fix / 2-pass upscale (`RefinerUpscale != 1`).** The value
  is read (`Refiner.Upscale`) and passed through, but `SdxlRecipePipeline`
  logs a warning and ignores it — StepSwap keeps the base latent resolution.
  Needs a tiled `VaeEncoder` (`EncodeTiled`) plus a latent-upscale-and-redenoise
  pass. Pixel-space upscale (Real-ESRGAN / SeedVR2) already ships as a
  separate feature (Tier 1) but isn't wired into this dropdown.
- [x] **P3 — Graceful refusal of unsupported prompt syntax.** `<object:>`,
  `<clear:>`, `<embed:>`, `<break>` are hard-refused at validation
  (`UnsupportedPromptSyntax` regex) — the Engine has no conditioning contract
  for them at all. `<region:>` / `<segment:>` are no longer regex-blocked:
  the extension builds a real `Regional` request (`BuildRegional` /
  `HasRegionalSyntax`) and lets the Engine's own per-family gate refuse it
  with a precise reason instead of a blanket "unsupported syntax" message.
- [ ] **P5 — `<segment:face>` via YOLO.** Still refused end-to-end: no image
  recipe declares `ImageFeatures.Regional` yet, so the `Regional` request P3
  builds is rejected for every architecture today. What's changed since the
  original plan: the vision runtime it needs already exists in the Engine, in
  pure C# — YOLO detection/pose/face, SAM2 mask refinement, text segmentation
  (`HartsyInference.Vision`) — so the ONNX-vs-native runtime decision is
  already made (native) and there's no new model-download infrastructure to
  build. The remaining work is one image recipe opting into
  `ImageFeatures.Regional`, not new infrastructure.
- [x] **P4 — Variation seed.** `BuildVariationSeed` slerps
  `SeedGenerator.CreateNoise(seed)` with the variation seed at the given
  strength, gated by `ImageFeatures.VariationSeed` (SD 1.5, SDXL, Flux.1).
- [~] **P6 — Architecture long-tail.** Wan 2.1 14B now has a registered
  recipe (`WanVideoRecipe` selects its preset by compat-class id, same as the
  5B/1.3B variants); Wan also has VACE, Animate, and S2V recipes gated by
  checkpoint header. Qwen-Image gained `ImageFeatures.RefEdit` (in-context
  reference editing on the same checkpoint) — verify whether that already
  covers the "Qwen Image Edit" ask before treating it as separate work.
  `ModelSupport._families` is the source of truth for what's mapped; don't
  re-enumerate it here.
- [~] **P7 — Ideogram 4.** Loader, Steps→preset mapping, chat-template
  tokenize, and the VRAM gate all shipped. Magic-prompt expansion (LLM
  rewrite of a plain prompt into Ideogram's structured JSON caption) is fully
  implemented in `Ideogram4MagicPrompt.cs` but compiled out behind
  `#if HARTSY_LLM_CORE`, pending SwarmUI core's expanded LLM API
  (`LLMParamInput.SystemPrompt`/`Temperature`/`Stream`,
  `AbstractLLMBackend.ListModels`) landing in the target core branch —
  without that symbol `Expand()` is a no-op and the plain prompt is sent
  as-is (Ideogram 4 accepts plain text; it's just out-of-distribution for its
  safety head). A no-LLM fallback (`WrapPlainAsJson`) exists in the same file
  but has no caller today; its own code comment notes that mechanically
  wrapping a short prompt in the JSON schema doesn't reliably clear the
  safety filter — only a genuinely elaborated caption does, which needs the
  LLM path. **Remaining:** E2E verify on a ≥24 GB host; wire `WrapPlainAsJson`
  in or drop it. License is "Ideogram 4 Non-Commercial" — keep surfacing that
  in the model description.

---

## Tier 1 — High-impact core features

- [x] **Inpainting / masks** — SDXL, Flux.1, SD3, SD 1.5, and Z-Image all
  blend-on-vanilla (`ImageFeatures.Inpaint` on all five recipes): per-step
  latent blend keeps the unmasked region on the source's noise/flow
  trajectory, plus a pixel-space recomposite at the end
  (`MaskBlendUtilities`, shared across pipelines). Mask handling covers
  `MaskImage` + `MaskGrow` (dilation) + `MaskBlur` (Gaussian). Deferred:
  `MaskShrinkGrow` (crop-to-bbox) and the dedicated 9-channel SDXL-Inpaint
  checkpoint variant — blend-on-vanilla covers the common case without a
  specialized checkpoint.
- [x] **ControlNet** — SD 1.5, SDXL, and Flux.1 carry `ImageFeatures.ControlNet`
  (SD3, Flux.2, and the rest don't — no CN checkpoints exist for them). SDXL
  union-type (`SdxlUnionControlType`) and per-slot start/end step-fraction
  gating (`ControlNetConditioning.StartFraction/EndFraction`) are both live,
  not full-range-only. Preprocessors are pure C#, no ONNX: Canny, Depth,
  OpenPose, SoftEdge/HED, Scribble, Lineart, Normal, Segmentation
  (`ControlNetPreprocessing.cs` dispatch). Stacking sums residuals (matches
  diffusers); CFG runs ControlNet once with the cond text embedding and
  shares residuals across both branches (`guess_mode=True` semantics) rather
  than a strict per-branch pass — an accuracy/perf tradeoff, not a bug.
- [x] **IP-Adapter** — SD 1.5 + SDXL carry the full set: standard, Plus,
  Plus-Face, and FaceID / FaceID-Plus / FaceID-PlusV2, the last three via a
  pure-C# ArcFace IR-50 implementation — no InsightFace runtime dependency.
  Weight-type math: "prompt is more important" scales encoder+mid cross-attn
  layers to 0.4× base while decoder stays at full base (prompt drives
  composition, IPA mainly contributes style at decode); "style transfer"
  zeros encoder + late-decoder layers and keeps only the middle third at full
  base (approximates Cubiq's block_3/4 SDXL schedule). Multi-image
  references average the CLIP-Vision embeddings pre-projection and run the
  projection once on the centroid rather than once per image. Flux gets
  image-prompting through **Redux** instead of classic IP-Adapter — a real
  IPA checkpoint is explicitly refused on Flux with a message pointing at
  Redux, because Flux's DiT has no cross-attention K/V slot for IPA's image
  tokens. **Still refused, real blockers:** Flux classic IP-Adapter (would
  need a new adapter class hooking DiT block-level modulation, not
  cross-attention); other architectures (SD3, Z-Image, Flux.2, AuraFlow,
  Chroma, F-Lite, Ernie) have no published IPA checkpoints to load;
  multi-adapter stacking (Swarm's UI exposes one IPA slot).
- [x] **Refiner StepSwap** — SDXL only (`ImageFeatures.Refiner`). Swaps
  base→refiner UNet at `(1-Strength)*totalSteps`, rebuilds ADM per branch
  (base cond=6.0/uncond=2.5 vs. the refiner's 5-value aesthetic-score-only
  ADM, since CrossAttentionDim differs: 2048 concat vs. refiner's 1280
  CLIP-G-only). ControlNet and IP-Adapter are both disabled during the
  refiner phase — their conditioning is sized for the base UNet, not the
  4-level refiner. `StepSwapNoisy` (re-noise at the swap point) stays
  deferred as a minor variant.
- [~] **Upscaling** — `HartsyInference.Vision/Upscale` (Real-ESRGAN, image)
  and SeedVR2 restoration (`engine.Restore`, video+image, wired into this
  extension's Video Restore group) both ship. The only remaining gap is the
  image-side "Refiner Upscale Method" dropdown — see P2.

## Tier 2 — Sampling / quality

- [!] **Scheduler-type selection (karras / exponential / …) and batch-size >
  1** — both are permanent gaps stated at the request-builder call site, not
  TODOs: the Engine resolves each family's own canonical schedule (no
  Comfy-Scheduler analogue), and Swarm drives batching itself with one
  `Generate` call per image (no latent-batched path). See P1.
- [ ] **Per-arch sampler & scheduler defaults registry** — declare allowed
  (sampler, scheduler) pairs per arch in `ModelSupport.cs` so Swarm's UI
  surfaces the right options.
- [ ] **CFG Rescaling / RenormCFG / CFGZeroStar / TCFG** — guidance-math
  variants, each a small loop tweak.
  ([Comfy ref: `WorkflowGeneratorSteps.cs:177-210`](../../../BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorSteps.cs#L177-L210))
- [ ] **PAG (Perturbed-Attention Guidance)** — attention-hook based. Swarm
  already exposes the param; ignored today.
- [ ] **SAG (Self-Attention Guidance)** — same shape as PAG, same status.

## Tier 3 — Ecosystem

- [ ] **Side-model registry expansion** — Comfy auto-downloads ~40 encoder
  variants; pure additions as new architectures need them.
- [ ] **SD3 LoRA path** — scaffolded upstream but untested (`Sd3Recipe` does
  not declare `ImageFeatures.Lora`).
- [ ] **TensorRT compile WebAPI endpoint** — replicate `DoTensorRTCreateWS`
  against HartsyInference's TRT path.
- [ ] **LoRA extraction utility** — diff two checkpoints, write a LoRA. New
  endpoint in `HartsyInferenceWebAPI.cs`.

## Tier 4 — Niche / advanced

- [x] ~~**Video architecture breadth**~~ — Wan (all mapped sizes plus
  VACE/Animate/S2V checkpoint-header variants), LTX-Video 0.9 + 2,
  HunyuanVideo, Kandinsky-5 video, and Lance all have recipes registered in
  `VideoRecipeRegistry.BuildDefaults()`; `VideoOutputEncoder` (ffmpeg mux)
  handles FPS/format/boomerang/trim for all of them. MiniMax-H3 is also
  registered but `Construct()` throws until MiniMax publishes the checkpoint
  — registered isn't usable there. Mochi, SVD, and Cosmos video have no
  family mapping at all — check `ModelSupport.cs` before assuming a video
  architecture is unsupported.
- [ ] **Textual inversion embeddings** — needs tokenizer token-injection; not
  present in any text encoder today.
- [ ] **Seamless tiling** — no padding-mode hooks in the conv path.
- [ ] **TeaCache / EasyCache step-skipping** — latent cache intercept points
  in the diffusion loop. Low ROI vs. quality tradeoff.
- [ ] **NAG (Normalized Attention Guidance)** — same shape as PAG; low demand.
- [ ] **GLIGEN spatial conditioning** — SD 1.5-only, superseded by ControlNet.
  Low priority.

## FLUX.1 Tools (BFL's official Flux conditioning suite)

- [x] **FLUX.1 Canny** — detected from `x_embedder.weight` input-dim shape
  (128 vs. 64) plus filename keyword; shares the `flux-1` family with vanilla
  Flux, no separate compat class.
- [x] **FLUX.1 Redux** — image-prompt adapter via SigLIP encoder + token-concat
  (not cross-attention K/V). `redux.*` Extra keys (`ReduxStyleModel`,
  `ReduxMultiply`, `ReduxMerge`, `ReduxApplyStart`) map from Comfy's
  style-model params.
- [ ] **FLUX.1 Depth** — detected at load time but refused: needs the
  existing ControlNet Depth preprocessor threaded through the Flux path
  specifically. Same pipeline shape as Canny otherwise.
- [ ] **FLUX.1 Fill** — detected at load time but refused: needs masked-image
  + mask preprocessing wired through the dedicated 32-channel input
  (different from the blend-on-vanilla mask path used elsewhere).

## Already shipped (not itemized above)

Single + multi LoRA (SD 1.5 / SDXL / Flux.1 / Wan); img2img; inpaint,
ControlNet, and IP-Adapter per the arch coverage above; refiner PostApply +
StepSwap; FLUX.1 Canny + Redux; tiled VAE decode; Real-ESRGAN + SeedVR2
upscale/restore; side-model auto-download; TAESD/latent2rgb live previews;
CUDA + Vulkan + CPU backends; cancellation; model hot-swap; pipeline cache.
For the current image/video/music architecture list — which grows
independently of this doc — read `Generation/ModelSupport.cs`'s `_families`
table, or query `SupportedArchitectures` / `PendingArchitectures`.

## Upstream-blocked items — file these as HartsyInference issues

1. Tiled `VaeEncoder` + latent-upscale-and-redenoise loop for hires-fix (P2)
   — pixel-space upscale already ships; this is the missing 2-pass path.
2. `ImageFeatures.Regional` support in at least one image recipe, to light up
   the `<region:>` / `<segment:>` plumbing that already exists (P5).
3. Textual inversion token injection.
4. Seamless tiling hooks in the latent loop.
5. FP4 GEMM in CudaBackend — unblocks Flux.2 Klein 9B / Dev with Comfy's
   canonical fp4-mixed encoders, and the Ideogram 4 nf4 checkpoint variant.
