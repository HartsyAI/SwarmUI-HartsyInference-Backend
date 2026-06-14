# 11 — Comfy Parity Punchlist

Canonical "what's left to ship" list. Supersedes the older
[`02-Comfy-Feature-Parity-Matrix.md`](./02-Comfy-Feature-Parity-Matrix.md) for
current status — that doc is kept as historical context.

Last refresh: 2026-06-10. Excludes Comfy-only features that don't apply
(workflow editor, custom-node packs, WebSocket passthrough, subprocess
self-start, ComfyUser session management).

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked
upstream (waiting on HartsyInference.Core).

---

## Production push P1–P7 (active order, set 2026-06-10)

The prioritized path to "credible default backend". Each item lists the exact
touch points so any of them can be picked up cold.

- [~] **P1 — Sampler / scheduler params + clip skip.**
  - *Sampler:* SD-family pipelines already route `request.Scheduler` through
    `SchedulerFactory` (euler / ddim / dpm++2m / lcm). Extension side: register
    a `Sampler` param (ID `sisampler`, flag `hartsyinference`, group Sampling),
    map Swarm names → factory names in `SamplingParamResolver`, thread into
    `Sd15Loader` / `SdxlLoader` / `RefinerLoader` request construction. Flow-match
    archs ignore it by design (log Verbose, don't refuse).
  - *Clip skip:* upstream — add `int layersFromEnd` to
    `ClipTextEncoder.Encode/EncodePenultimate` (today hardcoded final /
    penultimate), `TextToImageRequest.ClipSkip int?`, thread in SD1.5 + SDXL
    pipelines. Extension — read core `T2IParamTypes.ClipStopAtLayer`, pass
    through. SD1.5/SDXL only; others ignore.
- [ ] **P2 — Hires fix / 2-pass upscale (`RefinerUpscale != 1`).**
  Today refused at validation. Plan: (a) upstream tiled `VaeEncoder`
  (`EncodeTiled`, sibling of `DecodeTiled` — same im2col cliff); (b) latent
  upscale path first (bicubic on latent, re-denoise at `RefinerControl`
  creativity — no new models needed); (c) ESRGAN ONNX runner as a follow-up
  for pixel-space upscale parity (`ModelAutoDownloader` entry + ONNX runtime
  decision shared with P5).
- [ ] **P3 — Graceful refusal of unsupported prompt syntax.** Swarm prompt
  features Comfy services that we silently mangle today: `<segment:…>`,
  `<region:…>`, `<object:…>`, `<break>`, `<from:…>`/`<to:…>`, `<alternate:…>`,
  `<clear:…>`. Detect in `IsValidForThisBackend` (use
  `PromptRegion`-style scan of raw prompt), add refusal reason naming the
  feature. Lift each refusal as real support lands (P5 lifts `<segment:`).
- [ ] **P4 — Variation seed.** No upstream change needed:
  `TextToImageRequest.InitialNoise` overrides seed noise. Extension computes
  `slerp(noise(seed), noise(varseed), strength)` with upstream
  `SeedGenerator.CreateNoise` and passes it. Start with SD1.5 + SDXL
  (spatial `[1,4,H/8,W/8]` latents), then Flux/SD3 (packed shapes). Advertise
  `variation_seed` flag; refuse per-arch where unwired.
- [ ] **P5 — `<segment:face>` via YOLO.** The one segmentation feature casual
  users depend on. Needs: ONNX runtime decision (Microsoft.ML.OnnxRuntime in
  the extension, NOT upstream — keep HartsyInference pure), YOLOv8-face ONNX
  auto-download, detect → mask → MaskBlur/Grow → re-denoise crop via existing
  inpaint path. (b) ControlNet Depth/OpenPose preprocessors ride the same
  ONNX runtime once it exists.
- [ ] **P6 — Architecture long-tail by demand.** Order: Qwen Image Edit
  (+Plus) — needs upstream edit-conditioning in `QwenImagePipeline`
  (VL image input + latent concat); Wan 2.1/2.2 14B (needs upstream configs +
  block-streaming validation at 14B); Lumina 2 / OmniGen 2 / Lens (pipelines
  exist upstream, loaders unwired — cheap wins when asked for).
- [~] **P7 — Ideogram 4 loader.** Wired 2026-06-11: `Ideogram4Loader.cs` (folder
  walk → converter → dual DiT + Qwen3-VL-8B + Flux.2 VAE → pipeline), Steps→preset
  mapping, chat-template tokenize with right-pad trim, ≥22 GB CUDA VRAM gate at
  load, cache/dispatch/validation plumbed. **Remaining:** E2E verify on a ≥24 GB
  host (the 3060 can't run it); decide whether to surface structured-JSON prompt
  params (Swarm core's new ideogram region UI) → map to upstream `StructuredPrompt`;
  optional magic-prompt expansion. Upstream context:
  `Ideogram4Pipeline` (Qwen3-VL-8B 13-layer tap → dual 9.3B single-stream
  DiTs, asymmetric CFG, logit-normal Euler, Flux.2 VAE, fixed-constant latent
  norm), `Ideogram4CheckpointConverter` (diffusers + Comfy-Org layouts),
  `Ideogram4SamplerPreset` (Turbo12 / Default20 / Quality48), prompt dialect +
  structured-JSON `StructuredPrompt` subsystem, scheduler/prompt/generation
  tests. Extension work: (a) register `ideogram-4` model class + detection
  (key signature from converter: `transformer/` + `unconditional_transformer/`
  folders or `ideogram4*` single files) — Swarm core has no detection for it;
  (b) `Ideogram4Loader.cs` — Qwen3-VL-8B side-model auto-download, Qwen3
  chat-template tokenization, map Steps→nearest preset (12/20/48) or custom,
  plain-text → minimal StructuredPrompt JSON wrap (Magic Prompt API optional,
  off by default); (c) VRAM gate: TWO 9.3B DiTs resident → demands
  block-streaming both transformers; refuse below a measured VRAM floor with
  a clear message. fp8 checkpoint only (nf4 blocked on FP4 GEMM upstream).
  **License note:** weights are "Ideogram 4 Non-Commercial" — surface this in
  the model description/UI, relevant for Hartsy deployments.

---

## Tier 1 — High-impact core features

- [x] **Inpainting / masks** — Done 2026-05-07 for SDXL, Flux, SD3 via
  blend-on-vanilla path. Upstream `ImageToImageRequest` gained `Mask` +
  `RecompositeAtEnd`; pipelines blend re-noised source per step + pixel-space
  recomposite at end. Swarm-side `MaskResolver.cs` handles `MaskImage` +
  `MaskGrow` (separable max-filter dilation) + `MaskBlur` (Gaussian).
  `MaskShrinkGrow` ("inpaint only masked", crop-to-bbox) intentionally
  deferred — full-image inpaint covers the common path. SD 1.5 + Z-Image
  refused at validation until their pipelines get the same blend hooks
  (mechanical follow-up).
- [~] **ControlNet (single + stack)** — Phase B.1 done 2026-05-07: SDXL-base
  ControlNet (single + stack) wired end-to-end with Canny preprocessor.
  Upstream `ControlNet.LoadWeights` + `Forward` implemented (hint encoder,
  mirrored down/mid blocks, zero conv tower); `UNet.Forward` accepts optional
  residuals; `SdxlPipeline.GenerateFromTokens` accepts
  `IReadOnlyList<ControlNetConditioning>`. Swarm-side `ControlNetResolver` +
  `ControlNetWeightLoader` read the 3-slot Comfy ControlNet param holders.
  Stacking via summed residuals (matches diffusers). CFG runs CN once with
  cond text emb, residuals shared across both branches (guess_mode=True
  semantics) — strict per-branch CN passes are a future optimization.
  **Remaining for full parity:** SD 1.5 ControlNet wiring (upstream class
  supports it, just need SD15 pipeline integration); Flux ControlNet
  (different DiT architecture, separate adapter needed); CN start/end step
  ranges (params exist, currently always full-range).
- [~] **ControlNet preprocessors** — Phase B.1: Canny shipped (pure C#:
  Sobel + NMS + hysteresis, defaults match Comfy's `CannyEdgePreprocessor`
  thresholds 100/200). **Remaining:** Depth (DepthAnything ONNX), OpenPose
  (DWPose ONNX), Lineart (Sobel variant + optional ONNX), all need ONNX
  runtime wiring + bundled models — Phase B.2.
- [x] **IP-Adapter (standard + plus + plus-face)** — SD 1.5 + SDXL fully wired
  with weight types, start/end step gating, multi-image averaging.
  Upstream: `ClipVisionEncoder` (ViT-H/14), `IpAdapterStandardProjection`
  (MLP → 4 tokens), `IpAdapterPlusResampler` (Perceiver — 4 layers, learnable
  queries + cross-attn over concat[image_patches, latents] + FFN),
  `IpAdapterScaleSchedule` (per-cross-attn-layer scale arrays driven by
  weight-type + step gating). UNet paths thread `IReadOnlyList<float>?
  ipaScalePerLayer` plus K_ip / V_ip flat-list + image tokens through every
  cross-attention sub-layer; image-attention sums onto text-attention before
  `to_out`. Swarm side: multi-image averaging (CLIP-Vision outputs averaged
  pre-projection), reads <c>ipadapterweighttype</c> / <c>ipadapterstart</c> /
  <c>ipadapterend</c> Comfy params, cached per IPA file path. **Refused with
  technical justification (not implemented):** FaceID variants (need
  InsightFace ArcFace runtime — different image encoder, separate
  infrastructure), Flux IPA (DiT cross-attention layout — separate adapter
  class, ~1500 LOC), Z-Image / SD3 / Flux.2 / AuraFlow / Chroma / F-Lite /
  ERNIE IPA (no published checkpoints exist for any of these architectures).
- [x] **Refiner StepSwap** — Done 2026-05-07 for SDXL. Upstream
  `RefinerSwapConfig` record + per-step branch in `SdxlPipeline.RunDenoiseLoop`:
  swaps base → refiner UNet at `(1 - Strength) * totalSteps`, slices CLIP-G out
  of the concat'd embedding for the refiner's CrossAttentionDim=1280, rebuilds
  ADM with per-branch aesthetic scores (cond=6.0, uncond=2.5). ControlNet
  disabled during refiner phase (zero convs are base-shaped). Backend
  pre-loads refiner outside the lambda so `AddLoadStatus` surfaces in the UI;
  post-pass skipped when StepSwap was applied. `PostApply` mode unchanged.
  `StepSwapNoisy` deferred (re-noise at swap is a minor variant).
- [!] **Upscaling (RealESRGAN / latent upscale)** — blocked upstream:
  HartsyInference.Core has no upscaler loaders.

## Tier 2 — Sampling / quality

- [ ] **Per-arch sampler & scheduler defaults registry** — declare allowed
  (sampler, scheduler) pairs per arch in `ModelSupport.cs` so Swarm's UI
  surfaces the right options.
- [ ] **CFG Rescaling / RenormCFG / CFGZeroStar / TCFG** — guidance-math
  variants. Each is a small loop tweak.
  ([Comfy ref: `WorkflowGeneratorSteps.cs:177-210`](../../../BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorSteps.cs#L177-L210))
- [ ] **PAG (Perturbed-Attention Guidance)** — attention-hook based. Param is
  already in Swarm; we currently ignore it.
- [ ] **SAG (Self-Attention Guidance)** — same shape as PAG; param already
  exists Swarm-side.
- [ ] **Style Model (Flux.1 Redux)** — Flux-only image encoder; new
  `StyleModelResolver` + side-model entry.

## Tier 3 — Ecosystem

- [ ] **Side-model registry expansion** — Comfy auto-downloads ~40 encoder
  variants; we have ~12. Missing: LTX2 connector, HiDream CLIP variants, UMT5,
  ByteT5, HunyuanVideo LLaVA, Wan / Lumina text encoders. Pure additions to
  `SideModels.cs`.
- [ ] **SD3 LoRA path** — scaffolded upstream but untested.
- [ ] **TensorRT compile WebAPI endpoint** — replicate
  `DoTensorRTCreateWS` against HartsyInference's TRT path.
- [ ] **LoRA extraction utility** — diff two checkpoints, write LoRA. New
  endpoint in `HartsyInferenceWebAPI.cs`.

## Tier 4 — Niche / advanced

- [x] ~~**Video models (first wave)**~~ — Wan 2.2 TI2V-5B (T2V + I2V + LoRA)
  and LTX-Video 0.9 (T2V) shipped, with `VideoOutputEncoder` (ffmpeg mux),
  FPS/format/boomerang/trim params. Remaining video archs (Hunyuan family,
  LTXV2, Mochi, SVD, Cosmos, Wan 14B/VACE) blocked upstream — Wan 14B is P6.
- [!] **Regional prompting (SAM2)** — blocked upstream: no SAM2 segmentation.
- [!] **Seamless tiling** — blocked upstream: no latent tiling hooks.
- [ ] **TeaCache / EasyCache step-skipping** — latent cache intercept points in
  diffusion loop. Low ROI vs quality tradeoff.
- [!] **YOLO + SAM2 detection preprocessing** — blocked upstream.
- [ ] **NAG (Normalized Attention Guidance)** — same shape as PAG; post-2025
  research, low demand.
- [ ] **GLIGEN spatial conditioning** — SD1.5-only, superseded by ControlNet.
  Low priority.

## FLUX.1 Tools (BFL's official Flux conditioning suite)

- [x] **FLUX.1 Canny** — Done 2026-05-07. Detected from `x_embedder.weight`
  shape (input dim 128 vs 64) + filename keyword. `FluxConfig.Flux1Tools`
  preset; `FluxTransformer.XEmbedInputDim` exposes the input width;
  `FluxPipeline.GenerateFromTokens` accepts optional `Tensor? controlImage`,
  VAE-encodes once, packs, concatenates onto the packed noise per step
  (`ConcatPackedFeatureDim` helper). Swarm side: reads
  `Controlnets[0].Image` (or `InitImage` fallback) as the reference, runs
  the existing `CannyPreprocessor`, scales `[0, 1] → [-1, 1]` for Flux VAE,
  threads to pipeline. No new compat class — Flux Canny shares `flux-1`
  with vanilla Flux; pipeline mode is detected from checkpoint shape.
- [ ] **FLUX.1 Depth** — Detected at load time but refused with clear
  message: needs DepthAnything-V2 ONNX preprocessor (~700MB ONNX model).
  Same pipeline path as Canny, just different input preprocessing —
  follow-up.
- [ ] **FLUX.1 Fill** — Detected at load time but refused: needs masked-image
  + mask preprocessing wired through the Flux pipeline (analogous to my
  blend-on-vanilla mask path but for the dedicated 32-channel input). Same
  pipeline shape as Canny, follow-up.
- [ ] **FLUX.1 Redux** — Image-prompt adapter. Different from CLIP-Vision
  IPA: uses SigLIP encoder + token-concat (not cross-attention K/V). Needs
  a new `SiglipVisionEncoder`, a `FluxReduxAdapter` projection module, and
  `FluxPipeline` extension to inject Redux tokens at the right point.
  Follow-up.

## Already shipped (not in this list)

Single + multi LoRA (SD1.5 / SDXL / Flux / Wan), image arches SD1.5, SDXL,
SD3/3.5, Flux.1 (+Canny tool), Flux.2 base/Klein-4B, Z-Image, Chroma V1,
AuraFlow, F-Lite, Anima, HiDream-I1, Qwen-Image; video arches Wan 2.2 TI2V-5B +
LTX-Video; tiled VAE decode, side-model auto-download, refiner
PostApply + StepSwap, img2img, inpaint (SDXL/Flux/SD3), IP-Adapter
(SD1.5/SDXL), ControlNet SDXL+Canny, TAESD/latent2rgb live previews,
CUDA + Vulkan + CPU backends, cancellation, model hot-swap, pipeline cache.

## Upstream-blocked items — file these as HartsyInference issues

1. Upscaler loader infrastructure (RealESRGAN / latent upscalers).
2. Video pipeline framework (HunyuanVideo, LTX, Wan, HiDream).
3. SAM2 segmentation path.
4. Seamless tiling hooks in the latent loop.
5. YOLO model loader.
6. FP4 GEMM in CudaBackend (separately tracked — unblocks Flux.2 Klein 9B / Dev
   with Comfy's canonical fp4-mixed encoders).
