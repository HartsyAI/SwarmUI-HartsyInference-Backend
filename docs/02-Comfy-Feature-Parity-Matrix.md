# 02 — Comfy Feature Parity Matrix

This is the canonical comparison between what the ComfyUI backend extension
provides to SwarmUI users and what the HartsyInference backend provides today.
Each row is something the HartsyInference backend has to either match, defer,
or explicitly not support.

**Last full refresh: 2026-06-10** (code-level survey of both extensions —
feature flags, validation paths, and dispatch tables, not READMEs).
Working items live in [`11-Comfy-Parity-Punchlist.md`](./11-Comfy-Parity-Punchlist.md);
this doc is the status map, that doc is the work queue.

**Status legend:**

| Status | Meaning |
|--------|---------|
| ✅ Shipped | Wired end-to-end in the extension and serviced at generation time |
| 🟢 Partial | Works for some architectures / modes; the rest are refused cleanly at validation |
| 🟡 Planned | HartsyInference has the pieces (or they're cheap); extension wiring not started |
| 🔴 Blocked | Needs upstream HartsyInference work first |
| ⚫ Out of scope | Explicitly not pursuing |

## A. Core text-to-image

| Feature | Status | Notes |
|---------|--------|-------|
| Prompt + negative prompt | ✅ | All pipelines |
| CFG scale / Steps / Width / Height / Seed | ✅ | All pipelines |
| EndStepsEarly | ✅ | `SamplingParamResolver`, matches Comfy's truncating step math |
| Sampler selection | 🟢 | SD 1.5 / SDXL / SDXL-Refiner accept euler / ddim / dpm++2m / lcm via `SchedulerFactory`. Flow-match archs (Flux, SD3, Z-Image, …) use their canonical scheduler by design (same as Comfy's effective behavior for those). Param registration in progress — see punchlist P1. |
| Scheduler type (karras / exponential / …) | 🔴 | Sigma-schedule variants not implemented upstream; epsilon/v-pred schedulers use their built-in schedules |
| Variation seed | 🟡 | `TextToImageRequest.InitialNoise` override exists upstream — extension can slerp base+variation noise itself. Punchlist P4. |
| Clip skip (`CLIP Stop At Layer`) | 🟡 | `ClipTextEncoder` hardcodes final/penultimate today; needs a small upstream `layersFromEnd` param + request field. Punchlist P1. |
| Batch size > 1 | 🟢 | One image per pipeline call; Swarm's scheduler issues sequential calls. Latent-batched generation not implemented. |
| Sigma min/max, churn | 🔴 | Not surfaced upstream |
| Resolution presets / aspect ratio | ✅ | Swarm-side; we receive resolved width/height |

## B. Model architectures

### Image — shipped in the extension

| Architecture | Status | Notes |
|--------------|--------|-------|
| SD 1.5 | ✅ | LoRA ✅ img2img ✅ inpaint ❌ IPA ✅ CN ❌ |
| SDXL (+ official Refiner) | ✅ | LoRA ✅ img2img ✅ inpaint ✅ IPA ✅ CN ✅ (Canny) refiner PostApply/StepSwap ✅ |
| SD 3 / 3.5 Medium / 3.5 Large | ✅ | img2img ✅ inpaint ✅ LoRA ❌ (refused) |
| Flux.1 (Schnell/Dev/Krea) + FLUX.1 Canny | ✅ | LoRA ✅ img2img ✅ inpaint ✅; Canny tool-model auto-detected from x_embedder shape |
| Flux.2 (base, Klein 4B) | 🟢 | Klein 9B / Dev refused — FP4-mixed encoders need FP4 GEMM upstream |
| Chroma V1 | ✅ | Radiance / Zeta refused — need `ChromaConfig` presets upstream |
| AuraFlow v0.2/v0.3 | ✅ | Single-file checkpoints |
| F-Lite | ✅ | Diffusers-folder layout; wired-untested E2E |
| Z-Image (Turbo, Base) | ✅ | img2img ✅ |
| Anima (Cosmos-Predict2 2B) | ✅ | t2i only |
| HiDream-I1 | ✅ | 4 encoders incl. Llama-3.1-8B; needs Llama tokenizer assets in the HartsyInference build |
| Qwen-Image | ✅ | t2i only; Qwen2.5-VL-7B MLLM encoder |
| **Ideogram 4** | ✅ | Wired 2026-06-11 (t2i; Steps→official presets; ≥22 GB VRAM gate; non-commercial license). E2E verify pending on a ≥24 GB host. |

### Image — pending (refused with a clear message)

| Architecture | Blocker |
|--------------|---------|
| Ernie Image | No Ernie tokenizer upstream |
| HunyuanImage 2.1 | Upstream pipeline uses T5 stand-in instead of the real Qwen2.5-VL + byT5 glyph stream — output wouldn't be faithful |
| Chroma Radiance / Zeta-Chroma | `ChromaConfig` presets missing upstream |
| Flux.2 Klein 9B / Dev | FP4 GEMM missing upstream |

### Image — Comfy runs, HartsyInference has nothing (architecture long-tail)

SD2, Stable Cascade, PixArt α/Σ, Sana, Lumina 2 (pipeline exists upstream,
loader unwired), Ovis, Longcat, OmniGen 2 (pipeline exists upstream, loader
unwired), Qwen Image Edit / Edit Plus, Kandinsky 5 image, Lens (pipeline
exists upstream, loader unwired). Wire on demand — Qwen Edit is the
highest-demand item here (punchlist P6).

### Video

| Architecture | Status | Notes |
|--------------|--------|-------|
| Wan 2.2 TI2V-5B | ✅ | T2V + I2V (init-image first-frame conditioning), LoRA ✅; end-frame / extend / audio refused |
| LTX-Video 0.9 | ✅ | T2V only |
| Wan 2.1 / Wan 2.2 14B / VACE | 🔴 | Not upstream |
| Hunyuan Video family, Mochi, LTXV2, SVD, Cosmos video, Kandinsky video | 🔴 | Not upstream |
| Two-stage image-then-animate (Video Model param) | 🟡 | Refused today; mechanical once a second pipeline can chain |
| Frame interpolation (RIFE / FILM / GIMM) | 🔴 | Comfy-side installable; would need ONNX runners |

## C. LoRAs and adapters

| Feature | Status | Notes |
|---------|--------|-------|
| Single + multi LoRA w/ strength | ✅ | SD 1.5, SDXL, Flux, Wan 2.2 — per-gen non-destructive apply (shallow-clone weight dicts) |
| LoRA on SD3 / Z-Image / Flux.2 / others | 🟡 | SD3 scaffolded upstream, untested; others need `LoraTarget` mappings |
| LCM / Hyper / Lightning / Turbo LoRAs | ✅ | LoRA + low-step scheduler |
| Textual inversion embeddings | 🔴 | Needs tokenizer token-injection upstream |

## D. ControlNet / IP-Adapter / image conditioning

| Feature | Status | Notes |
|---------|--------|-------|
| ControlNet single + stack | 🟢 | SDXL only; summed residuals, 3-slot params honored |
| ControlNet preprocessors | 🟢 | Canny only (pure C#). Depth / OpenPose / Lineart need ONNX runners — punchlist P5b |
| ControlNet start/end ranges | 🟡 | Params read, currently full-range |
| IP-Adapter standard / Plus / Plus-Face | ✅ | SD 1.5 + SDXL; weight types + start/end gating + multi-image averaging |
| IP-Adapter FaceID / InstantID | 🔴 | Needs InsightFace ArcFace runtime |
| Flux Redux / style models | 🔴 | Needs SigLIP encoder upstream |
| ReVision | 🔴 | SDXL CLIP-Vision conditioning not wired |
| Img2img (init + creativity) | ✅ | SD 1.5 / SDXL / Flux / SD3 / Z-Image (Flux.2: mechanical TODO) |
| Inpaint (mask) | 🟢 | SDXL / Flux / SD3 blend-on-vanilla incl. MaskGrow/MaskBlur; SD 1.5 + Z-Image pending hooks |
| Outpaint | 🟢 | Works where inpaint works (Swarm-side mask construction) |
| GLIGEN | ⚫ | Superseded by ControlNet |

## E. Refiners and multi-stage

| Feature | Status | Notes |
|---------|--------|-------|
| SDXL Refiner PostApply | ✅ | Any base architecture |
| SDXL Refiner StepSwap | ✅ | SDXL base only |
| StepSwapNoisy | 🟡 | Minor variant, deferred |
| Refiner Upscale ≠ 1 (hires fix / 2-pass) | 🔴 | Needs tiled `VaeEncoder` + an upscaler (ESRGAN ONNX or latent) — punchlist P2. Refused at validation today. |
| Refiner VAE override | 🟡 | Refused; mechanical |

## F. Prompt syntax & segmentation (Swarm prompt features Comfy services)

| Feature | Status | Notes |
|---------|--------|-------|
| `<segment:…>` auto-refinement (YOLO face fix) | 🔴 | Needs YOLO runtime — punchlist P5. Until then: refuse, don't mangle (P3). |
| `<region:>` / `<object:>` regional prompting | 🔴 | Needs masked-conditioning loop support upstream |
| `<break>`, prompt from/to, alternation | 🔴 | Needs chunked text-encoder conditioning |
| CLIP-Seg / SAM2 masking | 🔴 | No segmentation models upstream |
| Graceful refusal of all of the above | 🟡 | Punchlist P3 — detect syntax at validation, refuse with message |

## G. Upscale / restore / postprocessing

| Feature | Status | Notes |
|---------|--------|-------|
| ESRGAN-family upscale models | 🔴 | Punchlist P2 (ONNX runner) |
| Face restore (CodeFormer / GFPGAN) | ⚫ | Not planned for v1 |
| Background removal (rembg) | ⚫ | Not planned for v1 |
| Seamless tiling | 🔴 | Needs padding-mode hooks in conv path upstream |

## H. Live progress and UX

| Feature | Status | Notes |
|---------|--------|-------|
| Per-step progress + ETA | ✅ | Heartbeat-cadence callbacks, 5% log throttle |
| Per-step preview image | ✅ | TAESD per-arch or latent2rgb fallback, ≤4/sec |
| Cancel mid-generation | ✅ | `CancellationToken` throughout |
| Final image streamed back | ✅ | Standard `takeOutput` |

## I. Backend management

| Feature | Status | Notes |
|---------|--------|-------|
| Device selection (CUDA ordinal / Vulkan / CPU, auto-probe) | ✅ | |
| FreeMemory / model eviction | ✅ | `PipelineCache` LRU + EvictAll |
| Multi-GPU (one backend per GPU) | ✅ | Swarm queues across instances |
| OverQueue | ✅ | Serialized via gen lock, mirrors Comfy semantics |
| Quantized checkpoint formats | 🟢 | fp8 (streamed) ✅; GGUF 🔴; NF4/FP4 🔴 (blocks Flux.2 Klein 9B + Ideogram 4 nf4 variant — fp8 variant is fine) |
| TeaCache / EasyCache step-skipping | 🟡 | Loop intercept points; low priority |

## J. Permissions and multi-user

All inherited from Swarm core — ✅ (`use_hartsyinference` / `admin_hartsyinference` registered).

## K. Out of scope (Comfy-specific by nature)

Workflow editor tab, stored API-format workflows, custom Comfy node packs,
`InstallableFeatures` node-pack installer, TensorRT compile, Comfy Manager
interop, HTTP/WebSocket bridge internals, `ComfyUser` session management.

## L. WebAPI

`HartsyInferenceWebAPI` is still a stub (probe-model / list-pipelines /
clear-cache / device-info / supported-archs routes planned, commented out).
Comfy's workflow-management routes are N/A; LoRA extraction may come later.

## Priority summary (June 2026)

See [`11-Comfy-Parity-Punchlist.md`](./11-Comfy-Parity-Punchlist.md) §"Production
push P1–P7" for the active order: P1 sampler/scheduler/clip-skip params,
P2 hires-fix + upscaler, P3 prompt-syntax graceful refusal, P4 variation seed,
P5 YOLO `<segment:face>`, P6 Qwen Edit + Wan 14B, P7 Ideogram 4 loader.
