# SwarmUI HartsyInference Backend

> **Status:** Working beta, broadly verified. The model fleet has been verified
> end-to-end with real weights (57 architectures across image/video/audio via
> coordinated verification passes), and a sustained performance campaign now has
> **three image models generating faster than ComfyUI on the same GPU** (RTX 4090,
> identical request through the SwarmUI API): Z-Image-Turbo 2.95s vs 3.1s,
> Krea2-Turbo 4.50s vs 6.5s, Qwen-Image 40.9s vs 54.8s. Video is live in production
> (Wan 2.x T2V/I2V/VACE/Animate/S2V, LTX incl. LTX-2 with audio, HunyuanVideo,
> Kandinsky-5). Img2img, inpaint, LoRA, refiner (verified live), ControlNet
> (SDXL + SD1.5 + union-type + FLUX-DiT, full in-engine preprocessor set incl.
> segmentation), IP-Adapter (standard/Plus/Plus-Face/FaceID/FaceID-PlusV2),
> FLUX.1 Kontext/Fill/Canny/Depth/Redux, GGUF checkpoints, live previews, and
> cancellation all work. Remaining gaps are tracked in [Known limitations](#known-limitations), the
> [parity matrix](./docs/02-Comfy-Feature-Parity-Matrix.md), and the engine's
> [benchmark log](https://github.com/HartsyAI/HartsyInference/blob/main/benchmarks/results/).

## What this is

A SwarmUI backend extension that runs Stable Diffusion / FLUX / SDXL / etc. inference
**entirely in C#** through the [HartsyInference](#hartsyinference) library — no Python,
no ComfyUI, no external process required.

The long-term goal is to **replace ComfyUI as the default SwarmUI backend** while
preserving full feature parity (see [`docs/02-Comfy-Feature-Parity-Matrix.md`](./docs/02-Comfy-Feature-Parity-Matrix.md)).

## Why

ComfyUI is excellent, but the way it's wired into Swarm has costs:

- **Python runtime + venv:** large install, slow startup, version drift on `torch` / `xformers` / CUDA.
- **Two processes:** Swarm (C#) talks HTTP/WebSocket to a separate Comfy process; every generation round-trips JSON + image bytes across that boundary.
- **Workflow JSON as IR:** Swarm builds a Comfy workflow graph, ships it across, Comfy interprets it. Translation overhead and a fragile contract (a Comfy node rename can break Swarm).
- **Custom-node sprawl:** users have to install `ComfyUI_IPAdapter_plus`, `was-node-suite-comfyui`, etc. piecemeal.

A pure-C# backend in-process means: one binary, one model loader, one cache, no IPC,
and direct access to Swarm's image/parameter/cache types.

## What works today

### Architectures

All benchmark times are end-to-end wall clock through the SwarmUI API — the identical
request routed to the ComfyUI backend, then this backend, on the same GPU. Warm median
of 3 runs, randomized seeds, outputs visually verified. "—" = no same-configuration
ComfyUI baseline recorded yet.

#### Image models (1024×1024)

| Model | Compat IDs | Status | Steps | Hartsy | ComfyUI | GPU |
|---|---|---|---:|---:|---:|---|
| Z-Image (Turbo, Base) | `z-image` | ✅ Verified | 8 | **2.95 s** | 3.1 s | RTX 4090 |
| Krea 2 (Turbo, Base) | `krea-2` | ✅ Verified | 8 | **4.50 s** | 6.5 s | RTX 4090 |
| Flux.1 Schnell | `flux-1` | ✅ Verified | 4 | 10.5 s | — | RTX 4090 |
| Flux.2 Klein 4B | `flux-2-klein-4b` | ✅ Verified | 10 | 15.1 s | — | RTX 4090 |
| Ideogram 4 (9.3B dual-DiT) | `ideogram-4` | ✅ Verified (≥22 GB VRAM; non-commercial license) | 20 | 19.5 s | 17.0 s | RTX 4090 |
| Flux.1 Dev / Krea / Canny / Kontext | `flux-1` | ✅ Verified (grind ongoing — was 72.4 s) | 20 | 31.0 s | 12.5 s | RTX 4090 |
| AuraFlow v0.2/v0.3 | `auraflow-v1` | ✅ e2e; perf round queued | 20 | 31.4 s | 14.0 s | RTX 4090 |
| SDXL + official Refiner | `stable-diffusion-xl-v1` | ✅ Verified incl. live refiner (scheduler-op work queued) | 20 | 33.0 s | 3.7 s | RTX 4090 |
| Qwen-Image (20B, GGUF) | `qwen-image` | ✅ Verified | 20 | **40.9 s** | 54.8 s | RTX 4090 |
| ERNIE-Image (8B fp8) | `ernie-image` | ✅ Verified (grind queued) | 20 | 50.6 s | 24.0 s | RTX 4090 |
| Chroma V1 | `chroma` | ✅ Verified (grind ongoing — was 550 s) | 20 | 63.2 s | 16.6 s | RTX 4090 |
| SD 1.5 / SD 3 / 3.5 | `stable-diffusion-v1`, `-v3*` | ✅ Verified (fleet pass) | — | not benched | — | — |
| Boogu (Base/Turbo/Edit) | `boogu` | ✅ e2e (needs current Swarm core for detection) | — | not benched | — | — |
| Flux.2 Dev (32B, Q4 GGUF) | `flux-2` | 🔧 Loader GGUF branch in progress | — | — | — | — |
| HiDream-I1 | `hidream-i1` | 🔧 Correctness debug in progress | — | — | — | — |
| Chroma Radiance / Zeta, F-Lite, Anima, OmniGen2, Lumina2 | various | ✅ e2e (fleet pass) / 🔧 detection pending (OmniGen2, Lumina2) | — | not benched | — | — |

#### Video models

| Model | Compat IDs | Status | Hartsy | ComfyUI | GPU |
|---|---|---|---:|---:|---|
| Wan 2.1 T2V 14B | `wan-21-14b` | ✅ Verified in production | **37 s** | ~30.6 s (1.2×) | RTX 4090 |
| Wan 2.2 TI2V-5B | `wan-22-5b` | ✅ Verified in production | 22 s | — | RTX 4090 |
| Wan 2.1 T2V 1.3B | `wan-2_1-*` | ✅ Verified in production | 17 s | — | RTX 4090 |
| Wan VACE / Animate / S2V / I2V | `wan-*` | ✅ Verified (control-video, driving-video, speech→video) | — | — | RTX 4090 |
| LTX-2.3 (with audio) | LTX-2 | ✅ Verified in production (was 451 s) | 57–105 s | — | RTX 4090 |
| LTX-Video 0.9 | `lightricks-ltx-video` | ✅ Verified | — | — | RTX 4090 |
| HunyuanVideo 13B | `hunyuan-video` | ✅ Verified (1.29 s/step) | — | — | RTX 4090 |
| Kandinsky-5 T2V | `kandinsky-5` | ✅ Verified (0.83 s/step) | — | — | RTX 4090 |

Per-architecture features (LoRA, img2img, inpaint, IP-Adapter, ControlNet, Kontext
reference images, FPS/format/boomerang/trim for video) are listed in
[Cross-cutting features](#cross-cutting-features) and the
[parity matrix](./docs/02-Comfy-Feature-Parity-Matrix.md).

Every published number is reproducible: methodology, request parameters, and the
living scoreboard are in the engine's
[Performance Guide](https://github.com/HartsyAI/HartsyInference/blob/main/docs/PERFORMANCE.md);
per-round narratives live in `benchmarks/results/`. The numbers require no
configuration — the engine's standard performance profile is default-on
(see [Performance out of the box](#performance-out-of-the-box)).

### Cross-cutting features

Working: prompt/negative/CFG/steps/seed, sampler selection (SD1.5/SDXL) + clip
skip (SD1.5), EndStepsEarly, img2img creativity (incl. Flux.2), inpaint masks
(+grow/blur), variation seed (SD1.5/SDXL/Flux), **`<segment:yolo->` auto-refinement
(SDXL/Flux/SD3, pure-C# YOLO)**, multi-LoRA with strengths, SDXL refiner (PostApply
any base, StepSwap SDXL), IP-Adapter standard/Plus/Plus-Face **+ FaceID + FaceID-Plus/PlusV2**
with weight types + step gating (SDXL + SD1.5, real-weight verified 07-16/17), ControlNet
stacking + start/end windows (SDXL + SD1.5; Canny / Depth-Anything-V2 / OpenPose / HED-softedge /
scribble / lineart / normal / **segmentation** preprocessors, LDM + diffusers checkpoint layouts),
**union-type SDXL ControlNet** (xinsir controlnet-union ProMax, per-slot control-type dropdown) and
**FLUX DiT ControlNet** (union + single-mode, contour-locked live-verified 07-17), FLUX.1 Kontext /
Fill / Canny / Depth / Redux (all verified e2e 07-16/17), **GGUF Flux transformers**,
Wan/LTX video with FPS/format/boomerang/trim, **Wan VACE control-video** (pose/depth/edge
clip → guided video), **Wan Animate** (driving video → character animation), **Wan S2V**
(speech audio → talking video), ACE-Step music, TAESD or latent2rgb
live previews, mid-gen cancel, FreeMemory, multi-GPU via one backend per GPU,
**admin WebAPI** (probe-model / list-pipelines / device-info / clear-cache).

Not yet — needs upstream HartsyInference engine work first (Category B): hires-fix
2-pass upscale + ESRGAN (tiled VaeEncoder + upscaler), guidance variants
(FreeU/SAG/PAG/NAG/RescaleCFG/CFGZero★), LoRA on SD3/Z-Image/Flux.2, SAM2
segment targets, seamless tiling, batch>1, variation seed on SD3,
FP4 (Flux.2 Klein 9B / Ideogram nf4), union-type segment/tile/repaint control types
(raw-map pass-through wired; dedicated preprocessing pending). (Since shipped — see the
architecture table: CLIP-Seg text segments, Wan 14B, the lineart/softedge/normal/**segmentation**
ControlNet preprocessors, **Flux-DiT ControlNet**, union-type SDXL ControlNet, **IP-Adapter
FaceID + FaceID-Plus/PlusV2**, and Qwen-Image-Edit.)

Not planned for v1: workflow editor, textual-inversion embeddings, full InstantID
pipeline (IP-Adapter FaceID covers the identity-transfer case), rembg, face restore,
TensorRT. TTS/STT (engine has Whisper/Bark/
Kokoro/F5/CosyVoice/etc) is deferred pending a Swarm UI surface for voice/
reference inputs — the current audio params are music-only.

## Performance out of the box

The engine's **standard performance profile** (cuDNN fused flash attention, fp8 tensor-core GEMM on
Ada+, F16 DiT activations, resident DiT weights, warm activation pool) is **default-on inside the
engine** — the extension configures nothing, and every install reproduces the published benchmark
times (several flagship image models faster than ComfyUI on the same GPU). Features degrade
gracefully on hardware that can't run them and each has a `HARTSY_<FEATURE>=0` kill-switch. The
authoritative specification — feature table, cuDNN/cuBLAS requirements and resolution order,
verification log lines, benchmark methodology — is the engine's
[Performance Guide](https://github.com/HartsyAI/HartsyInference/blob/main/docs/PERFORMANCE.md);
see also [docs/07-Parameters-And-Feature-Flags.md](docs/07-Parameters-And-Feature-Flags.md).

## How memory management works (current architecture)

The default backend runs Flux fp8 (~11.3 GB of weights) on a 12 GB GPU through a
combination of three techniques. All of these auto-engage based on detected VRAM —
the user never touches a flag.

1. **Per-block streaming** of transformer weights (`BlockStreamingController`).
   Only `prefetchAhead+1` blocks live on GPU at any moment; the rest stay on host RAM
   and are uploaded just-in-time on a side stream while the main stream computes.
   Resident weight footprint during denoising: ~600 MB instead of 11.3 GB.

2. **F16 GEMM path** for fp8 weights (`ResolveGemmDtype(input, weight)` resolves to
   F16 whenever fp8 is in play). Halves the per-Linear cast workspace and runs the
   GEMM on Ampere Tensor Cores. Critical fix vs the original F32 path which
   exploded the cast buffer to 151 MB per Linear.

3. **Tiled VAE decode** (`VaeDecoder.DecodeTiled`). Slices the latent into
   64-latent / 512-RGB tiles with 64-pixel RGB blend overlap. Decodes each tile
   independently, then blends overlapping regions with a tent-function weight mask.
   Caps the worst-case `im2col` workspace at ~2.4 GB per tile (F32) regardless of
   final resolution. Without this, Flux's VAE at 1024² needed a single 9.7 GB conv
   workspace.

The OOM-retry path in `CudaMemory.Allocate` provides self-healing: if a sync alloc
fails, drain both streams + trim the device mempool, then retry. You may see
`[Warning] [CudaMemory] OOM on first attempt` lines in the log during VAE decode —
that's normal under tight memory and the retry is recovering.

## Running on multiple GPUs

Setup mirrors the ComfyUI backend exactly: **one backend instance per GPU**, added
manually (neither backend auto-spawns per GPU). In `Server` → `Backends`, click
`HartsyInference (Pure C# Inference)` once per GPU and set each instance's `GPU_ID`
to a distinct ordinal (`0`, `1`, …). Swarm's scheduler then load-balances generations
across the instances — it routes each request to the least-used instance that already
has (or can load) the model, and different instances generate fully in parallel.

Unlike ComfyUI (which isolates each GPU in a separate Python process via
`CUDA_VISIBLE_DEVICES`), HartsyInference runs all instances **in one process**, so each
instance's engine state must be per-device-isolated. It is: the engine binds its CUDA
context per calling thread and keys every stream / memory pool / kernel module / cuBLAS
handle by device, so two backends on two GPUs share no mutable state. (The one remaining
cross-backend hazard — a process-global tensor-finalizer cleanup queue that let one
backend's thread run another's cleanup — was fixed by partitioning that queue per CUDA
context; requires engine ≥ the build noted in the extension bump.)

A `GPU_ID` list like `0,1` is accepted but only the first ordinal is used — one
HartsyInference instance drives one GPU. For multi-GPU, add multiple instances.

**Requires engine ≥ `2.0.0-alpha.5`.** Before that version the engine always built its device on ordinal 0
and `GPU_ID` was logged and ignored, so every instance silently shared one GPU no matter what you set here.

**The ordinal is CUDA's, not `nvidia-smi`'s.** CUDA enumerates fastest-first by default, so on a mixed-GPU
machine `GPU_ID=0` is the *fastest* card, which need not be `nvidia-smi`'s index 0 (measured on a
4090 + 3060 box: ordinal 0 is the 4090, while `nvidia-smi` index 0 is the 3060). To confirm which physical
card an instance got, watch `nvidia-smi` memory while it generates rather than trusting the number.

## Known limitations

- **VAE decode is slow** when memory is tight (tens of seconds for 1024²) because of
  the OOM-retry pressure. Each retry costs a stream sync. See [TODO](#todo) for the
  pre-flight memory budget plan that will eliminate most retries.
- **F16 VAE produces black output on Flux Schnell** — pipeline runs without error but
  values come out NaN/saturated. F32 VAE works fine. See [TODO](#todo).
- **No T5 caching across generations.** T5-XXL (~5 GB) is uploaded fresh each gen
  because keeping it resident through transformer streaming would OOM a 12 GB card.
  Should auto-cache on cards with >18 GB total. See [TODO](#todo).
- **No SDXL/SD3 end-to-end verification** since the May 2026 memory overhaul.
- **Single-batch only.** Batch > 1 not validated; the streaming controller assumes
  B=1 in places.

> **Resolved 2026-05-06:** Z-Image black-output bug. Root cause was a F16 cuBLAS GEMM
> overflow in SwiGLU's `w2` Linear when an FP8 weight met an F32 activation — the F32→F16
> cast of `gated = silu(w1(x)) * w3(x)` produced +Inf for some positions starting at step 1.
> Fixed by routing FP8 + F32 GEMMs through BF16 instead of F16 (BF16 has F32's full dynamic
> range). See [PHASE_3_DEVIATIONS.md #36](../../../../HartsyInference/docs/Checklists/PHASE_3_DEVIATIONS.md)
> for the full troubleshooting journey.

## TODO

Tracked as `// TODO: ...` comments in the code where applicable.

### Memory + performance
- [ ] **F16 VAE precision**. Black output at F16 on Flux Schnell — debug whether the
  F16 GroupNorm kernel accumulates in F32 internally, and whether the F16 softmax
  subtracts max-before-exp. If both are clean, the issue is somewhere in the
  ResNetBlock / VaeAttention chain. F16 VAE is needed for 2K+ resolutions where even
  tiled F32 won't fit.
- [ ] **Pre-flight memory check before each VAE tile**. Currently we allocate
  optimistically and recover via OOM retry, which costs ~600 ms per retry. A
  pre-flight `cuMemGetInfo` + mempool trim would catch the tight cases and drain
  the pool before the alloc is even attempted, eliminating most retries.
- [ ] **`VramStrategy` foundation** (the auto-tier system discussed in the planning
  thread). Single source of truth for budget, used by every pipeline phase to plan
  load/evict decisions.
- [ ] **T5 caching across generations** when budget permits. Auto-enable on >18 GB
  cards. Trivial change once `VramStrategy` is in place.
- [ ] **LRU model eviction** for multi-architecture workflows (SDXL → Flux → SDXL).
  Currently, switching models leaks the prior one until the GC runs.

### Architecture coverage
Loaders done — wired into the extension and dispatched from `HartsyInferenceBackend.cs`:
- [x] ~~**`Flux2Loader.cs`**~~ — Flux.2 Klein 4B (Qwen3-4B + flux2-vae). Klein 9B / Dev are
  refused at runtime until `LlamaStyleEncoderConfig.Qwen3_8B` / Mistral presets land.
- [x] ~~**`ChromaLoader.cs`**~~ — Chroma V1 (T5-XXL via `T2IParamTypes.T5XXLModel` + Flux VAE
  auto-download). Radiance / Zeta variants need additional `ChromaConfig` presets.
- [x] ~~**`AuraFlowLoader.cs`**~~ — AuraFlow v0.3 (single-file: bundled T5 + SDXL VAE +
  transformer all in one safetensors).
- [x] ~~**`FLiteLoader.cs`**~~ — F-Lite v1 (diffusers-folder layout: `dit_model/` +
  `text_encoder/` + `vae/`; user picks any safetensors inside, loader walks up to find root).

**Refused at the dispatch boundary (with clear messages) — upstream blockers exist:**
- **Ernie Image** — pipeline + `Ministral3B` encoder preset exist, but there is no real
  Ernie tokenizer in `HartsyInference.Tokenizers`. Refused until upstream ships one.
- **HunyuanImage 2.1** — upstream pipeline substitutes T5-XXL for the real Qwen2.5-VL
  MLLM encoder (and drops the byT5 glyph stream); output wouldn't be faithful. Refused
  until the real encoder path lands.
- **Flux.2 Klein 9B / Dev** — refused at runtime: the released encoders are FP4-mixed
  and HartsyInference has no FP4 GEMM. Klein 4B works via the `Qwen3_4B` preset.
- **Ideogram 4** — upstream pipeline/converter/tests are in place; the extension loader,
  model-class detection, and a dual-9.3B-DiT VRAM gate are punchlist P7. fp8 variant
  only until FP4 GEMM lands. Non-commercial license.

Each refused architecture has a one-line entry in `ModelSupport._pendingArchs` with the
human-readable reason; the user gets a clear explanation in the UI when they pick one.

Existing wiring polish:
- [ ] **End-to-end verification** of SDXL, SD 1.5, SD3 on the new streaming + tiling path.
  Code paths exist; haven't been run since the May 2026 memory overhaul.
- [x] ~~**Tiled VAE for non-Flux pipelines.**~~ Done — every pipeline routes through
  `DecodeTiled`. The fast-path skips tiling at small resolutions, so this is a free win.
- [ ] **Img2img with `VaeEncoder`** on the tiled path. Encoder has the same im2col
  problem; needs a sibling `EncodeTiled`.

### Z-Image — fixed 2026-05-06
- [x] ~~**Open bug** — Z-Image generates without errors but RGB output is uniformly black.~~
  Fixed via BF16 GEMM dtype for FP8 + F32 operand pairs. See
  [PHASE_3_DEVIATIONS.md #36](../../../../HartsyInference/docs/Checklists/PHASE_3_DEVIATIONS.md)
  for the full troubleshooting journey (8+ trace iterations to localize, then a 30-line
  fix in `CudaBackend.ResolveGemmDtype`).

### Quality / correctness
- [ ] **Tile seam visibility audit.** 64-pixel RGB overlap with tent blending should
  be smooth, but worth a side-by-side vs an un-tiled F32 reference at a few
  resolutions to confirm.
- [ ] **Numerical comparison against ComfyUI** for the same prompt + seed at the
  same model. Identifies any silent precision drift in the F16 transformer path.

### Long-term (defer)
- [ ] **cuDNN wrapper** to replace the hand-rolled im2col + cuBLAS Conv2D path. Would
  give us Winograd, implicit-GEMM, and FFT algorithms with auto-selected heuristics
  — eliminates the workspace cliff that necessitated tiling in the first place.
  Estimated ~200 lines of P/Invoke + a Conv2D-strategy switch.
- [ ] **CPU-offloaded activations** for >24 GB models. Currently we only offload
  weights; activations always stay on GPU. Real "lowvram" mode would page activation
  tensors out too.

## HartsyInference

[HartsyInference](https://github.com/Hartsy/HartsyInference) is a sister project
(`/home/kalebbroo/Desktop/Projects/HartsyInference` locally) — a pure C# / .NET 10
inference engine. What's implemented today:

- **Backends:** `IBackend` (eager execution) implemented for **CPU** (AVX/SIMD), **CUDA** (PTX via Driver API P/Invoke), and **Vulkan** (FP16 compute shaders)
- **Diffusion pipelines:** SD 1.5, SDXL (+ inpaint, + refiner), SD3, Flux, Flux.2, AuraFlow, Chroma, Z-Image, Anima, HiDream, Qwen-Image, HunyuanImage, ErnieImage, F-Lite, Lumina 2, OmniGen 2, Lens, Kandinsky 5, **Ideogram 4** (dual-DiT asymmetric CFG)
- **Video pipelines:** Wan 2.2 TI2V-5B, LTX-Video
- **Schedulers:** Euler, DDIM, DPM++ 2M, LCM, FlowMatch Euler/DMD/UniPC, logit-normal (Ideogram 4)
- **Text encoders:** CLIP-L/G, CLIP-Vision (IPA), T5-XXL, LlamaStyle (Qwen3 / Qwen3-VL-8B / Mistral / Llama 3.1), GPT-OSS
- **Tokenizers:** CLIP, T5, Whisper, Qwen3
- **Adapters:** LoRA stack with per-component application, ControlNet (SDXL + SD1.5 + union-type ProMax + FLUX-DiT), IP-Adapter standard/Plus/Plus-Face/FaceID/FaceID-Plus/FaceID-PlusV2
- **Prompting:** structured-prompt subsystem (JSON dialects, bounding boxes, regions — built for Ideogram 4)
- **Memory mgmt:** `BlockStreamingController` (per-layer streaming), `CudaStreamingWeightCache` (async upload on side stream), tiled VAE decode
- **Cancellation:** `CancellationToken` threaded through pipeline loops

What's **planned but not yet implemented** in HartsyInference:

- ❌ Tiled `VaeEncoder` (blocks high-res img2img / hires-fix — punchlist P2)
- ❌ Upscaler model loaders (ESRGAN family)
- ❌ FP4/NF4 GEMM (blocks Flux.2 Klein 9B/Dev + Ideogram 4 nf4 variant)
- ❌ Configurable CLIP stop-layer (clip skip — punchlist P1)
- ❌ Segmentation models (YOLO / SAM2)
- ❌ `ModelRegistry.LoadAsync()` HuggingFace auto-loader / `PipelineFactory.Create()` façade
- ❌ `HartsyInference.Server` OpenAI-compatible REST endpoints

> **Compatibility note:** HartsyInference targets `net10.0`; SwarmUI extensions target
> `net8.0`. Currently resolved via HartsyInference multi-targeting both.

## Documentation

The [`docs/`](./docs/) folder is the source of truth for the build plan:

| # | Document | Purpose |
|---|----------|---------|
| 00 | [Overview](./docs/00-Overview.md) | Vision, scope, non-goals |
| 01 | [Architecture](./docs/01-Architecture.md) | Layers, components, data flow |
| 02 | [Comfy Feature Parity Matrix](./docs/02-Comfy-Feature-Parity-Matrix.md) | Every Comfy feature, mapped to a HartsyInference plan |
| 03 | [Implementation Roadmap](./docs/03-Implementation-Roadmap.md) | Phased delivery plan with milestones |
| 04 | [HartsyInference Integration](./docs/04-HartsyInference-Integration.md) | The API surface HartsyInference must expose |
| 05 | [Pipeline Translation](./docs/05-Pipeline-Translation.md) | The `WorkflowGenerator` equivalent — params → HartsyInference calls |
| 06 | [Backend Lifecycle](./docs/06-Backend-Lifecycle.md) | Init / Generate / Shutdown contract |
| 07 | [Parameters & Feature Flags](./docs/07-Parameters-And-Feature-Flags.md) | What params we own, what flags we advertise |
| 08 | [Web API Routes](./docs/08-Web-API-Routes.md) | Extra HTTP routes the extension adds |
| 09 | [Testing Strategy](./docs/09-Testing-Strategy.md) | How we validate correctness + perf |
| 10 | [Risks & Open Questions](./docs/10-Risks-And-Open-Questions.md) | What's unknown, what's risky |

## Logging conventions

The extension forwards HartsyInference's internal log calls to SwarmUI's logger. Levels
are mapped 1:1 (`Verbose → Verbose`, `Debug → Debug`, `Info → Info`, `Warning →
Warning`, `Error → Error`) by `EnsureLoggerWired()` in `HartsyInferenceBackend.cs`.

What goes where:

- **Info** — major milestones only: model loaded, generation started, generation
  complete, image saved. One or two lines per generation.
- **Verbose** — phase-level detail: text encoding done, denoising step N/M, VAE tile
  N/M, OOM retry recovered, tensor stats. Useful when debugging a specific generation.
- **Debug** — per-block / per-tile internals: which block streamed when, individual
  tensor allocation sizes, cuBLAS workspace decisions. Heavy; only enable when
  hunting a specific bug.
- **Warning** — non-fatal anomalies that the user should know about: OOM on first
  attempt (recovered), tile seam mismatch, F16 precision fallback.
- **Error** — only on actual generation failure.

Almost no `Logs.Info` in the inference hot paths — those are reserved for SwarmUI's
own UI-visible status. Developers chasing bugs should run with `--log-level Verbose`.

## License

MIT — see [`LICENSE`](./LICENSE).
