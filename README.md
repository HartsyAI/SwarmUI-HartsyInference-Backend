# SwarmUI SharpInference Backend

> **Status:** Working beta. 14 image architectures + 2 video architectures dispatch
> end-to-end, with Flux and Z-Image verified at 1024×1024 on a 12 GB consumer GPU
> (RTX 3060). Img2img, inpaint, LoRA, refiner, ControlNet (SDXL+Canny), IP-Adapter
> (SD1.5/SDXL), live previews, and cancellation all work. There are still rough edges
> (slow VAE under memory pressure, several architectures wired but unverified
> post-bring-up) — see [Known limitations](#known-limitations) and the
> [parity matrix](./docs/02-Comfy-Feature-Parity-Matrix.md).

## What this is

A SwarmUI backend extension that runs Stable Diffusion / FLUX / SDXL / etc. inference
**entirely in C#** through the [SharpInference](#sharpinference) library — no Python,
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

| Architecture | Compat IDs | Status | Per-arch features |
|---|---|---|---|
| Flux.1 (Schnell, Dev, Krea) + FLUX.1 Canny | `flux-1` | **✅ Verified** at 1024² on 12 GB | LoRA, img2img, inpaint |
| Z-Image (Turbo, Base) | `z-image` | **✅ Verified** on CUDA | img2img |
| SD 1.5 | `stable-diffusion-v1` | ⚠ Wired-untested since memory overhaul | LoRA, img2img, IP-Adapter |
| SDXL + official Refiner | `stable-diffusion-xl-v1` | ⚠ Wired-untested since memory overhaul | LoRA, img2img, inpaint, IP-Adapter, ControlNet (Canny), refiner PostApply/StepSwap |
| SD 3 / 3.5 (Medium, Large) | `stable-diffusion-v3*` | ⚠ Wired-untested | img2img, inpaint |
| Flux.2 (base, Klein 4B) | `flux-2`, `flux-2-klein-4b` | ⚠ Wired-untested; Klein 9B/Dev refused (FP4 GEMM) | t2i |
| Chroma V1 | `chroma` | ⚠ Wired-untested | t2i |
| AuraFlow v0.2/v0.3 | `auraflow-v1` | ⚠ Wired-untested | t2i |
| F-Lite v1 | `f-lite` | ⚠ Wired-untested (diffusers-folder layout) | t2i |
| Anima (Cosmos-Predict2 2B) | `anima` | ⚠ Wired-untested | t2i |
| HiDream-I1 | `hidream-i1` | ⚠ Wired; needs Llama-3.1 tokenizer assets | t2i |
| Qwen-Image | `qwen-image` | ⚠ Wired-untested | t2i |
| **Wan 2.2 TI2V-5B (video)** | `wan-22-5b` | ⚠ Wired | T2V, I2V (init image), LoRA |
| **LTX-Video 0.9 (video)** | `lightricks-ltx-video` | ⚠ Wired | T2V |
| **Ideogram 4 (9.3B dual-DiT)** | `ideogram-4` | ⚠ Wired-untested; **needs ≥22 GB free VRAM** (gated at load) | t2i; Steps→official presets (12/20/48); negative prompt + CFG ignored by design (asymmetric CFG). **Non-commercial license.** |

Refused with a clear in-UI message (blockers tracked in
[the punchlist](./docs/11-Comfy-Parity-Punchlist.md)): Ernie Image (no tokenizer
upstream), HunyuanImage 2.1 (encoder stand-in not faithful), Chroma Radiance /
Zeta-Chroma (config presets), Flux.2 Klein 9B / Dev (FP4 GEMM).

### Cross-cutting features

Working: prompt/negative/CFG/steps/seed, EndStepsEarly, img2img creativity,
inpaint masks (+grow/blur), multi-LoRA with strengths, SDXL refiner (PostApply
any base, StepSwap SDXL), IP-Adapter standard/Plus/Plus-Face with weight types +
step gating, ControlNet stacking (SDXL, Canny preprocessor), Wan/LTX video with
FPS/format/boomerang/trim, TAESD or latent2rgb live previews, mid-gen cancel,
FreeMemory, multi-GPU via one backend per GPU.

Not yet (planned, in priority order — see
[punchlist P1–P7](./docs/11-Comfy-Parity-Punchlist.md)): sampler/scheduler
selection + clip skip (P1), hires-fix 2-pass upscale + ESRGAN (P2), graceful
refusal of `<segment:>`/`<region:>`-style prompt syntax (P3), variation seed
(P4), `<segment:face>` YOLO auto-refinement (P5), Qwen Image Edit + Wan 14B
(P6), **Ideogram 4** (P7 — 9.3B dual-DiT, Qwen3-VL-8B encoder; upstream
pipeline/converter/tests already in SharpInference; non-commercial license).

Not planned for v1: workflow editor, textual-inversion embeddings, FaceID/
InstantID, Flux Redux, seamless tiling, rembg, face restore, TensorRT.

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
> range). See [PHASE_3_DEVIATIONS.md #36](../../../../SharpInference/docs/Checklists/PHASE_3_DEVIATIONS.md)
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
Loaders done — wired into the extension and dispatched from `SharpInferenceBackend.cs`:
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
  Ernie tokenizer in `SharpInference.Tokenizers`. Refused until upstream ships one.
- **HunyuanImage 2.1** — upstream pipeline substitutes T5-XXL for the real Qwen2.5-VL
  MLLM encoder (and drops the byT5 glyph stream); output wouldn't be faithful. Refused
  until the real encoder path lands.
- **Flux.2 Klein 9B / Dev** — refused at runtime: the released encoders are FP4-mixed
  and SharpInference has no FP4 GEMM. Klein 4B works via the `Qwen3_4B` preset.
- **Chroma Radiance / Zeta-Chroma** — refused: `ChromaConfig` only has `V1` preset.
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
  [PHASE_3_DEVIATIONS.md #36](../../../../SharpInference/docs/Checklists/PHASE_3_DEVIATIONS.md)
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

## SharpInference

[SharpInference](https://github.com/Hartsy/SharpInference) is a sister project
(`/home/kalebbroo/Desktop/Projects/SharpInference` locally) — a pure C# / .NET 10
inference engine. What's implemented today:

- **Backends:** `IBackend` (eager execution) implemented for **CPU** (AVX/SIMD), **CUDA** (PTX via Driver API P/Invoke), and **Vulkan** (FP16 compute shaders)
- **Diffusion pipelines:** SD 1.5, SDXL (+ inpaint, + refiner), SD3, Flux, Flux.2, AuraFlow, Chroma, Z-Image, Anima, HiDream, Qwen-Image, HunyuanImage, ErnieImage, F-Lite, Lumina 2, OmniGen 2, Lens, Kandinsky 5, **Ideogram 4** (dual-DiT asymmetric CFG)
- **Video pipelines:** Wan 2.2 TI2V-5B, LTX-Video
- **Schedulers:** Euler, DDIM, DPM++ 2M, LCM, FlowMatch Euler/DMD/UniPC, logit-normal (Ideogram 4)
- **Text encoders:** CLIP-L/G, CLIP-Vision (IPA), T5-XXL, LlamaStyle (Qwen3 / Qwen3-VL-8B / Mistral / Llama 3.1), GPT-OSS
- **Tokenizers:** CLIP, T5, Whisper, Qwen3
- **Adapters:** LoRA stack with per-component application, ControlNet (SDXL), IP-Adapter standard/Plus
- **Prompting:** structured-prompt subsystem (JSON dialects, bounding boxes, regions — built for Ideogram 4)
- **Memory mgmt:** `BlockStreamingController` (per-layer streaming), `CudaStreamingWeightCache` (async upload on side stream), tiled VAE decode
- **Cancellation:** `CancellationToken` threaded through pipeline loops

What's **planned but not yet implemented** in SharpInference:

- ❌ Tiled `VaeEncoder` (blocks high-res img2img / hires-fix — punchlist P2)
- ❌ Upscaler model loaders (ESRGAN family)
- ❌ FP4/NF4 GEMM (blocks Flux.2 Klein 9B/Dev + Ideogram 4 nf4 variant)
- ❌ Configurable CLIP stop-layer (clip skip — punchlist P1)
- ❌ Segmentation models (YOLO / SAM2)
- ❌ `ModelRegistry.LoadAsync()` HuggingFace auto-loader / `PipelineFactory.Create()` façade
- ❌ `SharpInference.Server` OpenAI-compatible REST endpoints

> **Compatibility note:** SharpInference targets `net10.0`; SwarmUI extensions target
> `net8.0`. Currently resolved via SharpInference multi-targeting both.

## Documentation

The [`docs/`](./docs/) folder is the source of truth for the build plan:

| # | Document | Purpose |
|---|----------|---------|
| 00 | [Overview](./docs/00-Overview.md) | Vision, scope, non-goals |
| 01 | [Architecture](./docs/01-Architecture.md) | Layers, components, data flow |
| 02 | [Comfy Feature Parity Matrix](./docs/02-Comfy-Feature-Parity-Matrix.md) | Every Comfy feature, mapped to a SharpInference plan |
| 03 | [Implementation Roadmap](./docs/03-Implementation-Roadmap.md) | Phased delivery plan with milestones |
| 04 | [SharpInference Integration](./docs/04-SharpInference-Integration.md) | The API surface SharpInference must expose |
| 05 | [Pipeline Translation](./docs/05-Pipeline-Translation.md) | The `WorkflowGenerator` equivalent — params → SharpInference calls |
| 06 | [Backend Lifecycle](./docs/06-Backend-Lifecycle.md) | Init / Generate / Shutdown contract |
| 07 | [Parameters & Feature Flags](./docs/07-Parameters-And-Feature-Flags.md) | What params we own, what flags we advertise |
| 08 | [Web API Routes](./docs/08-Web-API-Routes.md) | Extra HTTP routes the extension adds |
| 09 | [Testing Strategy](./docs/09-Testing-Strategy.md) | How we validate correctness + perf |
| 10 | [Risks & Open Questions](./docs/10-Risks-And-Open-Questions.md) | What's unknown, what's risky |

## Logging conventions

The extension forwards SharpInference's internal log calls to SwarmUI's logger. Levels
are mapped 1:1 (`Verbose → Verbose`, `Debug → Debug`, `Info → Info`, `Warning →
Warning`, `Error → Error`) by `EnsureLoggerWired()` in `SharpInferenceBackend.cs`.

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
