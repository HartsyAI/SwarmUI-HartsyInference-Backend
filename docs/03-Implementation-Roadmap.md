# 03 — Implementation Roadmap

Phased plan to get from "empty scaffold" to "could replace Comfy as default for the
common path." Each phase has a concrete acceptance test (it's runnable, not
"feels-done").

## Phase 0 — Compatibility unblock (PREREQUISITE)

**Cannot start phase 1 until this is resolved.** HartsyInference targets `net10.0`;
SwarmUI extensions target `net8.0`. The extension cannot reference HartsyInference at
compile time without resolving this.

**Options (pick one before phase 1):**

1. **Multi-target HartsyInference.** PR upstream `<TargetFrameworks>net8.0;net10.0</TargetFrameworks>`. Lowest risk to Swarm; requires HartsyInference to verify it doesn't use net10-only APIs (likely it does — `nuint` semantics, ref struct improvements, etc.).
2. **Bump SwarmUI to net10.** Upstream PR to mcmonkey4eva. Coordinated risk; affects all extensions. Realistic timeline once .NET 10 is GA but probably not the path for this milestone.
3. **Out-of-process bridge.** Run HartsyInference as a separate `dotnet` process (net10) and talk to it over a local Unix socket / named pipe / loopback HTTP. Defeats the "in-process" goal but unblocks immediately.

**Decision needed.** Default plan in this roadmap is **option 1 (multi-target)**.

**Acceptance:** This extension's `.csproj` can `dotnet build` referencing HartsyInference projects from a net8.0 host.

---

## Phase 1 — Hello World (text-to-image, single architecture)

**Scope:** SD 1.5 only, CPU backend only, single-LoRA, no img2img, no controlnet,
no live previews, no cancel.

**Tasks:**

1. Wire up the extension class — register backend type, register a minimal "use HartsyInference" permission, register the HartsyInference param group.
2. Implement `HartsyInferenceBackend.Init()` — instantiate `CpuBackend`, set `Status=IDLE`.
3. Implement `HartsyInferenceBackend.LoadModel()` — for SD15, load tokenizer + CLIP text encoder + UNet + VAE decoder via `SafeTensorsLoader`, store in `PipelineCache`.
4. Implement `HartsyInferenceBackend.Generate()` — for SD15:
   - Resolve checkpoint via `T2IParamInput`
   - Load via `LoadModel` if not cached
   - Tokenize prompt + negative prompt
   - Call `pipeline.GenerateFromTokens(...)` synchronously
   - Convert returned `byte[] rgbData` → `SwarmUI.Image`
   - Return `Image[]` with one image
5. Implement `SupportedFeatures` — yield `"hartsyinference"`, `"comfyui"` (for parameter compatibility — see [`07-Parameters-And-Feature-Flags.md`](./07-Parameters-And-Feature-Flags.md) for why).

**Acceptance test:**

- Launch Swarm with the extension installed
- Configure a HartsyInference backend in Server → Backends
- Pick an SD 1.5 .safetensors model from `Models/`
- Type a prompt, click Generate
- A 512×512 image appears in the gallery within a sane time (CPU, so minutes are fine)

**Out of phase 1:**
- SDXL / Flux / SD3
- LoRAs
- Live progress
- Cancellation
- GPU backends

---

## Phase 2 — Architecture breadth + LoRA

**Scope:** Add SDXL and Flux. Add LoRA. Still synchronous, still no live progress.

**Tasks:**

1. Add per-architecture loaders (`Generation/ModelSupport.cs`):
   - SDXL — dual CLIP (L + G), `SdxlPipeline`
   - Flux schnell / dev — CLIP-L + T5-XXL, `FluxPipeline`, FlowMatch scheduler
2. Architecture detection — read `T2IModel.ModelClass.CompatClass` and dispatch.
3. LoRA: build a `LoraStack` from the user's selected LoRAs, call `ApplyToWeights`. **Critical:** decide LoRA cache strategy (see Risks doc) — likely keep an unmodified weights cache and reapply per gen.
4. Tiled VAE decode — switch to `VaeTiledDecoder` for large outputs (>1024 on SDXL, >1024 on Flux).
5. CUDA backend support — settings toggle for backend choice; `CudaBackend(deviceOrdinal)`.

**Acceptance test:**

- SDXL t2i works end-to-end
- Flux schnell t2i works (4 steps, FlowMatch)
- One LoRA at strength 0.8 visibly affects the output
- CUDA backend produces an image faster than CPU on the same hardware

---

## Phase 3 — Live progress + cancellation + refiner

**Scope:** Make the UX match Comfy: % bar updates, partial previews, stop button works. Plus SDXL refiner.

**Tasks:**

1. Live progress: in `GenerateLive`, build a `Action<GenerationProgress>` that calls `takeOutput(JObject {batch_index, overall_percent, current_percent})`.
2. Cancellation: until upstream adds `CancellationToken`, set a volatile `_cancelRequested` flag in the backend; check it inside the progress callback and throw `OperationCanceledException` from there. Upstream PR for `CancellationToken` parameter on `GenerateFromTokens` is the right long-term fix.
3. Per-step preview: every N steps (configurable), run `VaeDecoder` on the current latent, encode as JPEG, push as preview JObject. (Latent has to be exposed by HartsyInference's pipeline progress callback — currently the progress only carries (Step, Total, Elapsed). **Upstream change needed:** expose the in-flight latent.)
4. SDXL refiner: chain SdxlPipeline with `refiner_strength` cutoff → second SdxlPipeline-with-refiner-UNet runs the remainder.
5. End-step early-out (`endstepsearly` feature flag).

**Acceptance test:**

- Progress bar advances smoothly
- A preview image appears in the UI mid-generation
- Hitting Cancel during step 12/30 halts within ~1 step
- SDXL with refiner produces a different image than SDXL base alone

---

## Phase 4 — Img2img / inpaint + remaining architectures

**Scope:** Image-to-image and inpainting. SD3, Flux.2, AuraFlow, Chroma, Z-Image, Hunyuan-Image.

**Tasks:**

1. **Upstream:** add `VaeEncoder` to HartsyInference. Without this, no img2img path exists. This is a substantial chunk of work.
2. Once `VaeEncoder` lands, add img2img path: encode init image → noise to denoise-strength → continue from there.
3. Inpaint: same plus mask handling. `SdxlInpaintPipeline` already exists; just feed it the encoded source + mask.
4. SD3 / Flux.2 / AuraFlow / Chroma / Z-Image / Hunyuan-Image — wire each pipeline class into the architecture handler. Mostly boilerplate once SDXL/Flux are working.
5. Test models from each architecture against ComfyUI parity (see [`09-Testing-Strategy.md`](./09-Testing-Strategy.md)).

**Acceptance test:**

- Img2img with denoise strength 0.6 produces a visible variation of the source
- Inpaint (mask + replacement prompt) only modifies masked region
- One generation each from SD3, Flux.2, AuraFlow, Chroma, Z-Image succeeds

---

## Phase 5 — ControlNet + IP-Adapter + advanced UX

**Scope:** Conditioning adapters, hi-res fix, embeddings.

**Tasks:**

1. ControlNet: pipeline integration — pipelines need to accept a `ControlNet` argument. **Upstream:** wire `ControlNet` into at least `StableDiffusion15Pipeline`, `SdxlPipeline`, `FluxPipeline`. Plus a way to pass control-image conditioning per-step.
2. ControlNet preprocessors: at minimum canny + depth (depth needs a small ZoeDepth or DepthAnything model). **Upstream:** preprocessor module in HartsyInference.Vision.
3. IP-Adapter beyond the stub: face / plus / style variants.
4. Hi-res fix: 2-pass with VAE upscale.
5. Textual inversion embeddings: wire into the tokenizer and text encoder. **Upstream:** new token API.
6. LRU eviction on pipeline cache + idle unload.
7. Multi-GPU data parallelism (round-robin if user configured 2+ backends — this is mostly Swarm-side already).

**Acceptance test:**

- ControlNet canny visibly conditions an SDXL gen on an edge map
- IP-Adapter face transfers a face from a reference image
- Hi-res fix produces a 1.5×-upscaled image with maintained detail
- A textual-inversion embedding from `Models/Embeddings/` activates when its name appears in the prompt

---

## Phase 6 — "Could be the default" milestone

**Scope:** Polish. Performance. Bug squashing. Parity tests.

**Tasks:**

1. Run the full Swarm parameter test matrix against ComfyUI — every feature flag we advertise has to work or be honestly disabled.
2. Performance: profile vs Comfy on the same hardware. Identify slow spots in the pipeline translator (likely re-tokenization, re-LoRA-applying, fp32 casts).
3. Free-memory behaviour: verify memory steady-state after 100 alternating SDXL/Flux generations.
4. Documentation: complete the README installation guide; add troubleshooting.
5. Public beta — announce on the Swarm Discord, collect feedback, fix top issues.

**Acceptance test:**

- A clean Swarm install + this extension can run for 24 hours of continuous use without OOM, leak, or hang
- Image quality matches Comfy within visible-tolerance on a curated 50-prompt benchmark
- Generation latency is within 1.5× of Comfy on identical hardware (target: 1.0× or better)

---

## Phase 7+ — Stretch

Things we'd like but aren't blocking the "replace Comfy" goal:

- Audio extension (Whisper STT, Kokoro TTS) as a separate Swarm tab
- Vision tools (YOLO bounding boxes, SAM segmentation) as Swarm post-processing
- Video pipelines (LTX-Video, Wan)
- Quantized model loading (GGUF) for low-VRAM users
- Hosted HartsyInference.Server as an optional remote backend type (`hartsyinference_remote`)

## Critical-path summary

```
┌────────────────────────┐
│ Phase 0 (TFM unblock)  │ ← MUST resolve first
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ Phase 1 (SD15 t2i)     │
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ Phase 2 (SDXL/Flux/    │
│ LoRA, GPU)             │
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ Phase 3 (live progress │
│ cancellation, refiner) │
└────────────┬───────────┘
             ▼
┌────────────────────────┐         ┌──────────────────────────┐
│ Phase 4 (img2img,      │ ◄──────  │ Upstream: VaeEncoder     │
│ inpaint, more archs)   │          └──────────────────────────┘
└────────────┬───────────┘
             ▼
┌────────────────────────┐         ┌──────────────────────────┐
│ Phase 5 (ControlNet,   │ ◄──────  │ Upstream: pipeline       │
│ IP-Adapter, hi-res)    │          │ ControlNet integration   │
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ Phase 6 (parity ship)  │
└────────────────────────┘
```

The two upstream blockers (VaeEncoder, pipeline ControlNet integration) should be filed
as HartsyInference issues during phase 1 so they're in flight before we need them.
