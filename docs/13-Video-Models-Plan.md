# 13 — Video Models Plan (Wan 2.2 TI2V-5B, LTX-Video)

Goal: select a video model in SwarmUI and generate T2V / I2V through HartsyInference's
in-process pipelines, with the same parameters and behavior the ComfyUI backend
provides for those models. Comfy's handling (`WorkflowGeneratorModelSupport.cs`,
`WGNodeData.SaveOutput`, `SwarmSaveAnimationWS.py`) is the baseline we mirror.

This is the Wan 2.2 TI2V-5B / LTX-Video 0.9 build log — the extension has since grown
video support well past these two (Wan 2.1 14B/1.3B + VACE/Animate/S2V variants,
HunyuanVideo, LTX-2, Lance Video, MiniMax-H3). See
[`11-Comfy-Parity-Punchlist.md`](./11-Comfy-Parity-Punchlist.md) for the current full
architecture list; this doc stays scoped to the two families it was written for.

## Guiding principle: reuse Swarm, don't rebuild it

Swarm core already provides everything except the actual inference and the final
frame muxing. We must NOT re-implement:

| Concern | Already handled by | We do |
|---|---|---|
| Model arch detection | `T2IModelClassSorter` already detects `wan-2_2-ti2v-5b` (compat `wan-22-5b`) and `lightricks-ltx-video` from safetensors keys | Just accept those compat classes in `ModelSupport` |
| Video params | `T2IParamTypes`: `Text2VideoFrames`, `VideoFrames`, `VideoSteps`, `VideoCFG`, `VideoFPS`, `VideoFormat`, `VideoResolution`, `VideoBoomerang`, `TrimVideoStart/EndFrames`, feature flags `"text2video"` / `"video"` | Advertise the flags, read the params — register nothing new |
| Side-model selection | `T2IParamTypes.T5XXLModel` (subtype Clip, "Also used for Wan's umt5"), `T2IParamTypes.VAE` | Extension forwards the pick as `VideoRequest.Components`, but `WanVideoRecipe`/`LtxVideoRecipe` don't read it yet (`TODO(E-IMG-4/5)`) — the canonical side model always loads regardless of what the user picked |
| Side-model downloads | Swarm `CommonModels.Known["wan22-vae"]` (+ engine `SideModels`/`ModelDownloader.EnsureSideModelAsync`, into the same shared `Models/Clip`, `Models/VAE` folders) | Add entries using Comfy's exact canonical filenames so files are shared with a Comfy install |
| Video output type | `MediaType.VideoMp4/VideoWebm/VideoMov`; videos are just `Image(bytes, type)`; `Session.SaveImage` keys off `MediaType`, previews via `OutputMetadataTracker` + ffmpeg | Return `new Image(encodedBytes, MediaType.X)` |
| ffmpeg | `Utilities.FfmegLocation` (system ffmpeg or Comfy's vendored imageio-ffmpeg) | Spawn it; never bundle our own encoder |
| Resolution fitting | `VideoResolution` semantics + `Utilities.ResToModelFit`, model `StandardWidth/Height` (Wan 960×960, LTX 768×512) | Apply the same logic Comfy does in `WorkflowGeneratorSteps` |

What Swarm does NOT provide (genuinely ours to build — "ours" meaning the Hartsy stack;
loaders and prompt encoding have since moved from the extension into the engine, see
Phase V1.2):
1. In-process loaders for the two architectures (checkpoint → HartsyInference pipeline).
2. Prompt encoding (Comfy loads the text encoder inside the Comfy process; we load
   UMT5/T5 via HartsyInference — same model *files*, selected by the same params).
3. Raw RGB frames → encoded video container (Comfy does this in the
   `SwarmSaveAnimationWS` python node; for us it's a small ffmpeg-subprocess util —
   this one is still extension-side, `VideoOutputEncoder.cs`).

## HartsyInference building blocks (already exist)

- `WanVideoPipeline(backend, WanVideoTransformer, Wan22VaeDecoder, WanVideoConfig)`
  — `GenerateFromEmbeddings(promptEmbeds, negEmbeds, request, numFrames, onProgress, firstFrameLatent?)`
  → `byte[][]` RGB frames. Frames rule `(F-1) % 4 == 0`, res ÷16. I2V is live (start-frame
  and, on concat-I2V checkpoints, end-frame too — see Phase V2).
- `LtxVideoPipeline(backend, LtxVideoTransformer, LtxVideoVaeDecoder, LtxVideoConfig)`
  — `GenerateFromEmbeddings(..., numFrames, frameRate, onProgress)`. Frames rule
  `(F-1) % 8 == 0`, res ÷32, FPS feeds RoPE (mirrors Comfy's `LTXVConditioning.frame_rate`).
  T2V only; the converter has since grown to also detect 0.9.5 and 0.9.7/13B checkpoints
  via VAE-shape sniffing (timestep-conditioned decoder), same T2V-only pipeline.
- Converters: `WanVideoCheckpointConverter` (single-file original naming — exactly the
  files Swarm's sorter classifies — and diffusers shards), `LtxVideoCheckpointConverter`
  (single-file bundles DiT+VAE; T5 ships separately).
- Text encoders: `T5TextEncoderConfig.Umt5Xxl` (per-layer rel-bias, 256k vocab) and
  `.Xxl`; `T5TextEncoder.Encode(backend, tokenIds, masks)`.
- Both pipelines: per-step `GenerationProgress`, cancellation. `GenerateFramesAsync`
  exists on the low-level pipelines but has no caller — the recipe layer was
  deliberately redesigned to return a buffered frame list rather than stream (see
  Phase V3, streaming decode).

---

## Phase V1 — Text-to-Video (Wan 2.2 TI2V-5B + LTX-Video)

### V1.0 Engine prereq (HartsyInference repo)
- [x] umT5 SentencePiece (256k vocab) embedded in `HartsyInference.Tokenizers`;
      `T5Tokenizer.CreateUmt5(maxLength)` factory.
- [x] fp8-scaled UMT5 load via `CheckpointConvertUtils.ApplyFp8ScaledDequant` — this is
      now the production dequant path for every Wan variant (and most other fp8
      architectures engine-wide), not just Wan.
- [x] `Wan22VaeDecoder` key naming vs Comfy-Org's `wan2.2_vae.safetensors` — resolved:
      the converter matches the real header (nested `decoder.upsamples.{i}.upsamples.{j}`,
      time_conv on stages 0-1, `model.` prefix stripped) and this is the production load
      path for every Wan family.

### V1.1 Side models (`SideModels.cs`, now in `HartsyInference.Engine`)
Entries reuse Comfy's canonical names/URLs/hashes so files are shared with Comfy installs:
- [x] `Umt5Xxl` → `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (Clip folder); `Wan22Vae` →
      `Wan/wan2.2_vae.safetensors` (VAE folder) — both matching Swarm core's own entries.
- [x] ~~`LtxvVae`~~ — dropped: Swarm's registered LTX VAE targets the 0.9.7 variant, not
      the bundled-VAE single-file checkpoints this recipe requires; a DiT-only file gets
      a clear error instead.
- [x] LTX text encoder reuses the existing `T5XxlEnconly` entry (same file Comfy uses).

### V1.2 Loaders
- [x] The extension's old `WanVideoLoader.cs`/`LtxVideoLoader.cs` were lifted into
      `HartsyInference.Engine`'s `WanVideoRecipe`/`LtxVideoRecipe` (`Recipes/Video/`),
      resolved via `VideoRecipeRegistry` and cached by the engine's `VideoService`; the
      extension is now a thin dispatcher (`HartsyInferenceBackend.GenerateVideo` builds a
      `VideoRequest`, calls `_engine.Video.GenerateAsync`). The recipe itself still does:
      checkpoint → converted transformer + VAE + side-model text encoder → tokenize →
      encode → free encoder GPU weights (VRAM headroom for the DiT) →
      `GenerateFromEmbeddings(...)`.

### V1.3 Parameter mapping (`HartsyInferenceBackend.BuildVideoRequest`/`ResolveFrames`)
- [x] Frames come from `Text2VideoFrames`/`VideoFrames` when set; when unset the
      extension sends a flat 25 for Wan/LTX (only MiniMax-H3 defers to the recipe), so
      `WanVideoRecipe`/`LtxVideoRecipe`'s own `VideoDefaults` (33 / 97 frames) never
      actually apply on this path — deliberate per `ResolveFrames`' own comment, not an
      oversight, but worth knowing. Steps/CFG (main `Steps`/`CFGScale` — Comfy's
      `VideoSteps`/`VideoCFG` belong to the I2V flow only), FPS (`VideoFPS`, default 24;
      also feeds LTX's `frameRate`), and resolution (snapped ÷16 Wan / ÷32 LTX) all
      mirror Comfy's exact reads — frame counts snap to the model's `4n+1`/`8n+1` rule
      and log the adjustment rather than rejecting. Sigma shift / scheduler defaults are
      baked into the pipelines (Wan flow-shift 5.0; LTX dynamic μ shift), no extra
      params exposed.

### V1.4 Video output (`Generation/VideoOutputEncoder.cs`)
- [x] C# twin of `SwarmSaveAnimationWS.py`: resolves ffmpeg via `Utilities.FfmegLocation`,
      pipes frames to stdin (`-f rawvideo -pix_fmt rgb24`), per-format args copied from
      the python node (h264-mp4, h265-mp4, webm, prores, gif/gif-hd, webp).
      `VideoBoomerang`/`TrimVideoStart/EndFrames` applied on the frame array pre-encode;
      result wrapped as `Image(bytes, MediaType.…)`, single frame short-circuits to PNG.

### V1.5 Backend wiring (`HartsyInferenceBackend.cs`, `ModelSupport.cs`)
- [x] `ModelSupport` maps `wan-22-5b` + `lightricks-ltx-video` to their engine family
      ids; `"video"` feature advertised (`"text2video"` is derived client-side from the
      compat class). `LoadModel`/`GenerateLive` dispatch to `_engine.Video.GenerateAsync`
      with progress bridged (no latent preview yet at this point — that's V3).
      `IsValidForThisBackend`/`ValidateVideo` refuse per-checkpoint now, not per-family:
      each conditioning object (InitImage, EndFrame, references, driving video, LoRAs)
      is checked against `ModelSupport.SupportedVideoFeatures(compat, checkpointPath)`,
      which asks the resolved recipe's `Supports`/`SupportsFor` rather than hard-coding
      what's refused — refiners over video are refused unconditionally. LTX-2 compat IDs
      fall to the standard unsupported-architecture refusal. Cancellation (`_cancelCts`)
      checked per progress callback and in the muxer.

### V1.6 Validation
- [ ] Real-checkpoint smoke through the Swarm UI (Wan2.2 TI2V-5B + LTX-Video 0.9 T2V,
      mp4 + webp outputs, metadata sidecar, history thumbnail) as a formal QA pass.
      Separately, Wan 2.2's MoE expert-pair swap (Phase V3) and Wan-Animate driving
      (built engine-side, outside this doc's Wan2.2/LTX scope) are both still awaiting
      their own real-weight validation.
- [ ] Confirm interrupt works mid-generation and VRAM is reclaimed on model swap.

## Phase V2 — Image-to-Video (Wan TI2V first)

- [x] **Engine:** `Wan22VaeEncoder` (+ `AvgDown3D` shortcut, `Wan22Resample` downsample
      modes) with `EncodeRgbFrame(backend, rgb24, w, h)` → normalized
      `[1,48,1,H/16,W/16]` latent. Key naming verified against the real
      `wan2.2_vae.safetensors` header, including the F16 dtype (cast to F32 via
      `VaePrecisionHelper`).
- [x] **Extension:** init-image path on the MAIN Wan model (the two-stage Comfy
      `VideoModel` flow is explicitly refused for all archs): `InitImage` →
      `VideoResolution` sizing (`ResToModelFit` / `Model Preferred` / `Image`, snapped
      ÷16) → resize → `Wan22VaeEncoder` → `firstFrameLatent` → pipeline TI2V path
      (diffusers `expand_timesteps` semantics — same `Text2VideoFrames`/`Steps`/`CFGScale`
      as T2V). Encoder GPU weights freed after the encode, like the umT5 encoder.
- [ ] **`VideoEndFrame` (FLF2V) gap on TI2V-5B:** wired for Wan's *concat*-I2V
      checkpoints (`GenerateImageToVideoConcat`) and MiniMax-H3's fl2va — but TI2V-5B is
      the *non-concat* path (VAE-encoded `firstFrameLatent`), which reads `InitImage`
      only and never `VideoRequest.VideoEndFrame`. `Supports` advertises `EndFrame`
      family-wide, so an end-frame request against TI2V-5B is silently a no-op today.
      Needs either a per-variant `Supports`/`SupportsFor` narrowing or wiring the
      non-concat path to consume the end frame.
- [ ] LTX I2V: still refused — no image hook anywhere in the LTX transformer or
      pipeline (`LtxVideoRecipePipeline` carries an open TODO for it); same for LTX-2.
- [ ] Real-checkpoint I2V smoke (with V1.6).

## Phase V3 — Parity polish

- [x] Latent previews: `LatentArchitecture.Wan/Ltx` added with Comfy's published factor
      tables (Wan22 48×3, LTXV 128×3); `DecodeLatent2Rgb` handles rank-5 video latents
      (middle frame); both pipelines hand their live latent to `PreviewEncoder` with no
      extra changes needed on its side (TAESD falls back to latent2rgb until
      taew2.2/taeLTX side models are added).
- [ ] Streaming decode: the engine's video contract was redesigned to return a buffered
      `VideoGenerationResult` rather than stream frames — `GenerateFramesAsync` still
      exists on the low-level pipelines but is unused everywhere (engine and extension).
      Frames are fully buffered in memory before ffmpeg mux, which remains an
      unaddressed memory-bound risk on long or high-resolution clips.
- [ ] LoRA support for Wan: `WanVideoRecipe.Supports` omits `VideoFeatures.Lora`
      (explicit `TODO(E-IMG-4/5)`), so `ValidateVideo`'s generic feature gate refuses any
      LoRA selection on Wan outright. The extension's old
      `WanVideoLoader.GenerateWithLoras` merge (DiT-only, both MoE experts) was dropped
      in the loader lift and never re-wired engine-side. `MiniMaxH3Recipe` is the only
      video family that currently declares and merges LoRAs.
- [x] Wan2.2 A14B MoE expert pairs (base A14B, AnimeGen-T2V, and other high/low-noise
      finetune pairs): high-noise file as the main Model, low-noise file via the Refiner
      Model slot or Video Swap Model. `WanVideoRecipe.Construct` loads both DiTs and
      hands `transformer2` to `WanVideoPipeline`, which runs the real MoE — high-noise
      expert while `timestep ≥ boundary·1000`, one expert-swap at the crossing, only the
      active expert GPU-resident. Boundary defaults to Wan2.2's official 0.875 (T2V) /
      0.9 (I2V); a user-moved Refiner Control Percentage / Video Swap Percent is read as
      "fraction of steps for the low expert" and mapped through the shifted flow
      schedule (`boundary = s·p/(1+(s−1)·p)`, s = Sigma Shift, default 8). Pairs cache
      under an extended key (`base::moe::low::b<boundary>`). Real-weight validation still
      pending (V1.6).
- [x] Lance T2V: fully wired now. `ModelClassRegistrations` registers the `lance-t2v`
      model class under a dedicated `lance-video` compat class (Lance's image and video
      variants ship byte-identical configs, told apart by folder name); `ModelSupport`
      maps it to the engine's `lance-video` family, which has a real `LanceVideoRecipe` /
      `LanceVideoPipeline` (not a stub) and real-weight-verified output.
- [x] Video2Video / VideoExtend / audio-input: `VideoRequest.VideoExtendModel` was
      removed outright (never built on either side) — Video2Video/VideoExtend stay out
      of scope. Audio-input params (`VideoAudioInput`, `VideoAudioReference`, reference
      audio) have since shipped, mainly for MiniMax-H3.
