# 13 — Video Models Plan (Wan 2.2 TI2V-5B, LTX-Video)

Goal: select a video model in SwarmUI and generate T2V / I2V through HartsyInference's
in-process pipelines, with the same parameters and behavior the ComfyUI backend
provides for those models. Comfy's handling (`WorkflowGeneratorModelSupport.cs`,
`WGNodeData.SaveOutput`, `SwarmSaveAnimationWS.py`) is the baseline we mirror.

## Guiding principle: reuse Swarm, don't rebuild it

Swarm core already provides everything except the actual inference and the final
frame muxing. We must NOT re-implement:

| Concern | Already handled by | We do |
|---|---|---|
| Model arch detection | `T2IModelClassSorter` already detects `wan-2_2-ti2v-5b` (compat `wan-22-5b`) and `lightricks-ltx-video` from safetensors keys | Just accept those compat classes in `ModelSupport` |
| Video params | `T2IParamTypes`: `Text2VideoFrames`, `VideoFrames`, `VideoSteps`, `VideoCFG`, `VideoFPS`, `VideoFormat`, `VideoResolution`, `VideoBoomerang`, `TrimVideoStart/EndFrames`, feature flags `"text2video"` / `"video"` | Advertise the flags, read the params — register nothing new |
| Side-model selection | `T2IParamTypes.T5XXLModel` (subtype Clip, "Also used for Wan's umt5"), `T2IParamTypes.VAE` | Honor user picks via existing `ModelAutoDownloader.EnsureSideModel` |
| Side-model downloads | Swarm `CommonModels.Known["wan22-vae"]` (+ our existing `SideModels`/`ModelAutoDownloader` which wraps `Utilities.DownloadFile` into the shared `Models/Clip`, `Models/VAE` folders) | Add entries using Comfy's exact canonical filenames so files are shared with a Comfy install |
| Video output type | `MediaType.VideoMp4/VideoWebm/VideoMov`; videos are just `Image(bytes, type)`; `Session.SaveImage` keys off `MediaType`, previews via `OutputMetadataTracker` + ffmpeg | Return `new Image(encodedBytes, MediaType.X)` |
| ffmpeg | `Utilities.FfmegLocation` (system ffmpeg or Comfy's vendored imageio-ffmpeg) | Spawn it; never bundle our own encoder |
| Resolution fitting | `VideoResolution` semantics + `Utilities.ResToModelFit`, model `StandardWidth/Height` (Wan 960×960, LTX 768×512) | Apply the same logic Comfy does in `WorkflowGeneratorSteps` (~line 1926-1953) |

What Swarm does NOT provide (genuinely ours to build):
1. In-process loaders for the two architectures (checkpoint → HartsyInference pipeline).
2. Prompt encoding (Comfy loads the text encoder inside the Comfy process; we load
   UMT5/T5 via HartsyInference — same model *files*, selected by the same params).
3. Raw RGB frames → encoded video container (Comfy does this in the
   `SwarmSaveAnimationWS` python node; for us it's a small ffmpeg-subprocess util).

## HartsyInference building blocks (already exist)

- `WanVideoPipeline(backend, WanVideoTransformer, Wan22VaeDecoder, WanVideoConfig)`
  — `GenerateFromEmbeddings(promptEmbeds, negEmbeds, request, numFrames, onProgress, firstFrameLatent?)`
  → `byte[][]` RGB frames. Frames rule `(F-1) % 4 == 0`, res ÷16. I2V hook exists
  (`firstFrameLatent` `[1,48,1,H/16,W/16]`, `Wan22VaeLatentNorm.Normalize`d) but the
  **VAE encoder is not built yet** → I2V is Phase V2.
- `LtxVideoPipeline(backend, LtxVideoTransformer, LtxVideoVaeDecoder, LtxVideoConfig)`
  — `GenerateFromEmbeddings(..., numFrames, frameRate, onProgress)`. Frames rule
  `(F-1) % 8 == 0`, res ÷32, FPS feeds RoPE (mirrors Comfy's `LTXVConditioning.frame_rate`). T2V only.
- Converters: `WanVideoCheckpointConverter` (single-file original naming — exactly the
  files Swarm's sorter classifies — and diffusers shards), `LtxVideoCheckpointConverter`
  (single-file bundles DiT+VAE; T5 ships separately).
- Text encoders: `T5TextEncoderConfig.Umt5Xxl` (per-layer rel-bias, 256k vocab) and
  `.Xxl`; `T5TextEncoder.Encode(backend, tokenIds, masks)`.
- Both pipelines: per-step `GenerationProgress`, cancellation, streaming decode
  (`GenerateFramesAsync`) for memory-bounded output.

---

## Phase V1 — Text-to-Video (Wan 2.2 TI2V-5B + LTX-Video)

### V1.0 Engine prereq (HartsyInference repo)
- [x] Embed the umT5 SentencePiece (256k, ~4.4 MB) in `HartsyInference.Tokenizers`
      (`Resources/umt5_spiece.model`) — `T5Tokenizer.CreateUmt5(maxLength)` factory added;
      Wan E2E test switched to it; embedded-resource tests verify the 256k vocab.
- [x] fp8-scaled UMT5 load: already handled — `CheckpointConvertUtils.ApplyFp8ScaledDequant`
      exists and the upstream Wan E2E test loads the exact Comfy fp8-scaled file through it.
      The extension's WanVideoLoader uses the same call.
- [ ] Verify `Wan22VaeDecoder.LoadWeights` accepts the key naming inside Comfy-Org's
      `wan2.2_vae.safetensors` — `LanceCheckpointConverter.LoadVae` passes original
      state-dict keys through (stripping an optional `model.` prefix), so the repackage
      should load as-is; **confirm on the first real-checkpoint run** (V1.6).

### V1.1 Side models (`SideModels.cs`)
Entries reuse Comfy's canonical names/URLs/hashes (already documented in Swarm core)
so files are shared with Comfy installs; user overrides come from `T5XXLModel`/`VAE` params:
- [x] `Umt5Xxl` → `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, folder `Clip`,
      Comfy-Org Wan_2.1 repackage URL + hash (matches `GetUniMaxT5XXLModel`).
- [x] `Wan22Vae` → `Wan/wan2.2_vae.safetensors`, folder `VAE`, URL + hash matching
      Swarm `CommonModels.Known["wan22-vae"]`.
- [x] ~~`LtxvVae`~~ — dropped: Swarm's registered LTX VAE is the 0.9.7 variant, which our
      base-0.9 decoder config doesn't target. V1 requires full single-file LTX checkpoints
      (bundled VAE); a DiT-only file gets a clear error message instead.
- [x] LTX text encoder = existing `T5XxlEnconly` entry (same file Comfy uses).

### V1.2 Loaders (`Generation/WanVideoLoader.cs`, `Generation/LtxVideoLoader.cs`)
Follow the established loader pattern (Load → cache entry, Generate → output):
- [x] `WanVideoLoader.Load`: `WanVideoCheckpointConverter.LoadAndConvert` →
      `WanVideoTransformer`; UMT5 from side model (`T5TextEncoderConfig.Umt5Xxl`,
      fp8-scaled dequant); `Wan22VaeDecoder` via `LanceCheckpointConverter.LoadVae`.
- [x] `LtxVideoLoader.Load`: `LtxVideoCheckpointConverter.LoadAndConvert` (DiT+VAE
      from the checkpoint, VAE cast to F32); T5-XXL from `T5XxlEnconly`.
- [x] `Generate`: tokenize → `T5TextEncoder.Encode` (prompt + negative, masks) →
      `CfgHelper.SliceBatchElement` → free encoder GPU weights (VRAM headroom for the
      DiT preload, mirroring the upstream E2E tests) →
      `pipeline.GenerateFromEmbeddings(...)` → `VideoParamResolver.FinishVideo` →
      single video `Image`.
- [x] `PipelineCache`: both entry types added to the global LRU + EvictAll.

### V1.3 Parameter mapping (`Generation/VideoParamResolver.cs`)
Mirror Comfy's exact reads (it is the contract users already know):
- [x] Frames: `Text2VideoFrames` for T2V (defaults: Wan 81, LTX 97), snapped to the
      model rule (4n+1 / 8n+1) — snap, don't reject, and log the adjustment.
- [x] Steps/CFG: main `Steps`/`CFGScale` for the T2V sampler (verified: Comfy's
      `VideoSteps`/`VideoCFG` belong to the dedicated I2V flow only). `EndStepsEarly`
      honored via the shared `SamplingParamResolver`.
- [x] FPS: `VideoFPS` (default 24, matching Comfy's `Text2VideoFPS()`) → muxer fps;
      for LTX also passed as pipeline `frameRate` (Comfy: `LTXVConditioning.frame_rate`).
- [x] Resolution: width/height snapped ÷16 (Wan) / ÷32 (LTX), logged when adjusted;
      `VideoResolution` modes apply to I2V in V2.
- [x] Sigma shift / scheduler defaults: pipelines already encode the right schedules
      (Wan flow-shift 5.0; LTX dynamic μ shift); no extra params exposed.

### V1.4 Video output (`Generation/VideoOutputEncoder.cs`)
The only new subsystem — C# twin of `SwarmSaveAnimationWS.py`:
- [x] Resolve ffmpeg via `Utilities.FfmegLocation`; null → `SwarmUserErrorException`.
- [x] Pipe frames to stdin as `-f rawvideo -pix_fmt rgb24 -s WxH -r fps -i -`, per-format
      args copied from the python node: `h264-mp4` (libx264, yuv420p, crf 19), `h265-mp4`,
      `webm` (crf 23), `prores` (prores_ks p3, yuv422p10le), `gif`/`gif-hd` (palettegen),
      `webp` (libwebp — PIL-encoded in the python node, ffmpeg here).
- [x] `VideoBoomerang` + `TrimVideoStart/EndFrames` applied on the frame array pre-encode.
- [x] Result wrapped as `new Image(bytes, MediaType.…)`; single frame short-circuits to PNG
      like the Comfy node. Swarm's save path + preview generation handle the rest.

### V1.5 Backend wiring (`HartsyInferenceBackend.cs`, `ModelSupport.cs`)
- [x] `ModelSupport`: `wan-22-5b` + `lightricks-ltx-video` added to supported archs.
- [x] `SupportedFeatures`: `"video"` advertised (exposes VideoFPS/VideoFormat/etc.).
      `"text2video"` needs no advertising — the UI derives it client-side from the model's
      compat class (`main.js doAnyCompatFeature`) and it's in `DisregardedFeatureFlags`.
- [x] `LoadModel` / `GenerateLive`: dispatch branches for both loaders; progress bridged
      (no latent preview yet — `LatentArchitecture` has no video entries; previews are V3).
      Also fixed `IsCached` missing HiDream/Qwen-Image entries (they reloaded every gen).
- [x] `IsValidForThisBackend`: video block refuses InitImage (I2V pending the Wan VAE
      encoder), `VideoEndFrame`, `VideoExtendModel`, audio params, and refiners over video.
      LTX-2 classes are distinct compat IDs (`lightricks-ltx-video-2*`) and fall to the
      standard unsupported-architecture refusal.
- [x] Cancellation: `_cancelCts` token checked per progress callback and in the muxer.

### V1.6 Validation
- [ ] Real-checkpoint smoke: Wan2.2 TI2V-5B and LTX-Video 0.9 T2V through the Swarm UI,
      mp4 + webp outputs, metadata sidecar, history preview thumbnail. Also confirms the
      Comfy-Org `wan2.2_vae.safetensors` key naming (V1.0 carry-over).
- [ ] Confirm interrupt works mid-generation and VRAM is reclaimed on model swap.

## Phase V2 — Image-to-Video (Wan TI2V first)

- [x] **Engine:** `Wan22VaeEncoder` built in HartsyInference (+ `AvgDown3D` shortcut,
      `Wan22Resample` downsample modes) with `EncodeRgbFrame(backend, rgb24, w, h)` →
      normalized `[1,48,1,H/16,W/16]` latent. Encoder key naming verified against the
      actual Comfy-Org `wan2.2_vae.safetensors` header (HTTP range fetch) — including
      the F16 dtype, which the loader now casts to F32 via `VaePrecisionHelper`.
- [x] **Extension:** init-image path on the MAIN Wan model (not Comfy's two-stage
      `VideoModel` flow, which is now explicitly refused for all archs): `InitImage` →
      `VideoResolution` sizing (`Image Aspect, Model Res` via `ResToModelFit` /
      `Model Preferred` / `Image`, snapped ÷16) → resize → `Wan22VaeEncoder` →
      `firstFrameLatent` → pipeline TI2V path (diffusers `expand_timesteps` semantics:
      same `Text2VideoFrames`/`Steps`/`CFGScale` params as T2V, matching Comfy's
      `Wan22ImageToVideoLatent` start_image wiring). Encoder GPU weights freed after
      the encode, like the umT5 encoder.
- [ ] LTX I2V: needs engine-side conditioning work (`LTXVImgToVideo` equivalent —
      transformer currently has no image hook); InitImage on LTX stays refused.
- [ ] Real-checkpoint I2V smoke (with V1.6).

## Phase V3 — Parity polish (ordered, each independent)

- [x] Latent previews: `LatentArchitecture.Wan/Ltx` added with Comfy's published factor
      tables (Wan22 48×3, LTXV 128×3, machine-extracted from `latent_formats.py`);
      `DecodeLatent2Rgb` handles rank-5 video latents (middle frame); WanVideoPipeline
      passes its live latent zero-copy, LtxVideoPipeline hands a small unpacked
      middle-frame slice. The extension's `PreviewEncoder` consumes them with zero
      changes (TAESD falls back to latent2rgb — taew2.2/taeLTX SideModels entries are a
      possible future upgrade).
- [ ] Streaming decode: use `GenerateFramesAsync` and feed ffmpeg stdin as frames
      arrive (bounds memory on long videos).
- [ ] `VideoEndFrame` (FLF2V-style) — engine support required; evaluate then.
- [x] LoRA support for Wan: engine `LoraStack` auto-detects kohya/musubi/Comfy/diffusers-PEFT
      formats; extension follows the Flux pattern — cache entry retains the converted DiT weight
      dict, `WanVideoLoader.GenerateWithLoras` shallow-clones + merges via `LoraApplier` and runs
      a fresh per-gen transformer/pipeline. DiT-only (Wan's class declares LorasTargetTextEnc=false).
      LTX LoRAs remain refused (no engine validation yet).
- [ ] Lance T2V: register a custom `T2IModelClass` (Swarm has none for Lance) + loader.
- [ ] Video2Video / VideoExtend / audio-input params: explicitly out of scope until the
      base paths are proven.

## Risks / open questions

1. **fp8-scaled UMT5 load** — if the scaled-fp8 dequant isn't quick, ship V1 on the
   fp16 file (~9 GB download) and revisit.
2. **Comfy VAE file key naming** vs `Wan22VaeDecoder` expectations (V1.0 verify item).
3. **LTX version coverage** — our converter targets base 0.9-style single files
   (0.9.5+ VAE renames documented but unvalidated); LTX-2 is a different architecture
   and stays rejected.
4. **VRAM headroom** — 5B DiT + UMT5-XXL + VAE resident together; mitigation: encode
   prompt first, allow encoder weights to be dropped before denoise (later optimization),
   rely on `PipelineCache` eviction.
5. **Feature-flag gating mechanics** — confirm `"text2video"` flag exposure matches
   how the params appear when a Comfy backend is live, so the UI looks identical.
