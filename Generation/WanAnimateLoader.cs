using System.IO;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Vision.Detection;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Wan-Animate (character animation). Like VACE it rides the plain Wan2.1 CompatClass, but the checkpoint
/// adds a <c>pose_patch_embedding</c> (driving-pose latent) and a face pathway (<c>motion_encoder</c> →
/// <c>face_encoder</c> → <c>face_adapter</c>); <see cref="WanModelVariants.Detect"/> routes it here off those
/// signature weights.
///
/// <para><b>Driving-video mode:</b> the Init Image slot carries one driving video. We feed it to the engine as the
/// <c>pose</c> clip (<c>[1,3,T,H,W]</c>) and, since the engine defers pose-estimation + face-cropping preprocessors,
/// also derive the <c>face</c> clip from it by resampling to the motion-encoder resolution (<see cref="MotionEncoderSize"/>²).
/// For faithful results the user should supply an already pose-rendered driving video — same "you preprocess the
/// control" model as VACE. The face clip is decoded to <c>numFrames−1</c> frames so the face encoder's 4× temporal
/// downsample (+1 pad) lands exactly on the <c>gt</c> latent frames the face adapter cross-attends per-frame.</para>
///
/// <para><b>Inputs:</b> Init Image = the driving pose/motion video; <c>Animate Reference Image</c> (HartsyInference
/// param group) = the character identity image, VAE-encoded to the reference latent + (when the checkpoint ships the
/// i2v embedder) CLIP-ViT-H context. Background/replace-mode conditioning is not modeled (animate mode only), and
/// <c>continue_motion</c> chunked extension is not wired.</para>
/// </summary>
public static class WanAnimateLoader
{
    private const int TokenLength = 512;

    /// <summary>Wan-Animate motion-encoder input resolution (the face crop is square at this size).</summary>
    public const int MotionEncoderSize = 512;

    public static WanAnimateCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Wan Animate model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Wan Animate checkpoint not found: {model.RawFilePath}");

        string compat = model.ModelClass?.CompatClass?.ID ?? WanVideoLoader.Wan21_14BCompatClassId;

        T2IModel umt5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel), entry: SideModels.Umt5Xxl, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.Wan21Vae, log: log);

        // ── 1. Load + convert the Animate DiT (diffusers-named pose/face keys pass through verbatim) ──
        log($"Loading Wan Animate DiT: {model.Name} (compat {compat})");
        var (conv, ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (!conv.Transformer.ContainsKey("pose_patch_embedding.weight"))
        {
            ditLoader.Dispose();
            throw new SwarmUserErrorException(
                $"HartsyInference: '{model.Name}' is tagged as Wan Animate but has no 'pose_patch_embedding' weights "
                + "after conversion — it may be a plain Wan checkpoint, or a layout the converter doesn't map yet.");
        }
        // Weight-derived config via the engine's detector (sets IsAnimate, dims, ImageDim for the i2v CLIP
        // embedder, and the CFG defaults) — matches the engine harness that validated Animate e2e.
        WanVideoConfig config = WanConfigDetector.Detect(conv.Transformer);
        bool hasClipEmbedder = conv.Transformer.ContainsKey("condition_embedder.image_embedder.norm1.weight");
        log($"  Converted: {conv.Transformer.Count} transformer keys (Animate: pose + face/motion pathway, "
            + $"inner {config.InnerDim}{(hasClipEmbedder ? ", i2v CLIP" : "")})");

        // Engine API (c9603f1): the face/motion/pose hyperparameters moved into the transformer's
        // internal weight-derived setup; the ctor takes the config alone.
        WanAnimateTransformer transformer = new WanAnimateTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        try
        {
            // ── 2. Wan2.1 VAE (z=16; decoder + encoder share one weight dict; cast to F32) ──
            log($"Loading Wan2.1 VAE: {vaeModel.Name}");
            var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder(); vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder(); vaeEncoder.LoadWeights(vaeWeights);

            // ── 2b. CLIP-ViT-H image encoder (Animate ships the i2v image embedder — reference-identity context) ──
            ClipVisionEncoder clipVision = null;
            SafeTensorsLoader clipLoader = null;
            if (hasClipEmbedder)
            {
                T2IModel clipModel = ModelAutoDownloader.EnsureSideModel(
                    userPick: input?.Get(T2IParamTypes.ClipVisionModel), entry: SideModels.ClipVisionH14, log: log);
                log($"Loading CLIP-ViT-H image encoder: {clipModel.Name}");
                clipLoader = new SafeTensorsLoader();
                clipLoader.Load(clipModel.RawFilePath);
                clipVision = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
                clipVision.LoadWeights(clipLoader.GetAllTensors());
            }

            // ── 3. umT5-XXL + tokenizer ──
            log($"Loading umT5-XXL: {umt5Model.Name}");
            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Model.RawFilePath);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: TokenLength);

            log("Building Wan Animate pipeline...");
            WanAnimatePipeline pipeline = new WanAnimatePipeline(backend, transformer, vaeDecoder, vaeEncoder, config);

            log($"Wan Animate ready ({compat}, driving-video).");
            return new WanAnimateCacheEntry
            {
                ModelName = model.Name,
                CompatClass = compat,
                Pipeline = pipeline,
                Config = config,
                Tokenizer = tokenizer,
                Umt5 = umt5,
                Transformer = transformer,
                Vae = vaeDecoder,
                VaeEncoder = vaeEncoder,
                ClipVision = clipVision,
                ClipLoader = clipLoader,
                CheckpointLoader = ditLoader,
                VaeLoaders = vaeLoaders,
                Umt5Loader = umt5Loader,
            };
        }
        catch
        {
            transformer.Dispose();
            ditLoader.Dispose();
            throw;
        }
    }

    public static Image[] Generate(
        WanAnimateCacheEntry entry, IBackend backend, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 81, step: entry.Config.VaeTemporalCompression);
        if (numFrames < 5)
        {
            throw new SwarmUserErrorException("HartsyInference: Wan Animate needs at least 5 frames (the face pathway downsamples 4×).");
        }
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        Image driving = input.Get(T2IParamTypes.InitImage)
            ?? throw new SwarmUserErrorException(
                "HartsyInference: Wan Animate needs a driving video in the Init Image slot (the pose/motion sequence to animate).");

        var (width, height) = VideoParamResolver.ResolveResolution(input, multiple: entry.Config.VaeSpatialCompression);

        // ── umT5 prompt cache keyed on the (pos, neg) token ids (the WanVideoLoader round-6 pattern): a HIT skips
        // the whole umT5 phase including its multi-GB weight upload. Cached tensors are host-materialized
        // (ZeroPaddedRows touches them), so they survive activation sweeps and re-fault to device on use. ──
        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        bool textHit = entry.CachedPromptTokens is not null && entry.CachedNegTokens is not null
            && promptTokens.AsSpan().SequenceEqual(entry.CachedPromptTokens)
            && negTokens.AsSpan().SequenceEqual(entry.CachedNegTokens);
        Tensor promptEmbeds, negEmbeds;
        if (textHit)
        {
            promptEmbeds = entry.CachedPromptEmbeds;
            negEmbeds = entry.CachedNegEmbeds;
            Logs.Info("[HartsyInference][Animate] [wan-phase] umT5 prompt cache HIT — text encode skipped.");
        }
        else
        {
            long teStart = Environment.TickCount64;
            EnsureEncoderHeadroom(backend, entry, entry.Umt5.EnumerateWeights(), "umT5");
            Tensor batch = entry.Umt5.Encode(backend,
                [promptTokens, negTokens],
                [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
            promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, TokenLength, entry.Config.TextDim);
            negEmbeds = CfgHelper.SliceBatchElement(batch, 1, TokenLength, entry.Config.TextDim);
            batch.Dispose();
            // Wan cross-attends all 512 context rows with NO text mask — umT5 pad rows are garbage that drowns
            // the prompt (same fix as WanVideoLoader; this loader predates it).
            WanVideoLoader.ZeroPaddedRows(promptEmbeds, promptTokens, entry.Config.TextDim);
            WanVideoLoader.ZeroPaddedRows(negEmbeds, negTokens, entry.Config.TextDim);
            backend.Sync();
            backend.FreeWeights(entry.Umt5.EnumerateWeights());
            entry.CachedPromptEmbeds?.Dispose();
            entry.CachedNegEmbeds?.Dispose();
            entry.CachedPromptEmbeds = promptEmbeds;
            entry.CachedNegEmbeds = negEmbeds;
            entry.CachedPromptTokens = promptTokens;
            entry.CachedNegTokens = negTokens;
            Logs.Info($"[HartsyInference][Animate] [wan-phase] umT5 prompt cache MISS — encode+free {Environment.TickCount64 - teStart}ms.");
        }

        // Reference identity: who performs the driving motion. Required by the conditioning
        // (reference latent frame + mask concat + optional CLIP context), matching ComfyUI's WanAnimateToVideo.
        Image refImage = input.Get(SwarmUIHartsyInference.AnimateReferenceImageParam)
            ?? throw new SwarmUserErrorException(
                "HartsyInference: Wan Animate needs a character image in the 'Animate Reference Image' param "
                + "(HartsyInference group) — the Init Image slot carries the driving pose/motion video.");

        // ── CLIP-ViT-H reference-embeds cache keyed on the raw reference-image bytes (the CLIP preprocessor resizes
        // to 224² regardless of the output resolution, so the identity context is resolution-independent). A HIT
        // skips the CLIP-ViT-H upload + encode; host-materialized to survive activation sweeps. ──
        Tensor clipEmbeds = null;
        if (entry.ClipVision is not null)
        {
            string clipKey = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(refImage.RawData));
            if (entry.CachedClipEmbeds is not null && clipKey == entry.CachedClipKey)
            {
                clipEmbeds = entry.CachedClipEmbeds;
                Logs.Info("[HartsyInference][Animate] [wan-phase] CLIP reference cache HIT — image encode skipped.");
            }
            else
            {
                long clipStart = Environment.TickCount64;
                EnsureEncoderHeadroom(backend, entry, entry.ClipVision.EnumerateWeights(), "CLIP-ViT-H");
                backend.PreloadWeights(entry.ClipVision.EnumerateWeights());
                Tensor pixels = ClipImagePreprocessor.Process(refImage, imageSize: 224);
                Tensor clipBatched = entry.ClipVision.EncodeHiddenStates(backend, pixels);   // [1, 257, 1280]
                pixels.Dispose();
                backend.Sync();
                backend.FreeWeights(entry.ClipVision.EnumerateWeights());
                Tensor dropped = WanVideoLoader.DropBatch(clipBatched);
                clipBatched.Dispose();
                clipEmbeds = HostCopy(dropped);
                dropped.Dispose();
                entry.CachedClipEmbeds?.Dispose();
                entry.CachedClipEmbeds = clipEmbeds;
                entry.CachedClipKey = clipKey;
                Logs.Info($"[HartsyInference][Animate] [wan-phase] CLIP reference cache MISS — encode+free {Environment.TickCount64 - clipStart}ms.");
            }
        }

        // ── Driving preprocessing mode: pre-rendered pose/face overrides always win; otherwise auto-preprocess
        // (default) derives the skeleton + face crop in-backend (Phase 0: falls back to the raw clip until the pose/
        // face vision models land). See docs/Design/WAN_ANIMATE_PREPROCESSING.md. ──
        bool autoPreprocess = !input.TryGet(SwarmUIHartsyInference.AnimateAutoPreprocessParam, out bool ap) || ap;
        Image poseOverride = input.Get(SwarmUIHartsyInference.AnimatePoseVideoParam);
        Image faceOverride = input.Get(SwarmUIHartsyInference.AnimateFaceVideoParam);

        // ── Conditioning cache keyed on every input that shapes the conditioning: the driving clip, any pre-rendered
        // pose/face overrides, the preprocessing mode, the reference image, and the geometry. A HIT skips the whole
        // preprocess + VAE + motion-encode phase (and the driving decode). Host-materialized by the pipeline. ──
        string condKey = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(driving.RawData))
            + (poseOverride is not null ? Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(poseOverride.RawData)) : "-")
            + (faceOverride is not null ? Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(faceOverride.RawData)) : "-")
            + Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(refImage.RawData))
            + $":auto{(autoPreprocess ? 1 : 0)}:{width}x{height}:{numFrames}";
        bool condHit = entry.CachedConditioning is not null && condKey == entry.CachedConditioningKey;
        Tensor poseClip = null, faceClip = null, referenceRgb = null;
        WanAnimateConditioning cachedConditioning = null;
        if (condHit)
        {
            cachedConditioning = entry.CachedConditioning;
            Logs.Info("[HartsyInference][Animate] [wan-phase] conditioning cache HIT — preprocess + pose/gray VAE + motion encode + driving decode skipped.");
        }
        else
        {
            long condStart = Environment.TickCount64;
            // Both auto branches share one YOLO11-pose pipeline (cached on the entry): the face branch crops from its
            // keypoints, the pose branch renders the OpenPose skeleton from them.
            YoloPosePipeline pose = (autoPreprocess && (faceOverride is null || poseOverride is null))
                ? GetOrCreatePose(backend, entry) : null;
            (poseClip, faceClip) = WanAnimateDrivingPreprocessor.BuildDrivingClips(
                backend, pose, driving, poseOverride, faceOverride, autoPreprocess, width, height, numFrames, MotionEncoderSize, cancel);
            referenceRgb = RefImageToTensor(refImage, width, height);
            Logs.Info($"[HartsyInference][Animate] [wan-phase] conditioning cache MISS — driving prepared ({Environment.TickCount64 - condStart}ms), encoding this gen.");
        }

        // Sigma Shift: honor the user's param; default 8.0 = ComfyUI's Wan sampling shift (WAN22_Animate
        // inherits WAN21_T2V's sampling_settings).
        float flowShift = input.TryGet(T2IParamTypes.SigmaShift, out double sigmaShift) ? (float)sigmaShift : 8f;
        VideoGenerationRequest request = new VideoGenerationRequest
        {
            Prompt = prompt, NegativePrompt = negative, Width = width, Height = height,
            Steps = steps, CfgScale = cfgScale, Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
            FlowShift = flowShift,
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p => { cancel.ThrowIfCancellationRequested(); onProgress(p); };
        try
        {
            var (frames, outW, outH, _, usedConditioning) = entry.Pipeline.GenerateAnimation(
                promptEmbeds, negEmbeds, referenceRgb, poseClip, faceClip, request,
                clipImageEmbeds: clipEmbeds, cachedConditioning: cachedConditioning, onProgress: bridge);
            if (!condHit)
            {
                // Store the freshly-encoded conditioning for the next same-driving-video/reference gen (host-owned).
                entry.CachedConditioning?.Dispose();
                entry.CachedConditioning = usedConditioning;
                entry.CachedConditioningKey = condKey;
            }
            Logs.Verbose($"[HartsyInference][Animate] Pipeline returned {frames.Length} frames {outW}x{outH} "
                + $"({numFrames}f pose / {numFrames - 1}f face) in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            poseClip?.Dispose();
            faceClip?.Dispose();
            referenceRgb?.Dispose();
            // promptEmbeds/negEmbeds/clipEmbeds are the cross-generation caches (host-materialized; freed on the
            // next cache miss or in WanAnimateCacheEntry.Dispose) — never disposed here.
        }
    }

    /// <summary>Frees the (possibly KEEP_MODELS-resident) DiT device weights before an encoder upload when the
    /// encoder doesn't fit beside it (measured free VRAM, 2 GB margin); trims the pool first so slack from the
    /// previous generation doesn't make the reading pessimistic. Mirrors <see cref="WanS2VLoader"/>'s staging.</summary>
    private static void EnsureEncoderHeadroom(IBackend backend, WanAnimateCacheEntry entry, IEnumerable<Tensor> encoderWeights, string label)
    {
        backend.TrimMemoryPool();
        long need = 0;
        foreach (Tensor t in encoderWeights) need += t.DType.ComputeByteCount(t.ElementCount);
        long free = backend.FreeMemoryBytes();
        if (free < need + (2L << 30))
        {
            Logs.Info($"[HartsyInference][Animate] [wan-phase] {label} upload: evicting resident DiT " +
                $"(free {free >> 20} MB < {need >> 20} MB + 2048 MB margin).");
            backend.FreeWeights(entry.Transformer.EnumerateWeights());
            backend.TrimMemoryPool();
        }
        else
        {
            Logs.Verbose($"[HartsyInference][Animate] [wan-phase] {label} fits beside the resident DiT " +
                $"(free {free >> 20} MB ≥ {need >> 20} MB + 2048 MB margin).");
        }
    }

    /// <summary>Fresh host-materialized F32 copy of a device-produced tensor (call after <c>backend.Sync()</c>) —
    /// the cross-generation cache form that survives per-step activation sweeps and re-faults to device on use.</summary>
    private static unsafe Tensor HostCopy(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, DType.F32);
        long bytes = x.Shape.ElementCount * 4;
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Decodes a reference image to the pipeline's <c>[1, 3, 1, H, W]</c> tensor in [-1, 1], resized to
    /// the pose clip's resolution (the engine requires reference and pose sizes to match).</summary>
    private static unsafe Tensor RefImageToTensor(Image image, int width, int height)
    {
        byte[] rgb24 = RgbToImage.ToHwcRgbResized(image, width, height);
        Tensor t = new Tensor(new TensorShape([1L, 3, 1, height, width]), DType.F32);
        float* p = (float*)t.DataPointer;
        long frame = (long)height * width;
        for (long pix = 0; pix < frame; pix++)
            for (int c = 0; c < 3; c++)
                p[c * frame + pix] = rgb24[(int)(pix * 3 + c)] / 127.5f - 1f;
        return t;
    }

    private const string PoseWeightsFile = "yolo11n-pose-folded.safetensors";

    /// <summary>Lazily builds + caches the YOLO11-pose pipeline used to auto-crop the face (and, later, render the
    /// pose skeleton) from the driving video. Cached on the entry so its ~11 MB weights load once per model.</summary>
    private static YoloPosePipeline GetOrCreatePose(IBackend backend, WanAnimateCacheEntry entry)
    {
        if (entry.PosePipeline is not null) return entry.PosePipeline;
        string path = ResolvePoseWeights();
        entry.PosePipeline = new YoloPosePipeline(backend, YoloConfig.YoloV11nPose, path, inputSize: 640);
        Logs.Info($"[HartsyInference][Animate] loaded YOLO11n-pose for face preprocessing ({path}).");
        return entry.PosePipeline;
    }

    /// <summary>Resolves the folded YOLO11n-pose safetensors from the shared side-model folder. TODO(host): add an
    /// auto-download <see cref="SideModels"/> entry once the folded file is hosted; until then it is converted locally.</summary>
    private static string ResolvePoseWeights()
    {
        if (!Program.T2IModelSets.TryGetValue("Clip", out T2IModelHandler handler) || handler.FolderPaths.Length == 0)
            throw new SwarmUserErrorException("HartsyInference: cannot locate the model folder for the YOLO11n-pose weights.");
        string path = Path.Combine(handler.FolderPaths[0], PoseWeightsFile);
        if (!File.Exists(path))
            throw new SwarmUserErrorException(
                $"HartsyInference: Wan-Animate auto face-preprocessing needs the YOLO11n-pose weights at '{path}'. "
                + "Convert Ultralytics 'yolo11n-pose.pt' with tests/python-reference/convert_yolov8_pt_to_safetensors.py, "
                + "or turn off 'Animate Auto-Preprocess Driving' and supply an 'Animate Face Video' instead.");
        return path;
    }
}

public sealed class WanAnimateCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required WanAnimatePipeline Pipeline { get; init; }
    public required WanVideoConfig Config { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder Umt5 { get; init; }
    public required WanAnimateTransformer Transformer { get; init; }
    public required IWanVaeDecoder Vae { get; init; }
    public required IWanVaeEncoder VaeEncoder { get; init; }
    /// <summary>CLIP-ViT-H reference-image encoder; null when the checkpoint has no i2v image embedder.</summary>
    public ClipVisionEncoder ClipVision { get; init; }
    public SafeTensorsLoader ClipLoader { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required IReadOnlyList<SafeTensorsLoader> VaeLoaders { get; init; }
    public required SafeTensorsLoader Umt5Loader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;

    /// <summary>Cross-generation umT5 prompt cache: token-id keys plus the two zero-padded host embedding tensors. A hit skips the whole umT5 upload+encode phase.</summary>
    public int[] CachedPromptTokens { get; set; }
    public int[] CachedNegTokens { get; set; }
    public Tensor CachedPromptEmbeds { get; set; }
    public Tensor CachedNegEmbeds { get; set; }

    /// <summary>Cross-generation CLIP-ViT-H reference-identity embeds cache keyed on the SHA-256 of the raw reference-image bytes (resolution-independent — the CLIP preprocessor resizes to 224²).</summary>
    public string CachedClipKey { get; set; }
    public Tensor CachedClipEmbeds { get; set; }

    /// <summary>Cross-generation conditioning cache (VAE-encoded pose/reference/gray latents + face-motion features), keyed on the SHA-256 of the driving-video bytes + reference-image bytes + geometry. A hit skips the whole VAE + motion-encode phase and the driving-video decode.</summary>
    public string CachedConditioningKey { get; set; }
    public WanAnimateConditioning CachedConditioning { get; set; }

    /// <summary>Lazily-created YOLO11-pose pipeline for auto face/pose preprocessing; loaded once, disposed with the entry.</summary>
    public YoloPosePipeline PosePipeline { get; set; }

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        CachedPromptEmbeds?.Dispose();
        CachedNegEmbeds?.Dispose();
        CachedClipEmbeds?.Dispose();
        CachedConditioning?.Dispose();
        PosePipeline?.Dispose();
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        Umt5?.Dispose();
        Transformer?.Dispose();
        (ClipVision as object as IDisposable)?.Dispose();
        ClipLoader?.Dispose();
        CheckpointLoader?.Dispose();
        Umt5Loader?.Dispose();
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
