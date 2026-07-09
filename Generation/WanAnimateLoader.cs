using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
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

        // Pose clip at full res (T frames); face clip at the motion-encoder res (T−1 frames → gt face groups).
        Tensor poseClip = ControlVideoDecoder.DecodeControlClip(driving, width, height, numFrames, cancel);
        Tensor faceClip = ControlVideoDecoder.DecodeControlClip(driving, MotionEncoderSize, MotionEncoderSize, numFrames - 1, cancel);

        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        Tensor batch = entry.Umt5.Encode(backend,
            [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, TokenLength, entry.Config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, TokenLength, entry.Config.TextDim);
        batch.Dispose();
        // Wan cross-attends all 512 context rows with NO text mask — umT5 pad rows are garbage that drowns
        // the prompt (same fix as WanVideoLoader; this loader predates it).
        WanVideoLoader.ZeroPaddedRows(promptEmbeds, promptTokens, entry.Config.TextDim);
        WanVideoLoader.ZeroPaddedRows(negEmbeds, negTokens, entry.Config.TextDim);
        backend.Sync();
        backend.FreeWeights(entry.Umt5.EnumerateWeights());

        // Reference identity: who performs the driving motion. Required by the conditioning
        // (reference latent frame + mask concat + optional CLIP context), matching ComfyUI's WanAnimateToVideo.
        Image refImage = input.Get(SwarmUIHartsyInference.AnimateReferenceImageParam)
            ?? throw new SwarmUserErrorException(
                "HartsyInference: Wan Animate needs a character image in the 'Animate Reference Image' param "
                + "(HartsyInference group) — the Init Image slot carries the driving pose/motion video.");
        Tensor referenceRgb = RefImageToTensor(refImage, width, height);

        Tensor clipEmbeds = null;
        if (entry.ClipVision is not null)
        {
            backend.PreloadWeights(entry.ClipVision.EnumerateWeights());
            Tensor pixels = ClipImagePreprocessor.Process(refImage, imageSize: 224);
            Tensor clipBatched = entry.ClipVision.EncodeHiddenStates(backend, pixels);   // [1, 257, 1280]
            pixels.Dispose();
            backend.Sync();
            backend.FreeWeights(entry.ClipVision.EnumerateWeights());
            clipEmbeds = WanVideoLoader.DropBatch(clipBatched);
            clipBatched.Dispose();
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
            var (frames, outW, outH, _) = entry.Pipeline.GenerateAnimation(
                promptEmbeds, negEmbeds, referenceRgb, poseClip, faceClip, request,
                clipImageEmbeds: clipEmbeds, onProgress: bridge);
            Logs.Verbose($"[HartsyInference][Animate] Pipeline returned {frames.Length} frames {outW}x{outH} "
                + $"({numFrames}f pose / {numFrames - 1}f face) in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            poseClip.Dispose();
            faceClip.Dispose();
            referenceRgb.Dispose();
            clipEmbeds?.Dispose();
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
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
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
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
