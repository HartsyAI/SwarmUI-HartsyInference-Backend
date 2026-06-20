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
/// <para><b>Status:</b> the engine flags Animate numerics as first-run-validation pending, and reference-image /
/// background / replace-mode conditioning is not modeled (pose + face only). The SwarmUI wiring here is complete.</para>
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
        // Animate ships as a Wan2.1-14B backbone; honor the 1.3B compat if a small variant ever appears.
        WanVideoConfig config = compat == WanVideoLoader.Wan21_1_3BCompatClassId
            ? WanVideoConfig.T2V_1_3B
            : WanVideoConfig.T2V_14B;

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
        log($"  Converted: {conv.Transformer.Count} transformer keys (Animate: pose + face/motion pathway, inner {config.InnerDim})");

        // Real Wan-Animate-14B face/motion/pose hyperparameters (engine constructor defaults).
        WanAnimateTransformer transformer = new WanAnimateTransformer(config, poseLatentChannels: 16,
            motionEncoderSize: MotionEncoderSize, motionDim: 512, faceHiddenDim: 1024, faceNumHeads: 4,
            injectFaceLatentsBlocks: 5, motionVecDim: 20, motionBlocks: 5);
        transformer.LoadWeights(conv.Transformer);

        try
        {
            // ── 2. Wan2.1 VAE (z=16; decoder + encoder share one weight dict; cast to F32) ──
            log($"Loading Wan2.1 VAE: {vaeModel.Name}");
            var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder(); vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder(); vaeEncoder.LoadWeights(vaeWeights);

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
        backend.Sync();
        backend.FreeWeights(entry.Umt5.EnumerateWeights());

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt, NegativePrompt = negative, Width = width, Height = height,
            Steps = steps, CfgScale = cfgScale, Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p => { cancel.ThrowIfCancellationRequested(); onProgress(p); };
        try
        {
            var (frames, outW, outH, _) = entry.Pipeline.GenerateAnimation(
                promptEmbeds, negEmbeds, poseClip, faceClip, request, bridge);
            Logs.Verbose($"[HartsyInference][Animate] Pipeline returned {frames.Length} frames {outW}x{outH} "
                + $"({numFrames}f pose / {numFrames - 1}f face) in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            poseClip.Dispose();
            faceClip.Dispose();
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
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
        CheckpointLoader?.Dispose();
        Umt5Loader?.Dispose();
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
