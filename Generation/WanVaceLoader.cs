using System.IO;
using SwarmUI.Media;
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
/// Loads the Wan2.1 <b>VACE</b> (Video All-in-one Creation and Editing) control DiT. VACE shares the plain
/// Wan2.1 CompatClass (<c>wan-21-14b</c> / <c>wan-21-1_3b</c>) but adds a parallel control branch
/// (<c>vace_patch_embedding</c> + <c>vace_blocks.*</c>); SwarmUI's class sorter tags it with the
/// <c>wan-2_1-vace-*</c> model-class IDs, which <see cref="WanModelVariants.IsVace"/> uses to route here
/// instead of <see cref="WanVideoLoader"/>.
///
/// <para><b>Control-video mode:</b> the Init Image slot carries the control clip (a pose/depth/edge/sketch
/// video — or a still, tiled). <see cref="ControlVideoDecoder"/> turns it into the <c>[1,3,T,H,W]</c> tensor
/// the engine VAE-encodes into the control context that conditions every denoise step (alongside umT5 text).
/// This is the engine's native VACE entry point (<see cref="WanVacePipeline.GenerateFromControl"/>) and is
/// strictly more capable than the ComfyUI extension's reference-image-only wiring.</para>
///
/// <para>Side models (auto-downloaded; user picks win): umT5-XXL (<see cref="SideModels.Umt5Xxl"/>) and the
/// z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>). VACE is text+control only — no CLIP image encoder.</para>
///
/// <para><b>Status:</b> the engine flags VACE numerics as first-run-validation pending (control composition
/// + mask layout unverified vs the real checkpoint); the SwarmUI wiring here is complete.</para>
/// </summary>
public static class WanVaceLoader
{
    /// <summary>Wan's umT5 context length (matches diffusers' 512-token encode).</summary>
    private const int TokenLength = 512;

    /// <summary>Uniform control-hint scale passed to every VACE layer. ComfyUI hard-codes 1.0; per-layer /
    /// UI-driven scaling is a future enhancement.</summary>
    private const float ControlScale = 1.0f;

    public static WanVaceCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Wan VACE model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Wan VACE checkpoint not found: {model.RawFilePath}");

        string compat = model.ModelClass?.CompatClass?.ID ?? WanVideoLoader.Wan21_14BCompatClassId;
        WanVideoConfig config = compat == WanVideoLoader.Wan21_1_3BCompatClassId
            ? WanVideoConfig.Vace_1_3B
            : WanVideoConfig.Vace_14B;

        T2IModel umt5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel),
            entry: SideModels.Umt5Xxl, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: SideModels.Wan21Vae, log: log);

        // ── 1. Load + convert the VACE DiT (original/diffusers naming → diffusers; vace_* keys pass through) ──
        log($"Loading Wan VACE DiT: {model.Name} (compat {compat})");
        var (conv, ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            ditLoader.Dispose();
            throw new InvalidOperationException(
                $"Wan VACE checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        if (!conv.Transformer.ContainsKey("vace_patch_embedding.weight"))
        {
            ditLoader.Dispose();
            throw new SwarmUserErrorException(
                $"HartsyInference: '{model.Name}' is tagged as Wan VACE but has no 'vace_patch_embedding' weights "
                + "after conversion — it may be a plain Wan2.1 checkpoint mislabeled, or a layout we don't parse yet.");
        }
        log($"  Converted: {conv.Transformer.Count} transformer keys (VACE, {config.VaceLayers.Length} control layers, inner {config.InnerDim})");

        WanVaceTransformer transformer = new WanVaceTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        try
        {
            // ── 2. Wan2.1 VAE (z=16; decoder + encoder share one weight dict; cast to F32) ──
            log($"Loading Wan2.1 VAE: {vaeModel.Name}");
            var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder(); vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder(); vaeEncoder.LoadWeights(vaeWeights);

            // ── 3. umT5-XXL (fp8-scaled folded to plain dtype) ──
            log($"Loading umT5-XXL: {umt5Model.Name}");
            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Model.RawFilePath);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);

            // ── 4. Tokenizer (embedded umT5 256k SentencePiece) ──
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: TokenLength);

            log("Building Wan VACE pipeline...");
            WanVacePipeline pipeline = new WanVacePipeline(backend, transformer, vaeDecoder, vaeEncoder, config);

            log($"Wan VACE ready ({compat}, control-video).");
            return new WanVaceCacheEntry
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
        WanVaceCacheEntry entry, IBackend backend, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 81, step: entry.Config.VaeTemporalCompression);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        Image control = input.Get(T2IParamTypes.InitImage)
            ?? throw new SwarmUserErrorException(
                "HartsyInference: Wan VACE needs a control video (or image) in the Init Image slot — that's the "
                + "pose/depth/edge/sketch sequence the generation follows.");

        // Resolution: a control video resamples to the user's WxH; a still fits the model res by aspect.
        int width, height;
        if (control.Type?.MetaType == MediaMetaType.Video)
        {
            (width, height) = VideoParamResolver.ResolveResolution(input, multiple: entry.Config.VaeSpatialCompression);
        }
        else
        {
            var (imgW, imgH) = RgbToImage.GetDimensions(control);
            (width, height) = VideoParamResolver.ResolveI2VResolution(
                input, input.Get(T2IParamTypes.Model), imgW, imgH, multiple: entry.Config.VaeSpatialCompression);
        }

        // Decode the control clip to [1,3,T,H,W] in [-1,1] before touching the GPU-resident encoders.
        Tensor controlClip = ControlVideoDecoder.DecodeControlClip(control, width, height, numFrames, cancel);

        // Encode the prompt pair, then free the encoder before the DiT preload.
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
            var (frames, outW, outH, _) = entry.Pipeline.GenerateFromControl(
                promptEmbeds, negEmbeds, controlClip, request, ControlScale, bridge);
            Logs.Verbose($"[HartsyInference][VACE] Pipeline returned {frames.Length} frames {outW}x{outH} "
                + $"({numFrames}f control) in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            controlClip.Dispose();
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }
}

public sealed class WanVaceCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required WanVacePipeline Pipeline { get; init; }
    public required WanVideoConfig Config { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder Umt5 { get; init; }
    public required WanVaceTransformer Transformer { get; init; }
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
