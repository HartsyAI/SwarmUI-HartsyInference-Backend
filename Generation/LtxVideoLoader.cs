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
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads LTX-Video (Lightricks; SwarmUI compat class <c>lightricks-ltx-video</c>). Targets the
/// original 0.9-era single-file checkpoints (e.g. <c>ltx-video-2b-v0.9.safetensors</c>) which
/// bundle DiT + VAE — <c>LtxVideoCheckpointConverter</c> splits and renames both. LTX-2
/// (<c>lightricks-ltx-video-2*</c>) is a different architecture and is refused at validation.
///
/// Required side model (auto-downloaded; user pick via <see cref="T2IParamTypes.T5XXLModel"/>
/// takes priority): plain T5-XXL (<see cref="SideModels.T5XxlEnconly"/> — the same file Flux,
/// SD3, and Chroma use; LTX uses standard T5, not Wan's umT5).
///
/// The user's <c>VideoFPS</c> feeds the pipeline itself (RoPE interpolation — the same value
/// Comfy injects via <c>LTXVConditioning.frame_rate</c>), not just the output muxer.
/// </summary>
public static class LtxVideoLoader
{
    public const string LtxVideoCompatClassId = "lightricks-ltx-video";

    /// <summary>LTX's T5 context length (diffusers uses 128 tokens).</summary>
    private const int TokenLength = 128;

    public static LtxVideoCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("LTX-Video model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"LTX-Video checkpoint not found: {model.RawFilePath}");

        T2IModel t5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel),
            entry: SideModels.T5XxlEnconly,
            log: log);

        // ── 1. Load + convert the LTX single file (DiT + VAE bundled) ──
        log($"Loading LTX-Video checkpoint: {model.Name}");
        var (conv, ckptLoader) = LtxVideoCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            ckptLoader.Dispose();
            throw new InvalidOperationException(
                $"LTX checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        if (conv.Vae.Count == 0)
        {
            ckptLoader.Dispose();
            throw new InvalidOperationException(
                $"LTX checkpoint '{model.Name}' has no bundled VAE weights. HartsyInference currently requires "
                + "a full single-file LTX-Video checkpoint (DiT + VAE in one file, e.g. ltx-video-2b-v0.9.safetensors).");
        }
        log($"  Converted: {conv.Transformer.Count} DiT keys, {conv.Vae.Count} VAE keys");

        LtxVideoConfig config = LtxVideoConfig.V09;
        LtxVideoTransformer transformer = new LtxVideoTransformer(config);
        transformer.LoadWeights(conv.Transformer);
        LtxVideoVaeDecoder vae = new LtxVideoVaeDecoder();
        vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.Vae, DType.F32));

        // ── 2. Load T5-XXL (standard T5 — shared file with Flux/SD3/Chroma) ──
        log($"Loading T5-XXL: {t5Model.Name}");
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Model.RawFilePath);
        Dictionary<string, Tensor> t5Weights = StripStandalonePrefix(t5Loader.GetAllTensors(), "text_encoders.t5xxl.transformer.");
        if (t5Weights.Count == 0)
        {
            t5Loader.Dispose();
            ckptLoader.Dispose();
            throw new InvalidOperationException($"T5 model file '{t5Model.Name}' has no usable T5 tensors.");
        }
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);

        // ── 3. Tokenizer (embedded T5 spiece) ──
        log("Loading T5 tokenizer (embedded)...");
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: TokenLength);

        log("Building LTX-Video pipeline...");
        LtxVideoPipeline pipeline = new LtxVideoPipeline(backend, transformer, vae, config);

        log("LTX-Video ready (text-to-video).");
        return new LtxVideoCacheEntry
        {
            ModelName = model.Name,
            CompatClass = LtxVideoCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Tokenizer = tokenizer,
            T5 = t5,
            Transformer = transformer,
            Vae = vae,
            CheckpointLoader = ckptLoader,
            T5Loader = t5Loader,
        };
    }

    public static Image[] Generate(
        LtxVideoCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        var (width, height) = VideoParamResolver.ResolveResolution(input, multiple: entry.Config.VaeSpatialCompression);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 97, step: entry.Config.VaeTemporalCompression);
        int frameRate = VideoParamResolver.ResolveFps(input);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        // Encode the prompt pair, then drop the encoder's GPU weights before the DiT preload
        // (mirrors the upstream LTX E2E test).
        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        Tensor batch = entry.T5.Encode(backend,
            [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, TokenLength, entry.Config.CaptionChannels);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, TokenLength, entry.Config.CaptionChannels);
        batch.Dispose();
        backend.Sync();
        backend.FreeWeights(entry.T5.EnumerateWeights());

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        try
        {
            var (frames, outW, outH, _) = entry.Pipeline.GenerateFromEmbeddings(
                promptEmbeds, negEmbeds, request, numFrames, frameRate, bridge);
            Logs.Verbose($"[HartsyInference][LTX] Pipeline returned {frames.Length} frames {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <summary>Standalone T5-XXL files store keys as-is or wrapped under Comfy's
    /// <c>text_encoders.t5xxl.transformer.</c> prefix — strip if present (same handling as Chroma/SD3).</summary>
    private static Dictionary<string, Tensor> StripStandalonePrefix(Dictionary<string, Tensor> raw, string comfyPrefix)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var kv in raw)
        {
            if (kv.Key.StartsWith(comfyPrefix, StringComparison.Ordinal))
                result[kv.Key[comfyPrefix.Length..]] = kv.Value;
            else
                result[kv.Key] = kv.Value;
        }
        return result;
    }

}

public sealed class LtxVideoCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required LtxVideoPipeline Pipeline { get; init; }
    public required LtxVideoConfig Config { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required LtxVideoTransformer Transformer { get; init; }
    public required LtxVideoVaeDecoder Vae { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader T5Loader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        T5?.Dispose();
        Transformer?.Dispose();
        CheckpointLoader?.Dispose();
        T5Loader?.Dispose();
    }
}
