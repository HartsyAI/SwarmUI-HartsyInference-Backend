using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Chroma Radiance (<c>lodestones/Chroma-Radiance</c>) — the pixel-space, VAE-free
/// Chroma variant. Same T5-XXL conditioning and checkpoint converter as <see cref="ChromaLoader"/>,
/// but the transformer carries a NeRF-style pixel decoder head (<c>nerf_image_embedder.*</c>) and
/// the pipeline outputs RGB directly, so there is NO VAE to resolve.
///
/// <para>Engine status: the Radiance pipeline is wired but the checkpoint is mid-pretraining
/// upstream — output quality is validation-gated. First real generation is the test.</para>
/// </summary>
public static class ChromaRadianceLoader
{
    public const string ChromaRadianceCompatClassId = "chroma-radiance";

    public static ChromaRadianceCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Chroma Radiance model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Chroma Radiance checkpoint not found: {model.RawFilePath}");

        T2IModel t5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel),
            entry: SideModels.T5XxlEnconly,
            log: log);

        // ── 1. Load + convert (same converter as Chroma; radiance keys pass through) ──
        log($"Loading Chroma Radiance transformer: {model.Name}");
        var (conv, loader) = ChromaCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            loader.Dispose();
            throw new InvalidOperationException(
                $"Chroma Radiance checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        if (!ChromaRadianceConfig.IsRadiance(conv.Transformer))
        {
            loader.Dispose();
            throw new InvalidOperationException(
                $"'{model.Name}' converted but has no Radiance pixel-decoder keys (nerf_image_embedder.*). " +
                "This looks like a vanilla Chroma checkpoint — use it as a 'chroma' model instead.");
        }
        ChromaRadianceConfig config = ChromaRadianceConfig.FromWeights(conv.Transformer);
        log($"Architecture: Chroma Radiance (pixel-space, patch={config.PatchSize}, nerf hidden={config.NerfHidden}).");
        ChromaRadianceTransformer transformer = new ChromaRadianceTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        // ── 2. T5-XXL ──
        log($"Loading T5-XXL: {t5Model.Name}");
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Model.RawFilePath);
        Dictionary<string, Tensor> t5Weights = ChromaSideModelKeys.NormalizeT5(t5Loader.GetAllTensors());
        if (t5Weights.Count == 0)
        {
            t5Loader.Dispose();
            loader.Dispose();
            throw new InvalidOperationException($"T5 model file '{t5Model.Name}' has no usable T5 tensors.");
        }
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);

        // ── 3. Tokenizer (embedded). No VAE — Radiance is pixel-space. ──
        log("Loading T5 tokenizer (embedded)...");
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 512);

        log("Building Chroma Radiance pipeline...");
        ChromaRadiancePipeline pipeline = new ChromaRadiancePipeline(backend, t5, transformer, config);

        log("Chroma Radiance ready (mid-pretraining checkpoint — output is validation-gated).");
        return new ChromaRadianceCacheEntry
        {
            ModelName = model.Name,
            CompatClass = ChromaRadianceCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Tokenizer = tokenizer,
            T5 = t5,
            Transformer = transformer,
            CheckpointLoader = loader,
            T5Loader = t5Loader,
        };
    }

    public static Image[] Generate(
        ChromaRadianceCacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.DefaultSteps);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.DefaultCfgScale : (float)cfgRaw;

        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
        int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);

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

        var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(
            promptTokens, negTokens, promptMask, negMask, request, bridge);

        Logs.Verbose($"[HartsyInference][ChromaRadiance] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }
}

/// <summary>
/// Loads Zeta-Chroma (<c>lodestones/Zeta-Chroma</c>) — the pixel-space, VAE-free Chroma variant
/// that swaps T5 for Qwen3-4B caption conditioning (same encoder path as <see cref="ZImageLoader"/>)
/// and predicts x0 directly in pixel space. No CLIP, no VAE.
///
/// <para>Engine status: mid-pretraining checkpoint, validation-gated output.</para>
/// </summary>
public static class ZetaChromaLoader
{
    public const string ZetaChromaCompatClassId = "zeta-chroma";

    /// <summary>Qwen3 right-pads EncodeChat output with BosTokenId (151643).</summary>
    private const int Qwen3PadTokenId = 151643;

    public static ZetaChromaCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Zeta-Chroma model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Zeta-Chroma checkpoint not found: {model.RawFilePath}");

        T2IModel qwenModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.QwenModel),
            entry: SideModels.Qwen3_4B,
            log: log);

        // ── 1. Load + convert (Z-Image-derived converter) ──
        log($"Loading Zeta-Chroma transformer: {model.Name}");
        var (conv, loader) = ZetaChromaCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            loader.Dispose();
            throw new InvalidOperationException(
                $"Zeta-Chroma checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        ZetaChromaConfig config = ZetaChromaConfig.FromWeights(conv.Transformer);
        log($"Architecture: Zeta-Chroma (pixel-space, patch={config.PatchSize}, x0-prediction).");
        ZetaChromaTransformer transformer = new ZetaChromaTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        // ── 2. Qwen3-4B caption encoder (owned by caller, freed per-gen like Z-Image) ──
        log($"Loading Qwen3-4B encoder: {qwenModel.Name}");
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenModel.RawFilePath);
        Dictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
        if (qwenWeights.Count == 0)
        {
            qwenLoader.Dispose();
            loader.Dispose();
            throw new InvalidOperationException($"Qwen3 model file '{qwenModel.Name}' has no tensors.");
        }
        LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        qwen.LoadWeights(qwenWeights);

        log("Loading Qwen3 tokenizer (embedded)...");
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 256);

        log("Building Zeta-Chroma pipeline...");
        ZetaChromaPipeline pipeline = new ZetaChromaPipeline(backend, transformer, config);

        log("Zeta-Chroma ready (mid-pretraining checkpoint — output is validation-gated).");
        return new ZetaChromaCacheEntry
        {
            ModelName = model.Name,
            CompatClass = ZetaChromaCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Tokenizer = tokenizer,
            Qwen = qwen,
            Transformer = transformer,
            CheckpointLoader = loader,
            QwenLoader = qwenLoader,
        };
    }

    public static Image[] Generate(
        ZetaChromaCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.DefaultSteps);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfg = cfgRaw <= 0 ? entry.Config.DefaultCfgScale : (float)cfgRaw;

        int[] tokenIds = entry.Tokenizer.EncodeChat(prompt);
        int realLen = ComputeRealLength(tokenIds);
        int penultimateIdx = entry.Qwen.NumLayers - 1;
        Tensor encodedFull = entry.Qwen.EncodeMultiLayer(backend, new[] { tokenIds }, new[] { penultimateIdx });
        Tensor positiveEmbeddings = SliceFirstSeqF32(encodedFull, realLen);

        Tensor negativeEmbeddings = null;
        if (cfg > 1.0f)
        {
            int[] negTokens = entry.Tokenizer.EncodeChat(negative);
            int negRealLen = ComputeRealLength(negTokens);
            Tensor negEncodedFull = entry.Qwen.EncodeMultiLayer(backend, new[] { negTokens }, new[] { penultimateIdx });
            negativeEmbeddings = SliceFirstSeqF32(negEncodedFull, negRealLen);
        }

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromEmbeddings(
            positiveEmbeddings, request, cfgScale: cfg,
            negativeCaptionEmbeddings: negativeEmbeddings, onProgress: bridge);

        Logs.Verbose($"[HartsyInference][ZetaChroma] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }

    private static int ComputeRealLength(int[] tokenIds)
    {
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] == Qwen3PadTokenId) return i;
        }
        return tokenIds.Length;
    }

    private static unsafe Tensor SliceFirstSeqF32(Tensor source, int realLen)
    {
        if (source.Shape.Rank != 3)
            throw new ArgumentException($"Expected 3D tensor, got rank {source.Shape.Rank}.");
        if (source.DType != DType.F32)
            throw new ArgumentException($"SliceFirstSeqF32 expects F32, got {source.DType}.");
        long batch = source.Shape[0];
        long fullLen = source.Shape[1];
        long hidden = source.Shape[2];
        if (realLen <= 0 || realLen > fullLen)
            throw new ArgumentOutOfRangeException(nameof(realLen), $"realLen {realLen} out of range [1..{fullLen}].");
        TensorShape outShape = new TensorShape(batch, realLen, hidden);
        Tensor result = new Tensor(outShape, source.DType);
        long elemSize = source.DType.SizeInBytes;
        long fullRowBytes = fullLen * hidden * elemSize;
        long sliceRowBytes = realLen * hidden * elemSize;
        byte* src = (byte*)source.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        for (long b = 0; b < batch; b++)
        {
            Buffer.MemoryCopy(src + b * fullRowBytes, dst + b * sliceRowBytes, sliceRowBytes, sliceRowBytes);
        }
        return result;
    }
}

/// <summary>Shared side-model key normalization for the Chroma family (T5-XXL prefix stripping).</summary>
internal static class ChromaSideModelKeys
{
    public static Dictionary<string, Tensor> NormalizeT5(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (var kv in raw)
        {
            if (kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal))
                result[kv.Key[ComfyPrefix.Length..]] = kv.Value;
            else
                result[kv.Key] = kv.Value;
        }
        return result;
    }
}

public sealed class ChromaRadianceCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required ChromaRadiancePipeline Pipeline { get; init; }
    public required ChromaRadianceConfig Config { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required ChromaRadianceTransformer Transformer { get; init; }
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
        CheckpointLoader?.Dispose();
        T5Loader?.Dispose();
    }
}

public sealed class ZetaChromaCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required ZetaChromaPipeline Pipeline { get; init; }
    public required ZetaChromaConfig Config { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder Qwen { get; init; }
    public required ZetaChromaTransformer Transformer { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader QwenLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        Qwen?.Dispose();
        CheckpointLoader?.Dispose();
        QwenLoader?.Dispose();
    }
}
