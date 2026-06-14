using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Chroma (Lodestone Rock's 8.9B Flux derivative; <c>lodestones/Chroma</c> on
/// HuggingFace). Chroma differs from Flux.1: NO CLIP-L / NO pooled conditioning, T5-only,
/// MMDIT replaced by joint-attention DiT, but **same VAE as Flux.1**.
///
/// Required side models (configured via SwarmUI parameters):
///   - <see cref="T2IParamTypes.T5XXLModel"/>: a T5-XXL safetensors (text-encoders/t5xxl_fp8.safetensors works)
///   - <see cref="T2IParamTypes.VAE"/>: the Flux ae.safetensors (auto-downloaded if missing)
///
/// The Chroma checkpoint itself is the user-selected main model. Currently only <c>chroma</c>
/// is supported — Radiance and Zeta variants would need additional <see cref="ChromaConfig"/>
/// presets in HartsyInference.
/// </summary>
public static class ChromaLoader
{
    public const string ChromaCompatClassId = "chroma";

    public static ChromaCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Chroma model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Chroma checkpoint not found: {model.RawFilePath}");

        // Auto-fetch T5-XXL if the user didn't pick one (same fallback the VAE block below uses).
        // Falls back to the encoder-only fp8 entry — verified hash, ~5 GB.
        T2IModel t5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel),
            entry: SideModels.T5XxlEnconly,
            log: log);

        // ── 1. Load + convert the Chroma transformer ──
        log($"Loading Chroma transformer: {model.Name}");
        var (zConv, zLoader) = ChromaCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (zConv.Transformer.Count == 0)
        {
            zLoader.Dispose();
            throw new InvalidOperationException(
                $"Chroma checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        log($"  Converted: {zConv.Transformer.Count} transformer keys");

        ChromaConfig config = ChromaConfig.V1;
        log($"Architecture: {config.HiddenSize} hidden, {config.Depth} double / {config.DepthSingleBlocks} single (Chroma V1)");
        ChromaTransformer transformer = new ChromaTransformer(config);
        transformer.LoadWeights(zConv.Transformer);

        // ── 2. Load T5-XXL ──
        log($"Loading T5-XXL: {t5Model.Name}");
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Model.RawFilePath);
        Dictionary<string, Tensor> t5Weights = LoadT5FromStandalone(t5Loader.GetAllTensors());
        if (t5Weights.Count == 0)
        {
            t5Loader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"T5 model file '{t5Model.Name}' has no usable T5 tensors.");
        }
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);

        // ── 3. Resolve + load the Flux VAE (Chroma reuses it verbatim) ──
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: SideModels.FluxAe,
            log: log);
        log($"Loading Flux VAE: {vaeModel.Name}");
        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaeModel.RawFilePath);
        Dictionary<string, Tensor> vaeWeights = LoadVaeFromStandalone(vaeLoader.GetAllTensors());
        if (vaeWeights.Count == 0)
        {
            vaeLoader.Dispose();
            t5Loader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"VAE file '{vaeModel.Name}' has no usable VAE tensors.");
        }
        VaeDecoder vae = new VaeDecoder(VaeConfig.Chroma);
        vae.LoadWeights(vaeWeights);

        // ── 4. Tokenizer (embedded) ──
        log("Loading T5 tokenizer (embedded)...");
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 512);

        log("Building Chroma pipeline...");
        ChromaPipeline pipeline = new ChromaPipeline(backend, t5, transformer, vae, config);

        log("Chroma ready.");
        return new ChromaCacheEntry
        {
            ModelName = model.Name,
            CompatClass = ChromaCompatClassId,
            Pipeline = pipeline,
            ChromaConfig = config,
            Tokenizer = tokenizer,
            T5 = t5,
            Transformer = transformer,
            Vae = vae,
            CheckpointLoader = zLoader,
            T5Loader = t5Loader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        ChromaCacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 28);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.ChromaConfig.DefaultCfgScale : (float)cfgRaw;

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

        Logs.Verbose($"[HartsyInference][Chroma] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }

    /// <summary>Standalone T5-XXL safetensors files store keys either as-is (<c>encoder.embed_tokens.weight</c>)
    /// or wrapped under Comfy's <c>text_encoders.t5xxl.transformer.</c> prefix. Strip the prefix if present.</summary>
    private static Dictionary<string, Tensor> LoadT5FromStandalone(Dictionary<string, Tensor> raw)
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

    /// <summary>Same VAE-key normalization as Flux/Z-Image: strip Comfy/LDM prefixes and route
    /// through CheckpointConvertUtils.ConvertVaeKey which handles both LDM-named and
    /// already-diffusers-named files.</summary>
    private static Dictionary<string, Tensor> LoadVaeFromStandalone(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var (key, tensor) in raw)
        {
            string ldmKey = key;
            if (ldmKey.StartsWith("first_stage_model.", StringComparison.Ordinal))
                ldmKey = ldmKey["first_stage_model.".Length..];
            else if (ldmKey.StartsWith("vae.", StringComparison.Ordinal))
                ldmKey = ldmKey["vae.".Length..];

            string diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
            if (diffusersKey is not null)
            {
                result[diffusersKey] = tensor;
            }
        }
        return result;
    }
}

public sealed class ChromaCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required ChromaPipeline Pipeline { get; init; }
    public required ChromaConfig ChromaConfig { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required ChromaTransformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader T5Loader { get; init; }
    public required SafeTensorsLoader VaeLoader { get; init; }

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
        VaeLoader?.Dispose();
    }
}
