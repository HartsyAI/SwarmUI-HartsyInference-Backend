using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Krea 2 (Krea, 12.9B single-stream MMDiT flow-match T2I, open weights). The user picks the diffusion model
/// (<c>diffusion_models/krea2_{raw,turbo}_*.safetensors</c>); the Qwen3-VL-4B text encoder and the Qwen-Image VAE come
/// from the central <see cref="SideModels"/> registry (auto-downloaded, user-overridable):
/// <list type="bullet">
///   <item><c>T2IParamTypes.QwenModel</c> (Models/clip/) — Qwen3-VL-4B text encoder (<see cref="SideModels.Qwen3VL_4B"/>).</item>
///   <item><c>T2IParamTypes.VAE</c> (Models/vae/) — Qwen-Image VAE (<see cref="SideModels.QwenImageVae"/>, shared with Qwen-Image/Anima).</item>
/// </list>
///
/// <para>The released checkpoints use Krea 2's <b>raw</b> key names (<c>blocks.N.*</c>, <c>txtfusion.*</c>, <c>tmlp</c>,
/// <c>tproj</c>, <c>last.*</c>); <see cref="Krea2CheckpointConverter.RemapTransformerKey"/> rewrites them to the engine's
/// diffusers convention (and passes an already-diffusers folder through). The Qwen3-VL-4B keys are remapped to the
/// <see cref="LlamaStyleEncoder"/> convention (vision tower + lm_head dropped).</para>
///
/// <para><b>Base vs Turbo:</b> auto-detected from the filename (<c>turbo</c>/<c>tdm</c>/<c>distill</c> → Turbo). Base
/// uses the resolution-aware flow-match shift + CFG (28 steps / CFG 4.5); Turbo pins the shift to <c>mu=1.15</c> and
/// runs guidance-free (8 steps, single pass). Prompt template + prefix-drop are identical to Qwen-Image.</para>
/// </summary>
public static class Krea2Loader
{
    public const string Krea2CompatClassId = "krea-2";

    public static Krea2CacheEntry Load(IBackend backend, T2IModel model, T2IParamInput input, Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Krea 2 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Krea 2 checkpoint not found: {model.RawFilePath}");

        bool isTurbo = IsTurbo(model.Name) || IsTurbo(Path.GetFileName(model.RawFilePath));
        Krea2Config config = isTurbo ? Krea2Config.Turbo : Krea2Config.Base;
        log($"Loading Krea 2 model: {model.Name} ({(isTurbo ? "Turbo/TDM" : "Base")})");

        T2IModel teModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.QwenModel), entry: SideModels.Qwen3VL_4B, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.QwenImageVae, log: log);

        List<SafeTensorsLoader> loaders = [];
        try
        {
            log($"Loading Krea 2 transformer (12.9B): {Path.GetFileName(model.RawFilePath)}");
            (Dictionary<string, Tensor> ditW, SafeTensorsLoader ditL) =
                LoadComponent(model.RawFilePath, key => Krea2CheckpointConverter.RemapTransformerKey(StripTransformerPrefix(key)), applyFp8Dequant: true);
            loaders.Add(ditL);

            log($"Loading Qwen3-VL-4B text encoder: {teModel.Name}");
            (Dictionary<string, Tensor> teW, SafeTensorsLoader teL) =
                LoadComponent(teModel.RawFilePath, RemapQwenKey, applyFp8Dequant: true);
            loaders.Add(teL);

            log($"Loading Qwen-Image VAE: {vaeModel.Name}");
            (Dictionary<string, Tensor> vaeW, SafeTensorsLoader vaeL) =
                LoadComponent(vaeModel.RawFilePath, key => key, applyFp8Dequant: false);
            loaders.Add(vaeL);

            log("Building Krea 2 models (single-stream MMDiT, 28 blocks, text-fusion stage)...");
            Krea2Transformer transformer = new(config);
            transformer.LoadWeights(ditW);

            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_4B);
            textEncoder.LoadWeights(teW);

            QwenImageVaeDecoder vae = new(VaeConfig.QwenImage);
            vae.LoadWeights(CastToF32(vaeW));

            Qwen3Tokenizer tokenizer = new(maxLength: 512);
            Krea2Pipeline pipeline = new(backend, textEncoder, transformer, vae, config);

            log("Krea 2 ready.");
            return new Krea2CacheEntry
            {
                ModelName = model.Name,
                CompatClass = Krea2CompatClassId,
                Pipeline = pipeline,
                Config = config,
                IsTurbo = isTurbo,
                Tokenizer = tokenizer,
                TextEncoder = textEncoder,
                Transformer = transformer,
                Vae = vae,
                Loaders = loaders,
            };
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    public static Image[] Generate(Krea2CacheEntry entry, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel)
    {
        string prompt = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.Prompt));
        string negative = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.NegativePrompt));
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);

        // Width/height multiples of 16 (2×2 patchify × 8× VAE), 128–4096.
        int snappedW = Math.Clamp(width / 16 * 16, 128, 4096);
        int snappedH = Math.Clamp(height / 16 * 16, 128, 4096);
        if (snappedW != width || snappedH != height)
            Logs.Info($"[HartsyInference][Krea2] Snapped {width}x{height} → {snappedW}x{snappedH} (multiple of 16, 128–4096).");

        // Turbo: 8 steps, guidance off. Base: 28 steps, CFG 4.5.
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.IsTurbo ? 8 : 28);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfg = entry.IsTurbo ? 1.0f : (cfgRaw <= 0 ? 4.5f : (float)cfgRaw);

        var (promptTokens, promptDrop) = EncodeWithTemplate(entry.Tokenizer, prompt);
        var (negTokens, negDrop) = EncodeWithTemplate(entry.Tokenizer, negative);
        bool useCfg = cfg > 1.0f;

        TextToImageRequest request = new()
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = snappedW,
            Height = snappedH,
            Steps = steps,
            CfgScale = cfg,
            Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(
            promptTokens, useCfg ? negTokens : null, request, bridge,
            promptDropIndex: promptDrop, negativeDropIndex: negDrop);

        Logs.Verbose($"[HartsyInference][Krea2] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return [RgbToImage.FromHwcRgb(rgbBytes, outW, outH)];
    }

    private static bool IsTurbo(string name)
    {
        if (string.IsNullOrEmpty(name)) return false;
        string n = name.ToLowerInvariant();
        return n.Contains("turbo") || n.Contains("tdm") || n.Contains("distill");
    }

    /// <summary>Krea 2's prompt template is byte-identical to Qwen-Image's. Same system prompt + prefix-drop design.</summary>
    private const string Krea2SystemPrompt =
        "system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    private static (int[] tokens, int dropIndex) EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        const int MaxTokens = 512;
        List<int> ids = new(64);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw(Krea2SystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n"));
        int dropIndex = ids.Count;
        ids.AddRange(tokenizer.EncodeRaw(prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        if (ids.Count > MaxTokens)
            ids.RemoveRange(MaxTokens, ids.Count - MaxTokens);
        return (ids.ToArray(), dropIndex);
    }

    private static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadComponent(
        string filePath, Func<string, string> keyTransform, bool applyFp8Dequant)
    {
        SafeTensorsLoader loader = new();
        loader.Load(filePath);
        try
        {
            Dictionary<string, Tensor> merged = new();
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                if (kvp.Key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || kvp.Key == "scaled_fp8") continue;
                string mapped = keyTransform(kvp.Key);
                if (mapped is not null) merged[mapped] = kvp.Value;
            }
            return (applyFp8Dequant ? CheckpointConvertUtils.ApplyFp8ScaledDequant(merged) : merged, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private static string StripTransformerPrefix(string key)
    {
        if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
            return key["model.diffusion_model.".Length..];
        if (key.StartsWith("diffusion_model.", StringComparison.Ordinal))
            return key["diffusion_model.".Length..];
        if (key.StartsWith("transformer.", StringComparison.Ordinal))
            return key["transformer.".Length..];
        return key;
    }

    /// <summary>Maps a Qwen3-VL checkpoint key to the <c>LlamaStyleEncoder</c> convention (drops vision tower + lm_head;
    /// returns null to drop a key).</summary>
    private static string RemapQwenKey(string key)
    {
        if (key.Contains(".visual.") || key.StartsWith("visual.", StringComparison.Ordinal)) return null;
        if (key.Contains("lm_head")) return null;

        int lm = key.LastIndexOf("language_model.", StringComparison.Ordinal);
        string suffix = lm >= 0 ? key[(lm + "language_model.".Length)..] : key;
        if (suffix.StartsWith("model.", StringComparison.Ordinal))
            suffix = suffix["model.".Length..];

        if (suffix.StartsWith("layers.", StringComparison.Ordinal)
            || suffix.StartsWith("embed_tokens.", StringComparison.Ordinal)
            || suffix == "norm.weight")
        {
            return "model." + suffix;
        }
        return null;
    }

    private static Dictionary<string, Tensor> CastToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (var (key, tensor) in weights)
            f32[key] = (tensor.DType == DType.F16 || tensor.DType == DType.BF16) ? tensor.CastTo(DType.F32) : tensor;
        return f32;
    }
}

public sealed class Krea2CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required Krea2Pipeline Pipeline { get; init; }
    public required Krea2Config Config { get; init; }
    public required bool IsTurbo { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder TextEncoder { get; init; }
    public required Krea2Transformer Transformer { get; init; }
    public required QwenImageVaeDecoder Vae { get; init; }
    public required List<SafeTensorsLoader> Loaders { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        (TextEncoder as IDisposable)?.Dispose();
        (Transformer as IDisposable)?.Dispose();
        // QwenImageVaeDecoder holds no owned native handles (weights owned by Loaders).
        foreach (SafeTensorsLoader l in Loaders) l.Dispose();
    }
}
