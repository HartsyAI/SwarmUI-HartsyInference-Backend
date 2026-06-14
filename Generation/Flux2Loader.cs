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
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Flux.2 family checkpoints (Klein 4B, Klein 9B, Dev). Flux.2 uses Qwen-style text
/// encoding (Klein 4B = Qwen3-4B, Klein 9B / Dev = Qwen3-8B per the released configs)
/// and a NEW VAE that's distinct from Flux.1's (32 latent channels + BatchNorm running
/// stats stored alongside the VAE weights, vs Flux.1's 16-channel autoencoder).
///
/// Variant detection is hidden-size-based on the converted transformer:
///   3072 → Klein 4B, 4096 → Klein 9B, 6144 → Dev.
///
/// SwarmUI side-models picked through the standard parameters:
///   - Qwen text encoder: <see cref="T2IParamTypes.QwenModel"/> (Models/clip/)
///   - Flux.2 VAE:        <see cref="T2IParamTypes.VAE"/>      (Models/vae/Flux2/)
///
/// Both have auto-download fallbacks to the canonical Comfy-Org distributions.
/// </summary>
public static class Flux2Loader
{
    public const string Flux2BaseCompatClassId = "flux-2";
    public const string Flux2Klein4BCompatClassId = "flux-2-klein-4b";
    public const string Flux2Klein9BCompatClassId = "flux-2-klein-9b";

    public static bool IsFlux2Compat(string compatClass) =>
        compatClass == Flux2BaseCompatClassId
        || compatClass == Flux2Klein4BCompatClassId
        || compatClass == Flux2Klein9BCompatClassId;

    public static Flux2CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Flux.2 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Flux.2 checkpoint not found: {model.RawFilePath}");

        // Resolve variant by filename heuristic first; if unclear, fall back to hidden-size detection
        // after the converter runs (which knows the actual shapes). This lets us pick the right
        // text-encoder size up front (4B vs 8B Qwen) — important because they're separate downloads.
        Flux2Config config = GuessConfigFromFilename(model.Name) ?? Flux2Config.Klein4B;
        log($"Loading Flux.2 transformer: {model.Name} (initial guess: {DescribeConfig(config)})");

        // ── 1. Load and convert the transformer (Klein BFL layout → canonical) ──
        SafeTensorsLoader transformerLoader = new SafeTensorsLoader();
        transformerLoader.Load(model.RawFilePath);
        Dictionary<string, Tensor> rawWeights = transformerLoader.GetAllTensors();

        // Pre-cast BF16 → F16 on CPU (CudaBackend doesn't yet have a BF16↔F32 GPU cast for
        // every op; F16 keeps the same 8 GB footprint as BF16 and F16↔F32 IS supported).
        Dictionary<string, Tensor> castWeights = new(rawWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in rawWeights)
        {
            DType d = kvp.Value.DType;
            castWeights[kvp.Key] = (d == DType.F32 || d == DType.F16) ? kvp.Value : kvp.Value.CastTo(DType.F16);
        }
        rawWeights.Clear();

        // Confirm the variant by inspecting the converted hidden size before building the transformer.
        config = DetectConfigFromTransformerWeights(castWeights, config);
        log($"Architecture: hidden={config.HiddenSize}, depth={config.Depth} double + " +
            $"{config.DepthSingleBlocks} single → {DescribeConfig(config)}");

        int mlpInner = (int)(config.HiddenSize * config.MlpRatio);
        Flux2CheckpointConverter converter = new Flux2CheckpointConverter(config.HiddenSize, mlpInner);
        Dictionary<string, Tensor> converted = converter.ConvertTransformer(castWeights);
        castWeights.Clear();

        log("Building Flux2Transformer...");
        Flux2Transformer transformer = new Flux2Transformer(config);
        transformer.LoadWeights(converted);
        converted.Clear();

        // ── 2. Resolve + load the right text encoder for this variant ──
        // Klein 4B → Qwen3-4B, Klein 9B → Qwen3-8B, Dev → Mistral-Small-3.
        // The user's QwenModel param can override; otherwise auto-download from SideModels.
        (LlamaStyleEncoderConfig encoderConfig, SideModels.Entry encoderEntry, string encoderLabel) =
            ResolveTextEncoderForVariant(config);

        T2IModel encoderModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.QwenModel),
            entry: encoderEntry,
            log: log);
        log($"Loading {encoderLabel}: {encoderModel.Name}");

        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(encoderModel.RawFilePath);
        Dictionary<string, Tensor> qwenRaw = qwenLoader.GetAllTensors();
        // Refuse FP4 weights at load time — HartsyInference doesn't have an FP4 GEMM path yet,
        // so a Comfy fp4_mixed file would silently load with zeros and produce a black image.
        foreach (KeyValuePair<string, Tensor> kvp in qwenRaw)
        {
            string dtypeName = kvp.Value.DType.Name;
            if (dtypeName.StartsWith("F4", StringComparison.Ordinal) || dtypeName.Contains("FP4", StringComparison.Ordinal))
            {
                qwenLoader.Dispose();
                transformerLoader.Dispose();
                throw new NotSupportedException(
                    $"{encoderLabel} weights at '{encoderModel.Name}' contain FP4 tensors " +
                    $"(e.g. '{kvp.Key}' is {dtypeName}). HartsyInference doesn't support FP4 GEMM yet — " +
                    "swap to an fp8 or fp16 variant of the same encoder, or pick a different Flux.2 variant. " +
                    "Klein 4B (hidden=3072) uses Qwen3-4B which IS shipped as fp8-mixed and works today.");
            }
        }
        Dictionary<string, Tensor> qwenCast = new(qwenRaw.Count);
        foreach (KeyValuePair<string, Tensor> kvp in qwenRaw)
        {
            DType d = kvp.Value.DType;
            qwenCast[kvp.Key] = (d == DType.F32 || d == DType.F16) ? kvp.Value : kvp.Value.CastTo(DType.F16);
        }
        qwenRaw.Clear();

        LlamaStyleEncoder encoder = new LlamaStyleEncoder(encoderConfig);
        encoder.LoadWeights(qwenCast);
        qwenCast.Clear();

        // ── 3. Resolve + load the Flux.2 VAE (separate from Flux.1 ae) ──
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: SideModels.Flux2Vae,
            log: log);
        log($"Loading Flux.2 VAE: {vaeModel.Name}");

        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaeModel.RawFilePath);
        Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();

        if (!vaeWeights.TryGetValue("bn.running_mean", out Tensor bnMean))
            throw new InvalidOperationException(
                $"Flux.2 VAE '{vaeModel.Name}' is missing 'bn.running_mean'. " +
                "Verify this is a Flux.2 VAE checkpoint, not a Flux.1 ae.safetensors.");
        if (!vaeWeights.TryGetValue("bn.running_var", out Tensor bnVar))
            throw new InvalidOperationException(
                $"Flux.2 VAE '{vaeModel.Name}' is missing 'bn.running_var'.");

        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux2);
        vaeDecoder.LoadWeights(vaeWeights);

        // Encoder half of the same VAE — required for img2img (source → 32ch latent). The Flux.2
        // VAE checkpoint carries both encoder.* and decoder.* keys, so this reuses vaeWeights.
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Flux2);
        vaeEncoder.LoadWeights(vaeWeights);

        // ── 4. Tokenizer (embedded Qwen3 vocab/merges; same for 4B and 8B) ──
        log("Loading Qwen3 tokenizer (embedded)...");
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);

        log("Building Flux.2 pipeline...");
        Flux2Pipeline pipeline = new Flux2Pipeline(
            backend, encoder, transformer, vaeDecoder, vaeEncoder,
            bnMean, bnVar, config,
            hiddenLayers: null,    // null → defaults [9, 18, 27]
            bnEps: 1e-5f);

        log($"Flux.2 ready ({DescribeConfig(config)}).");
        return new Flux2CacheEntry
        {
            ModelName = model.Name,
            CompatClass = model.ModelClass?.CompatClass?.ID ?? Flux2BaseCompatClassId,
            Pipeline = pipeline,
            Flux2Config = config,
            Tokenizer = tokenizer,
            Encoder = encoder,
            Transformer = transformer,
            Vae = vaeDecoder,
            BnMean = bnMean,
            BnVar = bnVar,
            CheckpointLoader = transformerLoader,
            QwenLoader = qwenLoader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        Flux2CacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Flux2Config.GuidanceEmbed ? 28 : 10);
        // Flux.2 rounds image dims down to a multiple of 16 (VAE 8× × 2×2 patch). Resolve img2img
        // against those rounded dims so the source tensor matches the pipeline's shape check.
        int width = (input.Get(T2IParamTypes.Width) / 16) * 16;
        int height = (input.Get(T2IParamTypes.Height) / 16) * 16;
        long seedLong = input.Get(T2IParamTypes.Seed);
        int? seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF);

        // Klein has no guidance embedding; Dev uses guidance ~3.5 (BFL distillation target).
        float guidance = entry.Flux2Config.GuidanceEmbed ? 3.5f : 0f;

        int[] tokenIds = entry.Tokenizer.EncodeChat(prompt);

        Img2ImgResolver.Img2ImgSpec img2img = Img2ImgResolver.Resolve(input, width, height);
        TextToImageRequest request;
        if (img2img is not null)
        {
            request = new ImageToImageRequest
            {
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = seed,
                SourceImage = img2img.SourceTensor,
                Strength = img2img.Strength,
            };
        }
        else
        {
            request = new TextToImageRequest
            {
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = seed,
            };
        }
        Logs.Verbose($"[HartsyInference][Flux.2] Tokenized prompt: {tokenIds.Length} tokens, steps={steps}, guidance={guidance:F2}, mode={(img2img is not null ? "img2img" : "txt2img")}");

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        try
        {
            var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(
                tokenIds, request, guidanceScale: guidance, onProgress: bridge);

            Logs.Verbose($"[HartsyInference][Flux.2] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
        }
        finally
        {
            img2img?.SourceTensor?.Dispose();
        }
    }

    /// <summary>Filename-based variant guess. Used as the initial config to pick which Qwen
    /// encoder to download, before the full converter runs and confirms via hidden_size.</summary>
    private static Flux2Config GuessConfigFromFilename(string name)
    {
        if (string.IsNullOrEmpty(name)) return null;
        string n = name.ToLowerInvariant();
        if (n.Contains("klein-4b") || n.Contains("klein_4b") || n.Contains("klein4b"))
            return Flux2Config.Klein4B;
        if (n.Contains("klein-9b") || n.Contains("klein_9b") || n.Contains("klein9b"))
            return Flux2Config.Klein9B;
        if (n.Contains("flux2_dev") || n.Contains("flux-2-dev") || n.Contains("flux2-dev"))
            return Flux2Config.Dev;
        return null;
    }

    /// <summary>Confirms the variant by reading the actual hidden size from the converted
    /// transformer's <c>img_in.weight</c> (or <c>x_embedder.weight</c>). This is the
    /// authoritative answer — filename hints can lie.</summary>
    private static Flux2Config DetectConfigFromTransformerWeights(
        Dictionary<string, Tensor> weights, Flux2Config fallback)
    {
        // Try the canonical post-conversion key first, then a couple fallbacks.
        Tensor probe = null;
        foreach (string k in new[] { "img_in.weight", "x_embedder.weight", "patch_embed.proj.weight" })
        {
            if (weights.TryGetValue(k, out probe)) break;
        }
        if (probe is null || probe.Shape.Rank < 1)
        {
            return fallback;
        }
        int hidden = (int)probe.Shape[0];
        return hidden switch
        {
            3072 => Flux2Config.Klein4B,
            4096 => Flux2Config.Klein9B,
            6144 => Flux2Config.Dev,
            _ => fallback,
        };
    }

    private static string DescribeConfig(Flux2Config config)
    {
        return config.HiddenSize switch
        {
            3072 => "Klein 4B",
            4096 => "Klein 9B",
            6144 => "Dev (32B)",
            _ => $"unknown variant (hidden={config.HiddenSize})",
        };
    }

    /// <summary>Picks the right LlamaStyleEncoder preset + auto-download <see cref="SideModels.Entry"/>
    /// for the given Flux.2 variant. Klein 4B → Qwen3-4B (only verified path);
    /// Klein 9B → Qwen3-8B (will refuse at runtime if file is FP4 quantized — HartsyInference
    /// doesn't have FP4 GEMM yet); Dev → Mistral-Small-3 (same FP4 caveat).</summary>
    private static (LlamaStyleEncoderConfig encoder, SideModels.Entry sideModel, string label)
        ResolveTextEncoderForVariant(Flux2Config config)
    {
        return config.HiddenSize switch
        {
            3072 => (LlamaStyleEncoderConfig.Qwen3_4B,      SideModels.Qwen3_4B,           "Qwen3-4B encoder"),
            4096 => (LlamaStyleEncoderConfig.Qwen3_8B,      SideModels.Qwen3_8B_Fp4Mixed,  "Qwen3-8B encoder (Klein 9B)"),
            6144 => (LlamaStyleEncoderConfig.MistralSmall3, SideModels.MistralSmallFlux2,  "Mistral-Small-3 encoder (Flux.2 Dev)"),
            _ => throw new NotSupportedException(
                $"Flux.2 {DescribeConfig(config)} (hidden={config.HiddenSize}) is not a recognized variant. " +
                "Expected hidden ∈ {3072, 4096, 6144} for Klein 4B / Klein 9B / Dev respectively."),
        };
    }
}

public sealed class Flux2CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required Flux2Pipeline Pipeline { get; init; }
    public required Flux2Config Flux2Config { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder Encoder { get; init; }
    public required Flux2Transformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required Tensor BnMean { get; init; }
    public required Tensor BnVar { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader QwenLoader { get; init; }
    public required SafeTensorsLoader VaeLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        Encoder?.Dispose();
        CheckpointLoader?.Dispose();
        QwenLoader?.Dispose();
        VaeLoader?.Dispose();
    }
}
