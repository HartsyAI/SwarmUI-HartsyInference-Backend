using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Loads Stable Diffusion 3 / 3.5 (Medium, 3.5 Medium, 3.5 Large). Three text encoders:
/// CLIP-L + CLIP-G + (optional) T5-XXL. T5 can be skipped for ~10GB VRAM savings.
///
/// Mirrors Comfy's SD3 path which reads <c>ClipLModel</c> / <c>ClipGModel</c> /
/// <c>T5XXLModel</c> / <c>VAE</c> from Swarm's parameters when split-file checkpoints
/// are configured, and falls back to single-file all-in-one extraction via
/// <see cref="Sd3CheckpointConverter"/> otherwise (matches Stability/civitai distributions).
/// </summary>
public static class Sd3Loader
{
    public const string Sd3MediumCompatClassId = "stable-diffusion-v3-medium";
    public const string Sd35MediumCompatClassId = "stable-diffusion-v3.5-medium";
    public const string Sd35LargeCompatClassId = "stable-diffusion-v3.5-large";

    public static bool IsSd3Compat(string compatClass) =>
        compatClass == Sd3MediumCompatClassId
        || compatClass == Sd35MediumCompatClassId
        || compatClass == Sd35LargeCompatClassId;

    public static Sd3CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("SD3 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"SD3 checkpoint not found: {model.RawFilePath}");

        T2IModel clipLModel = input?.Get(T2IParamTypes.ClipLModel);
        T2IModel clipGModel = input?.Get(T2IParamTypes.ClipGModel);
        T2IModel t5Model = input?.Get(T2IParamTypes.T5XXLModel);
        T2IModel vaeModel = input?.Get(T2IParamTypes.VAE);

        bool splitMode = clipLModel is not null && clipGModel is not null && vaeModel is not null;

        Dictionary<string, Tensor> transformerWeights;
        Dictionary<string, Tensor> clipLWeights;
        Dictionary<string, Tensor> clipGWeights;
        Dictionary<string, Tensor> t5Weights = null;
        Dictionary<string, Tensor> vaeWeights;
        SafeTensorsLoader mainLoader;
        SafeTensorsLoader clipLLoader = null, clipGLoader = null, t5Loader = null, vaeLoader = null;

        log($"Loading SD3 checkpoint: {model.Name} ({(splitMode ? "split-file" : "all-in-one")} mode)");
        var (converted, mLoader) = Sd3CheckpointConverter.LoadAndConvert(model.RawFilePath);
        mainLoader = mLoader;

        if (converted.Transformer.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException("SD3 checkpoint has no transformer weights.");
        }
        transformerWeights = converted.Transformer;

        if (splitMode)
        {
            log($"  CLIP-L: {clipLModel.Name}");
            clipLLoader = new SafeTensorsLoader();
            clipLLoader.Load(clipLModel.RawFilePath);
            clipLWeights = StripStandalonePrefix(clipLLoader.GetAllTensors(), "text_encoders.clip_l.transformer.");

            log($"  CLIP-G: {clipGModel.Name}");
            clipGLoader = new SafeTensorsLoader();
            clipGLoader.Load(clipGModel.RawFilePath);
            clipGWeights = StripStandalonePrefix(clipGLoader.GetAllTensors(), "text_encoders.clip_g.transformer.");

            if (t5Model is not null)
            {
                log($"  T5-XXL: {t5Model.Name}");
                t5Loader = new SafeTensorsLoader();
                t5Loader.Load(t5Model.RawFilePath);
                t5Weights = StripStandalonePrefix(t5Loader.GetAllTensors(), "text_encoders.t5xxl.transformer.");
            }
            else
            {
                log("  T5-XXL: not configured (running in 2-encoder VRAM-saver mode).");
            }

            log($"  VAE: {vaeModel.Name}");
            vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaeModel.RawFilePath);
            vaeWeights = LoadVaeFromStandalone(vaeLoader.GetAllTensors());
        }
        else
        {
            clipLWeights = converted.ClipL;
            clipGWeights = converted.ClipG;
            t5Weights = converted.T5.Count > 0 ? converted.T5 : null;
            vaeWeights = converted.Vae;

            if (clipLWeights.Count == 0 || clipGWeights.Count == 0 || vaeWeights.Count == 0)
            {
                mainLoader.Dispose();
                throw new InvalidOperationException(
                    "SD3 all-in-one mode: this checkpoint is missing CLIP-L/CLIP-G/VAE components. " +
                    "Either pick a complete SD3 file, or configure CLIP-L Model + CLIP-G Model + " +
                    "T5-XXL Model (optional) + VAE parameters in Swarm to load from separate files.");
            }
            if (t5Weights is null)
            {
                log("  T5-XXL not bundled in this all-in-one checkpoint (running in 2-encoder VRAM-saver mode).");
            }
        }

        // Auto-detect arch from the transformer's patch_embed shape.
        int patchEmbedOutChannels = DetectPatchEmbedOutChannels(transformerWeights);
        Sd3Config sd3Config = Sd3Config.FromWeightShape(patchEmbedOutChannels);
        log($"Architecture: patch_embed out_channels={patchEmbedOutChannels} → SD3 config selected");

        log("Building MMDiT transformer...");
        Sd3Transformer transformer = new Sd3Transformer(sd3Config);
        transformer.LoadWeights(transformerWeights);

        log("Building CLIP-L encoder...");
        ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        clipL.LoadWeights(clipLWeights, prefix: "text_model");

        log("Building CLIP-G encoder...");
        ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
        clipG.LoadWeights(clipGWeights, prefix: "text_model");

        T5TextEncoder t5 = null;
        if (t5Weights is not null && t5Weights.Count > 0)
        {
            log("Building T5-XXL encoder...");
            t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);
        }

        log("Building VAE decoder (SD3 config)...");
        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd3);
        vaeDecoder.LoadWeights(vaeWeights);

        log("Building VAE encoder (SD3 config)...");
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Sd3);
        vaeEncoder.LoadWeights(vaeWeights);

        log("Building pipeline...");
        Sd3Pipeline pipeline = new Sd3Pipeline(backend, clipL, clipG, t5, transformer, vaeDecoder, vaeEncoder);

        log("Loading tokenizers (embedded)...");
        ClipTokenizer clipTokenizer = new ClipTokenizer();

        T5Tokenizer t5Tokenizer = null;
        if (t5 is not null)
        {
            t5Tokenizer = new T5Tokenizer(maxLength: 256);
        }

        log($"SD3 ready (T5={t5 is not null}).");
        return new Sd3CacheEntry
        {
            ModelName = model.Name,
            CompatClass = model.ModelClass?.CompatClass?.ID ?? Sd3MediumCompatClassId,
            Pipeline = pipeline,
            Sd3Config = sd3Config,
            ClipTokenizer = clipTokenizer,
            T5Tokenizer = t5Tokenizer,
            ClipL = clipL,
            ClipG = clipG,
            T5 = t5,
            Transformer = transformer,
            Vae = vaeDecoder,
            VaeEncoder = vaeEncoder,
            CheckpointLoader = mainLoader,
            ClipLLoader = clipLLoader,
            ClipGLoader = clipGLoader,
            T5Loader = t5Loader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        Sd3CacheEntry entry,
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

        // Both CLIPs share the same BPE tokenizer; encode once and reuse.
        int[] promptTokensCLIP = entry.ClipTokenizer.Encode(prompt);
        int[] negTokensCLIP = entry.ClipTokenizer.Encode(negative);
        int promptEosL = ClipTokenizer.FindEosPosition(promptTokensCLIP);
        int negEosL = ClipTokenizer.FindEosPosition(negTokensCLIP);
        int promptEosG = promptEosL;
        int negEosG = negEosL;

        int[] promptTokensT5 = null, negTokensT5 = null, promptMaskT5 = null, negMaskT5 = null;
        if (entry.T5 is not null && entry.T5Tokenizer is not null)
        {
            promptTokensT5 = entry.T5Tokenizer.Encode(prompt);
            negTokensT5 = entry.T5Tokenizer.Encode(negative);
            promptMaskT5 = T5Tokenizer.CreateAttentionMask(promptTokensT5);
            negMaskT5 = T5Tokenizer.CreateAttentionMask(negTokensT5);
        }

        Img2ImgResolver.Img2ImgSpec img2img = Img2ImgResolver.Resolve(input, width, height);
        TextToImageRequest request;
        if (img2img is not null)
        {
            request = new ImageToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfgRaw <= 0 ? 4.5f : (float)cfgRaw,
                Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
                SourceImage = img2img.SourceTensor,
                Strength = img2img.Strength,
                Mask = img2img.MaskTensor,
            };
        }
        else
        {
            request = new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfgRaw <= 0 ? 4.5f : (float)cfgRaw,
                Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
            };
        }

        try
        {
            long start = Environment.TickCount64;
            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                onProgress(p);
            };

            var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(
                promptTokensCLIP, negTokensCLIP,
                promptTokensCLIP, negTokensCLIP,
                promptEosL, negEosL,
                promptEosG, negEosG,
                promptTokensT5, negTokensT5,
                promptMaskT5, negMaskT5,
                request, bridge);

            Logs.Verbose($"[SharpInference][SD3] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
        }
        finally
        {
            img2img?.Dispose();
        }
    }

    /// <summary>SD3 has 16-channel latent, MMDiT patch_embed.weight shape is [embedDim, 16, 2, 2].
    /// SD3 Medium uses embedDim=1536, SD3.5 Medium=1536, SD3.5 Large=2432.</summary>
    private static int DetectPatchEmbedOutChannels(Dictionary<string, Tensor> transformer)
    {
        if (transformer.TryGetValue("pos_embed.proj.weight", out var t) && t.Shape.Rank >= 1)
            return (int)t.Shape[0];
        // diffusers naming
        if (transformer.TryGetValue("patch_embed.proj.weight", out var t2) && t2.Shape.Rank >= 1)
            return (int)t2.Shape[0];
        // Fallback: assume Medium.
        return 1536;
    }

    private static Dictionary<string, Tensor> StripStandalonePrefix(Dictionary<string, Tensor> raw, string comfyPrefix)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var kv in raw)
        {
            if (kv.Key.StartsWith(comfyPrefix, StringComparison.Ordinal))
            {
                string rest = kv.Key[comfyPrefix.Length..];
                if (!rest.EndsWith("position_ids", StringComparison.Ordinal))
                    result[rest] = kv.Value;
            }
            else
            {
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }

    private static Dictionary<string, Tensor> LoadVaeFromStandalone(Dictionary<string, Tensor> raw)
    {
        bool hasComfyPrefix = false;
        foreach (var k in raw.Keys)
        {
            if (k.StartsWith("vae.", StringComparison.Ordinal) || k.StartsWith("first_stage_model.", StringComparison.Ordinal))
            {
                hasComfyPrefix = true;
                break;
            }
        }
        if (hasComfyPrefix)
        {
            // Sd3 converter would handle this, but for a pure-VAE file just strip the prefix.
            Dictionary<string, Tensor> result = new(raw.Count);
            foreach (var kv in raw)
            {
                string k = kv.Key;
                if (k.StartsWith("vae.", StringComparison.Ordinal)) k = k["vae.".Length..];
                else if (k.StartsWith("first_stage_model.", StringComparison.Ordinal)) k = k["first_stage_model.".Length..];
                result[k] = kv.Value;
            }
            return result;
        }
        return new Dictionary<string, Tensor>(raw);
    }
}

public sealed class Sd3CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required Sd3Pipeline Pipeline { get; init; }
    public required Sd3Config Sd3Config { get; init; }
    public required ClipTokenizer ClipTokenizer { get; init; }
    public T5Tokenizer T5Tokenizer { get; init; }
    public required ClipTextEncoder ClipL { get; init; }
    public required ClipTextEncoder ClipG { get; init; }
    public T5TextEncoder T5 { get; init; }
    public required Sd3Transformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required VaeEncoder VaeEncoder { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public SafeTensorsLoader ClipLLoader { get; init; }
    public SafeTensorsLoader ClipGLoader { get; init; }
    public SafeTensorsLoader T5Loader { get; init; }
    public SafeTensorsLoader VaeLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        ClipTokenizer?.Dispose();
        T5Tokenizer?.Dispose();
        CheckpointLoader?.Dispose();
        ClipLLoader?.Dispose();
        ClipGLoader?.Dispose();
        T5Loader?.Dispose();
        VaeLoader?.Dispose();
    }
}
