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
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads F-Lite (Freepik's lightweight Flux Schnell-derivative). F-Lite ships ONLY in the
/// diffusers folder layout — there is no single-file safetensors release. Expected directory
/// structure:
/// <code>
///   F-Lite/
///   ├── dit_model/        ← MMDiT transformer shards
///   ├── text_encoder/     ← T5-XXL safetensors shards
///   └── vae/              ← Flux Schnell VAE in diffusers format
/// </code>
///
/// The user picks any <c>.safetensors</c> inside one of those subfolders in SwarmUI's model
/// picker; the loader walks up two levels to find the F-Lite root. F-Lite has no end-to-end
/// test in HartsyInference yet — first generation through this loader is the test.
/// </summary>
public static class FLiteLoader
{
    public const string FLiteCompatClassId = "f-lite";

    public static FLiteCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("F-Lite model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"F-Lite checkpoint not found: {model.RawFilePath}");

        string folderPath = ResolveFolderRoot(model.RawFilePath, "dit_model", "text_encoder", "vae");
        log($"Loading F-Lite folder: {Path.GetFileName(folderPath)} (from picked file {Path.GetFileName(model.RawFilePath)})");

        var (converted, handle) = FLiteCheckpointConverter.LoadAndConvert(folderPath);
        log($"  Converted: {converted.Transformer.Count} dit / {converted.T5.Count} T5 / {converted.Vae.Count} VAE keys");

        if (converted.Transformer.Count == 0 || converted.T5.Count == 0 || converted.Vae.Count == 0)
        {
            handle.Dispose();
            throw new InvalidOperationException(
                $"F-Lite folder '{folderPath}' is missing dit_model / text_encoder / vae components. " +
                "Ensure the folder has all three subdirectories with .safetensors files.");
        }

        FLiteConfig config = FLiteConfig.V1;
        log($"Architecture: F-Lite V1 (hidden={config.HiddenSize}, depth={config.Depth})");

        log("Building FLiteTransformer...");
        FLiteTransformer transformer = new FLiteTransformer(config);
        transformer.LoadWeights(converted.Transformer);

        log("Building T5-XXL encoder...");
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(converted.T5);

        log("Building VAE decoder (Flux Schnell config)...");
        VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
        vae.LoadWeights(converted.Vae);

        log("Loading T5 tokenizer (embedded)...");
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 512);

        log("Building F-Lite pipeline...");
        FLitePipeline pipeline = new FLitePipeline(backend, t5, transformer, vae, config);

        log("F-Lite ready.");
        return new FLiteCacheEntry
        {
            ModelName = model.Name,
            CompatClass = FLiteCompatClassId,
            Pipeline = pipeline,
            FLiteConfig = config,
            Tokenizer = tokenizer,
            T5 = t5,
            Transformer = transformer,
            Vae = vae,
            LoaderHandle = handle,
        };
    }

    public static Image[] Generate(
        FLiteCacheEntry entry,
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
        float cfgScale = cfgRaw <= 0 ? 4.5f : (float)cfgRaw;

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
            promptTokens, promptMask, negTokens, negMask, request, bridge);

        Logs.Verbose($"[HartsyInference][F-Lite] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }

    /// <summary>Walks up from a single-file model path until it finds a directory containing
    /// at least one of the expected subfolders. Errors with a clear message if none of them
    /// are found within 3 parent levels.</summary>
    internal static string ResolveFolderRoot(string filePath, params string[] expectedSubfolders)
    {
        string dir = Path.GetDirectoryName(filePath);
        for (int depth = 0; depth < 4 && dir is not null; depth++)
        {
            foreach (string sub in expectedSubfolders)
            {
                if (Directory.Exists(Path.Combine(dir, sub))) return dir;
            }
            dir = Path.GetDirectoryName(dir);
        }
        throw new InvalidOperationException(
            $"Could not find any of the expected subfolders [{string.Join(", ", expectedSubfolders)}] " +
            $"by walking up from '{filePath}'. " +
            "This architecture requires a diffusers-style folder layout — pick any .safetensors " +
            "inside the model's folder structure.");
    }
}

public sealed class FLiteCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required FLitePipeline Pipeline { get; init; }
    public required FLiteConfig FLiteConfig { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required FLiteTransformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required FLiteCheckpointConverter.LoaderHandle LoaderHandle { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        LoaderHandle?.Dispose();
    }
}
