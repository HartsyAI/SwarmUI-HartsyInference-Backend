using System.IO;
using SwarmUI.Core;
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
/// Loads AuraFlow v0.2 / v0.3 (fal/AuraFlow). Single-file checkpoints bundle the
/// MMDiT transformer, the Pile-T5-XL text encoder, and the SDXL VAE under one
/// safetensors — <see cref="AuraFlowCheckpointConverter"/> splits them apart.
///
/// Pile-T5-XL's SentencePiece vocab is compatible with the standard T5-XXL one, so
/// the embedded tokenizer in <see cref="T5Tokenizer"/> works without an external file.
/// </summary>
public static class AuraFlowLoader
{
    public const string AuraFlowCompatClassId = "auraflow-v1";

    public static AuraFlowCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("AuraFlow model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"AuraFlow checkpoint not found: {model.RawFilePath}");

        log($"Loading AuraFlow checkpoint: {model.Name}");
        var (converted, mainLoader) = AuraFlowCheckpointConverter.LoadAndConvert(model.RawFilePath);
        log($"  Converted: {converted.Transformer.Count} transformer / {converted.T5.Count} T5 / {converted.Vae.Count} VAE keys");

        if (converted.Transformer.Count == 0 || converted.T5.Count == 0 || converted.Vae.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException(
                $"AuraFlow checkpoint '{model.Name}' is missing one of transformer / T5 / VAE. " +
                "AuraFlow expects a complete bundled file (the v0.3 fal-released format). " +
                "Split-file AuraFlow isn't supported — pick a complete checkpoint.");
        }

        AuraFlowConfig config = AuraFlowConfig.V03;
        log($"Architecture: AuraFlow {config.NumDoubleBlocks} double + {config.NumSingleBlocks} single (V03 preset)");

        log("Building AuraFlow transformer...");
        AuraFlowTransformer transformer = new AuraFlowTransformer(config);
        transformer.LoadWeights(converted.Transformer);

        log("Building Pile-T5-XL encoder...");
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.PileT5Xl);
        t5.LoadWeights(converted.T5);

        // AuraFlow reuses SDXL VAE â same F16-overflow problem. Cast weights to BF16
        // on Ampere+, F32 elsewhere. See VaePrecisionHelper for rationale.
        DType vaeDtype = VaePrecisionHelper.PreferredSdxlVaeDtype(backend);
        Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(converted.Vae, vaeDtype);
        log($"Building VAE decoder (SDXL config, dtype={vaeDtype})...");
        VaeDecoder vae = new VaeDecoder(VaeConfig.AuraFlow);
        vae.LoadWeights(vaeWeights);

        // AuraFlow needs the Pile-T5-XL SentencePiece — same vocab size (32128) as Google T5 v1.1 but
        // different token-ID assignments. Using the embedded Google T5 spiece produces semantically
        // wrong embeddings (model still denoises into a coherent image, just not the prompted one).
        // Resolve under <ModelRoot>/Tokenizers/T5/, fall back to a one-shot HF download.
        string spiecePath = EnsurePileT5Spiece(log);
        log($"Loading Pile-T5 tokenizer ({Path.GetFileName(spiecePath)})...");
        T5Tokenizer tokenizer = new T5Tokenizer(spiecePath, maxLength: 256);

        log("Building AuraFlow pipeline...");
        AuraFlowPipeline pipeline = new AuraFlowPipeline(backend, t5, transformer, vae, config);

        log("AuraFlow ready.");
        return new AuraFlowCacheEntry
        {
            ModelName = model.Name,
            CompatClass = AuraFlowCompatClassId,
            Pipeline = pipeline,
            AuraFlowConfig = config,
            Tokenizer = tokenizer,
            T5 = t5,
            Transformer = transformer,
            Vae = vae,
            CheckpointLoader = mainLoader,
        };
    }

    /// <summary>Pile-T5-XL SentencePiece (~488 KB). Looked up under <c>&lt;ModelRoot&gt;/Tokenizers/T5/</c>
    /// across all configured roots; fetched from the official EleutherAI HF repo on first miss.</summary>
    private const string PileT5SpieceFilename = "pile_t5xl_spiece.model";
    private const string PileT5SpieceUrl = "https://huggingface.co/EleutherAI/pile-t5-xl/resolve/main/spiece.model";
    private const string PileT5SpieceSha256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347";
    private static readonly object _spieceDownloadLock = new();

    private static string EnsurePileT5Spiece(Action<string> log)
    {
        // Search every configured model root for an existing copy under Tokenizers/T5/.
        foreach (string root in Program.ServerSettings.Paths.ActualModelRoots)
        {
            string candidate = Path.Combine(root, "Tokenizers", "T5", PileT5SpieceFilename);
            if (File.Exists(candidate)) return candidate;
        }

        // Not found — download into the first model root. Lock so concurrent loads don't race.
        string targetRoot = Program.ServerSettings.Paths.ActualModelRoots.First();
        string targetDir = Path.Combine(targetRoot, "Tokenizers", "T5");
        string targetPath = Path.Combine(targetDir, PileT5SpieceFilename);
        lock (_spieceDownloadLock)
        {
            if (File.Exists(targetPath)) return targetPath;
            Directory.CreateDirectory(targetDir);
            log($"Pile-T5 SentencePiece not found locally — downloading from {PileT5SpieceUrl}");
            string tmpPath = targetPath + ".tmp";
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
            try
            {
                Utilities.DownloadFile(PileT5SpieceUrl, tmpPath, (bytes, total, _) => { },
                    verifyHash: PileT5SpieceSha256).Wait();
                File.Move(tmpPath, targetPath);
            }
            catch
            {
                if (File.Exists(tmpPath)) try { File.Delete(tmpPath); } catch { }
                throw;
            }
        }
        return targetPath;
    }

    public static Image[] Generate(
        AuraFlowCacheEntry entry,
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
        float cfgScale = cfgRaw <= 0 ? 5.0f : (float)cfgRaw;

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

        Logs.Verbose($"[SharpInference][AuraFlow] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }
}

public sealed class AuraFlowCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required AuraFlowPipeline Pipeline { get; init; }
    public required AuraFlowConfig AuraFlowConfig { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required AuraFlowTransformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        CheckpointLoader?.Dispose();
    }
}
