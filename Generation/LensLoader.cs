using System.IO;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Core.Backends;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Tokenizers;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Loads Microsoft Lens (3.8B dual-stream MMDiT; SwarmUI compat class <c>lens</c>) from the
/// ComfyUI split files (<c>Comfy-Org/Lens</c>): the user's main model is the DiT
/// (<c>lens_bf16/lens_mxfp8/lens_turbo_*</c> — MXFP8 quants are dequanted by the engine converter),
/// and the GPT-OSS-20B text encoder (NVFP4) + Flux.2 VAE auto-download as side models.
///
/// Variant handling: Lens, Lens-Turbo, and Lens-Base share one architecture — a "turbo" in the
/// checkpoint filename selects <see cref="LensConfig.Turbo"/> (4 steps, no CFG); everything else
/// uses <see cref="LensConfig.Default"/> (20 steps, CFG 5). The GPT-OSS tokenizer's vocab/merges
/// export from <c>microsoft/Lens/tokenizer/tokenizer.json</c> on first use; prompts are rendered
/// through Lens' Harmony chat template (<see cref="GptOssTokenizer.BuildChatInputs"/>).
/// </summary>
public static class LensLoader
{
    public const string LensCompatClassId = "lens";

    private const string TokenizerJsonUrl = "https://huggingface.co/microsoft/Lens/resolve/main/tokenizer/tokenizer.json";

    public static LensCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Lens model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Lens checkpoint not found: {model.RawFilePath}");

        T2IModel textEncoderModel = ModelAutoDownloader.EnsureSideModel(
            userPick: null, entry: SideModels.LensGptOss20b, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.Flux2Vae, log: log);
        string tokenizerDir = Path.Combine(Program.T2IModelSets["Clip"].DownloadFolderPath, "Lens");
        (string vocabPath, string mergesPath) = TokenizerExport.EnsureVocabMerges(
            TokenizerJsonUrl, tokenizerDir, "gpt_oss", log);

        LensConfig config = model.RawFilePath.Contains("turbo", StringComparison.OrdinalIgnoreCase)
            ? LensConfig.Turbo
            : LensConfig.Default;
        log($"Loading Lens ({(config.DefaultCfgScale > 1 ? "standard" : "turbo")}): {model.Name}");
        log($"  Text encoder: {textEncoderModel.Name}, VAE: {vaeModel.Name}");

        LensPipelineBundle bundle = LensPipelineFactory.LoadFromComfyFiles(
            backend, model.RawFilePath, textEncoderModel.RawFilePath, vaeModel.RawFilePath, config);

        log("Loading GPT-OSS tokenizer...");
        GptOssTokenizer tokenizer = new GptOssTokenizer(vocabPath, mergesPath);

        log("Lens ready.");
        return new LensCacheEntry
        {
            ModelName = model.Name,
            CompatClass = LensCompatClassId,
            Bundle = bundle,
            Config = config,
            Tokenizer = tokenizer,
        };
    }

    public static Image[] Generate(
        LensCacheEntry entry,
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

        (int[] posTokens, _) = entry.Tokenizer.BuildChatInputs(prompt);
        // Negative tokens only matter when CFG is live; Lens-Turbo (cfg 1) skips the second pass.
        int[] negTokens = cfgScale > 1f ? entry.Tokenizer.BuildChatInputs(negative).tokenIds : null;

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

        var (rgbBytes, outW, outH, _) = entry.Bundle.Pipeline.GenerateFromTokens(
            posTokens, negTokens, request, bridge);

        Logs.Verbose($"[SharpInference][Lens] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }
}

public sealed class LensCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }

    /// <summary>Owns the pipeline + all component weights/loaders (factory-built).</summary>
    public required LensPipelineBundle Bundle { get; init; }
    public required LensConfig Config { get; init; }
    public required GptOssTokenizer Tokenizer { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Bundle?.Dispose();
        Tokenizer?.Dispose();
    }
}
