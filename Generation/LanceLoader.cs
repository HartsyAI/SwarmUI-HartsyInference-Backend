using System.IO;
using FreneticUtilities.FreneticExtensions;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Lance (ByteDance's 3B unified multimodal model; Apache-2.0). Lance ships as a FOLDER
/// checkpoint (model.safetensors or shards + llm_config.json + Qwen2 tokenizer files), which
/// SwarmUI core now surfaces as a single folder-model entry. Two variants share one architecture:
/// Lance_3B (image T2I) and Lance_3B_Video (image + video) — the video variant is detected by
/// folder name.
///
/// Required side model: the Wan2.2 48-channel video VAE (<see cref="SideModels.Wan22Vae"/> — the
/// same file the Wan2.2 loader uses; Lance's HF repo only ships the equivalent .pth).
///
/// Text conditioning: Qwen2.5 ChatML chat template via <see cref="Qwen2Tokenizer"/>, preferring the
/// vocab.json/merges.txt shipped inside the checkpoint folder. NOTE: the engine pipelines are
/// first-run-validation pending (no real-checkpoint numerics yet) — generation works end-to-end but
/// output fidelity is unverified until the engine's validation pass lands.
/// </summary>
public static class LanceLoader
{
    public const string LanceCompatClassId = "lance";
    public const string LanceVideoCompatClassId = "lance-video";
    public const string LanceT2IClassId = "lance-t2i";
    public const string LanceT2VClassId = "lance-t2v";

    /// <summary>Total downscale between pixels and transformer tokens (VAE 16x then patchify 2x).</summary>
    private const int SizeMultiple = 32;

    /// <summary>Registers the Lance compat + model classes (core has none). Call once at extension
    /// init, before model folders are scanned.</summary>
    public static void RegisterModelClass()
    {
        T2IModelCompatClass compatImage = T2IModelClassSorter.RegisterCompat(new() { ID = LanceCompatClassId, ShortCode = "Lance", LorasTargetTextEnc = false });
        T2IModelCompatClass compatVideo = T2IModelClassSorter.RegisterCompat(new() { ID = LanceVideoCompatClassId, ShortCode = "LanceV", LorasTargetTextEnc = false, IsText2Video = true });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = LanceT2IClassId,
            CompatClass = compatImage,
            Name = "Lance 3B (image)",
            StandardWidth = 768,
            StandardHeight = 768,
            IsThisModelOfClass = (model, header) => IsLanceFolder(model, header, video: false),
        });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = LanceT2VClassId,
            CompatClass = compatVideo,
            Name = "Lance 3B (video)",
            StandardWidth = 832,
            StandardHeight = 480,
            IsThisModelOfClass = (model, header) => IsLanceFolder(model, header, video: true),
        });
    }

    /// <summary>A Lance checkpoint is a folder with llm_config.json declaring the Qwen2.5-VL backbone
    /// and language_model.* transformer keys in its (first-shard) header. The video variant is told
    /// apart by folder name — both variants ship byte-identical configs.</summary>
    private static bool IsLanceFolder(T2IModel model, Newtonsoft.Json.Linq.JObject header, bool video)
    {
        if (model?.RawFilePath is null || !Directory.Exists(model.RawFilePath))
        {
            return false;
        }
        string llmConfig = $"{model.RawFilePath}/llm_config.json";
        if (!File.Exists(llmConfig) || !File.ReadAllText(llmConfig).Contains("Qwen2_5_VL"))
        {
            return false;
        }
        if (header is null || !header.Properties().Any(p => p.Name.StartsWith("language_model.")))
        {
            return false;
        }
        bool isVideoVariant = model.RawFilePath.Replace('\\', '/').AfterLast('/').ToLowerInvariant().Contains("video");
        return video == isVideoVariant;
    }

    public static LanceCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Lance model has no folder path.");
        if (!Directory.Exists(model.RawFilePath))
            throw new DirectoryNotFoundException($"Lance checkpoint folder not found: {model.RawFilePath}");

        bool isVideo = model.ModelClass?.ID == LanceT2VClassId;

        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: SideModels.Wan22Vae,
            log: log);

        // ── 1. Transformer (sharded-folder aware converter) ──
        log($"Loading Lance backbone: {model.Name}");
        var (conv, loaders) = LanceCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw new InvalidOperationException($"Lance checkpoint '{model.Name}' has no language_model transformer weights.");
        }
        log($"  Converted: {conv.Transformer.Count} transformer keys ({conv.Vit.Count} ViT keys ignored — understanding path unused)");
        LanceConfig config = isVideo ? LanceConfig.Video : LanceConfig.Image;
        LanceTransformer transformer = new LanceTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        // ── 2. Wan2.2 VAE (decoder; computes in F32) ──
        log($"Loading Wan2.2 VAE: {vaeModel.Name}");
        var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
        Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
        Wan22VaeDecoder vae = new Wan22VaeDecoder();
        vae.LoadWeights(vaeWeights);

        // ── 3. Qwen2 chat tokenizer — prefer the checkpoint's own vocab/merges ──
        string vocabPath = $"{model.RawFilePath}/vocab.json";
        string mergesPath = $"{model.RawFilePath}/merges.txt";
        Qwen2Tokenizer tokenizer;
        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            log("Loading Qwen2 tokenizer from checkpoint folder...");
            tokenizer = new Qwen2Tokenizer(vocabPath, mergesPath);
        }
        else
        {
            log("Checkpoint folder has no vocab.json/merges.txt — using embedded Qwen2 tokenizer.");
            tokenizer = new Qwen2Tokenizer();
        }

        log("Building Lance pipeline...");
        LanceImagePipeline imagePipeline = new LanceImagePipeline(backend, transformer, vae, config);
        LanceVideoPipeline videoPipeline = isVideo ? new LanceVideoPipeline(backend, transformer, vae, config) : null;

        log($"Lance ready ({(isVideo ? "text-to-image + text-to-video" : "text-to-image")}). " +
            "Note: engine numerics are first-run-validation pending.");
        return new LanceCacheEntry
        {
            ModelName = model.Name,
            CompatClass = isVideo ? LanceVideoCompatClassId : LanceCompatClassId,
            ImagePipeline = imagePipeline,
            VideoPipeline = videoPipeline,
            Config = config,
            Transformer = transformer,
            Vae = vae,
            Tokenizer = tokenizer,
            CheckpointLoaders = loaders,
            VaeLoaders = vaeLoaders,
        };
    }

    public static Image[] Generate(
        LanceCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumTimesteps);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.CfgTextScale : (float)cfgRaw;

        bool wantVideo = entry.VideoPipeline is not null;
        int width, height;
        if (wantVideo)
        {
            (width, height) = VideoParamResolver.ResolveResolution(input, multiple: SizeMultiple);
        }
        else
        {
            width = input.Get(T2IParamTypes.Width);
            height = input.Get(T2IParamTypes.Height);
            if (width % SizeMultiple != 0 || height % SizeMultiple != 0)
            {
                int fixedW = Math.Max(SizeMultiple, width / SizeMultiple * SizeMultiple);
                int fixedH = Math.Max(SizeMultiple, height / SizeMultiple * SizeMultiple);
                throw new InvalidOperationException(
                    $"Lance requires width/height divisible by {SizeMultiple}. Got {width}x{height}; nearest valid is {fixedW}x{fixedH}.");
            }
        }

        int[] promptTokens = entry.Tokenizer.EncodeChat(prompt);
        int[] negativeTokens = entry.Tokenizer.EncodeChat(negative ?? "");

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

        if (wantVideo)
        {
            // 4x temporal VAE compression → frame counts of 4k+1; Lance tops out at 121 (~5s @ 24fps).
            int numFrames = Math.Min(121, VideoParamResolver.ResolveFrames(input, modelDefault: 81, step: 4));
            var (frames, outW, outH, _) = entry.VideoPipeline.GenerateFromTokens(
                promptTokens, negativeTokens, request, numFrames, bridge);
            Logs.Verbose($"[HartsyInference][Lance] T2V returned {frames.Length} frames {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return [VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel)];
        }

        var (rgbData, imgW, imgH, _) = entry.ImagePipeline.GenerateFromTokens(
            promptTokens, negativeTokens, request, bridge);
        Logs.Verbose($"[HartsyInference][Lance] T2I returned {imgW}x{imgH} in {Environment.TickCount64 - start}ms.");
        return [RgbToImage.FromHwcRgb(rgbData, imgW, imgH)];
    }
}

public sealed class LanceCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required LanceImagePipeline ImagePipeline { get; init; }
    public required LanceVideoPipeline VideoPipeline { get; init; }
    public required LanceConfig Config { get; init; }
    public required LanceTransformer Transformer { get; init; }
    public required Wan22VaeDecoder Vae { get; init; }
    public required Qwen2Tokenizer Tokenizer { get; init; }
    public required IReadOnlyList<SafeTensorsLoader> CheckpointLoaders { get; init; }
    public required IReadOnlyList<SafeTensorsLoader> VaeLoaders { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (ImagePipeline as IDisposable)?.Dispose();
        (VideoPipeline as IDisposable)?.Dispose();
        Transformer?.Dispose();
        Tokenizer?.Dispose();
        if (CheckpointLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in CheckpointLoaders) loader?.Dispose();
        }
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
