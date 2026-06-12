using System.IO;
using FreneticUtilities.FreneticExtensions;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Audio.Models.Codecs;
using SharpInference.Audio.Models.Music;
using SharpInference.Audio.Pipelines;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.Tokenizers;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Loads YuE (m-a-p's full-song lyrics-to-music model; HF <c>m-a-p/YuE-s1-7B-anneal-*</c>).
/// YuE checkpoints are FOLDER models (sharded LLaMA-layout safetensors), surfaced by the core
/// folder-model scanning support; detection requires "yue" in the folder name since the weights
/// are otherwise LLaMA-shaped.
///
/// Required sibling files inside (or next to) the checkpoint folder — these ship in
/// <c>m-a-p/xcodec_mini_infer</c> and must be converted/placed by the user (no canonical
/// single-file safetensors hosting exists yet):
///   - <c>tokenizer.model</c> — the YuE "mm" SentencePiece tokenizer.
///   - <c>xcodec.safetensors</c> — the X-Codec audio codec converted from the upstream .pth.
///
/// Stage-1 only (vocal + accompaniment codebook-0 streams → X-Codec decode @ 16 kHz mono);
/// stage-2 refinement is an engine TODO. Output encoded to MP3 like ACE-Step.
/// </summary>
public static class YueLoader
{
    public const string YueCompatClassId = "yue";
    public const string YueClassId = "yue-s1";

    /// <summary>Registers the YuE compat + model class. Call once at extension init, before model
    /// folders are scanned.</summary>
    public static void RegisterModelClass()
    {
        T2IModelCompatClass compat = T2IModelClassSorter.RegisterCompat(new() { ID = YueCompatClassId, ShortCode = "YuE", IsAudioModel = true, LorasTargetTextEnc = false });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = YueClassId,
            CompatClass = compat,
            Name = "YuE Stage-1",
            IsThisModelOfClass = (model, header) =>
                model?.RawFilePath is not null
                && Directory.Exists(model.RawFilePath)
                && model.RawFilePath.Replace('\\', '/').AfterLast('/').ToLowerInvariant().Contains("yue")
                && header is not null
                && header.ContainsKey("model.layers.0.self_attn.q_proj.weight"),
        });
    }

    public static YueCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("YuE model has no folder path.");
        if (!Directory.Exists(model.RawFilePath))
            throw new DirectoryNotFoundException($"YuE checkpoint folder not found: {model.RawFilePath}");

        string tokenizerPath = FindSibling(model.RawFilePath, "tokenizer.model")
            ?? throw new InvalidOperationException(
                $"YuE needs the mm tokenizer: place 'tokenizer.model' (from m-a-p/xcodec_mini_infer, mm_tokenizer_v0.2_hf/) "
                + $"inside the checkpoint folder '{model.RawFilePath}'.");
        string xcodecPath = FindSibling(model.RawFilePath, "xcodec.safetensors")
            ?? throw new InvalidOperationException(
                $"YuE needs the X-Codec weights: convert m-a-p/xcodec_mini_infer's codec checkpoint to safetensors and place "
                + $"'xcodec.safetensors' inside the checkpoint folder '{model.RawFilePath}'.");

        // ── 1. Stage-1 LM (7B LLaMA layout; sharded folder) ──
        log($"Loading YuE stage-1 LM: {model.Name}");
        var (stage1Weights, stage1Loader) = YueCheckpointConverter.LoadStage1(model.RawFilePath, castToF32: true);
        if (stage1Weights.Count == 0)
        {
            stage1Loader.Dispose();
            throw new InvalidOperationException($"YuE checkpoint '{model.Name}' has no recognized stage-1 weights.");
        }
        YueConfig config = YueConfig.V1;
        YueStage1Lm stage1 = new YueStage1Lm(config);
        stage1.LoadWeights(stage1Weights);

        // ── 2. X-Codec (16 kHz, 8 codebooks) ──
        log($"Loading X-Codec: {xcodecPath.AfterLast('/')}");
        var (codecWeights, codecLoader) = YueCheckpointConverter.LoadXCodec(xcodecPath, castToF32: true);
        XCodec xcodec = new XCodec(XCodecConfig.XCodec16kHz);
        xcodec.LoadWeights(codecWeights);

        // ── 3. mm tokenizer ──
        log("Loading YuE mm tokenizer...");
        YueTokenizer tokenizer = new YueTokenizer(tokenizerPath);

        log("Building YuE pipeline...");
        YuePipeline pipeline = new YuePipeline(config, stage1, xcodec);

        log("YuE stage-1 ready (lyrics-to-music, 16 kHz mono; stage-2 refinement is an engine TODO).");
        return new YueCacheEntry
        {
            ModelName = model.Name,
            CompatClass = YueCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Stage1 = stage1,
            Codec = xcodec,
            Tokenizer = tokenizer,
            Stage1Loader = stage1Loader,
            CodecLoader = codecLoader,
        };
    }

    public static Image[] Generate(
        YueCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string lyrics = input.Get(T2IParamTypes.Prompt) ?? "";
        string genre = input.Get(T2IParamTypes.Text2AudioStyle, "pop");
        double duration = Math.Clamp(input.Get(T2IParamTypes.Text2AudioDuration, 60d), 5d, 300d);
        long seedLong = input.Get(T2IParamTypes.Seed);
        int seed = seedLong < 0 ? Random.Shared.Next() : (int)(seedLong & 0x7FFFFFFF);
        int maxFrames = (int)(duration * entry.Config.FrameRateHz);

        int[] promptIds = entry.Tokenizer.EncodeStage1Prompt(genre, lyrics);

        long start = Environment.TickCount64;
        cancel.ThrowIfCancellationRequested();
        float[] samples = entry.Pipeline.Synthesize(backend, promptIds, maxFrames: maxFrames, seed: seed);
        cancel.ThrowIfCancellationRequested();
        Logs.Verbose($"[SharpInference][YuE] {samples.Length} samples @ {entry.Config.SampleRate} Hz " +
            $"({duration:0}s requested, {promptIds.Length} prompt tokens) in {Environment.TickCount64 - start}ms.");
        return [AudioOutputEncoder.EncodeMp3(samples, samples, entry.Config.SampleRate, cancel)];
    }

    /// <summary>Looks for a file inside the checkpoint folder, then one directory up (so several
    /// YuE variants can share one tokenizer/codec copy).</summary>
    private static string FindSibling(string folder, string fileName)
    {
        string inside = $"{folder}/{fileName}";
        if (File.Exists(inside))
        {
            return inside;
        }
        string parent = folder.Replace('\\', '/').BeforeLast('/');
        string beside = $"{parent}/{fileName}";
        return File.Exists(beside) ? beside : null;
    }
}

public sealed class YueCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required YuePipeline Pipeline { get; init; }
    public required YueConfig Config { get; init; }
    public required YueStage1Lm Stage1 { get; init; }
    public required XCodec Codec { get; init; }
    public required YueTokenizer Tokenizer { get; init; }
    public required IDisposable Stage1Loader { get; init; }
    public required IDisposable CodecLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Tokenizer?.Dispose();
        Stage1Loader?.Dispose();
        CodecLoader?.Dispose();
    }
}
