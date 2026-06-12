using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Audio.Models.Codecs.Oobleck;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Music;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Loads ACE-Step v1.5 (2B turbo flow-matching music DiT over 25 Hz Oobleck latents). This is the
/// architecture Swarm core's <c>ace-step-1_5</c> compat class natively detects — checkpoints that do
/// NOT match the extension's v1 model class route here (previously they were refused).
///
/// Required side models (auto-downloaded; user picks take priority):
///   - <see cref="SideModels.AceStep15Vae"/>: the Oobleck audio VAE (Comfy-Org repackage; same file
///     as Swarm core's "ace-step-15-vae" CommonModels entry).
///   - <see cref="SideModels.Qwen3Embedding06B"/>: Qwen3-Embedding-0.6B — encodes BOTH the style
///     prompt and the lyrics to the 1024-d states the v1.5 condition encoder consumes.
///
/// Turbo defaults: fixed 8-step shift-3.0 Euler, no CFG (the CFG slider is ignored). Engine
/// numerics are first-run-validation pending. Reference-audio timbre + LM code hints are engine
/// Phase-2 hooks, not exposed here yet.
/// </summary>
public static class AceStep15Loader
{
    /// <summary>Qwen3-Embedding EOS/pad id (<c>&lt;|endoftext|&gt;</c>).</summary>
    private const int QwenEosId = 151643;

    public static AceStep15CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("ACE-Step 1.5 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"ACE-Step 1.5 checkpoint not found: {model.RawFilePath}");

        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.AceStep15Vae, log: log);
        T2IModel qwenModel = ModelAutoDownloader.EnsureSideModel(
            userPick: null, entry: SideModels.Qwen3Embedding06B, log: log);

        // ── 1. Main model (677 keys: decoder DiT + condition encoder share one dict) ──
        log($"Loading ACE-Step 1.5 model: {model.Name}");
        var (weights, mainLoader) = AceStepCheckpointConverter.LoadModel15(model.RawFilePath, castToF32: true);
        if (weights.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException($"ACE-Step 1.5 checkpoint '{model.Name}' has no usable tensors.");
        }
        AceStep15Config config = new AceStep15Config();
        AceStep15Dit dit = new AceStep15Dit(config);
        dit.LoadWeights(weights);
        AceStep15ConditionEncoder conditionEncoder = new AceStep15ConditionEncoder(config);
        conditionEncoder.LoadWeights(weights);

        // ── 2. Oobleck VAE (48 kHz stereo ↔ 64-ch 25 Hz latents; fuses its own weight norm) ──
        log($"Loading Oobleck VAE: {vaeModel.Name}");
        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaeModel.RawFilePath);
        OobleckVae vae = new OobleckVae(OobleckConfig.AceStep15);
        vae.LoadWeights(vaeLoader.GetAllTensors());

        // ── 3. Qwen3-Embedding-0.6B (style + lyric conditioning, 1024-d states) ──
        log($"Loading Qwen3-Embedding-0.6B: {qwenModel.Name}");
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenModel.RawFilePath);
        LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_Embedding_0_6B);
        qwen.LoadWeights(qwenLoader.GetAllTensors());
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();

        log("Building ACE-Step 1.5 pipeline...");
        AceStepPipeline15 pipeline = new AceStepPipeline15(backend, dit, conditionEncoder, vae, config);

        log("ACE-Step 1.5 ready (text/lyrics-to-music, turbo 8-step). Numerics first-run-validation pending.");
        return new AceStep15CacheEntry
        {
            ModelName = model.Name,
            CompatClass = AceStepLoader.AceStepCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Dit = dit,
            ConditionEncoder = conditionEncoder,
            Vae = vae,
            Qwen = qwen,
            Tokenizer = tokenizer,
            CheckpointLoader = mainLoader,
            VaeLoader = vaeLoader,
            QwenLoader = qwenLoader,
        };
    }

    public static Image[] Generate(
        AceStep15CacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string lyrics = input.Get(T2IParamTypes.Prompt) ?? "";
        string style = ComposeStyleText(input);
        double duration = Math.Clamp(input.Get(T2IParamTypes.Text2AudioDuration, 120d), 1d, 600d);
        long seedLong = input.Get(T2IParamTypes.Seed);

        // Style prompt + lyrics both go through Qwen3-Embedding-0.6B → [T, 1024] states; drop the
        // encoder's GPU weights before the 2B DiT preload (mirrors the other audio/video loaders).
        Tensor textHidden = EncodeQwen(entry, backend, style);
        Tensor lyricHidden = string.IsNullOrWhiteSpace(lyrics) ? null : EncodeQwen(entry, backend, lyrics);
        backend.Sync();
        backend.FreeWeights(entry.Qwen.EnumerateWeights());

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        try
        {
            var (left, right, sampleRate, _) = entry.Pipeline.Generate(
                textHidden, lyricHidden, duration,
                seed: seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
                onProgress: bridge);
            Logs.Verbose($"[SharpInference][ACE15] {left.Length} samples/channel @ {sampleRate} Hz " +
                $"({duration:0}s requested) in {Environment.TickCount64 - start}ms.");
            return [AudioOutputEncoder.EncodeMp3(left, right, sampleRate, cancel)];
        }
        finally
        {
            textHidden.Dispose();
            lyricHidden?.Dispose();
        }
    }

    /// <summary>Encodes text with Qwen3-Embedding-0.6B (raw BPE + EOS, no chat template — embedding
    /// models consume plain text) and slices the batch row to the <c>[T, 1024]</c> layout the v1.5
    /// condition encoder expects.</summary>
    private static Tensor EncodeQwen(AceStep15CacheEntry entry, IBackend backend, string text)
    {
        IReadOnlyList<int> raw = entry.Tokenizer.EncodeRaw(text);
        int[] tokens = new int[raw.Count + 1];
        for (int i = 0; i < raw.Count; i++)
        {
            tokens[i] = raw[i];
        }
        tokens[^1] = QwenEosId;
        Tensor batch = entry.Qwen.Encode(backend, [tokens]);
        Tensor sliced = CfgHelper.SliceBatchElement(batch, 0, tokens.Length, entry.Config.TextInDim);
        batch.Dispose();
        return sliced;
    }

    /// <summary>Builds the style prompt from the audio params — same free-text convention as v1.</summary>
    private static string ComposeStyleText(T2IParamInput input)
    {
        string style = input.Get(T2IParamTypes.Text2AudioStyle, "pop");
        long bpm = input.Get(T2IParamTypes.Text2AudioBPM, 120);
        string keyScale = input.Get(T2IParamTypes.Text2AudioKeyScale, "C");
        string timeSig = input.Get(T2IParamTypes.Text2AudioTimeSignature, "4");
        return $"{style}, {bpm} bpm, key of {keyScale}, {timeSig}/4 time";
    }
}

public sealed class AceStep15CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required AceStepPipeline15 Pipeline { get; init; }
    public required AceStep15Config Config { get; init; }
    public required AceStep15Dit Dit { get; init; }
    public required AceStep15ConditionEncoder ConditionEncoder { get; init; }
    public required OobleckVae Vae { get; init; }
    public required LlamaStyleEncoder Qwen { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader VaeLoader { get; init; }
    public required SafeTensorsLoader QwenLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Qwen?.Dispose();
        Tokenizer?.Dispose();
        CheckpointLoader?.Dispose();
        VaeLoader?.Dispose();
        QwenLoader?.Dispose();
    }
}
