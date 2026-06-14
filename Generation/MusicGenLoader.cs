using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads MusicGen (Meta's text-to-music autoregressive LM over EnCodec tokens; HF
/// <c>facebook/musicgen-{small,medium,large}</c>). The HF checkpoint is self-contained: one
/// safetensors bundling the T5-base text encoder (<c>text_encoder.*</c>), the 32 kHz EnCodec
/// (<c>audio_encoder.*</c>), and the decoder LM — so no side-model downloads are needed.
///
/// Size preset (Small/Medium/Large) is inferred from the decoder hidden width in the checkpoint.
/// Output is mono PCM at the codec sample rate (32 kHz), encoded to MP3 like ACE-Step.
/// </summary>
public static class MusicGenLoader
{
    public const string MusicGenCompatClassId = "musicgen";
    public const string MusicGenClassId = "musicgen";

    /// <summary>T5-base conditioning context length (HF MusicGen processors default to 256).</summary>
    private const int TextTokenLength = 256;

    /// <summary>Registers the MusicGen compat + model class (audio params group lights up via
    /// IsAudioModel). Call once at extension init, before model folders are scanned.</summary>
    public static void RegisterModelClass()
    {
        T2IModelCompatClass compat = T2IModelClassSorter.RegisterCompat(new() { ID = MusicGenCompatClassId, ShortCode = "MusicGen", IsAudioModel = true, LorasTargetTextEnc = false });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = MusicGenClassId,
            CompatClass = compat,
            Name = "MusicGen",
            IsThisModelOfClass = (model, header) =>
                header.ContainsKey("enc_to_dec_proj.weight") && header.ContainsKey("lm_heads.0.weight"),
        });
    }

    public static MusicGenCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("MusicGen model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"MusicGen checkpoint not found: {model.RawFilePath}");

        // TODO(engine-blocked): three engine pieces are missing for MusicGen —
        //   1. EnCodecConfig.EnCodec32kHz preset (only EnCodec24kHz exists; MusicGen's codec is 32 kHz)
        //   2. MusicGenCheckpointConverter.LoadTextEncoder (text_encoder.* extraction from the HF file)
        //   3. T5TextEncoderConfig.T5Base preset (v1.0 non-gated ReLU T5-base)
        // Fail fast before the decoder load; the EngineGap.Value placeholders below keep the
        // final wiring shape compiling. Lift by deleting this throw + restoring the three lines.
        EngineGap.Throw(
            "MusicGen: HartsyInference is missing the EnCodec-32kHz preset, the bundled-T5 converter path, "
            + "and the T5-Base encoder preset. The extension wiring is ready; this lifts when the engine adds them.");

        // ── 1. Decoder LM (size preset from the final layer-norm width) ──
        log($"Loading MusicGen decoder: {model.Name}");
        var (decoderWeights, decoderLoader) = MusicGenCheckpointConverter.LoadDecoder(model.RawFilePath, castToF32: true);
        if (decoderWeights.Count == 0)
        {
            decoderLoader.Dispose();
            throw new InvalidOperationException($"MusicGen checkpoint '{model.Name}' has no recognized decoder weights.");
        }
        MusicGenConfig config = ResolveConfig(decoderWeights, model.Name, log);
        MusicGenDecoder decoder = new MusicGenDecoder(config);
        decoder.LoadWeights(decoderWeights);

        // ── 2. EnCodec 32 kHz (bundled under audio_encoder.*) ──
        log("Loading bundled EnCodec (32 kHz)...");
        var (codecWeights, codecLoader) = MusicGenCheckpointConverter.LoadEnCodec(model.RawFilePath, castToF32: true);
        // TODO(engine-blocked): EnCodec codec = new EnCodec(EnCodecConfig.EnCodec32kHz);
        EnCodec codec = EngineGap.Value<EnCodec>("MusicGen: EnCodecConfig.EnCodec32kHz preset missing upstream.");
        codec.LoadWeights(codecWeights);

        // ── 3. T5-base text encoder (bundled under text_encoder.*; v1.0 non-gated ReLU) ──
        log("Loading bundled T5-base text encoder...");
        // TODO(engine-blocked): var (t5Weights, t5Loader) = MusicGenCheckpointConverter.LoadTextEncoder(model.RawFilePath, castToF32: true);
        var (t5Weights, t5Loader) = EngineGap.Value<(Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader)>(
            "MusicGen: MusicGenCheckpointConverter.LoadTextEncoder missing upstream.");
        // TODO(engine-blocked): T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.T5Base);
        T5TextEncoder t5 = EngineGap.Value<T5TextEncoder>("MusicGen: T5TextEncoderConfig.T5Base preset missing upstream.");
        t5.LoadWeights(t5Weights);
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: TextTokenLength);

        log("Building MusicGen pipeline...");
        MusicGenPipeline pipeline = new MusicGenPipeline(config, decoder, codec);

        log($"MusicGen ready ({config.Hidden}-hidden, {config.NumLayers} layers, mono @ {config.CodecSampleRate} Hz).");
        return new MusicGenCacheEntry
        {
            ModelName = model.Name,
            CompatClass = MusicGenCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Decoder = decoder,
            Codec = codec,
            T5 = t5,
            Tokenizer = tokenizer,
            CheckpointLoader = decoderLoader,
            CodecLoader = codecLoader,
            T5Loader = t5Loader,
        };
    }

    public static Image[] Generate(
        MusicGenCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string style = input.Get(T2IParamTypes.Text2AudioStyle, null);
        string text = string.IsNullOrWhiteSpace(style) ? prompt : (string.IsNullOrWhiteSpace(prompt) ? style : $"{style}, {prompt}");
        // MusicGen is trained on ≤30s windows; longer needs sliding-window continuation (not wired).
        double duration = Math.Clamp(input.Get(T2IParamTypes.Text2AudioDuration, 10d), 1d, 30d);
        long seedLong = input.Get(T2IParamTypes.Seed);
        int seed = seedLong < 0 ? Random.Shared.Next() : (int)(seedLong & 0x7FFFFFFF);

        // T5-base conditioning [1, T, 768]; drop encoder GPU weights before the LM runs.
        int[] tokens = entry.Tokenizer.Encode(text);
        Tensor t5States = entry.T5.Encode(backend, [tokens], [T5Tokenizer.CreateAttentionMask(tokens)]);
        backend.Sync();
        backend.FreeWeights(entry.T5.EnumerateWeights());

        long start = Environment.TickCount64;
        cancel.ThrowIfCancellationRequested();
        try
        {
            float[] samples = entry.Pipeline.Synthesize(backend, t5States, seconds: (float)duration, seed: seed);
            cancel.ThrowIfCancellationRequested();
            Logs.Verbose($"[HartsyInference][MusicGen] {samples.Length} samples @ {entry.Config.CodecSampleRate} Hz " +
                $"({duration:0}s requested) in {Environment.TickCount64 - start}ms.");
            // Mono source → duplicate the channel for the MP3 encoder.
            return [AudioOutputEncoder.EncodeMp3(samples, samples, entry.Config.CodecSampleRate, cancel)];
        }
        finally
        {
            t5States.Dispose();
        }
    }

    /// <summary>Infers Small/Medium/Large from the decoder's final layer-norm width (1024/1536/2048).</summary>
    private static MusicGenConfig ResolveConfig(Dictionary<string, Tensor> decoderWeights, string modelName, Action<string> log)
    {
        int hidden = 0;
        if (decoderWeights.TryGetValue("model.decoder.layer_norm.weight", out Tensor finalNorm))
        {
            hidden = (int)finalNorm.Shape[0];
        }
        MusicGenConfig config = hidden switch
        {
            1024 => MusicGenConfig.Small,
            1536 => MusicGenConfig.Medium,
            2048 => MusicGenConfig.Large,
            _ => throw new InvalidOperationException(
                $"MusicGen checkpoint '{modelName}' has unrecognized decoder width {hidden} — expected 1024 (small), 1536 (medium), or 2048 (large)."),
        };
        log($"  Detected MusicGen size: hidden={hidden}.");
        return config;
    }
}

public sealed class MusicGenCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required MusicGenPipeline Pipeline { get; init; }
    public required MusicGenConfig Config { get; init; }
    public required MusicGenDecoder Decoder { get; init; }
    public required EnCodec Codec { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader CodecLoader { get; init; }
    public required SafeTensorsLoader T5Loader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        T5?.Dispose();
        Tokenizer?.Dispose();
        CheckpointLoader?.Dispose();
        CodecLoader?.Dispose();
        T5Loader?.Dispose();
    }
}
