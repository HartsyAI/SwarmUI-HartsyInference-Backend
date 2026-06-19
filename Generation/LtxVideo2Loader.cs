using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads LTX-2.3 (Lightricks, 22B audiovisual; SwarmUI compat class <c>lightricks-ltx-video-2</c>). Targets the
/// bundled single-file (or sharded) checkpoint (<c>ltx-2.3-22b-dev.safetensors</c>) which carries the dual-stream
/// DiT, the per-modality text connectors, the video VAE, the audio VAE, the vocoder, and — when present — the
/// Gemma-3-12B text tower. <see cref="LtxVideo2CheckpointConverter"/> routes all of them.
///
/// <para>The Gemma-3-12B text encoder is loaded from the bundled <c>text_encoder.*</c> weights when present; the
/// SentencePiece tokenizer is read from a <c>tokenizer.model</c> / <c>gemma*.model</c> file next to the checkpoint
/// (Gemma ships no embedded tokenizer in the engine). Both the engine LTX-2 path and this loader are numerics
/// validation-pending against the real checkpoint.</para>
///
/// <para>Output is a video plus (when the audio VAE + vocoder are present) a 48 kHz MP3 audio track, returned as a
/// second SwarmUI <see cref="Image"/> keyed by <see cref="MediaType.AudioMp3"/>.</para>
/// </summary>
public static class LtxVideo2Loader
{
    public const string LtxVideo2CompatClassId = "lightricks-ltx-video-2";

    /// <summary>Gemma context length fed to the connectors (padded to a register multiple inside the pipeline).</summary>
    private const int TokenLength = 256;

    public static LtxVideo2CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("LTX-2 model has no file path.");

        log($"Loading LTX-2 checkpoint: {model.Name}");
        LtxVideo2CheckpointConverter.ConvertedWeights conv;
        IDisposable ckptLoader;
        if (File.Exists(model.RawFilePath))
        {
            var (c, loader) = LtxVideo2CheckpointConverter.LoadAndConvert(model.RawFilePath);
            conv = c; ckptLoader = loader;
        }
        else if (Directory.Exists(model.RawFilePath))
        {
            var (c, loaders) = LtxVideo2CheckpointConverter.LoadAndConvertShards(model.RawFilePath);
            conv = c; ckptLoader = new MultiLoaderHandle(loaders);
        }
        else
        {
            throw new FileNotFoundException($"LTX-2 checkpoint not found: {model.RawFilePath}");
        }

        try
        {
            if (conv.Transformer.Count == 0)
                throw new InvalidOperationException(
                    $"LTX-2 checkpoint '{model.Name}' has no recognized DiT weights after conversion.");
            if (conv.Connectors.Count == 0)
                throw new InvalidOperationException(
                    $"LTX-2 checkpoint '{model.Name}' has no text-connector weights — the bundle must include "
                    + "the per-modality embeddings connectors.");
            if (conv.Vae.Count == 0)
                throw new InvalidOperationException($"LTX-2 checkpoint '{model.Name}' has no bundled video VAE weights.");
            log($"  Converted: {conv.Transformer.Count} DiT, {conv.Connectors.Count} connector, {conv.Vae.Count} VAE, "
                + $"{conv.AudioVae.Count} audio-VAE, {conv.Vocoder.Count} vocoder, {conv.TextEncoder.Count} text-encoder keys");

            LtxVideo2Config config = LtxVideo2Config.V23;

            LtxVideo2Transformer transformer = new LtxVideo2Transformer(config);
            transformer.LoadWeights(conv.Transformer);

            LtxVideo2TextConnectors connectors = new LtxVideo2TextConnectors(config);
            connectors.LoadWeights(conv.Connectors);

            (float[] vMean, float[] vStd) = ReadStats(conv.Vae, config.InChannels);
            LtxVideo2VaeDecoder vae = new LtxVideo2VaeDecoder(latentsMean: vMean, latentsStd: vStd);
            vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.Vae, DType.F32));

            // Audio (optional): VAE decoder + BigVGAN vocoder.
            LtxAudioVaeDecoder audioVae = null;
            LtxAudioVocoder vocoder = null;
            float[] aMean = null, aStd = null;
            if (conv.AudioVae.Count > 0 && conv.Vocoder.Count > 0)
            {
                // Audio latent stats are stored over the packed feature axis (8 latent ch × 16 mel = 128).
                (aMean, aStd) = ReadStats(conv.AudioVae, config.AudioInChannels);
                audioVae = new LtxAudioVaeDecoder();
                audioVae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.AudioVae, DType.F32));
                vocoder = new LtxAudioVocoder();
                vocoder.LoadWeights(conv.Vocoder);
                log("  Audio decode wired (VAE + vocoder).");
            }
            else
            {
                log("  No bundled audio VAE/vocoder — video-only output.");
            }

            // Gemma-3-12B text tower (from the bundled text_encoder.* weights).
            if (conv.TextEncoder.Count == 0)
                throw new InvalidOperationException(
                    $"LTX-2 checkpoint '{model.Name}' has no bundled Gemma-3-12B text encoder (text_encoder.*). "
                    + "A standalone Gemma side-model is not yet wired for LTX-2.");
            log("Loading Gemma-3-12B text tower...");
            LlamaStyleEncoder gemma = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Gemma3_12B);
            gemma.LoadWeights(conv.TextEncoder);

            // Gemma SentencePiece tokenizer — read from a file next to the checkpoint.
            GemmaTokenizer tokenizer = new GemmaTokenizer(LocateGemmaTokenizer(model.RawFilePath), maxLength: TokenLength);

            log("Building LTX-2 pipeline...");
            LtxVideo2Pipeline pipeline = new LtxVideo2Pipeline(backend, transformer, connectors, vae, gemma, config,
                audioVae, vocoder, aMean, aStd);

            log("LTX-2 ready (text-to-video" + (vocoder is not null ? "+audio" : "") + ").");
            return new LtxVideo2CacheEntry
            {
                ModelName = model.Name,
                CompatClass = LtxVideo2CompatClassId,
                Pipeline = pipeline,
                Config = config,
                Tokenizer = tokenizer,
                Gemma = gemma,
                Transformer = transformer,
                Connectors = connectors,
                Vae = vae,
                AudioVae = audioVae,
                Vocoder = vocoder,
                CheckpointLoader = ckptLoader,
            };
        }
        catch
        {
            ckptLoader.Dispose();
            throw;
        }
    }

    public static Image[] Generate(
        LtxVideo2CacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        var (width, height) = VideoParamResolver.ResolveResolution(input, multiple: entry.Config.VaeSpatialCompression);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 121, step: entry.Config.VaeTemporalCompression);
        int frameRate = VideoParamResolver.ResolveFps(input);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);

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

        LtxVideo2Pipeline.Ltx2Result result = entry.Pipeline.GenerateFromTokens(
            promptTokens, negTokens, request, numFrames, frameRate, bridge);
        Logs.Verbose($"[HartsyInference][LTX-2] Pipeline returned {result.Frames.Length} frames " +
            $"{result.Width}x{result.Height} in {Environment.TickCount64 - start}ms.");

        Image video = VideoParamResolver.FinishVideo(result.Frames, result.Width, result.Height, input, cancel);
        if (result.Audio is null || result.Audio.Length == 0)
            return new[] { video };

        float[] left = result.Audio[0];
        float[] right = result.Audio.Length > 1 ? result.Audio[1] : result.Audio[0];
        Image audio = AudioOutputEncoder.EncodeMp3(left, right, result.AudioSampleRate, cancel);
        return new[] { video, audio };
    }

    /// <summary>Reads the per-channel latent normalization stats from the converted VAE bucket into F32 arrays,
    /// trying the original Lightricks names (<c>per_channel_statistics.mean-of-means</c>/<c>std-of-means</c>) first,
    /// then the diffusers names (<c>latents_mean</c>/<c>latents_std</c>). Returns nulls (no denormalization) when
    /// absent.</summary>
    private static (float[] Mean, float[] Std) ReadStats(Dictionary<string, Tensor> vae, int channels)
    {
        Tensor mean = Find(vae, "per_channel_statistics.mean-of-means", "latents_mean");
        Tensor std = Find(vae, "per_channel_statistics.std-of-means", "latents_std");
        if (mean is null || std is null) return (null, null);
        return (ToFloatArray(mean, channels), ToFloatArray(std, channels));
    }

    private static Tensor Find(Dictionary<string, Tensor> w, params string[] keys)
    {
        foreach (string k in keys) if (w.TryGetValue(k, out Tensor t)) return t;
        return null;
    }

    private static unsafe float[] ToFloatArray(Tensor t, int count)
    {
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        int n = (int)Math.Min(count, f.Shape.ElementCount);
        float[] outArr = new float[n];
        float* p = (float*)f.DataPointer;
        for (int i = 0; i < n; i++) outArr[i] = p[i];
        return outArr;
    }

    /// <summary>Finds the Gemma SentencePiece model next to the checkpoint (<c>tokenizer.model</c> or
    /// <c>gemma*.model</c> / <c>*.spm</c>). Gemma ships no embedded tokenizer in the engine.</summary>
    private static string LocateGemmaTokenizer(string checkpointPath)
    {
        string dir = File.Exists(checkpointPath) ? Path.GetDirectoryName(checkpointPath) : checkpointPath;
        foreach (string candidate in new[] { "tokenizer.model", "gemma.model", "gemma3.model" })
        {
            string path = Path.Combine(dir!, candidate);
            if (File.Exists(path)) return path;
        }
        string[] spm = Directory.GetFiles(dir!, "*.model");
        if (spm.Length > 0) return spm[0];
        throw new FileNotFoundException(
            $"LTX-2 needs the Gemma SentencePiece tokenizer (tokenizer.model) next to the checkpoint in '{dir}'.");
    }

    /// <summary>Disposes a set of safetensors loaders as one <see cref="IDisposable"/> (sharded checkpoints).</summary>
    private sealed class MultiLoaderHandle(List<SafeTensorsLoader> loaders) : IDisposable
    {
        public void Dispose() { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }
}

public sealed class LtxVideo2CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required LtxVideo2Pipeline Pipeline { get; init; }
    public required LtxVideo2Config Config { get; init; }
    public required GemmaTokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder Gemma { get; init; }
    public required LtxVideo2Transformer Transformer { get; init; }
    public required LtxVideo2TextConnectors Connectors { get; init; }
    public required LtxVideo2VaeDecoder Vae { get; init; }
    public LtxAudioVaeDecoder AudioVae { get; init; }
    public LtxAudioVocoder Vocoder { get; init; }
    public required IDisposable CheckpointLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        Gemma?.Dispose();
        Transformer?.Dispose();
        Connectors?.Dispose();
        (Vocoder as IDisposable)?.Dispose();
        CheckpointLoader?.Dispose();
    }
}
