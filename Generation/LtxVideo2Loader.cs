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

        SafeTensorsLoader gemmaLoaderOuter = null;
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

            // Gemma-3-12B text tower: bundled text_encoder.* weights when present, else the standalone
            // Comfy-repackage side model (fp8 scaled — LlamaStyleEncoder consumes it raw, weights stay
            // fp8-resident; the raw safetensors loader must stay alive for the pipeline's lifetime
            // because the tensors are memory-mapped views into it).
            LlamaStyleEncoder gemma = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Gemma3_12B);
            SafeTensorsLoader gemmaLoader = null;
            string gemmaSidePath = null;
            if (conv.TextEncoder.Count > 0)
            {
                log("Loading Gemma-3-12B text tower (bundled)...");
                gemma.LoadWeights(conv.TextEncoder);
            }
            else
            {
                T2IModel gemmaModel = ModelAutoDownloader.EnsureSideModel(
                    userPick: input?.Get(T2IParamTypes.T5XXLModel),
                    entry: SideModels.GemmaLtx2,
                    log: log);
                gemmaSidePath = gemmaModel.RawFilePath;
                log($"Loading Gemma-3-12B text tower (standalone: {Path.GetFileName(gemmaSidePath)})...");
                gemmaLoader = new SafeTensorsLoader();
                gemmaLoaderOuter = gemmaLoader;
                gemmaLoader.Load(gemmaSidePath);
                gemma.LoadWeights(gemmaLoader.GetAllTensors());
            }

            // Gemma SentencePiece tokenizer — next to the checkpoint, or next to the standalone encoder.
            GemmaTokenizer tokenizer = new GemmaTokenizer(
                LocateGemmaTokenizer(model.RawFilePath, gemmaSidePath), maxLength: TokenLength);

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
                GemmaLoader = gemmaLoader,
            };
        }
        catch
        {
            gemmaLoaderOuter?.Dispose();
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

    /// <summary>Finds the Gemma SentencePiece model (<c>tokenizer.model</c> / <c>gemma*.model</c> / any
    /// <c>*.model</c>) next to the checkpoint, or next to the standalone Gemma encoder when one was
    /// resolved. Gemma ships no embedded tokenizer in the engine.</summary>
    private static string LocateGemmaTokenizer(string checkpointPath, string gemmaSidePath = null)
    {
        List<string> dirs = [];
        string ckptDir = File.Exists(checkpointPath) ? Path.GetDirectoryName(checkpointPath) : checkpointPath;
        if (!string.IsNullOrEmpty(ckptDir)) dirs.Add(ckptDir);
        if (!string.IsNullOrEmpty(gemmaSidePath)) dirs.Add(Path.GetDirectoryName(gemmaSidePath));
        foreach (string dir in dirs)
        {
            foreach (string candidate in new[] { "tokenizer.model", "gemma.model", "gemma3.model" })
            {
                string path = Path.Combine(dir!, candidate);
                if (File.Exists(path)) return path;
            }
        }
        // Any loose SentencePiece file next to the checkpoint (legacy behavior; checkpoint dir only —
        // the Clip folder holds unrelated .model files).
        string[] spm = Directory.GetFiles(dirs[0]!, "*.model");
        if (spm.Length > 0) return spm[0];
        throw new FileNotFoundException(
            $"LTX-2 needs the Gemma SentencePiece tokenizer (tokenizer.model) next to the checkpoint in '{dirs[0]}' "
            + (dirs.Count > 1 ? $"or next to the Gemma encoder in '{dirs[1]}'." : "."));
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
    /// <summary>Keeps the standalone Gemma safetensors mapping alive (tensor views point into it). Null when the text tower came bundled in the checkpoint.</summary>
    public SafeTensorsLoader GemmaLoader { get; init; }

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
        GemmaLoader?.Dispose();
    }
}
