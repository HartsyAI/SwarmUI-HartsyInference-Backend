using System.IO;
using System.Linq;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Wan2.2-S2V (speech-to-video). Rides the plain Wan2.1-14B CompatClass; the checkpoint adds an audio
/// injector (<c>audio_injector.*</c>) and a causal audio encoder (<c>audio_encoder.*</c>) over stacked Wav2Vec2
/// features. <see cref="WanModelVariants.Detect"/> routes it here off those signature weights.
///
/// <para><b>Best-effort / validation-pending.</b> The engine builds S2V structurally (CPU-tested with synthetic
/// weights) but its own docs flag several things as unconfirmed vs the real checkpoint: the
/// <c>WanVideoCheckpointConverter</c> has no S2V key mapping yet, and the audio-inject block indices, the Wav2Vec2
/// variant, and the harvested-layer count are provisional. This loader derives every structural value it can from
/// the checkpoint's own audio-encoder weight shapes, picks the Wav2Vec2 preset from the derived feature dim, and
/// <b>refuses cleanly</b> if the converted checkpoint lacks the audio keys (i.e. until the engine's converter maps
/// them). The audio-inject layer indices are still a provisional "every-Nth" guess — see <see cref="DeriveInjectLayers"/>.</para>
///
/// <para><b>Input:</b> the driving speech goes in the <c>Video Audio Input</c> param; the single-clip path
/// (audio + text, no reference identity / multi-chunk) is used.</para>
/// </summary>
public static class WanS2VLoader
{
    private const int TokenLength = 512;

    public static WanS2VCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Wan S2V model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Wan S2V checkpoint not found: {model.RawFilePath}");

        string compat = model.ModelClass?.CompatClass?.ID ?? WanVideoLoader.Wan21_14BCompatClassId;
        WanVideoConfig config = WanVideoConfig.S2V_14B;

        T2IModel umt5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel), entry: SideModels.Umt5Xxl, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.Wan21Vae, log: log);

        // ── 1. Load + convert the S2V DiT ──
        log($"Loading Wan S2V DiT: {model.Name} (compat {compat})");
        var (conv, ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(model.RawFilePath);
        Dictionary<string, Tensor> w = conv.Transformer;

        // Graceful guard: the engine converter doesn't map S2V keys yet — refuse precisely rather than crash deep in LoadWeights.
        if (!w.ContainsKey("audio_encoder.layer_weights") || !w.ContainsKey("audio_encoder.conv1.weight"))
        {
            ditLoader.Dispose();
            throw new SwarmUserErrorException(
                $"HartsyInference: '{model.Name}' is tagged as Wan S2V but the converted checkpoint has no "
                + "'audio_encoder.*' weights. S2V is validation-pending — the HartsyInference checkpoint converter "
                + "doesn't map the S2V audio keys yet. This will work once the engine ships the S2V converter pass.");
        }
        int injectCount = CountInjectors(w);
        if (injectCount == 0)
        {
            ditLoader.Dispose();
            throw new SwarmUserErrorException(
                $"HartsyInference: '{model.Name}' has S2V audio-encoder weights but no 'audio_injector.*' blocks after "
                + "conversion — the converter's S2V key mapping is incomplete. S2V is validation-pending.");
        }

        // Derive the audio-encoder structure from its own weight shapes (robust to base-768 vs large-1024 etc.).
        Tensor conv1 = w["audio_encoder.conv1.weight"];   // Conv1d [dim, audioDim, kernel]
        int audioDim = (int)conv1.Shape[1];
        int audioKernel = (int)conv1.Shape[2];
        int numAudioLayers = (int)w["audio_encoder.layer_weights"].Shape[0];
        int tokensPerFrame = w.TryGetValue("audio_encoder.conv2.weight", out Tensor conv2) && config.InnerDim > 0
            ? Math.Max(1, (int)conv2.Shape[0] / config.InnerDim) : 1;
        int[] injectLayers = DeriveInjectLayers(injectCount, config.NumLayers);
        log($"  Converted: {w.Count} keys (S2V audio injector ×{injectCount} @ [{string.Join(",", injectLayers)}], "
            + $"audioDim {audioDim}, {numAudioLayers} harvested layers, {tokensPerFrame} tok/frame)");

        WanS2VTransformer transformer = new WanS2VTransformer(config, injectLayers);
        transformer.LoadWeights(w);
        WanS2VAudioEncoder audioEncoder = new WanS2VAudioEncoder(numAudioLayers, audioDim, config.InnerDim,
            tokensPerFrame: tokensPerFrame, kernel: audioKernel);
        audioEncoder.LoadWeights(w, "audio_encoder");

        Wav2Vec2Encoder wav2vec2 = null;
        SafeTensorsLoader wav2vec2Loader = null;
        SafeTensorsLoader umt5Loader = null;
        try
        {
            // ── 2. Wan2.1 VAE (decoder only — single-clip path doesn't encode) ──
            log($"Loading Wan2.1 VAE: {vaeModel.Name}");
            var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder(); vaeDecoder.LoadWeights(vaeWeights);

            // ── 3. Wav2Vec2 audio front-end (variant chosen from the derived feature dim) ──
            Wav2Vec2EncoderConfig w2vConfig = audioDim >= 1024 ? Wav2Vec2EncoderConfig.Large : Wav2Vec2EncoderConfig.Base;
            SideModels.Entry w2vEntry = audioDim >= 1024 ? SideModels.Wav2Vec2Large : SideModels.Wav2Vec2Base;
            T2IModel w2vModel = ModelAutoDownloader.EnsureSideModel(userPick: null, entry: w2vEntry, log: log);
            log($"Loading Wav2Vec2 ({(audioDim >= 1024 ? "large" : "base")}): {w2vModel.Name}");
            wav2vec2Loader = new SafeTensorsLoader();
            wav2vec2Loader.Load(w2vModel.RawFilePath);
            wav2vec2 = new Wav2Vec2Encoder(w2vConfig);
            wav2vec2.LoadWeights(wav2vec2Loader.GetAllTensors());

            // ── 4. umT5-XXL + tokenizer ──
            log($"Loading umT5-XXL: {umt5Model.Name}");
            umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Model.RawFilePath);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: TokenLength);

            log("Building Wan S2V pipeline...");
            WanS2VPipeline pipeline = new WanS2VPipeline(backend, transformer, audioEncoder, vaeDecoder, config);

            log($"Wan S2V ready ({compat}, audio+text single-clip). Numerics validation-pending.");
            return new WanS2VCacheEntry
            {
                ModelName = model.Name,
                CompatClass = compat,
                Pipeline = pipeline,
                Config = config,
                Tokenizer = tokenizer,
                Umt5 = umt5,
                Transformer = transformer,
                AudioEncoder = audioEncoder,
                Wav2Vec2 = wav2vec2,
                Vae = vaeDecoder,
                CheckpointLoader = ditLoader,
                VaeLoaders = vaeLoaders,
                Umt5Loader = umt5Loader,
                Wav2Vec2Loader = wav2vec2Loader,
            };
        }
        catch
        {
            transformer.Dispose();
            ditLoader.Dispose();
            umt5Loader?.Dispose();
            wav2vec2Loader?.Dispose();
            (wav2vec2 as IDisposable)?.Dispose();
            throw;
        }
    }

    public static Image[] Generate(
        WanS2VCacheEntry entry, IBackend backend, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 81, step: entry.Config.VaeTemporalCompression);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        AudioFile audio = input.Get(T2IParamTypes.VideoAudioInput)
            ?? throw new SwarmUserErrorException(
                "HartsyInference: Wan S2V needs speech in the Video Audio Input param (that's what drives the video).");
        float[] waveform = AudioDecoder.DecodeMono16k(audio, cancel);

        var (width, height) = VideoParamResolver.ResolveResolution(input, multiple: entry.Config.VaeSpatialCompression);

        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        Tensor batch = entry.Umt5.Encode(backend,
            [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, TokenLength, entry.Config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, TokenLength, entry.Config.TextDim);
        batch.Dispose();
        backend.Sync();
        backend.FreeWeights(entry.Umt5.EnumerateWeights());

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt, NegativePrompt = negative, Width = width, Height = height,
            Steps = steps, CfgScale = cfgScale, Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p => { cancel.ThrowIfCancellationRequested(); onProgress(p); };
        // The pipeline runs Wav2Vec2 + the audio encoder before the DiT; their weights must be resident first
        // (the pipeline only preloads the DiT itself).
        backend.PreloadWeights(entry.Wav2Vec2.EnumerateWeights());
        backend.PreloadWeights(entry.AudioEncoder.EnumerateWeights());
        try
        {
            var (frames, outW, outH, _) = entry.Pipeline.GenerateFromWaveform(
                promptEmbeds, negEmbeds, waveform, entry.Wav2Vec2, request, numFrames, bridge);
            Logs.Verbose($"[HartsyInference][S2V] Pipeline returned {frames.Length} frames {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(entry.Wav2Vec2.EnumerateWeights());
            backend.FreeWeights(entry.AudioEncoder.EnumerateWeights());
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <summary>Counts distinct <c>audio_injector.{i}.</c> indices in the converted weights.</summary>
    private static int CountInjectors(IReadOnlyDictionary<string, Tensor> w)
    {
        int max = -1;
        foreach (string key in w.Keys)
        {
            const string prefix = "audio_injector.";
            int p = key.IndexOf(prefix, StringComparison.Ordinal);
            if (p < 0) continue;
            int s = p + prefix.Length, e = s;
            while (e < key.Length && char.IsDigit(key[e])) e++;
            if (e > s && int.TryParse(key.AsSpan(s, e - s), out int idx)) max = Math.Max(max, idx);
        }
        return max + 1;
    }

    /// <summary>Provisional "every-Nth block" inject-layer indices for <paramref name="count"/> injectors over
    /// <paramref name="numLayers"/> DiT blocks. The exact indices are unconfirmed vs the real checkpoint (engine TODO);
    /// this matches the documented "every Nth of the 40 blocks" pattern.</summary>
    private static int[] DeriveInjectLayers(int count, int numLayers)
    {
        if (count <= 0) return [];
        int stride = Math.Max(1, numLayers / count);
        int[] layers = new int[count];
        for (int i = 0; i < count; i++) layers[i] = Math.Min(numLayers - 1, i * stride);
        return layers;
    }
}

public sealed class WanS2VCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required WanS2VPipeline Pipeline { get; init; }
    public required WanVideoConfig Config { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder Umt5 { get; init; }
    public required WanS2VTransformer Transformer { get; init; }
    public required WanS2VAudioEncoder AudioEncoder { get; init; }
    public required Wav2Vec2Encoder Wav2Vec2 { get; init; }
    public required IWanVaeDecoder Vae { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required IReadOnlyList<SafeTensorsLoader> VaeLoaders { get; init; }
    public required SafeTensorsLoader Umt5Loader { get; init; }
    public required SafeTensorsLoader Wav2Vec2Loader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        Umt5?.Dispose();
        Transformer?.Dispose();
        (Wav2Vec2 as IDisposable)?.Dispose();
        CheckpointLoader?.Dispose();
        Umt5Loader?.Dispose();
        Wav2Vec2Loader?.Dispose();
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
