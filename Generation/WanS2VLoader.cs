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
/// <para><b>Config source:</b> <see cref="WanConfigDetector.Detect"/> — the engine's parity-proven S2V layout
/// (numerically verified layer-by-layer vs the ComfyUI reference on the real fp8 checkpoint), including the
/// non-uniform audio-inject block indices and the empirical S2V CFG defaults. Refuses cleanly if the converted
/// checkpoint lacks the audio keys.</para>
///
/// <para><b>Inputs:</b> the driving speech goes in the <c>Video Audio Input</c> param; an optional identity
/// reference portrait goes in the <c>Init Image</c> slot (VAE-encoded to appended reference tokens). Single-clip
/// path (multi-chunk FramePackMotioner extension not wired).</para>
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

        // The engine's WanConfigDetector carries the PARITY-PROVEN S2V layout — notably the non-uniform
        // audio-inject indices ([0,4,8,...,24,27,30,...,39] for 40/12, verified layer-by-layer vs the ComfyUI
        // reference), the audio-encoder dims from the raw `casual_audio_encoder.*` keys (the original Wan
        // repo's spelling, passed through the converter unchanged), and the empirical S2V CFG defaults.
        // This exactly mirrors the engine's validated WanS2V_Gpu_E2E harness.
        config = WanConfigDetector.Detect(w);
        if (!config.HasAudioConditioning)
        {
            ditLoader.Dispose();
            throw new SwarmUserErrorException(
                $"HartsyInference: '{model.Name}' is tagged as Wan S2V but the converted checkpoint has no "
                + "'casual_audio_encoder.*'/'audio_injector.*' weights — it may be a plain Wan checkpoint, or a "
                + "layout the converter doesn't map yet.");
        }
        int audioDim = config.AudioDim;
        log($"  Converted: {w.Count} keys (S2V audio injector ×{config.AudioInjectLayers.Length} @ "
            + $"[{string.Join(",", config.AudioInjectLayers)}], audioDim {audioDim}, {config.AudioLayers} harvested "
            + $"layers, {config.AudioTokens} tok/frame, cfg {config.GuidanceScale}"
            + $"{(config.CfgRescale > 0 ? $" renorm {config.CfgRescale}" : "")})");
        WanS2VTransformer transformer = new WanS2VTransformer(config);
        transformer.LoadWeights(w);
        WanS2VAudioEncoder audioEncoder = new WanS2VAudioEncoder(config.AudioLayers, config.AudioDim,
            config.InnerDim, config.AudioTokens);
        audioEncoder.LoadWeights(w);   // default prefix: the raw `casual_audio_encoder` keys

        Wav2Vec2Encoder wav2vec2 = null;
        SafeTensorsLoader wav2vec2Loader = null;
        SafeTensorsLoader umt5Loader = null;
        try
        {
            // ── 2. Wan2.1 VAE (decoder + encoder — the encoder feeds the reference-identity latent) ──
            log($"Loading Wan2.1 VAE: {vaeModel.Name}");
            var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder(); vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder(); vaeEncoder.LoadWeights(vaeWeights);

            // ── 3. Wav2Vec2 audio front-end (variant chosen from the derived feature dim) ──
            Wav2Vec2EncoderConfig w2vConfig = audioDim >= 1024 ? Wav2Vec2EncoderConfig.Large : Wav2Vec2EncoderConfig.Base;
            SideModels.Entry w2vEntry = audioDim >= 1024 ? SideModels.Wav2Vec2Large : SideModels.Wav2Vec2Base;
            T2IModel w2vModel = ModelAutoDownloader.EnsureSideModel(userPick: null, entry: w2vEntry, log: log);
            log($"Loading Wav2Vec2 ({(audioDim >= 1024 ? "large" : "base")}): {w2vModel.Name}");
            wav2vec2Loader = new SafeTensorsLoader();
            wav2vec2Loader.Load(w2vModel.RawFilePath);
            wav2vec2 = new Wav2Vec2Encoder(w2vConfig);
            // The Comfy-Org audio-encoder file prefixes every key with "wav2vec2." (plus an unused lm_head) —
            // strip it, mirroring the engine's validated S2V harness.
            Dictionary<string, Tensor> wavW = new();
            foreach (var kvp in wav2vec2Loader.GetAllTensors())
                if (kvp.Key.StartsWith("wav2vec2.", StringComparison.Ordinal)) wavW[kvp.Key["wav2vec2.".Length..]] = kvp.Value;
            if (wavW.Count == 0) wavW = new Dictionary<string, Tensor>(wav2vec2Loader.GetAllTensors());   // unprefixed distribution
            wav2vec2.LoadWeights(wavW);

            // ── 4. umT5-XXL + tokenizer ──
            log($"Loading umT5-XXL: {umt5Model.Name}");
            umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Model.RawFilePath);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: TokenLength);

            log("Building Wan S2V pipeline...");
            WanS2VPipeline pipeline = new WanS2VPipeline(backend, transformer, audioEncoder, vaeDecoder, config, encoder: vaeEncoder);

            log($"Wan S2V ready ({compat}, audio+text, reference-identity capable).");
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
            (wav2vec2 as object as IDisposable)?.Dispose();
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
        // Wan cross-attends all 512 context rows with NO text mask — umT5 pad rows are garbage that drowns
        // the prompt (same fix as WanVideoLoader; this loader predates it).
        WanVideoLoader.ZeroPaddedRows(promptEmbeds, promptTokens, entry.Config.TextDim);
        WanVideoLoader.ZeroPaddedRows(negEmbeds, negTokens, entry.Config.TextDim);
        backend.Sync();
        backend.FreeWeights(entry.Umt5.EnumerateWeights());

        // Optional identity anchor: the Init Image slot carries the reference portrait (VAE-encoded to appended
        // reference tokens by the pipeline — ComfyUI WanSoundImageToVideo's ref_image path).
        Image initImage = input.Get(T2IParamTypes.InitImage);
        byte[] referenceRgb24 = initImage is not null ? RgbToImage.ToHwcRgbResized(initImage, width, height) : null;

        // Sigma Shift: honor the user's param; default 8.0 = ComfyUI's Wan sampling shift (WAN22_S2V inherits
        // WAN21_T2V's sampling_settings).
        float flowShift = input.TryGet(T2IParamTypes.SigmaShift, out double sigmaShift) ? (float)sigmaShift : 8f;
        VideoGenerationRequest request = new VideoGenerationRequest
        {
            Prompt = prompt, NegativePrompt = negative, Width = width, Height = height,
            Steps = steps, CfgScale = cfgScale, Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
            FlowShift = flowShift,
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
                promptEmbeds, negEmbeds, waveform, entry.Wav2Vec2, request, numFrames,
                referenceRgb24: referenceRgb24 ?? default(ReadOnlySpan<byte>), onProgress: bridge);
            Logs.Verbose($"[HartsyInference][S2V] Pipeline returned {frames.Length} frames {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            // Mux the driving speech into the container, trimmed to the clip's duration at Wan's native
            // 16 fps (the rate the audio features were bucketed at — see FinishVideo's defaultFps).
            int audioSamples = (int)Math.Min(waveform.Length, (long)Math.Ceiling(frames.Length / 16.0 * 16000));
            VideoOutputEncoder.AudioTrack speechTrack = new()
            {
                Left = waveform.AsSpan(0, audioSamples).ToArray(),
                SampleRate = 16000,
            };
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel, audio: speechTrack, defaultFps: 16) };
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
        (Wav2Vec2 as object as IDisposable)?.Dispose();
        CheckpointLoader?.Dispose();
        Umt5Loader?.Dispose();
        Wav2Vec2Loader?.Dispose();
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
