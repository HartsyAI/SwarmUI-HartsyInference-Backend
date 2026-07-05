using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads LTX-Video (Lightricks; SwarmUI compat class <c>lightricks-ltx-video</c>). Targets the
/// single-file checkpoints (e.g. <c>ltx-video-2b-v0.9.safetensors</c>) which bundle DiT + VAE —
/// <c>LtxVideoCheckpointConverter</c> splits and renames both. Auto-detects the variant from the
/// checkpoint: <b>0.9</b> (2B base, non-timestep VAE), <b>0.9.5</b> (2B, timestep-conditioned VAE),
/// and <b>0.9.7/0.9.8</b> (13B — 48 layers, head_dim 128, cross 4096, timestep VAE). LTX-2
/// (<c>lightricks-ltx-video-2*</c>) is a different architecture handled by <see cref="LtxVideo2Loader"/>.
///
/// Required side model (auto-downloaded; user pick via <see cref="T2IParamTypes.T5XXLModel"/>
/// takes priority): plain T5-XXL (<see cref="SideModels.T5XxlEnconly"/> — the same file Flux,
/// SD3, and Chroma use; LTX uses standard T5, not Wan's umT5).
///
/// The user's <c>VideoFPS</c> feeds the pipeline itself (RoPE interpolation — the same value
/// Comfy injects via <c>LTXVConditioning.frame_rate</c>), not just the output muxer.
/// </summary>
public static class LtxVideoLoader
{
    public const string LtxVideoCompatClassId = "lightricks-ltx-video";

    /// <summary>LTX's T5 context length (diffusers uses 128 tokens).</summary>
    private const int TokenLength = 128;

    public static LtxVideoCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("LTX-Video model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"LTX-Video checkpoint not found: {model.RawFilePath}");

        T2IModel t5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel),
            entry: SideModels.T5XxlEnconly,
            log: log);

        // ── 1. Load + convert the LTX single file (DiT + VAE bundled) ──
        log($"Loading LTX-Video checkpoint: {model.Name}");
        var (conv, ckptLoader) = LtxVideoCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            ckptLoader.Dispose();
            throw new InvalidOperationException(
                $"LTX checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        if (conv.Vae.Count == 0)
        {
            ckptLoader.Dispose();
            throw new InvalidOperationException(
                $"LTX checkpoint '{model.Name}' has no bundled VAE weights. HartsyInference currently requires "
                + "a full single-file LTX-Video checkpoint (DiT + VAE in one file, e.g. ltx-video-2b-v0.9.safetensors).");
        }
        log($"  Converted: {conv.Transformer.Count} DiT keys, {conv.Vae.Count} VAE keys");

        // ── Variant selection: 0.9 (2B base) vs 0.9.5 (2B, timestep VAE) vs 0.9.7/0.9.8 (13B) ──
        // The 13B transformer has 48 DiT layers (head_dim 128, cross-attn 4096); 0.9/0.9.5 have 28.
        // 0.9.5 and 13B share the timestep-conditioned VAE (residual channel-changing upsamplers,
        // decoder_block_out_channels (256,512,1024), decode at 0.05/0.025). Detect from the checkpoint
        // itself (layer count + VAE key shape), with the model name as a tiebreaker. Mirrors the engine's
        // LtxVideoGenerationTests variant recipe.
        int maxBlock = MaxTransformerBlockIndex(conv.Transformer.Keys);
        string nameLc = (model.Name ?? "").ToLowerInvariant();
        bool is13B = maxBlock >= 28 || nameLc.Contains("13b") || nameLc.Contains("0.9.7") || nameLc.Contains("0.9.8");
        bool timestepVae = is13B
            || LtxVideoCheckpointConverter.IsTimestepVae(ckptLoader.GetAllTensors().Keys)
            || nameLc.Contains("0.9.5");
        LtxVideoConfig config = is13B ? LtxVideoConfig.V097 : timestepVae ? LtxVideoConfig.V095 : LtxVideoConfig.V09;
        log($"  LTX-Video variant: {(is13B ? "0.9.7/13B" : timestepVae ? "0.9.5" : "0.9")} "
            + $"({maxBlock + 1} DiT layers, {(timestepVae ? "timestep-conditioned" : "base")} VAE).");

        LtxVideoTransformer transformer = new LtxVideoTransformer(config);
        transformer.LoadWeights(conv.Transformer);
        // The timestep-conditioned (0.9.5/13B) VAE has a different decoder block layout than the 0.9 base
        // one; build it with the 0.9.5 config when detected (matches the engine test's construction).
        LtxVideoVaeDecoder vae = timestepVae
            ? new LtxVideoVaeDecoder(blockOutChannels: [256, 512, 1024], spatioTemporalScaling: [true, true, true],
                layersPerBlock: [5, 5, 5, 5], patchSize: 4, isCausal: false, timestepConditioned: true,
                upsampleFactor: [2, 2, 2], upsampleResidual: [true, true, true])
            : new LtxVideoVaeDecoder();
        vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.Vae, DType.F32));

        // 13B fp8 weights stay fp8-resident (~13 GB); caching their F16 casts would ~double VRAM and OOM a
        // 24 GB card. Dequant transiently per GEMM instead — the verified 13B recipe (matches the Wan fp8 path).
        if (is13B && backend is HartsyInference.Cuda.CudaBackend cudaBackend)
        {
            cudaBackend.CacheWeightCasts = false;
            log("  13B: CacheWeightCasts disabled (fp8-resident, transient per-GEMM dequant).");
        }

        // ── 2. Load T5-XXL (standard T5 — shared file with Flux/SD3/Chroma) ──
        log($"Loading T5-XXL: {t5Model.Name}");
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Model.RawFilePath);
        Dictionary<string, Tensor> t5Weights = StripStandalonePrefix(t5Loader.GetAllTensors(), "text_encoders.t5xxl.transformer.");
        if (t5Weights.Count == 0)
        {
            t5Loader.Dispose();
            ckptLoader.Dispose();
            throw new InvalidOperationException($"T5 model file '{t5Model.Name}' has no usable T5 tensors.");
        }
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);

        // ── 3. Tokenizer (embedded T5 spiece) ──
        log("Loading T5 tokenizer (embedded)...");
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: TokenLength);

        log("Building LTX-Video pipeline...");
        LtxVideoPipeline pipeline = new LtxVideoPipeline(backend, transformer, vae, config);

        log("LTX-Video ready (text-to-video).");
        return new LtxVideoCacheEntry
        {
            ModelName = model.Name,
            CompatClass = LtxVideoCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Tokenizer = tokenizer,
            T5 = t5,
            Transformer = transformer,
            Vae = vae,
            CheckpointLoader = ckptLoader,
            T5Loader = t5Loader,
        };
    }

    public static Image[] Generate(
        LtxVideoCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = PromptConditioningResolver.VideoText(input.Get(T2IParamTypes.Prompt));
        string negative = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.NegativePrompt));
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        var (width, height) = VideoParamResolver.ResolveResolution(input, multiple: entry.Config.VaeSpatialCompression);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 97, step: entry.Config.VaeTemporalCompression);
        int frameRate = VideoParamResolver.ResolveFps(input);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        // Encode the prompt pair, then drop the encoder's GPU weights before the DiT preload
        // (mirrors the upstream LTX E2E test).
        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        Tensor batch = entry.T5.Encode(backend,
            [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, TokenLength, entry.Config.CaptionChannels);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, TokenLength, entry.Config.CaptionChannels);
        // The engine's LTX pipeline has no context attention mask (the reference masks pad tokens in
        // cross-attention); zeroing the pad rows is the closest approximation and prevents T5's
        // garbage pad-position outputs from drowning the prompt (same failure class as Wan's flat video).
        WanVideoLoader.ZeroPaddedRows(promptEmbeds, promptTokens, entry.Config.CaptionChannels);
        WanVideoLoader.ZeroPaddedRows(negEmbeds, negTokens, entry.Config.CaptionChannels);
        batch.Dispose();
        backend.Sync();
        backend.FreeWeights(entry.T5.EnumerateWeights());

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

        try
        {
            var (frames, outW, outH, _) = entry.Pipeline.GenerateFromEmbeddings(
                promptEmbeds, negEmbeds, request, numFrames, frameRate, bridge);
            Logs.Verbose($"[HartsyInference][LTX] Pipeline returned {frames.Length} frames {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <summary>Highest <c>transformer_blocks.{i}</c> index across the converted DiT keys (−1 if none).
    /// Used to tell the 48-layer 13B (0.9.7/0.9.8) apart from the 28-layer 2B (0.9/0.9.5).</summary>
    private static int MaxTransformerBlockIndex(IEnumerable<string> keys)
    {
        const string tok = "transformer_blocks.";
        int max = -1;
        foreach (string k in keys)
        {
            int at = k.IndexOf(tok, StringComparison.Ordinal);
            if (at < 0) continue;
            int s = at + tok.Length, e = s;
            while (e < k.Length && char.IsDigit(k[e])) e++;
            if (e > s) max = Math.Max(max, int.Parse(k.AsSpan(s, e - s)));
        }
        return max;
    }

    /// <summary>Standalone T5-XXL files store keys as-is or wrapped under Comfy's
    /// <c>text_encoders.t5xxl.transformer.</c> prefix — strip if present (same handling as Chroma/SD3).</summary>
    private static Dictionary<string, Tensor> StripStandalonePrefix(Dictionary<string, Tensor> raw, string comfyPrefix)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var kv in raw)
        {
            if (kv.Key.StartsWith(comfyPrefix, StringComparison.Ordinal))
                result[kv.Key[comfyPrefix.Length..]] = kv.Value;
            else
                result[kv.Key] = kv.Value;
        }
        return result;
    }

}

public sealed class LtxVideoCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required LtxVideoPipeline Pipeline { get; init; }
    public required LtxVideoConfig Config { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required LtxVideoTransformer Transformer { get; init; }
    public required LtxVideoVaeDecoder Vae { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader T5Loader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        T5?.Dispose();
        Transformer?.Dispose();
        CheckpointLoader?.Dispose();
        T5Loader?.Dispose();
    }
}
