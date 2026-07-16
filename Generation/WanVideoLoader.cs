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
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.Lora;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads the Wan-Video DiT family (Wan-AI; umT5-conditioned text/image-to-video) for every SwarmUI Wan compat class:
/// <c>wan-22-5b</c> (Wan2.2 TI2V-5B, z=48 VAE), <c>wan-21-1_3b</c> (Wan2.1 1.3B), and <c>wan-21-14b</c> (Wan2.1 14B
/// and Wan2.2 A14B 14B, z=16 VAE). The variant (size, T2V vs CLIP-I2V) is resolved from the compat class plus the
/// converted DiT keys (an <c>image_embedder</c>/36-channel patch embed ⇒ I2V).
///
/// <para>Side models (auto-downloaded; user picks take priority): umT5-XXL (<see cref="SideModels.Umt5Xxl"/>); the
/// z=48 Wan2.2 VAE (<see cref="SideModels.Wan22Vae"/>) or the z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>); and
/// for Wan2.1 I2V, CLIP-ViT-H (<see cref="SideModels.ClipVisionH14"/>).</para>
///
/// <para><b>Conditioning paths:</b> TI2V-5B I2V pins the VAE-encoded first frame at timestep 0 (expand_timesteps);
/// Wan2.1 I2V instead concatenates a 36-channel <c>[noise, mask, cond-latent]</c> input and cross-attends to the CLIP
/// image context.</para>
///
/// <para><b>Wan2.2 A14B MoE expert pairs</b> (base A14B, AnimeGen-T2V, and other high/low-noise finetune pairs):
/// select the high-noise file as the main Model and the low-noise file in the <b>Refiner Model</b> slot (Swarm's
/// documented Wan 2.2 pair convention — <c>docs/Video Model Support.md</c> §Wan 2.2), or via <b>Video Swap Model</b>
/// (whose dropdown only lists I2V-class files; the refiner slot works for both). The engine's
/// <see cref="WanVideoPipeline"/> then runs the full MoE: high-noise expert while <c>timestep ≥ boundary·1000</c>,
/// low-noise below, with a single expert-swap at the crossing. The boundary defaults to Wan2.2's official
/// 0.875 (T2V) / 0.9 (I2V); a user-moved Refiner Control Percentage / Video Swap Percent maps through the shifted
/// flow schedule (see <see cref="ResolveBoundary"/>).</para>
/// </summary>
public static class WanVideoLoader
{
    public const string Wan22_5BCompatClassId = "wan-22-5b";
    public const string Wan21_1_3BCompatClassId = "wan-21-1_3b";
    public const string Wan21_14BCompatClassId = "wan-21-14b";

    /// <summary>Wan's umT5 context length (matches diffusers' 512-token encode).</summary>
    private const int TokenLength = 512;

    /// <summary>Resolves the low-noise expert of a Wan2.2 A14B-style MoE pair, or null for single-expert
    /// generations. The Refiner Model slot is checked first (Swarm's documented Wan 2.2 pair convention),
    /// then Video Swap Model (API parity / the I2V-group convention). Both the base and the expert must be
    /// Wan-14B compat class — anything else is not a pair and returns null (validation refuses mismatches).</summary>
    public static T2IModel ResolveLowNoiseModel(T2IModel model, T2IParamInput input)
    {
        if (input is null || model?.ModelClass?.CompatClass?.ID != Wan21_14BCompatClassId) return null;
        T2IModel refiner = input.Get(T2IParamTypes.RefinerModel);
        if (refiner?.ModelClass?.CompatClass?.ID == Wan21_14BCompatClassId) return refiner;
        T2IModel swap = input.Get(T2IParamTypes.VideoSwapModel);
        if (swap?.ModelClass?.CompatClass?.ID == Wan21_14BCompatClassId) return swap;
        return null;
    }

    /// <summary>The user's explicit expert-split override — the fraction of steps given to the low-noise
    /// expert — or null for "use the architecture default boundary". The registered defaults (Video Swap
    /// Percent 0.5, Refiner Control Percentage 0.2) count as "auto": Swarm sends group params whenever the
    /// group is open, so an untouched slider must not override Wan2.2's official boundary.</summary>
    private static float? ExplicitSwapFraction(T2IParamInput input)
    {
        if (input.TryGet(T2IParamTypes.VideoSwapPercent, out double swapPct) && swapPct != 0.5) return (float)swapPct;
        if (input.TryGet(T2IParamTypes.RefinerControl, out double refCtl) && refCtl > 0 && refCtl != 0.2) return (float)refCtl;
        return null;
    }

    /// <summary>Maps the expert split to the engine's timestep boundary (<see cref="WanVideoConfig.BoundaryRatio"/>).
    /// Untouched sliders → Wan2.2's official 0.875 (T2V) / 0.9 (I2V). An explicit fraction p converts through the
    /// shifted flow schedule — boundary = s·p/(1+(s−1)·p), the same warp the UniPC sigmas use — so "fraction of
    /// steps for the low expert" lands on the matching timestep (p=0.5 at shift 8 ≈ 0.889, consistent with the
    /// official defaults).</summary>
    private static float ResolveBoundary(T2IParamInput input, bool isConcatI2V)
    {
        float? p = ExplicitSwapFraction(input);
        if (p is null) return isConcatI2V ? 0.9f : 0.875f;
        float shift = input.TryGet(T2IParamTypes.SigmaShift, out double s) ? (float)s : 8f;
        float frac = Math.Clamp(p.Value, 0.01f, 0.99f);
        return shift * frac / (1f + (shift - 1f) * frac);
    }

    /// <summary>Cache key for a Wan generation: the base model name alone for single-expert runs, extended
    /// with the low-noise expert name + resolved boundary for MoE pairs — so changing the pair or the split
    /// reloads instead of silently reusing a stale single-expert pipeline. Safe to call for any architecture
    /// (non-Wan-14B and null input just return <c>model.Name</c>).</summary>
    public static string EffectiveCacheKey(T2IModel model, T2IParamInput input)
    {
        T2IModel low = ResolveLowNoiseModel(model, input);
        if (low is null) return model.Name;
        float? p = ExplicitSwapFraction(input);
        string split = p is null ? "auto"
            : ResolveBoundary(input, isConcatI2V: false).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
        return $"{model.Name}::moe::{low.Name}::b{split}";
    }

    public static WanVideoCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Wan video model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Wan video checkpoint not found: {model.RawFilePath}");

        string compat = model.ModelClass?.CompatClass?.ID ?? Wan22_5BCompatClassId;
        bool isWan21 = compat is Wan21_1_3BCompatClassId or Wan21_14BCompatClassId;

        T2IModel umt5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel),
            entry: SideModels.Umt5Xxl, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: isWan21 ? SideModels.Wan21Vae : SideModels.Wan22Vae, log: log);

        // ── 1. Load + convert the Wan DiT (original naming → diffusers) ──
        log($"Loading Wan DiT: {model.Name} (compat {compat})");
        var (conv, ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            ditLoader.Dispose();
            throw new InvalidOperationException(
                $"Wan checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        bool isClipI2V = conv.Transformer.ContainsKey("condition_embedder.image_embedder.norm1.weight");
        int inChannels = conv.Transformer.TryGetValue("patch_embedding.weight", out Tensor pe) ? (int)pe.Shape[1] : 0;
        WanVideoConfig config = ResolveConfig(compat, isClipI2V, inChannels);
        // No CFG renormalization: ComfyUI runs the same fp8 checkpoints with plain CFG and stays clean.
        // The forced 0.7 renorm (2026-07-08, fp8 "DC bias" band-aid) std-matched the whole velocity tensor
        // and shifted the palette/texture without fixing the collapse — the real divergence was the sampling
        // shift (3.0 vs Comfy's 8.0, now the FlowShift default below).
        string mode = isClipI2V ? "CLIP-I2V" : config.InChannels > config.VaeLatentChannels ? "concat-I2V" : "T2V/TI2V";
        log($"  Converted: {conv.Transformer.Count} transformer keys ({mode}, in {inChannels}, inner {config.InnerDim}{(config.CfgRescale > 0 ? $", cfg-renorm {config.CfgRescale}" : "")})");

        // ── 1b. Wan2.2 A14B MoE: optional low-noise expert (Refiner slot per Swarm's Wan 2.2 pair convention,
        // or Video Swap Model). BoundaryRatio > 0 makes WanVideoPipeline run the high-noise expert while
        // timestep ≥ boundary·1000 and swap to the low-noise expert (freeing the other's VRAM) below it.
        T2IModel lowNoiseModel = ResolveLowNoiseModel(model, input);
        if (lowNoiseModel is not null)
        {
            if (string.IsNullOrWhiteSpace(lowNoiseModel.RawFilePath) || !File.Exists(lowNoiseModel.RawFilePath))
                throw new FileNotFoundException($"Wan low-noise expert checkpoint not found: {lowNoiseModel.RawFilePath}");
            if (WanModelVariants.Detect(lowNoiseModel) != WanModelVariants.Variant.Base)
                throw new InvalidOperationException(
                    $"Wan low-noise expert '{lowNoiseModel.Name}' is a VACE/Animate/S2V variant — the Wan2.2 expert pair needs plain T2V/I2V checkpoints.");
            config = config with { BoundaryRatio = ResolveBoundary(input, isConcatI2V: config.InChannels > config.VaeLatentChannels) };
        }

        WanVideoTransformer transformer = new WanVideoTransformer(config);
        transformer.LoadWeights(conv.Transformer);
        WanVideoTransformer transformer2 = null;
        Dictionary<string, Tensor> transformer2Weights = null;
        SafeTensorsLoader lowNoiseLoader = null;

        try
        {
            // ── 1c. Low-noise expert DiT (architecturally identical to the high-noise one) ──
            if (lowNoiseModel is not null)
            {
                log($"Loading Wan low-noise expert: {lowNoiseModel.Name} (boundary {config.BoundaryRatio:0.###})");
                (WanVideoCheckpointConverter.ConvertedWeights convLow, SafeTensorsLoader lowLoader) =
                    WanVideoCheckpointConverter.LoadAndConvert(lowNoiseModel.RawFilePath);
                lowNoiseLoader = lowLoader;
                if (convLow.Transformer.Count == 0)
                    throw new InvalidOperationException(
                        $"Wan low-noise expert '{lowNoiseModel.Name}' has no recognized transformer weights after conversion.");
                transformer2 = new WanVideoTransformer(config);
                transformer2.LoadWeights(convLow.Transformer);
                transformer2Weights = convLow.Transformer;
            }

            // ── 2. VAE (decoder + encoder share one weight dict; cast to F32) ──
            log($"Loading Wan VAE: {vaeModel.Name}");
            var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            IWanVaeDecoder vae;
            IWanVaeEncoder vaeEncoder;
            if (isWan21)
            {
                Wan21VaeDecoder d = new Wan21VaeDecoder(); d.LoadWeights(vaeWeights); vae = d;
                Wan21VaeEncoder e = new Wan21VaeEncoder(); e.LoadWeights(vaeWeights); vaeEncoder = e;
            }
            else
            {
                Wan22VaeDecoder d = new Wan22VaeDecoder(); d.LoadWeights(vaeWeights); vae = d;
                Wan22VaeEncoder e = new Wan22VaeEncoder(); e.LoadWeights(vaeWeights); vaeEncoder = e;
            }

            // ── 3. CLIP-ViT-H image encoder (Wan2.1 I2V only) ──
            ClipVisionEncoder clipVision = null;
            SafeTensorsLoader clipLoader = null;
            if (isClipI2V)
            {
                T2IModel clipModel = ModelAutoDownloader.EnsureSideModel(
                    userPick: input?.Get(T2IParamTypes.ClipVisionModel), entry: SideModels.ClipVisionH14, log: log);
                log($"Loading CLIP-ViT-H image encoder: {clipModel.Name}");
                clipLoader = new SafeTensorsLoader();
                clipLoader.Load(clipModel.RawFilePath);
                clipVision = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
                clipVision.LoadWeights(clipLoader.GetAllTensors());
            }

            // ── 4. umT5-XXL (fp8-scaled folded to plain dtype) ──
            log($"Loading umT5-XXL: {umt5Model.Name}");
            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Model.RawFilePath);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);

            // ── 5. Tokenizer (embedded umT5 256k SentencePiece) ──
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: TokenLength);

            log("Building Wan video pipeline...");
            WanVideoPipeline pipeline = new WanVideoPipeline(backend, transformer, vae, config, vaeEncoder, transformer2);

            log($"Wan ready ({compat}, {(isClipI2V ? "CLIP image-to-video" : "text/image-to-video")}{(transformer2 is not null ? ", MoE expert pair" : "")}).");
            return new WanVideoCacheEntry
            {
                ModelName = EffectiveCacheKey(model, input),
                CompatClass = compat,
                Pipeline = pipeline,
                Config = config,
                IsClipI2V = isClipI2V,
                Tokenizer = tokenizer,
                Umt5 = umt5,
                Transformer = transformer,
                TransformerWeights = conv.Transformer,
                Transformer2 = transformer2,
                Transformer2Weights = transformer2Weights,
                LowNoiseLoader = lowNoiseLoader,
                Vae = vae,
                VaeEncoder = vaeEncoder,
                ClipVision = clipVision,
                CheckpointLoader = ditLoader,
                VaeLoaders = vaeLoaders,
                Umt5Loader = umt5Loader,
                ClipLoader = clipLoader,
            };
        }
        catch
        {
            transformer.Dispose();
            transformer2?.Dispose();
            lowNoiseLoader?.Dispose();
            ditLoader.Dispose();
            throw;
        }
    }

    /// <summary>Maps a SwarmUI Wan compat class (+ the DiT's CLIP-image-embedder presence and patch-embed in_channels)
    /// to the engine config preset. 14B with in_channels 36 is I2V: CLIP keys ⇒ Wan2.1 I2V-14B, otherwise the
    /// Wan2.2 A14B I2V (concat-only); in_channels 16 ⇒ T2V (Wan2.1-14B or an A14B T2V expert).</summary>
    private static WanVideoConfig ResolveConfig(string compat, bool isClipI2V, int inChannels) => compat switch
    {
        Wan21_1_3BCompatClassId => WanVideoConfig.T2V_1_3B,
        Wan21_14BCompatClassId => inChannels == 36
            ? (isClipI2V ? WanVideoConfig.I2V_14B_480p : WanVideoConfig.I2V_A14B)
            : WanVideoConfig.T2V_14B,
        _ => WanVideoConfig.Ti2V5B,
    };

    public static Image[] Generate(
        WanVideoCacheEntry entry, IBackend backend, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel) =>
        RunPipeline(entry.Pipeline, entry, backend, input, onProgress, cancel);

    /// <summary>LoRA path: clone the cached DiT weights, merge the stack, run a fresh transformer + pipeline.
    /// For a MoE pair the same stack merges into BOTH experts (the Wan2.2 community convention — e.g. the
    /// lightx2v Lightning pairs ship one file per expert but plain single-file LoRAs target both).</summary>
    public static Image[] GenerateWithLoras(
        WanVideoCacheEntry entry, IReadOnlyList<LoraResolver.LoraSpec> loras, IBackend backend, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel)
    {
        // A KEEP_MODELS-resident base DiT from a prior no-LoRA generation can't coexist with the merged
        // transformer's preload — free it up front (the next no-LoRA generation re-uploads it).
        backend.FreeWeights(entry.Transformer.EnumerateWeights());
        if (entry.Transformer2 is not null) backend.FreeWeights(entry.Transformer2.EnumerateWeights());
        backend.TrimMemoryPool();
        Dictionary<string, Tensor> transformerWeights = LoraApplier.ShallowClone(entry.TransformerWeights);
        LoraStack stack = LoraApplier.BuildAndApply(loras, backend, transformerWeights: transformerWeights);
        WanVideoTransformer transformer = new WanVideoTransformer(entry.Config);
        WanVideoTransformer transformer2 = null;
        LoraStack stack2 = null;
        try
        {
            transformer.LoadWeights(transformerWeights);
            if (entry.Transformer2 is not null)
            {
                Dictionary<string, Tensor> lowWeights = LoraApplier.ShallowClone(entry.Transformer2Weights);
                stack2 = LoraApplier.BuildAndApply(loras, backend, transformerWeights: lowWeights);
                transformer2 = new WanVideoTransformer(entry.Config);
                transformer2.LoadWeights(lowWeights);
            }
            using WanVideoPipeline pipeline = new WanVideoPipeline(backend, transformer, entry.Vae, entry.Config, entry.VaeEncoder, transformer2);
            return RunPipeline(pipeline, entry, backend, input, onProgress, cancel);
        }
        finally
        {
            transformer?.Dispose();
            transformer2?.Dispose();
            stack?.Dispose();
            stack2?.Dispose();
        }
    }

    /// <summary>Zeroes embedding rows past the real tokens (content + EOS; pad id 0), matching the
    /// reference Wan pipelines which zero-pad the umT5 output to the 512-row context. See the call
    /// site in <see cref="RunPipeline"/> for why this is load-bearing.</summary>
    internal static unsafe void ZeroPaddedRows(Tensor embeds, int[] tokens, int dim)
    {
        int realLen = 0;
        while (realLen < tokens.Length && tokens[realLen] != 0) realLen++;
        int rows = (int)(embeds.Shape.ElementCount / dim);
        if (realLen >= rows) return;
        float* p = (float*)embeds.DataPointer;
        new Span<float>(p + (long)realLen * dim, (rows - realLen) * dim).Clear();
    }

    private static Image[] RunPipeline(
        WanVideoPipeline pipeline, WanVideoCacheEntry entry, IBackend backend, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel)
    {
        string prompt = PromptConditioningResolver.VideoText(input.Get(T2IParamTypes.Prompt));
        string negative = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.NegativePrompt));
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 81, step: entry.Config.VaeTemporalCompression);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        Image initImage = input.Get(T2IParamTypes.InitImage);
        int width, height;
        if (initImage is not null)
        {
            var (imgW, imgH) = RgbToImage.GetDimensions(initImage);
            (width, height) = VideoParamResolver.ResolveI2VResolution(
                input, input.Get(T2IParamTypes.Model), imgW, imgH, multiple: entry.Config.VaeSpatialCompression);
            Logs.Verbose($"[HartsyInference][Wan] I2V init image {imgW}x{imgH} → clip {width}x{height}.");
        }
        else
        {
            (width, height) = VideoParamResolver.ResolveResolution(input, multiple: entry.Config.VaeSpatialCompression);
        }

        // Encode the prompt pair — cached across generations keyed on the token ids (the LTX/Flux
        // prompt-cache pattern): a HIT skips the whole umT5 phase including its multi-GB weight upload.
        // The cached tensors are host-materialized (built host-side + ZeroPaddedRows touches them), so
        // they survive the pipeline's per-step FreeActivations sweeps and re-fault to device on use.
        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        bool textHit = entry.CachedPromptTokens is not null && entry.CachedNegTokens is not null
            && promptTokens.AsSpan().SequenceEqual(entry.CachedPromptTokens)
            && negTokens.AsSpan().SequenceEqual(entry.CachedNegTokens);
        Tensor promptEmbeds, negEmbeds;
        if (textHit)
        {
            promptEmbeds = entry.CachedPromptEmbeds;
            negEmbeds = entry.CachedNegEmbeds;
            Logs.Info("[HartsyInference][Wan] [wan-phase] umT5 prompt cache HIT — text encode skipped.");
        }
        else
        {
            long teStart = Environment.TickCount64;
            // The umT5 upload may not fit beside a KEEP_MODELS-resident DiT from a prior generation —
            // decide from measured free VRAM and evict the DiT when short (the denoise re-uploads it).
            EnsureEncoderHeadroom(backend, entry, entry.Umt5.EnumerateWeights(), "umT5");
            Tensor batch = entry.Umt5.Encode(backend,
                [promptTokens, negTokens],
                [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
            promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, TokenLength, entry.Config.TextDim);
            negEmbeds = CfgHelper.SliceBatchElement(batch, 1, TokenLength, entry.Config.TextDim);
            batch.Dispose();
            // Wan's DiT cross-attends over all 512 context rows with NO text mask — the reference
            // (diffusers WanPipeline / Comfy) zero-pads embeddings past the real tokens. umT5 emits
            // garbage at pad positions; leaving it in drowns the prompt and denoises to a flat clip.
            ZeroPaddedRows(promptEmbeds, promptTokens, entry.Config.TextDim);
            ZeroPaddedRows(negEmbeds, negTokens, entry.Config.TextDim);
            backend.Sync();
            backend.FreeWeights(entry.Umt5.EnumerateWeights());
            entry.CachedPromptEmbeds?.Dispose();
            entry.CachedNegEmbeds?.Dispose();
            entry.CachedPromptEmbeds = promptEmbeds;
            entry.CachedNegEmbeds = negEmbeds;
            entry.CachedPromptTokens = promptTokens;
            entry.CachedNegTokens = negTokens;
            Logs.Info($"[HartsyInference][Wan] [wan-phase] umT5 prompt cache MISS — encode+free {Environment.TickCount64 - teStart}ms.");
        }

        // Sigma Shift: honor the user's param when set; otherwise default to ComfyUI's Wan sampling shift
        // (8.0, comfy supported_models WAN21_T2V — inherited by I2V/VACE) rather than the official-repo
        // 3.0/5.0 presets. At Swarm's default step counts (15-20) the lower shift under-forms structure and
        // prints high-frequency comb texture; 8.0 is the reference look Wan users get through ComfyUI.
        float flowShift = input.TryGet(T2IParamTypes.SigmaShift, out double sigmaShift) ? (float)sigmaShift : 8f;
        VideoGenerationRequest request = new VideoGenerationRequest
        {
            Prompt = prompt, NegativePrompt = negative, Width = width, Height = height,
            Steps = steps, CfgScale = cfgScale, Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
            FlowShift = flowShift,
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p => { cancel.ThrowIfCancellationRequested(); onProgress(p); };

        // promptEmbeds/negEmbeds/imageEmbeds are intentionally never disposed here — they are the
        // cross-generation caches (tiny, host-materialized; freed on the next cache miss or in
        // WanVideoCacheEntry.Dispose).
        {
            // Concat-conditioned I2V (Wan2.1 I2V-14B with CLIP, or Wan2.2 I2V-A14B without): 36-ch
            // [noise, mask, cond-latent] input. CLIP embeds are added only when the variant has an image embedder.
            bool isConcatI2V = entry.Config.InChannels > entry.Config.VaeLatentChannels;
            if (isConcatI2V && initImage is not null)
            {
                // CLIP image embeddings cached keyed on the raw init-image bytes — a same-image repeat
                // skips the CLIP-ViT-H upload+encode (the embed depends only on the image; the 224²
                // preprocess is resolution-independent of the target clip size).
                Tensor imageEmbeds = null;
                if (entry.IsClipI2V && entry.ClipVision is not null)
                {
                    string imageKey = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(initImage.RawData));
                    if (entry.CachedImageEmbeds is not null && imageKey == entry.CachedImageKey)
                    {
                        imageEmbeds = entry.CachedImageEmbeds;
                        Logs.Info("[HartsyInference][Wan] [wan-phase] CLIP image cache HIT — vision encode skipped.");
                    }
                    else
                    {
                        long clipStart = Environment.TickCount64;
                        EnsureEncoderHeadroom(backend, entry, entry.ClipVision.EnumerateWeights(), "CLIP-ViT-H");
                        backend.PreloadWeights(entry.ClipVision.EnumerateWeights());
                        Tensor pixels = ClipImagePreprocessor.Process(initImage, imageSize: 224);
                        Tensor imageEmbedsBatched = entry.ClipVision.EncodeHiddenStates(backend, pixels);   // [1, 257, 1280]
                        pixels.Dispose();
                        backend.Sync();
                        backend.FreeWeights(entry.ClipVision.EnumerateWeights());
                        imageEmbeds = DropBatch(imageEmbedsBatched);   // host copy — survives activation sweeps
                        imageEmbedsBatched.Dispose();
                        entry.CachedImageEmbeds?.Dispose();
                        entry.CachedImageEmbeds = imageEmbeds;
                        entry.CachedImageKey = imageKey;
                        Logs.Info($"[HartsyInference][Wan] [wan-phase] CLIP image cache MISS — encode+free {Environment.TickCount64 - clipStart}ms.");
                    }
                }

                byte[] frameRgb = RgbToImage.ToHwcRgbResized(initImage, width, height);
                (byte[][] f2, int w2, int h2, _) = pipeline.GenerateImageToVideoConcat(
                    promptEmbeds, negEmbeds, imageEmbeds, frameRgb, request, numFrames, bridge);
                Logs.Verbose($"[HartsyInference][Wan] Concat-I2V returned {f2.Length} frames {w2}x{h2} in {Environment.TickCount64 - start}ms.");
                return new[] { VideoParamResolver.FinishVideo(f2, w2, h2, input, cancel) };
            }
            if (isConcatI2V && initImage is null)
            {
                input.RefusalReasons?.Add("HartsyInference: this Wan I2V model requires an Init Image.");
            }

            // TI2V-5B expand_timesteps I2V (first-frame latent) or plain T2V.
            Tensor firstFrameLatent = null;
            if (initImage is not null && !isConcatI2V)
            {
                byte[] frameRgb = RgbToImage.ToHwcRgbResized(initImage, width, height);
                firstFrameLatent = entry.VaeEncoder.EncodeRgbFrame(backend, frameRgb, width, height);
                backend.Sync();
                backend.FreeWeights(entry.VaeEncoder.EnumerateWeights());
            }
            try
            {
                var (frames, outW, outH, _) = pipeline.GenerateFromEmbeddings(
                    promptEmbeds, negEmbeds, request, numFrames, bridge, firstFrameLatent);
                Logs.Verbose($"[HartsyInference][Wan] Pipeline returned {frames.Length} frames {outW}x{outH} " +
                    $"({(firstFrameLatent is null ? "T2V" : "I2V")}) in {Environment.TickCount64 - start}ms.");
                return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
            }
            finally { firstFrameLatent?.Dispose(); }
        }
    }

    /// <summary>Frees the (possibly KEEP_MODELS-resident) DiT device weights before an encoder upload when the encoder doesn't fit beside it (measured free VRAM, 2 GB margin); trims the pool first so slack from the previous generation doesn't make the reading pessimistic.</summary>
    private static void EnsureEncoderHeadroom(IBackend backend, WanVideoCacheEntry entry, IEnumerable<Tensor> encoderWeights, string label)
    {
        backend.TrimMemoryPool();
        long need = 0;
        foreach (Tensor t in encoderWeights) need += t.DType.ComputeByteCount(t.ElementCount);
        long free = backend.FreeMemoryBytes();
        if (free < need + (2L << 30))
        {
            Logs.Info($"[HartsyInference][Wan] [wan-phase] {label} upload: evicting resident DiT " +
                $"(free {free >> 20} MB < {need >> 20} MB + 2048 MB margin).");
            backend.FreeWeights(entry.Transformer.EnumerateWeights());
            if (entry.Transformer2 is not null) backend.FreeWeights(entry.Transformer2.EnumerateWeights());
            backend.TrimMemoryPool();
        }
        else
        {
            Logs.Verbose($"[HartsyInference][Wan] [wan-phase] {label} fits beside the resident DiT " +
                $"(free {free >> 20} MB ≥ {need >> 20} MB + 2048 MB margin).");
        }
    }

    /// <summary>Copies a <c>[1, seq, dim]</c> tensor to a <c>[seq, dim]</c> tensor (the pipeline's image-embeds shape).</summary>
    internal static unsafe Tensor DropBatch(Tensor x)
    {
        int seq = (int)x.Shape[1], dim = (int)x.Shape[2];
        Tensor o = new Tensor(new TensorShape(seq, dim), DType.F32);
        long bytes = (long)seq * dim * 4;
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }
}

public sealed class WanVideoCacheEntry : IDisposable
{
    /// <summary>Cache key (<see cref="WanVideoLoader.EffectiveCacheKey"/>): the base model name, extended
    /// with the low-noise expert + boundary for Wan2.2 MoE pairs.</summary>
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required WanVideoPipeline Pipeline { get; init; }
    public required WanVideoConfig Config { get; init; }
    public required bool IsClipI2V { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder Umt5 { get; init; }
    public required WanVideoTransformer Transformer { get; init; }

    /// <summary>Converted (diffusers-named) DiT weight dict, retained for per-generation LoRA merging
    /// (<see cref="LoraApplier.ShallowClone"/> before mutating).</summary>
    public required Dictionary<string, Tensor> TransformerWeights { get; init; }

    /// <summary>Low-noise expert DiT (Wan2.2 A14B MoE pair); null for single-expert checkpoints.</summary>
    public WanVideoTransformer Transformer2 { get; init; }

    /// <summary>Converted weight dict of the low-noise expert, retained for LoRA merging (null when single-expert).</summary>
    public Dictionary<string, Tensor> Transformer2Weights { get; init; }

    /// <summary>mmap loader backing the low-noise expert (null when single-expert).</summary>
    public SafeTensorsLoader LowNoiseLoader { get; init; }
    public required IWanVaeDecoder Vae { get; init; }
    public required IWanVaeEncoder VaeEncoder { get; init; }
    public ClipVisionEncoder ClipVision { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required IReadOnlyList<SafeTensorsLoader> VaeLoaders { get; init; }
    public required SafeTensorsLoader Umt5Loader { get; init; }
    public SafeTensorsLoader ClipLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;

    /// <summary>Cross-generation umT5 prompt cache: token-id keys plus the two zero-padded host embedding tensors. A hit skips the whole umT5 upload+encode phase.</summary>
    public int[] CachedPromptTokens { get; set; }
    public int[] CachedNegTokens { get; set; }
    public Tensor CachedPromptEmbeds { get; set; }
    public Tensor CachedNegEmbeds { get; set; }

    /// <summary>Cross-generation CLIP image-embedding cache keyed on the SHA-256 of the raw init-image bytes (Wan2.1 CLIP-I2V only).</summary>
    public string CachedImageKey { get; set; }
    public Tensor CachedImageEmbeds { get; set; }

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        CachedPromptEmbeds?.Dispose();
        CachedNegEmbeds?.Dispose();
        CachedImageEmbeds?.Dispose();
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        Umt5?.Dispose();
        Transformer?.Dispose();
        Transformer2?.Dispose();
        CheckpointLoader?.Dispose();
        LowNoiseLoader?.Dispose();
        Umt5Loader?.Dispose();
        ClipLoader?.Dispose();
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
