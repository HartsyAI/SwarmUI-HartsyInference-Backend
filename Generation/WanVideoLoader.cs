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
/// image context. <b>A14B MoE caveat:</b> SwarmUI selects a single model file, so an A14B expert runs as a single
/// transformer here (no high/low-noise boundary switch — that needs both expert files; the engine's
/// <see cref="WanVideoPipeline"/> supports the full MoE when given <c>transformer2</c>). Numerics validation-pending.</para>
/// </summary>
public static class WanVideoLoader
{
    public const string Wan22_5BCompatClassId = "wan-22-5b";
    public const string Wan21_1_3BCompatClassId = "wan-21-1_3b";
    public const string Wan21_14BCompatClassId = "wan-21-14b";

    /// <summary>Wan's umT5 context length (matches diffusers' 512-token encode).</summary>
    private const int TokenLength = 512;

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
        // fp8 checkpoints carry a small velocity DC bias that CFG>=5 amplifies into a color-drifting /
        // dark trajectory over the clip; CFG renormalization (~0.7) corrects it. Mirrors the engine's
        // WanConfigDetector fp8 auto-detect, which the preset-based ResolveConfig path bypasses.
        if (conv.Transformer.TryGetValue("blocks.0.ffn.net.0.proj.weight", out Tensor ffn0) && ffn0.DType.IsFp8)
        {
            config = config with { CfgRescale = 0.7f };
        }
        string mode = isClipI2V ? "CLIP-I2V" : config.InChannels > config.VaeLatentChannels ? "concat-I2V" : "T2V/TI2V";
        log($"  Converted: {conv.Transformer.Count} transformer keys ({mode}, in {inChannels}, inner {config.InnerDim}{(config.CfgRescale > 0 ? $", cfg-renorm {config.CfgRescale}" : "")})");

        WanVideoTransformer transformer = new WanVideoTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        try
        {
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
            WanVideoPipeline pipeline = new WanVideoPipeline(backend, transformer, vae, config, vaeEncoder);

            log($"Wan ready ({compat}, {(isClipI2V ? "CLIP image-to-video" : "text/image-to-video")}).");
            return new WanVideoCacheEntry
            {
                ModelName = model.Name,
                CompatClass = compat,
                Pipeline = pipeline,
                Config = config,
                IsClipI2V = isClipI2V,
                Tokenizer = tokenizer,
                Umt5 = umt5,
                Transformer = transformer,
                TransformerWeights = conv.Transformer,
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

    /// <summary>LoRA path: clone the cached DiT weights, merge the stack, run a fresh transformer + pipeline.</summary>
    public static Image[] GenerateWithLoras(
        WanVideoCacheEntry entry, IReadOnlyList<LoraResolver.LoraSpec> loras, IBackend backend, T2IParamInput input,
        Action<GenerationProgress> onProgress, CancellationToken cancel)
    {
        Dictionary<string, Tensor> transformerWeights = LoraApplier.ShallowClone(entry.TransformerWeights);
        LoraStack stack = LoraApplier.BuildAndApply(loras, backend, transformerWeights: transformerWeights);
        WanVideoTransformer transformer = new WanVideoTransformer(entry.Config);
        try
        {
            transformer.LoadWeights(transformerWeights);
            using WanVideoPipeline pipeline = new WanVideoPipeline(backend, transformer, entry.Vae, entry.Config, entry.VaeEncoder);
            return RunPipeline(pipeline, entry, backend, input, onProgress, cancel);
        }
        finally
        {
            transformer?.Dispose();
            stack?.Dispose();
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
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
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

        // Encode the prompt pair, then free the encoder before the DiT preload.
        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);
        Tensor batch = entry.Umt5.Encode(backend,
            [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, TokenLength, entry.Config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, TokenLength, entry.Config.TextDim);
        batch.Dispose();
        // Wan's DiT cross-attends over all 512 context rows with NO text mask — the reference
        // (diffusers WanPipeline / Comfy) zero-pads embeddings past the real tokens. umT5 emits
        // garbage at pad positions; leaving it in drowns the prompt and denoises to a flat clip.
        ZeroPaddedRows(promptEmbeds, promptTokens, entry.Config.TextDim);
        ZeroPaddedRows(negEmbeds, negTokens, entry.Config.TextDim);
        backend.Sync();
        backend.FreeWeights(entry.Umt5.EnumerateWeights());

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt, NegativePrompt = negative, Width = width, Height = height,
            Steps = steps, CfgScale = cfgScale, Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p => { cancel.ThrowIfCancellationRequested(); onProgress(p); };

        try
        {
            // Concat-conditioned I2V (Wan2.1 I2V-14B with CLIP, or Wan2.2 I2V-A14B without): 36-ch
            // [noise, mask, cond-latent] input. CLIP embeds are added only when the variant has an image embedder.
            bool isConcatI2V = entry.Config.InChannels > entry.Config.VaeLatentChannels;
            if (isConcatI2V && initImage is not null)
            {
                Tensor imageEmbeds = null;
                if (entry.IsClipI2V && entry.ClipVision is not null)
                {
                    backend.PreloadWeights(entry.ClipVision.EnumerateWeights());
                    Tensor pixels = ClipImagePreprocessor.Process(initImage, imageSize: 224);
                    Tensor imageEmbedsBatched = entry.ClipVision.EncodeHiddenStates(backend, pixels);   // [1, 257, 1280]
                    pixels.Dispose();
                    backend.Sync();
                    backend.FreeWeights(entry.ClipVision.EnumerateWeights());
                    imageEmbeds = DropBatch(imageEmbedsBatched);
                    imageEmbedsBatched.Dispose();
                }

                byte[] frameRgb = RgbToImage.ToHwcRgbResized(initImage, width, height);
                (byte[][] f2, int w2, int h2, _) = pipeline.GenerateImageToVideoConcat(
                    promptEmbeds, negEmbeds, imageEmbeds, frameRgb, request, numFrames, bridge);
                imageEmbeds?.Dispose();
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
        finally
        {
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <summary>Copies a <c>[1, seq, dim]</c> tensor to a <c>[seq, dim]</c> tensor (the pipeline's image-embeds shape).</summary>
    private static unsafe Tensor DropBatch(Tensor x)
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
    public required IWanVaeDecoder Vae { get; init; }
    public required IWanVaeEncoder VaeEncoder { get; init; }
    public ClipVisionEncoder ClipVision { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required IReadOnlyList<SafeTensorsLoader> VaeLoaders { get; init; }
    public required SafeTensorsLoader Umt5Loader { get; init; }
    public SafeTensorsLoader ClipLoader { get; init; }

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
        CheckpointLoader?.Dispose();
        Umt5Loader?.Dispose();
        ClipLoader?.Dispose();
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
