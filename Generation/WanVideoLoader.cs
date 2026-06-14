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
/// Loads Wan2.2 TI2V-5B (Wan-AI's text/image-to-video DiT; SwarmUI compat class
/// <c>wan-22-5b</c>, detected from the Comfy-Org repackaged single file
/// <c>wan2.2_ti2v_5B_fp16.safetensors</c>).
///
/// Required side models (auto-downloaded; user picks via SwarmUI parameters take priority):
///   - <see cref="T2IParamTypes.T5XXLModel"/>: umT5-XXL (Comfy's canonical fp8-scaled file —
///     <see cref="SideModels.Umt5Xxl"/>). NOT the same as plain T5-XXL: 256k multilingual vocab
///     with per-layer relative attention bias. Pairs with the embedded umT5 tokenizer.
///   - <see cref="T2IParamTypes.VAE"/>: the Wan2.2 48-channel video VAE (<see cref="SideModels.Wan22Vae"/>,
///     shared with SwarmUI core's "wan22-vae" CommonModels entry).
///
/// Supports text-to-video AND image-to-video: with an Init Image set, the frame is fitted to the
/// <c>VideoResolution</c> target, VAE-encoded via <c>Wan22VaeEncoder</c>, and passed to the pipeline's
/// TI2V first-frame conditioning path (diffusers <c>expand_timesteps</c> — frame 0 pinned at
/// timestep 0 while the remaining frames denoise).
/// </summary>
public static class WanVideoLoader
{
    public const string Wan22_5BCompatClassId = "wan-22-5b";

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

        T2IModel umt5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel),
            entry: SideModels.Umt5Xxl,
            log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: SideModels.Wan22Vae,
            log: log);

        // ── 1. Load + convert the Wan DiT (original naming → diffusers) ──
        log($"Loading Wan2.2 DiT: {model.Name}");
        var (conv, ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (conv.Transformer.Count == 0)
        {
            ditLoader.Dispose();
            throw new InvalidOperationException(
                $"Wan checkpoint '{model.Name}' has no recognized transformer weights after conversion.");
        }
        log($"  Converted: {conv.Transformer.Count} transformer keys");

        WanVideoConfig config = WanVideoConfig.Ti2V5B;
        WanVideoTransformer transformer = new WanVideoTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        // ── 2. Load the Wan2.2 VAE — decoder + encoder share one weight dict (the Comfy-Org
        // repackage ships F16; the Wan22 blocks compute in F32, so cast up front) ──
        log($"Loading Wan2.2 VAE: {vaeModel.Name}");
        var (vaeWeightsRaw, vaeLoaders) = LanceCheckpointConverter.LoadVae(vaeModel.RawFilePath);
        Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
        Wan22VaeDecoder vae = new Wan22VaeDecoder();
        vae.LoadWeights(vaeWeights);
        Wan22VaeEncoder vaeEncoder = new Wan22VaeEncoder();
        vaeEncoder.LoadWeights(vaeWeights);

        // ── 3. Load umT5-XXL (fp8-scaled weights folded to plain dtype at load) ──
        log($"Loading umT5-XXL: {umt5Model.Name}");
        SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
        umt5Loader.Load(umt5Model.RawFilePath);
        Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
        if (umt5Weights.Count == 0)
        {
            umt5Loader.Dispose();
            foreach (var l in vaeLoaders) l.Dispose();
            ditLoader.Dispose();
            throw new InvalidOperationException($"umT5 model file '{umt5Model.Name}' has no usable tensors.");
        }
        T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
        umt5.LoadWeights(umt5Weights);

        // ── 4. Tokenizer (embedded umT5 256k SentencePiece) ──
        log("Loading umT5 tokenizer (embedded)...");
        T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: TokenLength);

        log("Building Wan video pipeline...");
        WanVideoPipeline pipeline = new WanVideoPipeline(backend, transformer, vae, config);

        log("Wan2.2 TI2V-5B ready (text-to-video + image-to-video).");
        return new WanVideoCacheEntry
        {
            ModelName = model.Name,
            CompatClass = Wan22_5BCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Tokenizer = tokenizer,
            Umt5 = umt5,
            Transformer = transformer,
            TransformerWeights = conv.Transformer,
            Vae = vae,
            VaeEncoder = vaeEncoder,
            CheckpointLoader = ditLoader,
            VaeLoaders = vaeLoaders,
            Umt5Loader = umt5Loader,
        };
    }

    public static Image[] Generate(
        WanVideoCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        return RunPipeline(entry.Pipeline, entry, backend, input, onProgress, cancel);
    }

    /// <summary>LoRA path: shallow-clone the cached DiT weight dict, merge the LoRA stack into it
    /// (HartsyInference's LoraStack auto-detects kohya/musubi, Comfy diffusion_model., and
    /// diffusers-PEFT key formats), and run a fresh transformer + pipeline for this generation.
    /// The cached no-LoRA pipeline stays untouched; merged tensors die with the stack. Wan LoRAs
    /// target the DiT only (the model class declares LorasTargetTextEnc=false), so umT5 and the
    /// VAE are reused from the cache entry as-is.</summary>
    public static Image[] GenerateWithLoras(
        WanVideoCacheEntry entry,
        IReadOnlyList<LoraResolver.LoraSpec> loras,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        Dictionary<string, Tensor> transformerWeights = LoraApplier.ShallowClone(entry.TransformerWeights);
        LoraStack stack = LoraApplier.BuildAndApply(loras, backend, transformerWeights: transformerWeights);
        WanVideoTransformer transformer = new WanVideoTransformer(entry.Config);
        try
        {
            transformer.LoadWeights(transformerWeights);
            using WanVideoPipeline pipeline = new WanVideoPipeline(backend, transformer, entry.Vae, entry.Config);
            return RunPipeline(pipeline, entry, backend, input, onProgress, cancel);
        }
        finally
        {
            // Stack last so merged tensors outlive the transformer that referenced them via LoadWeights.
            transformer?.Dispose();
            stack?.Dispose();
        }
    }

    /// <summary>Shared per-pipeline driver — same logic whether the pipeline is the cached one
    /// (no-LoRA) or a freshly-built per-gen one (LoRA).</summary>
    private static Image[] RunPipeline(
        WanVideoPipeline pipeline,
        WanVideoCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        int numFrames = VideoParamResolver.ResolveFrames(input, modelDefault: 81, step: entry.Config.VaeTemporalCompression);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfgScale = cfgRaw <= 0 ? entry.Config.GuidanceScale : (float)cfgRaw;

        // I2V: size the clip from the init image + VideoResolution mode; T2V: from the standard
        // Width/Height params. Both snapped to the VAE's 16-multiple.
        SwarmUI.Utils.Image initImage = input.Get(T2IParamTypes.InitImage);
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

        // Encode the prompt pair, then drop the encoder's GPU weights before the DiT preload —
        // the 5B transformer needs the VRAM headroom (mirrors the upstream Wan E2E test).
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

        // I2V conditioning: fit the init image to the clip size, VAE-encode to the normalized
        // [1,48,1,H/16,W/16] first-frame latent, then drop the encoder's GPU weights too.
        Tensor firstFrameLatent = null;
        if (initImage is not null)
        {
            byte[] frameRgb = RgbToImage.ToHwcRgbResized(initImage, width, height);
            firstFrameLatent = entry.VaeEncoder.EncodeRgbFrame(backend, frameRgb, width, height);
            backend.Sync();
            backend.FreeWeights(entry.VaeEncoder.EnumerateWeights());
        }

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
            var (frames, outW, outH, _) = pipeline.GenerateFromEmbeddings(
                promptEmbeds, negEmbeds, request, numFrames, bridge, firstFrameLatent);
            Logs.Verbose($"[HartsyInference][Wan] Pipeline returned {frames.Length} frames {outW}x{outH} " +
                $"({(firstFrameLatent is null ? "T2V" : "I2V")}) in {Environment.TickCount64 - start}ms.");
            return new[] { VideoParamResolver.FinishVideo(frames, outW, outH, input, cancel) };
        }
        finally
        {
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
            firstFrameLatent?.Dispose();
        }
    }
}

public sealed class WanVideoCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required WanVideoPipeline Pipeline { get; init; }
    public required WanVideoConfig Config { get; init; }
    public required T5Tokenizer Tokenizer { get; init; }
    public required T5TextEncoder Umt5 { get; init; }
    public required WanVideoTransformer Transformer { get; init; }

    /// <summary>Converted (diffusers-named) DiT weight dict, retained for per-generation LoRA
    /// merging — tensor refs over <see cref="CheckpointLoader"/>'s mmap, so the cost is the dict
    /// shell only. <see cref="LoraApplier.ShallowClone"/> before mutating.</summary>
    public required Dictionary<string, Tensor> TransformerWeights { get; init; }
    public required Wan22VaeDecoder Vae { get; init; }
    public required Wan22VaeEncoder VaeEncoder { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required IReadOnlyList<SafeTensorsLoader> VaeLoaders { get; init; }
    public required SafeTensorsLoader Umt5Loader { get; init; }

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
        if (VaeLoaders is not null)
        {
            foreach (SafeTensorsLoader loader in VaeLoaders) loader?.Dispose();
        }
    }
}
