using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Z-Image models. Z-Image checkpoints contain the transformer only; the Qwen3-4B
/// text encoder and VAE come from separate Swarm-registered models picked through the
/// normal parameter system:
///   - T2IParamTypes.QwenModel  (Models/clip/  — Comfy uses lumina2 type)
///   - T2IParamTypes.VAE        (Models/vae/   — Z-Image reuses the Flux VAE verbatim)
///
/// Mirrors how Comfy's WorkflowGeneratorModelSupport.cs:1210-1211 handles Z-Image.
/// </summary>
public static class ZImageLoader
{
    public const string ZImageCompatClassId = "z-image";

    /// <summary>Qwen3 right-pads EncodeChat output with BosTokenId (151643).</summary>
    private const int Qwen3PadTokenId = 151643;

    public static ZImageCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Z-Image model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Z-Image checkpoint not found: {model.RawFilePath}");

        // Resolve Qwen3-4B and the Flux VAE via the central SideModels registry — same
        // canonical filenames Comfy uses, so a side-by-side install ends up with the same
        // files in the same folders.
        T2IModel qwenModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.QwenModel),
            entry: SideModels.Qwen3_4B,
            log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: SideModels.FluxAe,
            log: log);

        // 1. Load Z-Image transformer.
        log($"Loading Z-Image transformer: {model.Name}");
        var (zConv, zLoader) = ZImageCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (zConv.Transformer.Count == 0)
        {
            zLoader.Dispose();
            throw new InvalidOperationException("Z-Image checkpoint has no transformer weights.");
        }

        var (numLayers, numRefiner, hidden, ffnDim, isFp8Mix) = ZImageCheckpointConverter.DetectArchitecture(zConv.Transformer);
        log($"Architecture: {numLayers} layers, {numRefiner} refiner, hidden={hidden}, ffn={ffnDim}, fp8_mix={isFp8Mix}");
        ZImageConfig zConfig = ZImageConfig.FromWeights(zConv.Transformer);

        log("Building Z-Image transformer...");
        ZImageTransformer transformer = new ZImageTransformer(zConfig);
        transformer.LoadWeights(zConv.Transformer);

        // 2. Load Qwen3-4B from Swarm-registered model.
        log($"Loading Qwen3-4B encoder weights: {qwenModel.Name}");
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenModel.RawFilePath);
        var qwenWeights = qwenLoader.GetAllTensors();
        if (qwenWeights.Count == 0)
        {
            qwenLoader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"Qwen3 model file '{qwenModel.Name}' has no tensors.");
        }

        log("Building Qwen3-4B encoder...");
        LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        qwen.LoadWeights(qwenWeights);

        log("Loading Qwen3 tokenizer (embedded)...");
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 256);

        // 3. Load VAE from Swarm-registered model. Z-Image reuses Flux VAE format.
        log($"Loading VAE: {vaeModel.Name}");
        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaeModel.RawFilePath);
        Dictionary<string, Tensor> vaeWeights = LoadVaeWeights(vaeLoader.GetAllTensors());
        if (vaeWeights.Count == 0)
        {
            vaeLoader.Dispose();
            qwenLoader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"VAE file '{vaeModel.Name}' has no usable VAE tensors.");
        }

        log("Building VAE decoder (Flux config)...");
        VaeDecoder vae = new VaeDecoder(VaeConfig.ZImage);
        vae.LoadWeights(vaeWeights);

        log("Building VAE encoder (img2img — Z-Image reuses Flux VAE)...");
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.ZImage);
        vaeEncoder.LoadWeights(vaeWeights);

        log("Building Z-Image pipeline...");
        ZImagePipeline pipeline = new ZImagePipeline(backend, transformer, vae, vaeEncoder, zConfig);

        log($"Z-Image ready (SchedulerShift={zConfig.SchedulerShift}).");
        return new ZImageCacheEntry
        {
            ModelName = model.Name,
            CompatClass = ZImageCompatClassId,
            Pipeline = pipeline,
            ZImageConfig = zConfig,
            Tokenizer = tokenizer,
            Qwen = qwen,
            Transformer = transformer,
            Vae = vae,
            VaeEncoder = vaeEncoder,
            ZImageCheckpointLoader = zLoader,
            QwenLoader = qwenLoader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        ZImageCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 8);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfg = cfgRaw <= 0 ? 1.0f : (float)cfgRaw;

        int[] tokenIds = entry.Tokenizer.EncodeChat(prompt);
        int realLen = ComputeRealLength(tokenIds);

        int penultimateIdx = entry.Qwen.NumLayers - 1;
        Tensor encodedFull = entry.Qwen.EncodeMultiLayer(backend, new[] { tokenIds }, new[] { penultimateIdx });
        Tensor positiveEmbeddings = SliceFirstSeqF32(encodedFull, realLen);

        Tensor negativeEmbeddings = null;
        if (cfg > 1.0f)
        {
            // Encode even when the negative is empty — Comfy does the same (passes "" through
            // the text encoder), which yields a short but valid unconditional embedding that
            // CFG needs. Z-Image-Base requires this; Z-Image-Turbo runs at cfg=1.0 and skips it.
            int[] negTokens = entry.Tokenizer.EncodeChat(negative);
            int negRealLen = ComputeRealLength(negTokens);
            Tensor negEncodedFull = entry.Qwen.EncodeMultiLayer(backend, new[] { negTokens }, new[] { penultimateIdx });
            negativeEmbeddings = SliceFirstSeqF32(negEncodedFull, negRealLen);
        }

        // Img2img: build an ImageToImageRequest if an init image is provided. The
        // upstream pipeline detects this on runtime type and switches behavior.
        // Caveat (per ZImagePipeline.cs): nonzero-strength Z-Image img2img produces
        // structurally-correct output but quality hasn't been validated against a
        // Python reference. Strength=0 pass-through is exact.
        Img2ImgResolver.Img2ImgSpec img2img = Img2ImgResolver.Resolve(input, width, height);
        TextToImageRequest request;
        if (img2img is not null)
        {
            request = new ImageToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
                SourceImage = img2img.SourceTensor,
                Strength = img2img.Strength,
            };
        }
        else
        {
            request = new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
            };
        }

        try
        {
            long start = Environment.TickCount64;
            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                onProgress(p);
            };

            var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromEmbeddings(
                positiveEmbeddings,
                request,
                cfgScale: cfg,
                negativeCaptionEmbeddings: negativeEmbeddings,
                onProgress: bridge);

            Logs.Verbose($"[HartsyInference][Z-Image] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
        }
        finally
        {
            img2img?.SourceTensor?.Dispose();
        }
    }

    /// <summary>Normalizes the keys of a standalone Flux/Z-Image VAE safetensors file into
    /// the diffusers naming <see cref="VaeDecoder.LoadWeights"/> expects. Standalone VAE
    /// files in the wild come in three flavours and we have to handle all of them:
    /// <list type="number">
    /// <item><description>Comfy combined-checkpoint extracts: <c>vae.*</c> or <c>first_stage_model.*</c> prefix.</description></item>
    /// <item><description>BFL-native LDM bare keys: <c>decoder.mid.block_1.norm1.weight</c>.</description></item>
    /// <item><description>Already diffusers: <c>decoder.mid_block.resnets.0.norm1.weight</c>.</description></item>
    /// </list>
    /// <para>The strategy: strip a Comfy prefix if present, then route every key through
    /// <see cref="CheckpointConvertUtils.ConvertVaeKey"/>. That function maps LDM names to
    /// diffusers names AND passes already-diffusers names through unchanged (its fall-through
    /// is <c>"decoder." + decoderKey</c>, which yields the same string for keys that already
    /// matched <c>decoder.</c>). Single uniform path, no format-detection guessing.</para>
    /// <para>Without this, the auto-downloaded mcmonkey <c>flux_ae.safetensors</c> (BFL-native
    /// keys) failed loading with <c>KeyNotFoundException</c> on
    /// <c>decoder.mid_block.resnets.0.norm1.weight</c> — those keys are present in LDM form
    /// but have to be remapped before VaeDecoder.LoadWeights can find them.</para>
    /// </summary>
    private static Dictionary<string, Tensor> LoadVaeWeights(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var (key, tensor) in raw)
        {
            string ldmKey = key;
            if (ldmKey.StartsWith("first_stage_model.", StringComparison.Ordinal))
                ldmKey = ldmKey["first_stage_model.".Length..];
            else if (ldmKey.StartsWith("vae.", StringComparison.Ordinal))
                ldmKey = ldmKey["vae.".Length..];

            var diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
            if (diffusersKey is not null)
            {
                result[diffusersKey] = tensor;
            }
        }
        return result;
    }

    private static int ComputeRealLength(int[] tokenIds)
    {
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] == Qwen3PadTokenId) return i;
        }
        return tokenIds.Length;
    }

    private static unsafe Tensor SliceFirstSeqF32(Tensor source, int realLen)
    {
        if (source.Shape.Rank != 3)
            throw new ArgumentException($"Expected 3D tensor, got rank {source.Shape.Rank}.");
        if (source.DType != DType.F32)
            throw new ArgumentException($"SliceFirstSeqF32 expects F32, got {source.DType}.");

        long batch = source.Shape[0];
        long fullLen = source.Shape[1];
        long hidden = source.Shape[2];
        if (realLen <= 0 || realLen > fullLen)
            throw new ArgumentOutOfRangeException(nameof(realLen), $"realLen {realLen} out of range [1..{fullLen}].");

        TensorShape outShape = new TensorShape(batch, realLen, hidden);
        Tensor result = new Tensor(outShape, source.DType);
        long elemSize = source.DType.SizeInBytes;
        long fullRowBytes = fullLen * hidden * elemSize;
        long sliceRowBytes = realLen * hidden * elemSize;

        byte* src = (byte*)source.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        for (long b = 0; b < batch; b++)
        {
            Buffer.MemoryCopy(src + b * fullRowBytes, dst + b * sliceRowBytes, sliceRowBytes, sliceRowBytes);
        }
        return result;
    }
}

public sealed class ZImageCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required ZImagePipeline Pipeline { get; init; }
    public required ZImageConfig ZImageConfig { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder Qwen { get; init; }
    public required ZImageTransformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required VaeEncoder VaeEncoder { get; init; }
    public required SafeTensorsLoader ZImageCheckpointLoader { get; init; }
    public required SafeTensorsLoader QwenLoader { get; init; }
    public required SafeTensorsLoader VaeLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        Qwen?.Dispose();
        ZImageCheckpointLoader?.Dispose();
        QwenLoader?.Dispose();
        VaeLoader?.Dispose();
    }
}
