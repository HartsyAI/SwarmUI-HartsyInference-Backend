using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.Lora;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Loads Stable Diffusion XL (base) from a single all-in-one .safetensors (LDM format,
/// the standard SDXL distribution that civitai/HF use). SDXL has dual CLIP encoders
/// (CLIP-L + CLIP-G), an SDXL-tuned UNet, and an SDXL-tuned VAE — all bundled.
///
/// Both CLIP encoders use the same OpenAI BPE tokenizer (same vocab.json + merges.txt
/// as Flux uses), so we share one ClipTokenizer instance between them.
/// </summary>
public static class SdxlLoader
{
    public const string SdxlCompatClassId = "stable-diffusion-xl-v1";
    public const string SdxlRefinerCompatClassId = "stable-diffusion-xl-v1-refiner";

    public static SdxlCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("SDXL model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"SDXL checkpoint not found: {model.RawFilePath}");

        log($"Loading SDXL checkpoint: {model.Name}");
        var (converted, mainLoader) = SdxlCheckpointConverter.LoadAndConvert(model.RawFilePath);

        if (converted.UNet.Count == 0 || converted.ClipL.Count == 0 || converted.ClipG.Count == 0 || converted.Vae.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException(
                "SDXL checkpoint is missing UNet/CLIP-L/CLIP-G/VAE components. Is this a complete SDXL base file?");
        }
        log($"  UNet={converted.UNet.Count}, CLIP-L={converted.ClipL.Count}, CLIP-G={converted.ClipG.Count}, VAE={converted.Vae.Count}");

        log("Building UNet (SDXL Base config)...");
        UNet unet = new UNet(UNetConfig.SdxlBase);
        unet.LoadWeights(converted.UNet);

        log("Building CLIP-L encoder...");
        ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        clipL.LoadWeights(converted.ClipL, prefix: "text_model");

        log("Building CLIP-G encoder...");
        ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
        clipG.LoadWeights(converted.ClipG, prefix: "text_model");

        // SDXL VAE F16 is famously broken (resnet activations overflow â NaN â black).
        // Cast weights to BF16 on Ampere+ (matches ComfyUI's policy) or F32 on older HW.
        // Per VaePrecisionHelper: BF16 has F32-equivalent range so cannot overflow, while
        // staying the same 2 bytes as F16 (â¼170 MB for SDXL VAE â no memory hit vs the
        // broken F16 path).
        DType vaeDtype = VaePrecisionHelper.PreferredSdxlVaeDtype(backend);
        Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(converted.Vae, vaeDtype);
        log($"Building VAE decoder (SDXL config, dtype={vaeDtype})...");
        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
        vaeDecoder.LoadWeights(vaeWeights);

        log("Building VAE encoder (img2img)...");
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Sdxl);
        vaeEncoder.LoadWeights(vaeWeights);

        log("Building pipeline (SDXL VAE scaling=0.13025)...");
        SdxlPipeline pipeline = new SdxlPipeline(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder);

        log("Loading CLIP tokenizer (embedded; shared for both CLIP-L and CLIP-G)...");
        ClipTokenizer tokenizer = new ClipTokenizer();

        log("SDXL ready.");
        return new SdxlCacheEntry
        {
            ModelName = model.Name,
            CompatClass = SdxlCompatClassId,
            Pipeline = pipeline,
            Tokenizer = tokenizer,
            ClipL = clipL,
            ClipG = clipG,
            UNet = unet,
            Vae = vaeDecoder,
            VaeEncoder = vaeEncoder,
            CheckpointLoader = mainLoader,
            // Retained for the LoRA path; mmap-backed via the loader.
            UnetWeights = converted.UNet,
            ClipLWeights = converted.ClipL,
            ClipGWeights = converted.ClipG,
        };
    }

    /// <summary>LoRA-merged generation path. Builds fresh CLIP-L + CLIP-G + UNet from
    /// shallow-cloned dicts with the LoraStack merged in, runs the SDXL pipeline,
    /// disposes everything per-gen. The cached entry's components are not touched.</summary>
    public static Image[] GenerateWithLoras(
        SdxlCacheEntry entry,
        IReadOnlyList<LoraResolver.LoraSpec> loras,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel,
        RefinerSwapConfig refinerSwap = null,
        IReadOnlyList<SharpInference.Diffusion.Adapters.IpAdapterConditioning> ipAdapters = null)
    {
        Dictionary<string, Tensor> unetWeights = LoraApplier.ShallowClone(entry.UnetWeights);
        Dictionary<string, Tensor> clipLWeights = LoraApplier.ShallowClone(entry.ClipLWeights);
        Dictionary<string, Tensor> clipGWeights = LoraApplier.ShallowClone(entry.ClipGWeights);

        LoraStack stack = LoraApplier.BuildAndApply(
            loras, backend,
            unetWeights: unetWeights,
            clipLWeights: clipLWeights,
            clipGWeights: clipGWeights);

        ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
        UNet unet = new UNet(UNetConfig.SdxlBase);
        try
        {
            clipL.LoadWeights(clipLWeights, prefix: "text_model");
            clipG.LoadWeights(clipGWeights, prefix: "text_model");
            unet.LoadWeights(unetWeights);

            using SdxlPipeline pipeline = new SdxlPipeline(backend, clipL, clipG, unet, entry.Vae, entry.VaeEncoder);
            return RunSdxlPipeline(pipeline, entry.Tokenizer, input, onProgress, cancel, refinerSwap, ipAdapters);
        }
        finally
        {
            // UNet / ClipTextEncoder don't implement IDisposable upstream — they
            // rely on tensor finalizers. Stack disposed last so merged tensors
            // outlive the model components that referenced them.
            stack?.Dispose();
        }
    }

    public static Image[] Generate(
        SdxlCacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel,
        RefinerSwapConfig refinerSwap = null,
        IReadOnlyList<SharpInference.Diffusion.Adapters.IpAdapterConditioning> ipAdapters = null)
    {
        return RunSdxlPipeline(entry.Pipeline, entry.Tokenizer, input, onProgress, cancel, refinerSwap, ipAdapters);
    }

    /// <summary>Shared per-pipeline driver — same logic whether the pipeline is the
    /// cached one (no-LoRA) or a freshly-built per-gen one (LoRA). When
    /// <paramref name="refinerSwap"/> is supplied, the pipeline performs a Comfy-style
    /// StepSwap mid-loop instead of running base alone.</summary>
    private static Image[] RunSdxlPipeline(
        SdxlPipeline pipeline,
        ClipTokenizer tokenizer,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel,
        RefinerSwapConfig refinerSwap = null,
        IReadOnlyList<SharpInference.Diffusion.Adapters.IpAdapterConditioning> ipAdapters = null)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 30);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);

        // Both encoders use the same BPE — encode once, feed to both. CLIP-G needs the
        // EOS position to extract the pooled vector for ADM conditioning.
        int[] promptTokensL = tokenizer.Encode(prompt);
        int[] negTokensL = tokenizer.Encode(negative);
        int[] promptTokensG = promptTokensL;
        int[] negTokensG = negTokensL;
        int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

        Img2ImgResolver.Img2ImgSpec img2img = Img2ImgResolver.Resolve(input, width, height);
        ControlNetResolver.ResolvedSpec controlnets = ControlNetResolver.Resolve(input, UNetConfig.SdxlBase, width, height, msg => Logs.Verbose($"[SharpInference][SDXL] {msg}"));
        string schedulerName = SamplingParamResolver.ResolveSchedulerName(input);
        int? seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF);
        // Variation seed: pre-blend the initial noise (pipeline takes ownership of the tensor).
        // No ClipSkip here — SDXL uses penultimate CLIP layers by spec (same as Comfy).
        SharpInference.Core.Tensors.Tensor variationNoise = VariationSeedResolver.Resolve(input, width, height, seed);
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
                CfgScale = cfgRaw <= 0 ? 7.5f : (float)cfgRaw,
                Seed = seed,
                Scheduler = schedulerName,
                InitialNoise = variationNoise,
                SourceImage = img2img.SourceTensor,
                Strength = img2img.Strength,
                Mask = img2img.MaskTensor,
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
                CfgScale = cfgRaw <= 0 ? 7.5f : (float)cfgRaw,
                Seed = seed,
                Scheduler = schedulerName,
                InitialNoise = variationNoise,
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

            var (rgbBytes, outW, outH, _) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request, bridge,
                controlnets?.Conditionings,
                refinerSwap,
                ipAdapters);

            Logs.Verbose($"[SharpInference][SDXL] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
        }
        finally
        {
            img2img?.Dispose();
            controlnets?.Dispose();
        }
    }
}

public sealed class SdxlCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required SdxlPipeline Pipeline { get; init; }
    public required ClipTokenizer Tokenizer { get; init; }
    public required ClipTextEncoder ClipL { get; init; }
    public required ClipTextEncoder ClipG { get; init; }
    public required UNet UNet { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required VaeEncoder VaeEncoder { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }

    /// <summary>UNet weights from the converter, retained for the LoRA path.</summary>
    public required Dictionary<string, Tensor> UnetWeights { get; init; }
    /// <summary>CLIP-L weights retained for the LoRA path.</summary>
    public required Dictionary<string, Tensor> ClipLWeights { get; init; }
    /// <summary>CLIP-G weights retained for the LoRA path.</summary>
    public required Dictionary<string, Tensor> ClipGWeights { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        CheckpointLoader?.Dispose();
    }
}
