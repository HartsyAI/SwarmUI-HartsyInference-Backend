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
/// Loads Stable Diffusion 1.5 from a single all-in-one .safetensors (the standard
/// LDM/CompVis distribution format that every SD 1.5 finetune on civitai uses).
/// SD 1.5 has a single CLIP-L text encoder, a UNet, and a VAE — all bundled.
/// </summary>
public static class Sd15Loader
{
    public const string Sd15CompatClassId = "stable-diffusion-v1";

    public static Sd15CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("SD 1.5 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"SD 1.5 checkpoint not found: {model.RawFilePath}");

        log($"Loading SD 1.5 checkpoint: {model.Name}");
        var (converted, mainLoader) = Sd15CheckpointConverter.LoadAndConvert(model.RawFilePath);

        if (converted.UNet.Count == 0 || converted.ClipL.Count == 0 || converted.Vae.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException(
                "SD 1.5 checkpoint is missing UNet/CLIP-L/VAE components. Is this a complete SD 1.5 file?");
        }
        log($"  UNet keys: {converted.UNet.Count}, CLIP-L keys: {converted.ClipL.Count}, VAE keys: {converted.Vae.Count}");

        log("Building UNet...");
        UNet unet = new UNet(UNetConfig.Sd15);
        unet.LoadWeights(converted.UNet);

        log("Building CLIP-L text encoder...");
        ClipTextEncoder textEncoder = new ClipTextEncoder(ClipTextEncoderConfig.Sd15);
        textEncoder.LoadWeights(converted.ClipL, prefix: "text_model");

        log("Building VAE decoder...");
        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd15);
        vaeDecoder.LoadWeights(converted.Vae);

        log("Building VAE encoder (img2img)...");
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Sd15);
        vaeEncoder.LoadWeights(converted.Vae);

        log("Building pipeline...");
        StableDiffusion15Pipeline pipeline = new StableDiffusion15Pipeline(backend, textEncoder, unet, vaeDecoder, vaeEncoder);

        log("Loading CLIP tokenizer (embedded)...");
        ClipTokenizer tokenizer = new ClipTokenizer();

        log("SD 1.5 ready.");
        return new Sd15CacheEntry
        {
            ModelName = model.Name,
            CompatClass = Sd15CompatClassId,
            Pipeline = pipeline,
            Tokenizer = tokenizer,
            TextEncoder = textEncoder,
            UNet = unet,
            Vae = vaeDecoder,
            VaeEncoder = vaeEncoder,
            CheckpointLoader = mainLoader,
            // Retained for the LoRA fast-path: the dicts reference mmap-backed tensors
            // owned by CheckpointLoader, so they're cheap to keep and safe as long as
            // the loader is alive (which it is, until this entry is disposed).
            UnetWeights = converted.UNet,
            ClipLWeights = converted.ClipL,
        };
    }

    /// <summary>LoRA-merged generation path. Builds a fresh CLIP-L + UNet from
    /// shallow-cloned weight dicts with the LoraStack merged in, runs the SD 1.5
    /// pipeline, disposes the per-gen models + stack on completion. The cached
    /// pipeline / model objects are not touched.</summary>
    public static Image[] GenerateWithLoras(
        Sd15CacheEntry entry,
        IReadOnlyList<LoraResolver.LoraSpec> loras,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel,
        IReadOnlyList<SharpInference.Diffusion.Adapters.IpAdapterConditioning> ipAdapters = null)
    {
        Dictionary<string, Tensor> unetWeights = LoraApplier.ShallowClone(entry.UnetWeights);
        Dictionary<string, Tensor> clipLWeights = LoraApplier.ShallowClone(entry.ClipLWeights);

        LoraStack stack = LoraApplier.BuildAndApply(
            loras, backend,
            unetWeights: unetWeights,
            clipLWeights: clipLWeights);

        // Per-gen components built from the merged dicts. The cached entry's CLIP-L /
        // UNet are untouched and remain available for the next no-LoRA generation.
        ClipTextEncoder textEncoder = new ClipTextEncoder(ClipTextEncoderConfig.Sd15);
        UNet unet = new UNet(UNetConfig.Sd15);
        try
        {
            textEncoder.LoadWeights(clipLWeights, prefix: "text_model");
            unet.LoadWeights(unetWeights);

            // Reuse the cached VAE halves — VAEs aren't a LoRA target.
            using StableDiffusion15Pipeline pipeline = new StableDiffusion15Pipeline(
                backend, textEncoder, unet, entry.Vae, entry.VaeEncoder);

            return RunSd15Pipeline(pipeline, entry.Tokenizer, input, onProgress, cancel, ipAdapters);
        }
        finally
        {
            // UNet / ClipTextEncoder don't implement IDisposable upstream — their
            // backing tensors are GC'd via finalizers. Disposed last so the merged
            // tensors outlive the model components that referenced them through
            // LoadWeights.
            stack?.Dispose();
        }
    }

    public static Image[] Generate(
        Sd15CacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel,
        IReadOnlyList<SharpInference.Diffusion.Adapters.IpAdapterConditioning> ipAdapters = null)
    {
        return RunSd15Pipeline(entry.Pipeline, entry.Tokenizer, input, onProgress, cancel, ipAdapters);
    }

    /// <summary>Shared per-pipeline driver — same logic whether the pipeline is the
    /// cached one (no-LoRA) or a freshly-built per-gen one (LoRA).</summary>
    private static Image[] RunSd15Pipeline(
        StableDiffusion15Pipeline pipeline,
        ClipTokenizer tokenizer,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel,
        IReadOnlyList<SharpInference.Diffusion.Adapters.IpAdapterConditioning> ipAdapters = null)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 20);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);

        int[] promptTokens = tokenizer.Encode(prompt);
        int[] negativeTokens = tokenizer.Encode(negative);

        // Img2img: build an ImageToImageRequest when an init image is provided.
        // The pipeline switches behavior based on the runtime type of `request`.
        Img2ImgResolver.Img2ImgSpec img2img = Img2ImgResolver.Resolve(input, width, height);
        string schedulerName = SamplingParamResolver.ResolveSchedulerName(input);
        int? clipSkip = SamplingParamResolver.ResolveClipSkip(input);
        int? seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF);
        // Variation seed: pre-blend the initial noise (pipeline takes ownership of the tensor).
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
                ClipSkip = clipSkip,
                InitialNoise = variationNoise,
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
                CfgScale = cfgRaw <= 0 ? 7.5f : (float)cfgRaw,
                Seed = seed,
                Scheduler = schedulerName,
                ClipSkip = clipSkip,
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
                promptTokens, negativeTokens, request, bridge, ipAdapters);

            Logs.Verbose($"[SharpInference][SD15] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
        }
        finally
        {
            img2img?.SourceTensor?.Dispose();
        }
    }
}

public sealed class Sd15CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required StableDiffusion15Pipeline Pipeline { get; init; }
    public required ClipTokenizer Tokenizer { get; init; }
    public required ClipTextEncoder TextEncoder { get; init; }
    public required UNet UNet { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required VaeEncoder VaeEncoder { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }

    /// <summary>UNet weights from the converter, retained for the LoRA path
    /// (re-merged per gen into a fresh UNet without disturbing the cached one).</summary>
    public required Dictionary<string, Tensor> UnetWeights { get; init; }
    /// <summary>CLIP-L text encoder weights retained for the LoRA path.</summary>
    public required Dictionary<string, Tensor> ClipLWeights { get; init; }

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
