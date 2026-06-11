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
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Loads the official SDXL Refiner checkpoint and runs the post-base "PostApply"
/// refine pass via SharpInference's <see cref="SdxlRefinerPipeline"/>. The refiner
/// owns its own CLIP-G encoder (no CLIP-L), refiner UNet (4-level config),
/// VAE encoder + decoder, and tokenizer — independent of the base pipeline.
///
/// Cross-architecture refining is supported because the refiner takes RGB pixel
/// data as input. Any base architecture's output (SD1.5 / SDXL / Flux / Z-Image)
/// can be refined.
///
/// <para>This loader covers the <c>PostApply</c> path (pixel-space second pass via
/// <see cref="SdxlRefinerPipeline"/>) which works against any base architecture's
/// output. <c>StepSwap</c> reuses the same loaded refiner UNet (exposed via
/// <see cref="RefinerCacheEntry.RefinerUnet"/>) but plugs it into the SDXL base
/// pipeline's denoise loop directly via
/// <see cref="SharpInference.Diffusion.Pipelines.RefinerSwapConfig"/>; that wiring
/// lives in <see cref="Hartsy.Extensions.SharpInferenceBackend.Backends.SharpInferenceBackend"/>.
/// <c>StepSwapNoisy</c> remains unimplemented — it would re-noise the latent at the
/// swap point, a minor variant we haven't surfaced yet.</para>
/// </summary>
public static class RefinerLoader
{
    /// <summary>Loads an SDXL refiner checkpoint and builds a fully-wired refiner pipeline.</summary>
    public static RefinerCacheEntry Load(
        IBackend backend,
        T2IModel refinerModel,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(refinerModel?.RawFilePath))
            throw new InvalidOperationException("Refiner model has no file path.");
        if (!File.Exists(refinerModel.RawFilePath))
            throw new FileNotFoundException($"Refiner checkpoint not found: {refinerModel.RawFilePath}");

        log($"Loading SDXL refiner checkpoint: {refinerModel.Name}");
        var (converted, mainLoader) = SdxlRefinerCheckpointConverter.LoadAndConvert(refinerModel.RawFilePath);

        if (converted.UNet.Count == 0 || converted.ClipG.Count == 0 || converted.Vae.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException(
                "SDXL refiner checkpoint missing UNet/CLIP-G/VAE components. Is this the official SDXL refiner file?");
        }
        log($"  Refiner UNet={converted.UNet.Count}, CLIP-G={converted.ClipG.Count}, VAE={converted.Vae.Count}");

        log("Building refiner UNet (SdxlRefiner config — 4 levels)...");
        UNet refinerUnet = new UNet(UNetConfig.SdxlRefiner);
        refinerUnet.LoadWeights(converted.UNet);

        log("Building CLIP-G encoder (refiner has no CLIP-L)...");
        ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
        clipG.LoadWeights(converted.ClipG, prefix: "text_model");

        log("Building VAE encoder (Sdxl config)...");
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Sdxl);
        vaeEncoder.LoadWeights(converted.Vae);

        log("Building VAE decoder (Sdxl config)...");
        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
        vaeDecoder.LoadWeights(converted.Vae);

        log("Building refiner pipeline...");
        SdxlRefinerPipeline pipeline = new SdxlRefinerPipeline(backend, clipG, refinerUnet, vaeEncoder, vaeDecoder);

        log("Loading CLIP tokenizer (embedded; shared with SDXL base — same OpenAI BPE)...");
        ClipTokenizer tokenizer = new ClipTokenizer();

        log("Refiner ready.");
        return new RefinerCacheEntry
        {
            ModelName = refinerModel.Name,
            Pipeline = pipeline,
            Tokenizer = tokenizer,
            ClipG = clipG,
            RefinerUnet = refinerUnet,
            VaeEncoder = vaeEncoder,
            VaeDecoder = vaeDecoder,
            CheckpointLoader = mainLoader,
        };
    }

    /// <summary>Run the SDXL refiner over a single base image. Encodes the base
    /// image's pixels via the VAE encoder, injects noise at <paramref name="strength"/>,
    /// and runs the refiner UNet for the remaining steps.</summary>
    /// <param name="entry">Loaded refiner pipeline.</param>
    /// <param name="baseImage">Base-stage Swarm Image to refine.</param>
    /// <param name="input">User input — provides Prompt / NegativePrompt / Seed.</param>
    /// <param name="steps">Step total used to compute the refiner sub-range. See
    /// <see cref="ImageToImageRequest.Strength"/> for how strength + steps map to
    /// the refiner sub-step count.</param>
    /// <param name="strength">Fraction of steps the refiner runs (Swarm's RefinerControl).</param>
    /// <param name="cfgScale">Refiner-specific CFG scale.</param>
    public static Image Refine(
        RefinerCacheEntry entry,
        Image baseImage,
        T2IParamInput input,
        int steps,
        float strength,
        float cfgScale,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        // 1. Pixels out of the base image into a [1,3,H,W] F32 tensor in [-1, 1].
        var (rgb, w, h) = RgbToImage.ToHwcRgb(baseImage);
        Tensor sourceTensor = ImagePostProcessor.RgbBytesToTensor(rgb, w, h);

        try
        {
            string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
            string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
            long seedLong = input.Get(T2IParamTypes.Seed);

            // CLIP-G tokenization (refiner uses ONLY CLIP-G).
            int[] promptTokens = entry.Tokenizer.Encode(prompt);
            int[] negTokens = entry.Tokenizer.Encode(negative);
            int promptEos = ClipTokenizer.FindEosPosition(promptTokens);
            int negEos = ClipTokenizer.FindEosPosition(negTokens);

            SdxlRefinerRequest request = new SdxlRefinerRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = w,
                Height = h,
                Steps = steps,
                CfgScale = cfgScale,
                Strength = strength,
                Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
                SourceImage = sourceTensor,
            };

            long start = Environment.TickCount64;
            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                onProgress(p);
            };

            var (refinedRgb, outW, outH, _) = entry.Pipeline.RefineFromTokens(
                promptTokens, negTokens, promptEos, negEos, request, bridge);

            Logs.Verbose($"[SharpInference][Refiner] Pass complete: {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return RgbToImage.FromHwcRgb(refinedRgb, outW, outH);
        }
        finally
        {
            sourceTensor.Dispose();
        }
    }
}

public sealed class RefinerCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required SdxlRefinerPipeline Pipeline { get; init; }
    public required ClipTokenizer Tokenizer { get; init; }
    public required ClipTextEncoder ClipG { get; init; }
    public required UNet RefinerUnet { get; init; }
    public required VaeEncoder VaeEncoder { get; init; }
    public required VaeDecoder VaeDecoder { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Pipeline?.Dispose();
        Tokenizer?.Dispose();
        CheckpointLoader?.Dispose();
        // Refiner UNet, VAE encoder/decoder, ClipTextEncoder are not IDisposable
        // upstream — their tensors are GC'd via finalizers (same pattern as the
        // per-architecture cache entries).
    }
}
