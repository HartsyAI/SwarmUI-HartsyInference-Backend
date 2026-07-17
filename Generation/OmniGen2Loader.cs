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
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads OmniGen 2 (VectorSpaceLab, 4B MMDiT, Apache-2.0). The picked file is the fp16 transformer
/// (Comfy-Org repackage naming, <c>omnigen2.safetensors</c>); the Qwen2.5-VL-3B text encoder and the
/// FLUX.1 16-channel VAE resolve through <see cref="SideModels"/>. Supports text-to-image and
/// reference-image editing (Init Image + Prompt Images → VAE ref latents with the reference triple-pass
/// dual guidance; conditioning stays text-only through the mllm, matching upstream <c>pipeline_omnigen2.py</c>).
///
/// <para>Conditioning is the LIVE Qwen2.5-VL-3B encode matching ComfyUI's <c>text_encoders/omnigen2.py</c>
/// exactly (that implementation produced the embeddings the engine's e2e run was verified against): the
/// prompt goes through the chat template
/// <c>&lt;|im_start|&gt;system\n{system}&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{prompt}&lt;|im_end|&gt;\n</c>,
/// the FULL templated sequence is kept (no prefix drop, unlike Qwen-Image), and the conditioning is the
/// final hidden state after the model's last RMSNorm (ComfyUI <c>layer="last"</c>) — i.e. exactly
/// <see cref="LlamaStyleEncoder.Encode"/>. The tokenizer is <see cref="Qwen3Tokenizer"/> (Qwen2.5 and Qwen3
/// share the base BPE — the Qwen-Image precedent).</para>
///
/// <para>The fp16 transformer is cast to BF16 at load (same footprint; F32 exponent range) — F16 weights
/// NaN'd to all-black under CFG-amplified activations in the engine's bring-up, and ComfyUI likewise runs
/// OmniGen2 in BF16. Mirrors <c>OmniGen2GenerationTests</c>.</para>
/// </summary>
public static class OmniGen2Loader
{
    public const string OmniGen2CompatClassId = "omnigen-2";

    /// <summary>ComfyUI's OmniGen2 system prompt, verbatim (comfy/text_encoders/omnigen2.py llama_template).</summary>
    private const string SystemPrompt =
        "You are a helpful assistant that generates high-quality images based on user instructions.";

    public static OmniGen2CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("OmniGen2 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"OmniGen2 checkpoint not found: {model.RawFilePath}");

        T2IModel teModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.QwenModel), entry: SideModels.Qwen2_5_VL_3B, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.FluxAe, log: log);

        List<SafeTensorsLoader> loaders = [];
        try
        {
            log($"Loading OmniGen2 transformer (4B): {Path.GetFileName(model.RawFilePath)}");
            (OmniGen2CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) =
                OmniGen2CheckpointConverter.LoadAndConvert(model.RawFilePath);
            loaders.Add(transformerLoader);

            // BF16 (not F16, not F32): F16 overflows under CFG (NaN → all-black at cfg≥5), F32 doubles the
            // footprint to ~16 GB. BF16 keeps 8 GB with F32's exponent range — the validated e2e recipe.
            Dictionary<string, Tensor> transformerWeights = CastWeightsToBf16(converted.Transformer);

            log("Building OmniGen2 transformer...");
            OmniGen2Config config = OmniGen2Config.V1;
            OmniGen2Transformer transformer = new(config);
            transformer.LoadWeights(transformerWeights);

            log($"Loading Qwen2.5-VL-3B text encoder: {teModel.Name}");
            SafeTensorsLoader teLoader = new();
            teLoader.Load(teModel.RawFilePath);
            loaders.Add(teLoader);
            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen2_5_VL_3B);
            textEncoder.LoadWeights(teLoader.GetAllTensors());

            log($"Loading FLUX.1 VAE: {vaeModel.Name}");
            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) =
                LoaderVaeUtils.LoadFluxVaeF32(vaeModel.RawFilePath);
            loaders.Add(vaeLoader);
            VaeDecoder vae = new(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);
            // Encoder half enables the reference-image edit path (Init Image / Prompt Images → VAE ref latents).
            VaeEncoder vaeEncoder = new(VaeConfig.Flux);
            vaeEncoder.LoadWeights(vaeWeights);

            Qwen3Tokenizer tokenizer = new(maxLength: 512);

            log("Building OmniGen2 pipeline...");
            OmniGen2Pipeline pipeline = new(backend, transformer, vae, vaeEncoder, config);

            log("OmniGen2 ready.");
            return new OmniGen2CacheEntry
            {
                ModelName = model.Name,
                CompatClass = OmniGen2CompatClassId,
                Pipeline = pipeline,
                Config = config,
                Tokenizer = tokenizer,
                TextEncoder = textEncoder,
                Transformer = transformer,
                Vae = vae,
                Loaders = loaders,
            };
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    /// <summary>Reference-pipeline image budget (<c>pipeline_omnigen2.py</c> call defaults):
    /// <c>max_pixels = 1024·1024</c>, <c>max_input_image_side_length = 1024</c>. References are downscaled with
    /// <c>ratio = min(sqrt(maxPixels/cur), maxSide/longSide, 1)</c> and floored to multiples of 16
    /// (<c>OmniGen2ImageProcessor.get_new_height_width</c>, vae_scale_factor 8 × patch 2).</summary>
    private const int RefMaxPixels = 1024 * 1024;
    private const int RefMaxSideLength = 1024;

    /// <summary>Reference edit default for <c>image_guidance_scale</c> (the OmniGen2 model card's editing
    /// recommendation) when the user leaves the IP2P CFG 2 slider untoggled.</summary>
    private const float DefaultImageGuidance = 2.0f;

    public static Image[] Generate(
        OmniGen2CacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfg = cfgRaw <= 0 ? 1.0f : (float)cfgRaw;
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 20);
        int? seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF);

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        // Reference images, in Picture order: the Init Image first (the primary edit subject), then any
        // Prompt Images (Swarm's Image Prompting group / <image:...> prompt syntax). OmniGen2 supports up to
        // 5 references (image_index_embedding slots); extras are dropped with a log.
        List<Image> refImagesRaw = new(4);
        Image initImageRaw = input.Get(T2IParamTypes.InitImage);
        if (initImageRaw is not null) refImagesRaw.Add(initImageRaw);
        List<Image> promptImages = input.Get(T2IParamTypes.PromptImages);
        if (promptImages is not null) refImagesRaw.AddRange(promptImages);
        if (refImagesRaw.Count > 5)
        {
            Logs.Warning($"[HartsyInference][OmniGen2] {refImagesRaw.Count} reference images given; " +
                "OmniGen2 supports at most 5 — using the first 5.");
            refImagesRaw.RemoveRange(5, refImagesRaw.Count - 5);
        }

        if (refImagesRaw.Count > 0)
        {
            return GenerateEdit(entry, backend, input, refImagesRaw, prompt, negative, width, height, steps,
                cfg, seed, bridge, cancel, start);
        }

        // ── Text-to-image ──
        // Live Qwen2.5-VL-3B conditioning with a single-slot prompt cache: repeat prompts (seed-only reruns)
        // skip both encoder passes. Cached tensors are host-materialized so activation reclaims can't kill them.
        Tensor condEmbeds = entry.GetOrEncode(backend, prompt);
        Tensor negEmbeds = cfg > 1.0f ? entry.GetOrEncodeNegative(backend, negative) : null;

        TextToImageRequest request = new()
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfg,
            Seed = seed,
        };

        (byte[] rgb, int outW, int outH, int _) = entry.Pipeline.GenerateFromEmbeddings(
            condEmbeds, request,
            cfgScale: cfg,
            negativeCaptionEmbeddings: negEmbeds,
            textGuidanceScale: cfg,
            onProgress: bridge);

        Logs.Verbose($"[HartsyInference][OmniGen2] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return [RgbToImage.FromHwcRgb(rgb, outW, outH)];
    }

    /// <summary>Reference-image edit (Init Image / Prompt Images → VAE ref latents + dual guidance). Conditioning
    /// stays TEXT-ONLY through the Qwen2.5-VL mllm — the upstream edit pipeline never feeds it vision tokens
    /// (those belong to the separate chat model); the references reach the transformer solely as VAE latents.
    /// Text guidance reuses CFG Scale; image guidance maps from the IP2P CFG 2 slider (default 2.0, the model
    /// card's edit recommendation). With ONE reference the output size snaps to the (budget-resized) reference
    /// size (<c>align_res</c>, the upstream default); multi-reference outputs use the requested size scaled into
    /// the 1 MP budget.</summary>
    private static Image[] GenerateEdit(
        OmniGen2CacheEntry entry, IBackend backend, T2IParamInput input, List<Image> refImagesRaw,
        string prompt, string negative, int width, int height, int steps, float cfg, int? seed,
        Action<GenerationProgress> bridge, CancellationToken cancel, long startTick)
    {
        float imageGuidance = input.TryGet(T2IParamTypes.IP2PCFG2, out double ip2p)
            ? (float)ip2p
            : DefaultImageGuidance;

        List<Tensor> refTensors = new(refImagesRaw.Count);
        try
        {
            foreach (Image refImage in refImagesRaw)
            {
                (byte[] nativeRgb, int nativeW, int nativeH) = RgbToImage.ToHwcRgb(refImage);
                (int refH, int refW) = ComputeReferenceSize(nativeH, nativeW);
                byte[] refRgb = refH == nativeH && refW == nativeW
                    ? nativeRgb
                    : RgbToImage.ToHwcRgbResized(refImage, refW, refH);
                refTensors.Add(ImagePostProcessor.RgbBytesToTensor(refRgb, refW, refH));
            }

            // Output size per the reference: align_res with one input image (output = the resized reference's
            // size); otherwise the requested size scaled into the max_pixels budget, floored to multiples of 16.
            int outW, outH;
            if (refTensors.Count == 1)
            {
                outH = (int)refTensors[0].Shape[2];
                outW = (int)refTensors[0].Shape[3];
            }
            else
            {
                double ratio = Math.Min(Math.Sqrt((double)RefMaxPixels / ((double)width * height)), 1.0);
                outW = Math.Max(16, (int)(width * ratio) / 16 * 16);
                outH = Math.Max(16, (int)(height * ratio) / 16 * 16);
            }
            if (outW != width || outH != height)
                Logs.Info($"[HartsyInference][OmniGen2] Edit resolution {width}x{height} → {outW}x{outH} " +
                    (refTensors.Count == 1 ? "(aligned to the reference image)." : "(scaled into the 1 MP budget)."));

            cancel.ThrowIfCancellationRequested();
            Tensor condEmbeds = entry.GetOrEncode(backend, prompt);
            Tensor negEmbeds = cfg > 1.0f ? entry.GetOrEncodeNegative(backend, negative) : null;
            backend.Sync();
            backend.FreeWeights(entry.TextEncoder.EnumerateWeights());
            backend.FreeActivations();

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = outW,
                Height = outH,
                Steps = steps,
                CfgScale = cfg,
                Seed = seed,
            };

            (byte[] rgb, int genW, int genH, int _) = entry.Pipeline.EditFromEmbeddings(
                condEmbeds, negEmbeds, refTensors, request,
                textGuidanceScale: cfg,
                imageGuidanceScale: imageGuidance,
                onProgress: bridge);

            Logs.Verbose($"[HartsyInference][OmniGen2] Edit returned {genW}x{genH} " +
                $"({refTensors.Count} ref(s), ig={imageGuidance}) in {Environment.TickCount64 - startTick}ms.");
            return [RgbToImage.FromHwcRgb(rgb, genW, genH)];
        }
        finally
        {
            foreach (Tensor t in refTensors) t.Dispose();
        }
    }

    /// <summary>OmniGen2ImageProcessor.get_new_height_width: downscale into the pixel/side budget, floored to
    /// multiples of 16; never upscales.</summary>
    private static (int h, int w) ComputeReferenceSize(int height, int width)
    {
        double maxSideRatio = height > width
            ? (double)RefMaxSideLength / height
            : (double)RefMaxSideLength / width;
        double maxPixelsRatio = Math.Sqrt((double)RefMaxPixels / ((double)height * width));
        double ratio = Math.Min(Math.Min(maxPixelsRatio, maxSideRatio), 1.0);
        int newH = Math.Max(16, (int)(height * ratio) / 16 * 16);
        int newW = Math.Max(16, (int)(width * ratio) / 16 * 16);
        return (newH, newW);
    }

    /// <summary>Tokenizes with ComfyUI's OmniGen2 chat template. The full sequence (system block included) is
    /// the conditioning — no prefix drop. Special tokens are inserted by id; text segments BPE'd separately,
    /// which matches HF tokenization (special tokens split the text at exactly these boundaries).</summary>
    internal static int[] EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        const int MaxTokens = 512;
        List<int> ids = new(96);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("system\n" + SystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n" + prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        if (ids.Count > MaxTokens)
            ids.RemoveRange(MaxTokens, ids.Count - MaxTokens);
        return [.. ids];
    }

    /// <summary>Casts F16/F32 weights to BF16 (others pass through) — same footprint as F16 with F32's
    /// exponent range, so CFG-amplified activations can't overflow (the verified OmniGen2 e2e recipe).</summary>
    private static Dictionary<string, Tensor> CastWeightsToBf16(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> bf16 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            bf16[kvp.Key] = (dt == DType.F16 || dt == DType.F32) ? kvp.Value.CastTo(DType.BF16) : kvp.Value;
        }
        return bf16;
    }
}

public sealed class OmniGen2CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required OmniGen2Pipeline Pipeline { get; init; }
    public required OmniGen2Config Config { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder TextEncoder { get; init; }
    public required OmniGen2Transformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required List<SafeTensorsLoader> Loaders { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    // Single-slot prompt-embedding caches (cond + negative), keyed on the raw prompt text. A hit skips the
    // whole 3B encoder forward. Host-materialized at store time so FreeActivations can't revert them.
    private string _cachedPrompt;
    private Tensor _cachedEmbeds;
    private string _cachedNegPrompt;
    private Tensor _cachedNegEmbeds;

    /// <summary>Returns the [1, T, 2048] final-hidden-state conditioning for the prompt (cached one-slot).</summary>
    public unsafe Tensor GetOrEncode(IBackend backend, string prompt)
    {
        if (_cachedEmbeds is not null && _cachedPrompt == prompt)
            return _cachedEmbeds;
        int[] tokens = OmniGen2Loader.EncodeWithTemplate(Tokenizer, prompt);
        Tensor embeds = TextEncoder.Encode(backend, [tokens]);
        _ = embeds.DataPointer;   // host-materialize: survives activation reclaims across generations
        _cachedEmbeds?.Dispose();
        _cachedEmbeds = embeds;
        _cachedPrompt = prompt;
        return embeds;
    }

    /// <summary>Negative-prompt twin of <see cref="GetOrEncode"/> (own slot — the pair alternates per gen).</summary>
    public unsafe Tensor GetOrEncodeNegative(IBackend backend, string negative)
    {
        if (_cachedNegEmbeds is not null && _cachedNegPrompt == negative)
            return _cachedNegEmbeds;
        int[] tokens = OmniGen2Loader.EncodeWithTemplate(Tokenizer, negative);
        Tensor embeds = TextEncoder.Encode(backend, [tokens]);
        _ = embeds.DataPointer;
        _cachedNegEmbeds?.Dispose();
        _cachedNegEmbeds = embeds;
        _cachedNegPrompt = negative;
        return embeds;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _cachedEmbeds?.Dispose();
        _cachedNegEmbeds?.Dispose();
        (Pipeline as IDisposable)?.Dispose();
        TextEncoder?.Dispose();
        Transformer?.Dispose();
        foreach (SafeTensorsLoader l in Loaders) l.Dispose();
    }
}
