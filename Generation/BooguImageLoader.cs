using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Boogu-Image-0.1 (<c>boogu-project/Boogu-Image</c>, 10B, Apache-2.0). An OmniGen2/Lumina-2 lineage DiT
/// (8 dual-stream + 32 single-stream blocks, GQA 28:7) conditioned on Qwen3-VL-8B and decoded by the FLUX.1 VAE.
/// Supports text-to-image (Base / Turbo) and reference-image editing (Edit).
///
/// <para>The user picks the Boogu transformer file; the Qwen3-VL-8B encoder (<see cref="SideModels.Qwen3VL_8B"/>) and
/// FLUX.1 VAE (<see cref="SideModels.FluxAe"/>) auto-resolve. For editing, the same Qwen3-VL file must carry the vision
/// tower (<c>visual.*</c> keys); the canonical Ideogram-4 repackage ships language-tower-only, so editing is enabled
/// only when a vision-capable Qwen3-VL is supplied (T2I always works).</para>
///
/// <para>Conditioning matches <c>pipeline_boogu.py</c>: the instruction is wrapped in the Qwen3-VL chat template with
/// the Boogu system prompt, and the encoder's final hidden state is the conditioning. T2I uses single CFG; Edit uses
/// the reference image both through the VAE latent stream and (when vision is available) the Qwen3-VL vision tower,
/// with double guidance.</para>
/// </summary>
public static class BooguImageLoader
{
    public const string BooguImageCompatClassId = "boogu-image";

    /// <summary>Boogu T2I system prompt (verbatim from <c>pipeline_boogu.py</c> <c>SYSTEM_PROMPT_4_T2I_UNIFIED</c>).</summary>
    private const string SystemPromptT2I =
        "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";

    /// <summary>Boogu TI2I (edit) system prompt (verbatim from <c>pipeline_boogu.py</c> <c>SYSTEM_PROMPT_4_TI2I_UNIFIED</c>).</summary>
    private const string SystemPromptTI2I =
        "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

    // Qwen3-VL special token ids (from mllm/config.json).
    private const int VisionStartTokenId = 151652;
    private const int VisionEndTokenId = 151653;
    private const int ImagePadTokenId = 151655;

    /// <summary>Minimum free VRAM to attempt a CUDA load — 10B DiT (fp8 ~10 GB) + Qwen3-VL-8B + VAE + headroom.</summary>
    private const double MinRequiredVramGb = 16.0;

    /// <summary>Qwen3-VL-8B language-tower hidden size — distinguishes the 8B encoder from the 4B/0.6B variants.</summary>
    private const long Qwen3Vl8BHiddenSize = 4096;

    public static BooguImageCacheEntry Load(IBackend backend, T2IModel model, T2IParamInput input, Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Boogu-Image model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Boogu-Image checkpoint not found: {model.RawFilePath}");

        if (backend is CudaBackend cuda)
        {
            (nuint freeBytes, nuint totalBytes) = cuda.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < MinRequiredVramGb)
            {
                throw new InvalidOperationException(
                    $"Boogu-Image needs ≥{MinRequiredVramGb:F0} GB free VRAM (10B DiT + Qwen3-VL-8B + VAE); " +
                    $"this GPU has {freeGb:F1} GB free of {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB total.");
            }
        }
        else
        {
            log("WARNING: Boogu-Image on a non-CUDA backend — a 10B DiT per step will be extremely slow.");
        }

        log($"Loading Boogu-Image model: {model.Name}");

        T2IModel teModel = ResolveQwen3Vl(input?.Get(T2IParamTypes.QwenModel), log);
        T2IModel vaeModel = ResolveFluxVae(input?.Get(T2IParamTypes.VAE), log);

        List<SafeTensorsLoader> loaders = [];
        try
        {
            log($"Loading Boogu transformer (10B): {Path.GetFileName(model.RawFilePath)}");
            (Dictionary<string, Tensor> transformerW, SafeTensorsLoader transformerL) =
                LoadComponent(model.RawFilePath, StripTransformerPrefix, applyFp8Dequant: true);
            loaders.Add(transformerL);

            log($"Loading Qwen3-VL-8B language tower: {teModel.Name}");
            (Dictionary<string, Tensor> teW, SafeTensorsLoader teL) =
                LoadComponent(teModel.RawFilePath, RemapQwenLanguageKey, applyFp8Dequant: true);
            loaders.Add(teL);

            // Vision tower (edit only). The Ideogram-4 Qwen3-VL repackage is language-only; if no visual.* keys
            // are present we leave the vision encoder null and editing is refused at generation time (T2I works).
            (Dictionary<string, Tensor> visionW, SafeTensorsLoader visionL) =
                LoadComponent(teModel.RawFilePath, RemapQwenVisionKey, applyFp8Dequant: true);
            bool hasVision = visionW.ContainsKey("patch_embed.proj.weight");
            if (hasVision) loaders.Add(visionL); else visionL.Dispose();

            log($"Loading FLUX.1 VAE: {vaeModel.Name}");
            (Dictionary<string, Tensor> vaeW, SafeTensorsLoader vaeL) =
                LoadComponent(vaeModel.RawFilePath, key => key, applyFp8Dequant: false);
            loaders.Add(vaeL);

            BooguImageConfig config = BooguImageConfig.V01;
            log("Building Boogu-Image models...");

            BooguImageTransformer transformer = new(config);
            transformer.LoadWeights(transformerW);

            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_8B);
            textEncoder.LoadWeights(teW);

            VaeDecoder vaeDecoder = new(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeW);
            VaeEncoder vaeEncoder = new(VaeConfig.Flux);
            vaeEncoder.LoadWeights(vaeW);

            Qwen3VlVisionEncoder visionEncoder = null;
            Qwen3VlMultimodalEncoder multimodal = null;
            if (hasVision)
            {
                Qwen3VlVisionConfig visionConfig = Qwen3VlVisionConfig.Qwen3Vl8B;
                visionEncoder = new Qwen3VlVisionEncoder(visionConfig);
                visionEncoder.LoadWeights(visionW);
                Qwen3VlImageProcessor imageProcessor = new(visionConfig);
                multimodal = new Qwen3VlMultimodalEncoder(textEncoder, visionEncoder, imageProcessor, visionConfig,
                    imageTokenId: ImagePadTokenId, textHeadDim: 128, ropeTheta: 5_000_000.0, mropeSection: [24, 20, 20]);
                log("Boogu-Image: Qwen3-VL vision tower loaded — image editing enabled.");
            }
            else
            {
                log("Boogu-Image: selected Qwen3-VL has no vision tower (language-only) — text-to-image only; " +
                    "supply a vision-capable Qwen3-VL-8B for image editing.");
            }

            log("Loading Qwen3 tokenizer (embedded)...");
            Qwen3Tokenizer tokenizer = new(maxLength: 4096);

            log("Building Boogu-Image pipeline...");
            BooguImagePipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, config);

            log("Boogu-Image ready.");
            return new BooguImageCacheEntry
            {
                ModelName = model.Name,
                CompatClass = BooguImageCompatClassId,
                Pipeline = pipeline,
                Config = config,
                Tokenizer = tokenizer,
                TextEncoder = textEncoder,
                Transformer = transformer,
                VisionEncoder = visionEncoder,
                Multimodal = multimodal,
                Loaders = loaders,
            };
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    public static Image[] Generate(
        BooguImageCacheEntry entry,
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
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 25);

        // Output size must be a multiple of 16 (2×2 patchify × 8× VAE).
        int snappedW = Math.Clamp(width / 16 * 16, 256, 2048);
        int snappedH = Math.Clamp(height / 16 * 16, 256, 2048);
        if (snappedW != width || snappedH != height)
            Logs.Info($"[HartsyInference][Boogu] Snapped resolution {width}x{height} → {snappedW}x{snappedH} (multiple of 16, 256–2048).");

        // Text guidance from CFG Scale (Boogu Base works well ~2–5; Turbo = 1). Negative prompt drives the uncond pass.
        float textGuidance = (float)input.Get(T2IParamTypes.CFGScale);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        TextToImageRequest request = new()
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = snappedW,
            Height = snappedH,
            Steps = steps,
            CfgScale = textGuidance,
            Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        Image initImage = input.Get(T2IParamTypes.InitImage);
        long startTick = Environment.TickCount64;

        if (initImage is not null)
        {
            return GenerateEdit(entry, backend, request, prompt, negative, initImage, snappedW, snappedH, textGuidance, bridge, cancel, startTick);
        }

        // ── Text-to-image ──
        int[] tokens = BuildTemplatedTokens(entry.Tokenizer, SystemPromptT2I, prompt, numImagePad: 0);
        using Tensor instr = entry.TextEncoder.Encode(backend, [tokens]);

        Tensor negEmb = null;
        try
        {
            if (textGuidance > 1.0f)
            {
                int[] negTokens = BuildTemplatedTokens(entry.Tokenizer, SystemPromptT2I, negative, numImagePad: 0);
                negEmb = entry.TextEncoder.Encode(backend, [negTokens]);
            }
            var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromEmbeddings(instr, request, textGuidance, negEmb, bridge);
            Logs.Verbose($"[HartsyInference][Boogu] T2I {outW}x{outH} in {Environment.TickCount64 - startTick}ms.");
            return [RgbToImage.FromHwcRgb(rgbBytes, outW, outH)];
        }
        finally
        {
            negEmb?.Dispose();
        }
    }

    private static Image[] GenerateEdit(
        BooguImageCacheEntry entry, IBackend backend, TextToImageRequest request,
        string prompt, string negative, Image initImage, int width, int height,
        float textGuidance, Action<GenerationProgress> bridge, CancellationToken cancel, long startTick)
    {
        if (entry.Multimodal is null || entry.VisionEncoder is null)
            throw new InvalidOperationException(
                "Boogu-Image editing requires a Qwen3-VL-8B with the vision tower, but the loaded encoder is " +
                "language-only. Select a vision-capable Qwen3-VL-8B (full model) to edit, or remove the Init Image " +
                "to run text-to-image.");

        // Reference image in two forms: [3,H0,W0] in [0,1] for the Qwen3-VL vision tower (it smart-resizes
        // internally), and [1,3,H,W] in [-1,1] at the generation size for the DiT latent stream (VAE-encoded).
        (byte[] nativeRgb, int nativeW, int nativeH) = RgbToImage.ToHwcRgb(initImage);
        using Tensor visionRgb = HwcRgbToChw01(nativeRgb, nativeW, nativeH);

        byte[] genRgb = RgbToImage.ToHwcRgbResized(initImage, width, height);
        using Tensor refLatentInput = ImagePostProcessor.RgbBytesToTensor(genRgb, width, height);

        // Image-pad placeholder count must equal the vision tower's merged token count. The merged grid is the
        // smart-resized image (multiples of patch·merge) divided by patch·merge; this uses the SAME processor
        // config the multimodal encoder runs internally, so the counts match.
        Qwen3VlVisionConfig visionCfg = Qwen3VlVisionConfig.Qwen3Vl8B;
        Qwen3VlImageProcessor countProc = new(visionCfg);
        (int resizedH, int resizedW) = countProc.SmartResize(nativeH, nativeW);
        int mergeFactor = visionCfg.PatchSize * visionCfg.SpatialMergeSize;
        int numMerged = (resizedH / mergeFactor) * (resizedW / mergeFactor);

        int[] condTokens = BuildTemplatedTokens(entry.Tokenizer, SystemPromptTI2I, prompt, numMerged);
        int[] dropTextTokens = BuildTemplatedTokens(entry.Tokenizer, SystemPromptTI2I, negative, numMerged);

        Tensor cond = null, dropText = null;
        try
        {
            cancel.ThrowIfCancellationRequested();
            cond = entry.Multimodal.Encode(backend, condTokens, [visionRgb]);
            cancel.ThrowIfCancellationRequested();
            dropText = entry.Multimodal.Encode(backend, dropTextTokens, [visionRgb]);

            // Single-guidance edit (image_guidance_scale = 1): pred = cond + (tg-1)·(cond - dropText). The drop-all
            // pass (image guidance > 1) is a future param; keep it off so editing stays a two-pass loop.
            var (rgbBytes, outW, outH, _) = entry.Pipeline.EditFromEmbeddings(
                cond, dropText, null, [refLatentInput], request, textGuidance, 1.0f, bridge);
            Logs.Verbose($"[HartsyInference][Boogu] Edit {outW}x{outH} ({numMerged} ref tokens) in {Environment.TickCount64 - startTick}ms.");
            return [RgbToImage.FromHwcRgb(rgbBytes, outW, outH)];
        }
        finally
        {
            cond?.Dispose();
            dropText?.Dispose();
        }
    }

    /// <summary>Builds the Qwen3-VL chat-templated token sequence: <c>&lt;|im_start|&gt;system\n{sys}&lt;|im_end|&gt;\n
    /// &lt;|im_start|&gt;user\n[&lt;|vision_start|&gt;&lt;|image_pad|&gt;×N&lt;|vision_end|&gt;]{instruction}&lt;|im_end|&gt;\n
    /// &lt;|im_start|&gt;assistant\n</c>. Text fragments use the raw BPE; structural tokens are the Qwen special ids.</summary>
    private static int[] BuildTemplatedTokens(Qwen3Tokenizer tok, string system, string instruction, int numImagePad)
    {
        List<int> ids = new(256);
        ids.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(ids, tok, "system\n" + system);
        ids.Add(Qwen3Tokenizer.ImEndId);
        AppendRaw(ids, tok, "\n");
        ids.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(ids, tok, "user\n");
        if (numImagePad > 0)
        {
            ids.Add(VisionStartTokenId);
            for (int i = 0; i < numImagePad; i++) ids.Add(ImagePadTokenId);
            ids.Add(VisionEndTokenId);
        }
        AppendRaw(ids, tok, instruction ?? "");
        ids.Add(Qwen3Tokenizer.ImEndId);
        AppendRaw(ids, tok, "\n");
        ids.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(ids, tok, "assistant\n");
        return [.. ids];
    }

    private static void AppendRaw(List<int> dst, Qwen3Tokenizer tok, string text)
    {
        foreach (int id in tok.EncodeRaw(text)) dst.Add(id);
    }

    /// <summary>HWC RGB bytes [0,255] → CHW <c>[3,H,W]</c> F32 in [0,1] (the Qwen3-VL image processor's input form).</summary>
    private static unsafe Tensor HwcRgbToChw01(byte[] rgb, int width, int height)
    {
        Tensor t = new(new TensorShape(3, height, width), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int src = (y * width + x) * 3;
                for (int c = 0; c < 3; c++)
                    p[(long)c * height * width + (long)y * width + x] = rgb[src + c] / 255.0f;
            }
        return t;
    }

    private static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadComponent(
        string filePath, Func<string, string> keyTransform, bool applyFp8Dequant)
    {
        SafeTensorsLoader loader = new();
        loader.Load(filePath);
        try
        {
            Dictionary<string, Tensor> merged = new();
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                if (kvp.Key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || kvp.Key == "scaled_fp8")
                    continue;
                string mapped = keyTransform(kvp.Key);
                if (mapped is not null)
                    merged[mapped] = kvp.Value;
            }
            return (applyFp8Dequant ? CheckpointConvertUtils.ApplyFp8ScaledDequant(merged) : merged, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private static string StripTransformerPrefix(string key)
    {
        if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
            return key["model.diffusion_model.".Length..];
        if (key.StartsWith("diffusion_model.", StringComparison.Ordinal))
            return key["diffusion_model.".Length..];
        if (key.StartsWith("transformer.", StringComparison.Ordinal))
            return key["transformer.".Length..];
        return key;
    }

    /// <summary>Maps a Qwen3-VL language-tower key to the <c>LlamaStyleEncoder</c> convention, or null to drop it
    /// (vision tower, <c>lm_head</c>).</summary>
    private static string RemapQwenLanguageKey(string key)
    {
        if (key.Contains(".visual.") || key.StartsWith("visual.", StringComparison.Ordinal)) return null;
        if (key.Contains("lm_head")) return null;
        int lm = key.LastIndexOf("language_model.", StringComparison.Ordinal);
        string suffix = lm >= 0 ? key[(lm + "language_model.".Length)..] : key;
        if (suffix.StartsWith("model.", StringComparison.Ordinal))
            suffix = suffix["model.".Length..];
        if (suffix.StartsWith("layers.", StringComparison.Ordinal)
            || suffix.StartsWith("embed_tokens.", StringComparison.Ordinal)
            || suffix == "norm.weight")
        {
            return "model." + suffix;
        }
        return null;
    }

    /// <summary>Maps a Qwen3-VL vision-tower key to bare keys (strips <c>…visual.</c>), or null for non-vision keys.</summary>
    private static string RemapQwenVisionKey(string key)
    {
        int v = key.LastIndexOf(".visual.", StringComparison.Ordinal);
        if (v >= 0) return key[(v + ".visual.".Length)..];
        if (key.StartsWith("visual.", StringComparison.Ordinal)) return key["visual.".Length..];
        return null;
    }

    private static T2IModel ResolveQwen3Vl(T2IModel userPick, Action<string> log)
    {
        if (userPick is not null && !IsQwen3Vl8B(userPick))
        {
            log($"Selected text encoder '{userPick.Name}' is not a Qwen3-VL-8B (no 4096-dim embed_tokens); " +
                "ignoring it and auto-resolving the Qwen3-VL-8B that Boogu-Image requires.");
            userPick = null;
        }
        return ModelAutoDownloader.EnsureSideModel(userPick, SideModels.Qwen3VL_8B, log);
    }

    private static bool IsQwen3Vl8B(T2IModel model)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath) || !File.Exists(model.RawFilePath)) return false;
        try
        {
            using SafeTensorsLoader probe = new();
            probe.Load(model.RawFilePath);
            bool hasEmbed = false, hasLayers = false;
            foreach (KeyValuePair<string, SafeTensorDescriptor> kvp in probe.Descriptors)
            {
                if (kvp.Key.Contains("embed_tokens.weight"))
                {
                    TensorShape shape = kvp.Value.Shape;
                    if ((shape.Rank >= 1 ? shape[shape.Rank - 1] : 0) != Qwen3Vl8BHiddenSize) return false;
                    hasEmbed = true;
                }
                if (kvp.Key.Contains(".layers.")) hasLayers = true;
            }
            return hasEmbed && hasLayers;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference][Boogu] Could not probe text encoder '{model.Name}' ({ex.Message}); treating as incompatible.");
            return false;
        }
    }

    private static T2IModel ResolveFluxVae(T2IModel userPick, Action<string> log)
    {
        if (userPick is not null && !IsFlux1Vae(userPick))
        {
            log($"Selected VAE '{userPick.Name}' is not a FLUX.1 VAE (16-channel ae); " +
                "ignoring it and auto-resolving the FLUX.1 VAE that Boogu-Image requires.");
            userPick = null;
        }
        return ModelAutoDownloader.EnsureSideModel(userPick, SideModels.FluxAe, log);
    }

    /// <summary>Header probe: a FLUX.1 KL VAE (16-channel) — has the diffusers <c>encoder.conv_in</c> and
    /// <c>decoder.conv_out</c> but NOT the Flux.2 BatchNorm stats (<c>bn.running_mean</c>).</summary>
    private static bool IsFlux1Vae(T2IModel model)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath) || !File.Exists(model.RawFilePath)) return false;
        try
        {
            using SafeTensorsLoader probe = new();
            probe.Load(model.RawFilePath);
            bool hasEncoder = probe.Descriptors.ContainsKey("encoder.conv_in.weight");
            bool isFlux2 = probe.Descriptors.ContainsKey("bn.running_mean");
            return hasEncoder && !isFlux2;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference][Boogu] Could not probe VAE '{model.Name}' ({ex.Message}); treating as incompatible.");
            return false;
        }
    }
}

public sealed class BooguImageCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required BooguImagePipeline Pipeline { get; init; }
    public required BooguImageConfig Config { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder TextEncoder { get; init; }
    public required BooguImageTransformer Transformer { get; init; }
    public Qwen3VlVisionEncoder VisionEncoder { get; init; }
    public Qwen3VlMultimodalEncoder Multimodal { get; init; }
    public required List<SafeTensorsLoader> Loaders { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        (TextEncoder as IDisposable)?.Dispose();
        (Transformer as IDisposable)?.Dispose();
        (VisionEncoder as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        foreach (SafeTensorsLoader l in Loaders) l.Dispose();
    }
}
