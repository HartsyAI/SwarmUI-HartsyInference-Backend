using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Diffusion.Utilities;
using SixLabors.ImageSharp.Processing;
using ISImage = SixLabors.ImageSharp.Image;
using SixLabors.ImageSharp.PixelFormats;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Qwen-Image (Alibaba, 20B MMDiT). A single text encoder — Qwen2.5-VL-7B run as a feature
/// extractor — plus the 16-channel Qwen-Image VAE. The diffusion_models checkpoint is transformer-only;
/// the encoder and VAE come from separate Swarm-registered models picked through the normal parameter
/// system, with central <see cref="SideModels"/> auto-download when the user leaves them blank:
/// <list type="bullet">
///   <item><c>T2IParamTypes.QwenModel</c> (Models/clip/) — Qwen2.5-VL-7B text encoder.</item>
///   <item><c>T2IParamTypes.VAE</c> (Models/vae/) — Qwen-Image VAE (shared with Anima).</item>
/// </list>
///
/// <para>Mirrors Comfy's Qwen-Image path (<c>WorkflowGeneratorModelSupport.cs:1153-1162</c>): a
/// <c>qwen_image</c> CLIPLoader feeding qwen_2.5_vl_7b + the qwen-image VAE.</para>
///
/// <para><b>Prompt template:</b> matches diffusers — the prompt is wrapped in Qwen-Image's system+user
/// template and the prefix hidden states are dropped (see <see cref="EncodeWithTemplate"/> +
/// <c>QwenImagePipeline</c>'s drop-index). The tokenizer is <see cref="Qwen3Tokenizer"/> — its base BPE
/// merges are identical to Qwen2.5's for ordinary text, so token IDs match; the template's special
/// tokens are inserted by id. Pending GPU verification of the exact BPE segment boundaries.</para>
/// </summary>
public static class QwenImageLoader
{
    public const string QwenImageCompatClassId = "qwen-image";

    public static QwenImageCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Qwen-Image model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Qwen-Image checkpoint not found: {model.RawFilePath}");

        // 1. Load + convert the transformer (and any bundled encoder/VAE in an all-in-one file).
        // GGUF checkpoints (e.g. qwen-image-edit-2511-Q5_K_M.gguf) route through the same converter via
        // the GGUF bridge, mirroring FluxLoader's split-mode GGUF path.
        log($"Loading Qwen-Image checkpoint: {model.Name}");
        bool isGguf = model.RawFilePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        QwenImageCheckpointConverter.ConvertedWeights converted;
        SafeTensorsLoader mainLoader = null;
        IDisposable ggufHandle = null;
        if (isGguf)
        {
            // Keep the quantized tensors NATIVE (mmap-backed): dequantizing a 20B Q4/Q5 checkpoint to F16 on the
            // host needs ~40 GB RAM and got the process OOM-killed. The engine's Linear path dequantizes GGUF
            // quants per-GEMM on the GPU instead. Rank-2 relabel converts ggml's [in, out] shape metadata to the
            // [out, in] order every converter/Linear assumes (pure metadata swap, valid for quantized dtypes).
            GgufModelLoader.LoadedGgufModel gguf = GgufModelLoader.Load(model.RawFilePath);
            ggufHandle = gguf;
            converted = QwenImageCheckpointConverter.Convert(
                GgufModelLoader.RelabelRank2ToPyTorchOrder(gguf.Weights));
        }
        else
        {
            (converted, mainLoader) = QwenImageCheckpointConverter.LoadAndConvert(model.RawFilePath);
        }
        if (converted.Transformer.Count == 0)
        {
            mainLoader?.Dispose();
            ggufHandle?.Dispose();
            throw new InvalidOperationException(
                $"Qwen-Image checkpoint '{model.Name}' contains no transformer weights " +
                "(looked for <c>transformer_blocks.*</c> / <c>img_in.*</c>).");
        }
        log($"Parsed checkpoint: {converted.Transformer.Count} transformer tensors.");

        // V1 is the released 20B Qwen-Image (depth=60, hidden=3072). V2 presets are speculative
        // placeholders for unreleased weights, so we don't auto-detect into them.
        QwenImageConfig config = QwenImageConfig.V1;

        log($"Building Qwen-Image transformer (depth={config.Depth}, hidden={config.HiddenSize})...");
        QwenImageTransformer transformer = new QwenImageTransformer(config);
        // Load transformer weights as-is (fp8/fp16 kept for the quantized GEMM path — matches the
        // HartsyInference reference test, which does NOT cast the transformer to F32).
        transformer.LoadWeights(converted.Transformer);

        // 2. Resolve + load the Qwen2.5-VL-7B text encoder (bundled-if-present, else side-model).
        SafeTensorsLoader encoderLoader = null, vaeLoader = null;

        log("Building Qwen2.5-VL-7B text encoder...");
        LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
        if (converted.TextEncoder.Count > 0)
        {
            textEncoder.LoadWeights(converted.TextEncoder);
        }
        else
        {
            T2IModel encoderModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.QwenModel), entry: SideModels.Qwen2_5_VL_7B, log: log);
            encoderLoader = new SafeTensorsLoader();
            encoderLoader.Load(encoderModel.RawFilePath);
            textEncoder.LoadWeights(encoderLoader.GetAllTensors());
        }

        // 3. Resolve + load the Qwen-Image VAE (shared with Anima). Both halves: the decoder for output,
        // the encoder for img2img/edit (ImageToImageRequest requires it on pipeline construction).
        log("Building Qwen-Image VAE decoder + encoder (16-channel)...");
        QwenImageVaeDecoder vae = new QwenImageVaeDecoder(VaeConfig.QwenImage);
        QwenImageVaeEncoder vaeEncoder = new QwenImageVaeEncoder(VaeConfig.QwenImage);
        if (converted.Vae.Count > 0)
        {
            Dictionary<string, Tensor> vaeWeights = CastToF32(converted.Vae);
            vae.LoadWeights(vaeWeights);
            vaeEncoder.LoadWeights(vaeWeights);
        }
        else
        {
            T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.QwenImageVae, log: log);
            vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaeModel.RawFilePath);
            Dictionary<string, Tensor> vaeWeights = CastToF32(vaeLoader.GetAllTensors());
            vae.LoadWeights(vaeWeights);
            vaeEncoder.LoadWeights(vaeWeights);
        }

        // 4. Qwen-Image-Edit: load the Qwen2.5-VL VISION tower from the same TE weights (the ComfyUI-style
        // TE files carry the full `visual.*` tower alongside the language model). Swarm classes the edit
        // checkpoints as qwen-image-edit (2509, ref "index" method) / qwen-image-edit-plus (2511,
        // "index_timestep_zero" — detected via the checkpoint's __index_timestep_zero__ marker key).
        string classId = model.ModelClass?.ID ?? "";
        bool isEditModel = classId is "qwen-image-edit" or "qwen-image-edit-plus";
        bool refTimestepZero = classId == "qwen-image-edit-plus";
        Qwen25VlVisionEncoder visionEncoder = null;
        Qwen25VlMultimodalEncoder multimodalEncoder = null;
        if (isEditModel)
        {
            IReadOnlyDictionary<string, Tensor> teDict = converted.TextEncoder.Count > 0
                ? converted.TextEncoder
                : encoderLoader.GetAllTensors();
            if (teDict.ContainsKey("visual.patch_embed.proj.weight"))
            {
                log("Building Qwen2.5-VL vision tower (edit conditioning)...");
                Qwen25VlVisionConfig visionConfig = Qwen25VlVisionConfig.Qwen2_5_VL_7B;
                visionEncoder = new Qwen25VlVisionEncoder(visionConfig);
                visionEncoder.LoadWeights(teDict);
                multimodalEncoder = new Qwen25VlMultimodalEncoder(
                    textEncoder, visionEncoder, new Qwen25VlImageProcessor(visionConfig));
            }
            else
            {
                log("WARNING: text-encoder weights carry no visual.* tower — edit runs text-only (degraded fidelity).");
            }
        }

        log("Building Qwen-Image pipeline...");
        QwenImagePipeline pipeline = new QwenImagePipeline(
            backend, textEncoder, transformer, vae, vaeEncoder, multimodalEncoder, config);

        log("Loading Qwen tokenizer (embedded)...");
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);

        log("Qwen-Image ready (Qwen2.5-VL-7B encoder; flow-match Euler, dynamic shift).");
        return new QwenImageCacheEntry
        {
            ModelName = model.Name,
            CompatClass = QwenImageCompatClassId,
            Pipeline = pipeline,
            QwenImageConfig = config,
            Tokenizer = tokenizer,
            TextEncoder = textEncoder,
            Transformer = transformer,
            Vae = vae,
            VaeEncoder = vaeEncoder,
            VisionEncoder = visionEncoder,
            MultimodalEncoder = multimodalEncoder,
            RefTimestepZero = refTimestepZero,
            CheckpointLoader = mainLoader,
            GgufHandle = ggufHandle,
            EncoderLoader = encoderLoader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        QwenImageCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.Prompt));
        string negative = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.NegativePrompt));
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 20);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfg = cfgRaw <= 0 ? 2.5f : (float)cfgRaw;

        // Build the Qwen-Image system-prompt template and the prefix-drop index (matches diffusers'
        // prompt_template_encode + prompt_template_encode_start_idx). Real length, no padding — the
        // pipeline has no attention mask, so padding would pollute conditioning; the encoder is causal
        // so real-token hidden states are unaffected by dropping the (absent) pad positions. Qwen2.5 and
        // Qwen3 share the same base BPE merges, so these IDs match the real Qwen2.5-VL tokenizer.
        // Qwen-Image-Edit with a reference: the Init Image is the EDIT REFERENCE, not an img2img source —
        // it conditions the generation twice (VAE reference latent appended to the DiT token stream + vision
        // tokens inside the Qwen2.5-VL chat template), while the output latent starts from pure noise
        // (ComfyUI TextEncodeQwenImageEdit[Plus] semantics).
        // Reference images, in Picture order: the Init Image first (the primary edit subject), then any
        // Prompt Images (Swarm's Image Prompting group / <image:...> prompt syntax). The edit-plus (2511)
        // checkpoints are trained with up to 3 references; extras are dropped with a log.
        List<SwarmUI.Utils.Image> refImagesRaw = new(4);
        SwarmUI.Utils.Image initImageRaw = input.Get(T2IParamTypes.InitImage);
        if (initImageRaw is not null) refImagesRaw.Add(initImageRaw);
        List<SwarmUI.Utils.Image> promptImages = input.Get(T2IParamTypes.PromptImages);
        if (promptImages is not null) refImagesRaw.AddRange(promptImages);
        if (refImagesRaw.Count > 3)
        {
            Logs.Warning($"[HartsyInference][Qwen-Image] {refImagesRaw.Count} reference images given; " +
                "the edit-plus checkpoints are trained with at most 3 — using the first 3.");
            refImagesRaw.RemoveRange(3, refImagesRaw.Count - 3);
        }
        bool editRoute = entry.MultimodalEncoder is not null && refImagesRaw.Count > 0;
        List<Tensor> editVaeRefs = null, editVisionRefs = null;
        int[] promptTokens, negTokens;
        int promptDrop, negDrop;
        if (editRoute)
        {
            editVaeRefs = new List<Tensor>(refImagesRaw.Count);
            editVisionRefs = new List<Tensor>(refImagesRaw.Count);
            int[] padCounts = new int[refImagesRaw.Count];
            for (int r = 0; r < refImagesRaw.Count; r++)
            {
                (Tensor vaeRef, Tensor visionRef) = BuildEditReferences(refImagesRaw[r]);
                editVaeRefs.Add(vaeRef);
                editVisionRefs.Add(visionRef);
                padCounts[r] = entry.MultimodalEncoder.CountImageTokens(visionRef);
            }
            (promptTokens, promptDrop) = EncodeWithEditTemplate(entry.Tokenizer, prompt, padCounts);
            (negTokens, negDrop) = EncodeWithEditTemplate(entry.Tokenizer, negative, padCounts);
            Logs.Verbose($"[HartsyInference][Qwen-Image] edit route: {refImagesRaw.Count} reference(s), " +
                $"[{string.Join(", ", padCounts)}] vision tokens, timestepZero={entry.RefTimestepZero}.");
        }
        else
        {
            (promptTokens, promptDrop) = EncodeWithTemplate(entry.Tokenizer, prompt);
            (negTokens, negDrop) = EncodeWithTemplate(entry.Tokenizer, negative);
        }

        // Img2img / inpaint (non-edit models): an Init Image routes through the Qwen-Image VAE encoder
        // (flow-matching AddNoise at the strength-selected step); an additional Mask Image enables the
        // blend-on-vanilla inpaint path — same request contract as Flux.
        Img2ImgResolver.Img2ImgSpec img2img = editRoute ? null : Img2ImgResolver.Resolve(input, width, height);
        int? seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF);
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
                CfgScale = cfg,
                Seed = seed,
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
                CfgScale = cfg,
                Seed = seed,
            };
        }

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        try
        {
            var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(
                promptTokens, negTokens, request, bridge,
                promptDropIndex: promptDrop, negativeDropIndex: negDrop,
                editRefImages: editVaeRefs, editRefTimestepZero: entry.RefTimestepZero,
                editRefVisionImages: editVisionRefs);

            Logs.Verbose($"[HartsyInference][Qwen-Image] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
        }
        finally
        {
            img2img?.Dispose();
            if (editVaeRefs is not null) foreach (Tensor t in editVaeRefs) t.Dispose();
            if (editVisionRefs is not null) foreach (Tensor t in editVisionRefs) t.Dispose();
        }
    }

    /// <summary>Builds the two edit-reference tensors from the init image (ComfyUI
    /// <c>TextEncodeQwenImageEditPlus</c> recipe): the VAE reference rescaled to ~1MP area (dims rounded to
    /// 16 — VAE 8× plus 2×2 packing) in the engine's [-1,1] NCHW contract, and the vision-tower reference
    /// rescaled to ~384² area in the Qwen image-processor's [0,1] contract (it smart-resizes to the
    /// 28-multiple grid itself). Box resampling ≈ upstream's "area" mode.</summary>
    private static unsafe (Tensor vaeRef, Tensor visionRef) BuildEditReferences(SwarmUI.Utils.Image initImage)
    {
        using var frame = ISImage.Load<Rgb24>(initImage.RawData);
        int srcW = frame.Width, srcH = frame.Height;

        double vaeScale = Math.Sqrt(1024.0 * 1024.0 / ((double)srcW * srcH));
        int vaeW = Math.Max(16, (int)Math.Round(srcW * vaeScale / 16.0) * 16);
        int vaeH = Math.Max(16, (int)Math.Round(srcH * vaeScale / 16.0) * 16);
        Tensor vaeRef;
        using (var vaeFrame = frame.Clone(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new SixLabors.ImageSharp.Size(vaeW, vaeH),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Box,
        })))
        {
            byte[] rgb = new byte[vaeW * vaeH * 3];
            vaeFrame.CopyPixelDataTo(rgb);
            vaeRef = ImagePostProcessor.RgbBytesToTensor(rgb, vaeW, vaeH);
        }

        double visScale = Math.Sqrt(384.0 * 384.0 / ((double)srcW * srcH));
        int visW = Math.Max(28, (int)Math.Round(srcW * visScale));
        int visH = Math.Max(28, (int)Math.Round(srcH * visScale));
        Tensor visionRef;
        using (var visFrame = frame.Clone(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new SixLabors.ImageSharp.Size(visW, visH),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Box,
        })))
        {
            byte[] rgb = new byte[visW * visH * 3];
            visFrame.CopyPixelDataTo(rgb);
            visionRef = new Tensor(new HartsyInference.Core.Tensors.TensorShape(1, 3, visH, visW), DType.F32);
            float* dst = (float*)visionRef.DataPointer;
            for (int y = 0; y < visH; y++)
                for (int x = 0; x < visW; x++)
                {
                    int po = (y * visW + x) * 3;
                    for (int c = 0; c < 3; c++)
                        dst[(long)c * visH * visW + (long)y * visW + x] = rgb[po + c] / 255.0f;
                }
        }
        return (vaeRef, visionRef);
    }

    /// <summary>The edit-mode system prompt (ComfyUI <c>QwenImageTokenizer.llama_template_images</c> /
    /// <c>TextEncodeQwenImageEditPlus</c>).</summary>
    private const string QwenImageEditSystemPrompt =
        "system\nDescribe the key features of the input image (color, shape, size, texture, objects, " +
        "background), then explain how the user's text instruction should alter or modify the image. " +
        "Generate a new image that meets the user's requirements while maintaining consistency with the " +
        "original input where appropriate.";

    /// <summary>Builds the Qwen-Image-Edit templated token sequence: edit system prompt, then a user turn of
    /// <c>Picture 1: &lt;|vision_start|&gt;&lt;|image_pad|&gt;×N&lt;|vision_end|&gt;</c> followed by the
    /// instruction. The drop index lands after the <c>user\n</c> header — the same position ComfyUI's
    /// dynamic rule (second <c>&lt;|im_start|&gt;</c> + 3) resolves to for this template, so the retained
    /// hidden states START at the Picture block (vision tokens are kept, template preamble dropped). No
    /// 512-token cap: the vision run alone is commonly 100-300 tokens.</summary>
    private static (int[] tokens, int dropIndex) EncodeWithEditTemplate(Qwen3Tokenizer tokenizer, string prompt, int[] imagePadCounts)
    {
        int totalPads = 0;
        foreach (int c in imagePadCounts) totalPads += c;
        List<int> ids = new(totalPads + 128);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw(QwenImageEditSystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n"));
        int dropIndex = ids.Count;                         // everything above is the discarded prefix
        for (int img = 0; img < imagePadCounts.Length; img++)
        {
            ids.AddRange(tokenizer.EncodeRaw($"Picture {img + 1}: "));
            ids.Add(Qwen25VlMultimodalEncoder.VisionStartId);
            for (int i = 0; i < imagePadCounts[img]; i++)
                ids.Add(Qwen25VlMultimodalEncoder.ImageTokenId);
            ids.Add(Qwen25VlMultimodalEncoder.VisionEndId);
        }
        ids.AddRange(tokenizer.EncodeRaw(prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        return (ids.ToArray(), dropIndex);
    }

    /// <summary>The exact system prompt Qwen-Image conditions on (diffusers
    /// <c>QwenImagePipeline.prompt_template_encode</c>). The encoder sees
    /// <c>&lt;|im_start|&gt;system\n{SystemPrompt}&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{prompt}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>;
    /// the system + user-header prefix hidden states are then dropped.</summary>
    private const string QwenImageSystemPrompt =
        "system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    /// <summary>Builds the Qwen-Image templated token sequence (real length, no padding) and the
    /// prefix-drop index — the count of leading tokens (system block + user header) whose hidden states
    /// the pipeline discards. Special tokens are inserted by id; text segments between them are BPE'd
    /// individually via <see cref="Qwen3Tokenizer.EncodeRaw"/>. This mirrors diffusers' fixed
    /// <c>prompt_template_encode_start_idx</c> design, which relies on the <c>user\n</c> header
    /// tokenizing independently of the following prompt content.</summary>
    private static (int[] tokens, int dropIndex) EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        const int MaxTokens = 512;
        List<int> ids = new(64);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw(QwenImageSystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n"));
        int dropIndex = ids.Count;                         // everything above is the discarded prefix
        ids.AddRange(tokenizer.EncodeRaw(prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        if (ids.Count > MaxTokens)
        {
            ids.RemoveRange(MaxTokens, ids.Count - MaxTokens); // truncate trailing (diffusers truncation)
        }
        return (ids.ToArray(), dropIndex);
    }

    private static Dictionary<string, Tensor> CastToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (var (key, tensor) in weights)
        {
            f32[key] = (tensor.DType == DType.F16 || tensor.DType == DType.BF16) ? tensor.CastTo(DType.F32) : tensor;
        }
        return f32;
    }
}

public sealed class QwenImageCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required QwenImagePipeline Pipeline { get; init; }
    public required QwenImageConfig QwenImageConfig { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder TextEncoder { get; init; }
    public required QwenImageTransformer Transformer { get; init; }
    public required QwenImageVaeDecoder Vae { get; init; }
    public QwenImageVaeEncoder VaeEncoder { get; init; }
    /// <summary>Qwen2.5-VL vision tower — non-null only for edit checkpoints whose TE file carried <c>visual.*</c>.</summary>
    public Qwen25VlVisionEncoder VisionEncoder { get; init; }
    /// <summary>Multimodal (vision-conditioned) encode path; non-null iff <see cref="VisionEncoder"/> is.</summary>
    public Qwen25VlMultimodalEncoder MultimodalEncoder { get; init; }
    /// <summary>2511 edit checkpoints modulate reference tokens at t=0 (qwen-image-edit-plus).</summary>
    public bool RefTimestepZero { get; init; }
    /// <summary>Null for GGUF checkpoints (see <see cref="GgufHandle"/>).</summary>
    public SafeTensorsLoader CheckpointLoader { get; init; }
    /// <summary>Owns the memory-mapped GGUF when the checkpoint was a .gguf; null for safetensors.</summary>
    public IDisposable GgufHandle { get; init; }
    public SafeTensorsLoader EncoderLoader { get; init; }
    public SafeTensorsLoader VaeLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        TextEncoder?.Dispose();
        Transformer?.Dispose();
        // VaeDecoder isn't IDisposable (no owned native handles freed here).
        VaeEncoder?.Dispose();
        VisionEncoder?.Dispose();
        CheckpointLoader?.Dispose();
        GgufHandle?.Dispose();
        EncoderLoader?.Dispose();
        VaeLoader?.Dispose();
    }
}
