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
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

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
        log($"Loading Qwen-Image checkpoint: {model.Name}");
        var (converted, mainLoader) = QwenImageCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (converted.Transformer.Count == 0)
        {
            mainLoader.Dispose();
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
        // SharpInference reference test, which does NOT cast the transformer to F32).
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

        // 3. Resolve + load the Qwen-Image VAE (shared with Anima).
        log("Building Qwen-Image VAE decoder (16-channel)...");
        VaeDecoder vae = new VaeDecoder(VaeConfig.QwenImage);
        if (converted.Vae.Count > 0)
        {
            vae.LoadWeights(CastToF32(converted.Vae));
        }
        else
        {
            T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.QwenImageVae, log: log);
            vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaeModel.RawFilePath);
            vae.LoadWeights(CastToF32(vaeLoader.GetAllTensors()));
        }

        log("Building Qwen-Image pipeline...");
        QwenImagePipeline pipeline = new QwenImagePipeline(backend, textEncoder, transformer, vae, config);

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
            CheckpointLoader = mainLoader,
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
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
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
        var (promptTokens, promptDrop) = EncodeWithTemplate(entry.Tokenizer, prompt);
        var (negTokens, negDrop) = EncodeWithTemplate(entry.Tokenizer, negative);

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfg,
            Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(
            promptTokens, negTokens, request, bridge,
            promptDropIndex: promptDrop, negativeDropIndex: negDrop);

        Logs.Verbose($"[SharpInference][Qwen-Image] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
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
    public required VaeDecoder Vae { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
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
        CheckpointLoader?.Dispose();
        EncoderLoader?.Dispose();
        VaeLoader?.Dispose();
    }
}
