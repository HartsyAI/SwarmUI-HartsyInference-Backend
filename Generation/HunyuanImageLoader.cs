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
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Loads HunyuanImage 2.1 checkpoints (17B MMDiT, 20 double + 40 single blocks, 32×/64-ch VAE).
/// GGUF repacks ship original-Tencent keys, remapped to diffusers naming by
/// <see cref="HunyuanImageCheckpointConverter.ConvertTencentToDiffusers"/> (quant-native, transient
/// per-GEMM dequant). Primary conditioning is Qwen2.5-VL-7B via <see cref="HunyuanImageQwenTextEncoder"/>;
/// the ByT5 glyph branch is not wired yet (optional at forward time).</summary>
public static class HunyuanImageLoader
{
    public const string HunyuanImageCompatClassId = "hunyuan-image-2_1";

    public static HunyuanImageCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("HunyuanImage model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"HunyuanImage checkpoint not found: {model.RawFilePath}");

        // ── 1. Transformer (GGUF: native quant, Tencent→diffusers remap in the converter) ──
        log($"Loading HunyuanImage transformer: {model.Name}");
        bool isGguf = model.RawFilePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        HunyuanImageCheckpointConverter.ConvertedWeights converted;
        SafeTensorsLoader mainLoader = null;
        IDisposable ggufHandle = null;
        if (isGguf)
        {
            GgufModelLoader.LoadedGgufModel gguf = GgufModelLoader.Load(model.RawFilePath);
            ggufHandle = gguf;
            Dictionary<string, Tensor> relabeled = GgufModelLoader.RelabelRank2ToPyTorchOrder(gguf.Weights);
            // BF16 GGUF tensors must become F16 — the transient weight path skips BF16 (blank image).
            Dictionary<string, Tensor> cast = new(relabeled.Count);
            foreach (KeyValuePair<string, Tensor> kvp in relabeled)
                cast[kvp.Key] = kvp.Value.DType == DType.BF16 ? kvp.Value.CastTo(DType.F16) : kvp.Value;
            converted = HunyuanImageCheckpointConverter.Convert(cast);
        }
        else
        {
            (converted, mainLoader) = HunyuanImageCheckpointConverter.LoadAndConvert(model.RawFilePath);
        }
        log($"Parsed checkpoint: {converted.Transformer.Count} transformer tensors (fp8={converted.IsFp8Mix}).");

        HunyuanImageConfig config = HunyuanImageConfig.V21;
        HunyuanImageTransformer transformer = new HunyuanImageTransformer(config);
        transformer.LoadWeights(converted.Transformer);

        // ── 2. Qwen2.5-VL-7B text encoder ──
        T2IModel qwenModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.QwenModel),
            entry: SideModels.Qwen25Vl7BHunyuan,
            log: log);
        log($"Loading Qwen2.5-VL-7B encoder: {qwenModel.Name}");
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenModel.RawFilePath);
        LlamaStyleEncoder llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
        llama.LoadWeights(qwenLoader.GetAllTensors());
        HunyuanImageQwenTextEncoder qwenEncoder = new HunyuanImageQwenTextEncoder(llama);

        // ── 3. HunyuanImage 32×/64-ch VAE ──
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE),
            entry: SideModels.HunyuanImageVae,
            log: log);
        log($"Loading HunyuanImage VAE: {vaeModel.Name}");
        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaeModel.RawFilePath);
        // The file ships old-LDM naming with up.0 = DEEPEST level (opposite of SD-LDM) — remap without
        // the usual index reversal.
        Dictionary<string, Tensor> vaeRaw = vaeLoader.GetAllTensors();
        Dictionary<string, Tensor> vaeWeights = new(vaeRaw.Count);
        foreach (KeyValuePair<string, Tensor> kvp in vaeRaw)
        {
            string mapped = CheckpointConvertUtils.ConvertVaeKey(kvp.Key, numUpLevels: 6, reverseUpIndices: false) ?? kvp.Key;
            vaeWeights[mapped] = kvp.Value;
        }
        HunyuanImageVaeDecoder vaeDecoder = new HunyuanImageVaeDecoder(VaeConfig.HunyuanImage);
        vaeDecoder.LoadWeights(vaeWeights);

        Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();

        log("Building HunyuanImage pipeline...");
        HunyuanImagePipeline pipeline = new HunyuanImagePipeline(backend, qwenEncoder, transformer, vaeDecoder, config);

        log("HunyuanImage 2.1 ready.");
        return new HunyuanImageCacheEntry
        {
            ModelName = model.Name,
            CompatClass = model.ModelClass?.CompatClass?.ID ?? HunyuanImageCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Tokenizer = tokenizer,
            Llama = llama,
            Transformer = transformer,
            Vae = vaeDecoder,
            CheckpointLoader = mainLoader,
            GgufHandle = ggufHandle,
            QwenLoader = qwenLoader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        HunyuanImageCacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.Prompt));
        string negative = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.NegativePrompt) ?? "");
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 20);
        float cfg = (float)input.Get(T2IParamTypes.CFGScale, 3.5);
        // 32× VAE + unit patches → dims must be a multiple of 32.
        int width = (input.Get(T2IParamTypes.Width) / 32) * 32;
        int height = (input.Get(T2IParamTypes.Height) / 32) * 32;
        long seedLong = input.Get(T2IParamTypes.Seed);
        int? seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF);

        (int[] ids, int[] mask) = TokenizePadded(entry.Tokenizer, prompt);
        bool useCfg = cfg > 1.0f;
        // An empty negative encodes to exactly the 34-token template, which the encoder's prefix-drop
        // rejects — give it one real token.
        if (useCfg && string.IsNullOrWhiteSpace(negative)) negative = ".";
        (int[] negIds, int[] negMask) = useCfg ? TokenizePadded(entry.Tokenizer, negative) : (null, null);

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            Width = width,
            Height = height,
            Steps = steps,
            Seed = seed,
            CfgScale = cfg,
        };
        Logs.Verbose($"[HartsyInference][HunyuanImage] steps={steps}, cfg={cfg}, {width}x{height}");

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };
        var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(
            ids, mask, negIds, negMask, request, onProgress: bridge);
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }

    /// <summary>Chat-template encode padded to the fixed 1034-token window with a matching attention mask (diffusers <c>_get_qwen_prompt_embeds</c>).</summary>
    private static (int[] ids, int[] mask) TokenizePadded(Qwen2Tokenizer tokenizer, string prompt)
    {
        int[] raw = tokenizer.EncodeChat(prompt, systemPrompt: HunyuanImageQwenTextEncoder.SystemPrompt, addGenerationPrompt: false);
        int realLen = Math.Min(raw.Length, HunyuanImageQwenTextEncoder.PaddedLength);
        int[] ids = Qwen2Tokenizer.PadToLength(raw, HunyuanImageQwenTextEncoder.PaddedLength);
        int[] mask = new int[HunyuanImageQwenTextEncoder.PaddedLength];
        for (int i = 0; i < realLen; i++) mask[i] = 1;
        return (ids, mask);
    }
}

public sealed class HunyuanImageCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required HunyuanImagePipeline Pipeline { get; init; }
    public required HunyuanImageConfig Config { get; init; }
    public required Qwen2Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder Llama { get; init; }
    public required HunyuanImageTransformer Transformer { get; init; }
    public required HunyuanImageVaeDecoder Vae { get; init; }
    /// <summary>Null for GGUF checkpoints (see <see cref="GgufHandle"/>).</summary>
    public SafeTensorsLoader CheckpointLoader { get; init; }
    /// <summary>Owns the memory-mapped GGUF when the checkpoint was a .gguf; null for safetensors.</summary>
    public IDisposable GgufHandle { get; init; }
    public required SafeTensorsLoader QwenLoader { get; init; }
    public required SafeTensorsLoader VaeLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Llama?.Dispose();
        CheckpointLoader?.Dispose();
        GgufHandle?.Dispose();
        QwenLoader?.Dispose();
        VaeLoader?.Dispose();
    }
}
