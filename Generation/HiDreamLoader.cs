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
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads HiDream-I1 (Full / Dev). HiDream is an MMDiT with FOUR text encoders — CLIP-L, CLIP-G,
/// T5-XXL, and Llama-3.1-8B (used as a feature extractor; the pipeline harvests hidden states from
/// every Llama layer). The diffusion_models checkpoint is transformer-only; the encoders and the
/// (Flux) VAE come from separate Swarm-registered models picked through the normal parameter system,
/// with central <see cref="SideModels"/> auto-download when the user leaves them blank:
/// <list type="bullet">
///   <item><c>T2IParamTypes.ClipLModel</c> — HiDream long-CLIP-L.</item>
///   <item><c>T2IParamTypes.ClipGModel</c> — HiDream long-CLIP-G.</item>
///   <item><c>T2IParamTypes.T5XXLModel</c> — T5-XXL encoder.</item>
///   <item><c>T2IParamTypes.LLaMAModel</c> — Llama-3.1-8B-Instruct.</item>
///   <item><c>T2IParamTypes.VAE</c> — Flux VAE (HiDream reuses it).</item>
/// </list>
///
/// <para>Mirrors Comfy's HiDream path (<c>WorkflowGeneratorModelSupport.cs:1111-1127</c>): a
/// QuadrupleCLIPLoader feeding clip_l_hidream + clip_g_hidream + t5xxl + llama_3.1_8b, plus the
/// flux-ae VAE.</para>
///
/// <para><b>Llama tokenizer requirement:</b> the Llama-3.1 branch needs a real 128k-vocab BPE
/// tokenizer. HartsyInference ships it as an embedded resource only when the asset files are present
/// in the build (see <see cref="EmbeddedTokenizerResources.HasLlama3Assets"/>). If they're absent,
/// loading fails here with a clear instruction rather than producing garbage from placeholder tokens.</para>
/// </summary>
public static class HiDreamLoader
{
    public const string HiDreamI1CompatClassId = "hidream-i1";

    public static HiDreamCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("HiDream model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"HiDream checkpoint not found: {model.RawFilePath}");

        // Fail fast with an actionable message if the Llama-3.1 tokenizer assets weren't embedded
        // into this HartsyInference build. Without them the Llama branch can't tokenize and HiDream
        // output would be garbage — better to refuse at load than silently produce noise.
        if (!EmbeddedTokenizerResources.HasLlama3Assets)
        {
            throw new InvalidOperationException(
                "HiDream needs the Llama-3.1 tokenizer, which isn't embedded in this HartsyInference build. " +
                "Extract vocab.json + merges.txt from the Llama-3.1 tokenizer.json and drop them into " +
                "HartsyInference/src/HartsyInference.Tokenizers/Resources/ as llama3_vocab.json + llama3_merges.txt, " +
                "then rebuild HartsyInference (and reload this extension).");
        }

        // 1. Load + convert the HiDream transformer (and any bundled components in an all-in-one file).
        log($"Loading HiDream checkpoint: {model.Name}");
        var (converted, mainLoader) = HiDreamCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (converted.Transformer.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException(
                $"HiDream checkpoint '{model.Name}' contains no transformer weights " +
                "(looked for <c>caption_projection.*</c> / <c>double_stream_blocks.*</c>).");
        }
        log($"Parsed checkpoint: {converted.Transformer.Count} transformer tensors, fp8_mix={converted.IsFp8Mix}.");

        // Auto-detect depth (Full and Dev share architecture; AutoDetect substitutes detected block counts).
        HiDreamConfig config = HiDreamConfig.AutoDetect(converted.Transformer);

        log("Building HiDream transformer...");
        HiDreamTransformer transformer = new HiDreamTransformer(config);
        transformer.LoadWeights(CastToF32(converted.Transformer));

        // 2. Resolve the four text encoders + VAE. Prefer components bundled inside an all-in-one
        // checkpoint; otherwise resolve a user pick or auto-download via SideModels, then load standalone.
        SafeTensorsLoader clipLLoader = null, clipGLoader = null, t5Loader = null, llamaLoader = null, vaeLoader = null;

        log("Building CLIP-L encoder...");
        ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        if (converted.ClipL.Count > 0)
        {
            clipL.LoadWeights(converted.ClipL, prefix: "text_model");
        }
        else
        {
            T2IModel clipLModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.ClipLModel), entry: SideModels.HiDreamClipL, log: log);
            clipLLoader = new SafeTensorsLoader();
            clipLLoader.Load(clipLModel.RawFilePath);
            clipL.LoadWeights(StripStandalonePrefix(clipLLoader.GetAllTensors(), "text_encoders.clip_l.transformer."), prefix: "text_model");
        }

        log("Building CLIP-G encoder...");
        ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
        if (converted.ClipG.Count > 0)
        {
            clipG.LoadWeights(converted.ClipG, prefix: "text_model");
        }
        else
        {
            T2IModel clipGModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.ClipGModel), entry: SideModels.HiDreamClipG, log: log);
            clipGLoader = new SafeTensorsLoader();
            clipGLoader.Load(clipGModel.RawFilePath);
            clipG.LoadWeights(StripStandalonePrefix(clipGLoader.GetAllTensors(), "text_encoders.clip_g.transformer."), prefix: "text_model");
        }

        log("Building T5-XXL encoder...");
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        if (converted.T5.Count > 0)
        {
            t5.LoadWeights(converted.T5);
        }
        else
        {
            T2IModel t5Model = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.T5XXLModel), entry: SideModels.T5XxlEnconly, log: log);
            t5Loader = new SafeTensorsLoader();
            t5Loader.Load(t5Model.RawFilePath);
            t5.LoadWeights(StripStandalonePrefix(t5Loader.GetAllTensors(), "text_encoders.t5xxl.transformer."));
        }

        log("Building Llama-3.1-8B encoder...");
        LlamaStyleEncoder llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Llama31_8B);
        if (converted.Llama.Count > 0)
        {
            llama.LoadWeights(converted.Llama);
        }
        else
        {
            T2IModel llamaModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.LLaMAModel), entry: SideModels.Llama31_8B, log: log);
            llamaLoader = new SafeTensorsLoader();
            llamaLoader.Load(llamaModel.RawFilePath);
            llama.LoadWeights(StripStandalonePrefix(llamaLoader.GetAllTensors(), "text_encoders.llama.transformer."));
        }

        log("Building VAE decoder (Flux config)...");
        VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
        if (converted.Vae.Count > 0)
        {
            vae.LoadWeights(CastToF32(converted.Vae));
        }
        else
        {
            T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.FluxAe, log: log);
            vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaeModel.RawFilePath);
            // Flux VAE ships fp16/bf16; VaeDecoder expects F32 (matches the HiDream reference test).
            vae.LoadWeights(CastToF32(LoadVaeFromStandalone(vaeLoader.GetAllTensors())));
        }

        log("Building HiDream pipeline...");
        HiDreamPipeline pipeline = new HiDreamPipeline(backend, clipL, clipG, t5, llama, transformer, vae, config);

        log("Loading tokenizers (CLIP + T5 + Llama, embedded)...");
        ClipTokenizer clipTokenizer = new ClipTokenizer();
        T5Tokenizer t5Tokenizer = new T5Tokenizer(maxLength: 256);
        LlamaTokenizer llamaTokenizer = new LlamaTokenizer(maxLength: 256);

        log("HiDream ready (4 text encoders; scheduler=flow-match).");
        return new HiDreamCacheEntry
        {
            ModelName = model.Name,
            CompatClass = HiDreamI1CompatClassId,
            Pipeline = pipeline,
            HiDreamConfig = config,
            ClipTokenizer = clipTokenizer,
            T5Tokenizer = t5Tokenizer,
            LlamaTokenizer = llamaTokenizer,
            ClipL = clipL,
            ClipG = clipG,
            T5 = t5,
            Llama = llama,
            Transformer = transformer,
            Vae = vae,
            CheckpointLoader = mainLoader,
            ClipLLoader = clipLLoader,
            ClipGLoader = clipGLoader,
            T5Loader = t5Loader,
            LlamaLoader = llamaLoader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        HiDreamCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.Prompt));
        string negative = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.NegativePrompt));
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 28);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfg = cfgRaw <= 0 ? 5.0f : (float)cfgRaw;

        // CLIP-L and CLIP-G share the same BPE tokenizer; encode once and reuse for both branches
        // (matches the HiDream reference, which feeds the same token ids to both CLIP encoders).
        int[] clipTokens = entry.ClipTokenizer.Encode(prompt);
        int[] negClipTokens = entry.ClipTokenizer.Encode(negative);
        int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
        int negEosPos = ClipTokenizer.FindEosPosition(negClipTokens);

        // T5 (with attention mask) and Llama-3.1. Always encode the negative too — the pipeline's
        // parameters are non-optional; it decides internally whether to run the negative pass (cfg>1).
        int[] t5Tokens = entry.T5Tokenizer.Encode(prompt);
        int[] negT5Tokens = entry.T5Tokenizer.Encode(negative);
        int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);
        int[] negT5Mask = T5Tokenizer.CreateAttentionMask(negT5Tokens);

        int[] llamaTokens = entry.LlamaTokenizer.Encode(prompt);
        int[] negLlamaTokens = entry.LlamaTokenizer.Encode(negative);

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
            clipTokens, negClipTokens,
            clipTokens, negClipTokens,
            eosPos, negEosPos,
            eosPos, negEosPos,
            t5Tokens, negT5Tokens,
            t5Mask, negT5Mask,
            llamaTokens, negLlamaTokens,
            request, bridge);

        Logs.Verbose($"[HartsyInference][HiDream] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
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

    private static Dictionary<string, Tensor> StripStandalonePrefix(Dictionary<string, Tensor> raw, string comfyPrefix)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var kv in raw)
        {
            if (kv.Key.StartsWith(comfyPrefix, StringComparison.Ordinal))
            {
                string rest = kv.Key[comfyPrefix.Length..];
                if (!rest.EndsWith("position_ids", StringComparison.Ordinal))
                    result[rest] = kv.Value;
            }
            else
            {
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }

    private static Dictionary<string, Tensor> LoadVaeFromStandalone(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var kv in raw)
        {
            string k = kv.Key;
            if (k.StartsWith("vae.", StringComparison.Ordinal)) k = k["vae.".Length..];
            else if (k.StartsWith("first_stage_model.", StringComparison.Ordinal)) k = k["first_stage_model.".Length..];
            result[k] = kv.Value;
        }
        return result;
    }
}

public sealed class HiDreamCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required HiDreamPipeline Pipeline { get; init; }
    public required HiDreamConfig HiDreamConfig { get; init; }
    public required ClipTokenizer ClipTokenizer { get; init; }
    public required T5Tokenizer T5Tokenizer { get; init; }
    public required LlamaTokenizer LlamaTokenizer { get; init; }
    public required ClipTextEncoder ClipL { get; init; }
    public required ClipTextEncoder ClipG { get; init; }
    public required T5TextEncoder T5 { get; init; }
    public required LlamaStyleEncoder Llama { get; init; }
    public required HiDreamTransformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public SafeTensorsLoader ClipLLoader { get; init; }
    public SafeTensorsLoader ClipGLoader { get; init; }
    public SafeTensorsLoader T5Loader { get; init; }
    public SafeTensorsLoader LlamaLoader { get; init; }
    public SafeTensorsLoader VaeLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        ClipTokenizer?.Dispose();
        T5Tokenizer?.Dispose();
        LlamaTokenizer?.Dispose();
        // ClipTextEncoder and VaeDecoder aren't IDisposable (no owned native handles freed here);
        // only the heavy encoders and the transformer are.
        T5?.Dispose();
        Llama?.Dispose();
        Transformer?.Dispose();
        CheckpointLoader?.Dispose();
        ClipLLoader?.Dispose();
        ClipGLoader?.Dispose();
        T5Loader?.Dispose();
        LlamaLoader?.Dispose();
        VaeLoader?.Dispose();
    }
}
