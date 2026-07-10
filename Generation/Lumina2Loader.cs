using System.IO;
using SwarmUI.Core;
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
/// Loads Lumina-Image-2.0 (Alpha-VLLM, 2B NextDiT, Apache-2.0). The picked file is the DIFFUSERS-format
/// transformer (the original AlphaVLLM single-file uses fused/legacy keys the converter rejects); the
/// Gemma-2-2B text encoder and FLUX.1 16-channel VAE resolve through <see cref="SideModels"/>.
///
/// <para>Conditioning is the LIVE Gemma-2-2B encode matching the verified reference recipe
/// (<c>tests/python-reference/encode_lumina2_prompt.py</c>, cross-checked against ComfyUI's
/// <c>text_encoders/lumina2.py</c>): the prompt is prefixed with the Lumina system prompt joined by
/// <c>" &lt;Prompt Start&gt; "</c> as ONE plain string (no role markers), tokenized with BOS and no EOS, and the
/// conditioning is <c>hidden_states[-2]</c> — the SECOND-to-last layer, tapped WITHOUT the final RMSNorm
/// (ComfyUI <c>layer="hidden", layer_idx=-2</c>) — via <see cref="LlamaStyleEncoder.EncodeMultiLayer"/> at
/// index <c>NumLayers - 1</c>. Real tokens only (both the engine pipeline and the recipe are mask-free).</para>
///
/// <para>The Gemma-2 encoder is the engine's documented approximation (no attention soft-cap / sliding
/// window); structurally faithful, small drift possible on very long prompts. Mirrors
/// <c>Lumina2GenerationTests</c> (F32 transformer + F32 Flux VAE).</para>
/// </summary>
public static class Lumina2Loader
{
    public const string Lumina2CompatClassId = "lumina-2";

    /// <summary>Lumina-Image-2.0's system prompt, verbatim from the official pipeline.</summary>
    private const string SystemPrompt =
        "You are an assistant designed to generate superior images with the superior degree of " +
        "image-text alignment based on textual prompts or user prompts.";

    /// <summary>Gemma-2 SentencePiece model (256k vocab — distinct from Gemma-3's). Ungated on the
    /// Lumina-Image-2.0 repo; downloaded once to Models/clip/Gemma2/.</summary>
    private const string TokenizerUrl =
        "https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/resolve/main/tokenizer/tokenizer.model";
    private const string TokenizerSha256 =
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2";

    public static Lumina2CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Lumina-2 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Lumina-2 checkpoint not found: {model.RawFilePath}");

        T2IModel teModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.GemmaModel), entry: SideModels.Gemma2_2B, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.FluxAe, log: log);

        List<SafeTensorsLoader> loaders = [];
        try
        {
            log($"Loading Lumina-2 transformer (2B): {Path.GetFileName(model.RawFilePath)}");
            (Lumina2CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) =
                Lumina2CheckpointConverter.LoadAndConvert(model.RawFilePath);
            loaders.Add(transformerLoader);
            Dictionary<string, Tensor> transformerWeights = CastWeightsToF32(converted.Transformer);

            log("Building Lumina-2 transformer...");
            Lumina2Config config = Lumina2Config.FromWeights(transformerWeights);
            Lumina2Transformer transformer = new(config);
            transformer.LoadWeights(transformerWeights);

            log($"Loading Gemma-2-2B text encoder: {teModel.Name}");
            SafeTensorsLoader teLoader = new();
            teLoader.Load(teModel.RawFilePath);
            loaders.Add(teLoader);
            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Gemma2_2B);
            textEncoder.LoadWeights(teLoader.GetAllTensors());

            log($"Loading FLUX.1 VAE: {vaeModel.Name}");
            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) =
                LoaderVaeUtils.LoadFluxVaeF32(vaeModel.RawFilePath);
            loaders.Add(vaeLoader);
            VaeDecoder vae = new(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);

            log("Resolving Gemma-2 tokenizer.model...");
            string tokenizerPath = EnsureTokenizer(teModel, log);
            GemmaTokenizer tokenizer = new(tokenizerPath, maxLength: 512);

            log("Building Lumina-2 pipeline...");
            Lumina2Pipeline pipeline = new(backend, transformer, vae, config);

            log("Lumina-2 ready.");
            return new Lumina2CacheEntry
            {
                ModelName = model.Name,
                CompatClass = Lumina2CompatClassId,
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

    public static Image[] Generate(
        Lumina2CacheEntry entry,
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
        float cfg = cfgRaw <= 0 ? 4.0f : (float)cfgRaw;
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 25);

        Tensor condEmbeds = entry.GetOrEncode(backend, prompt);
        // The Lumina-2 pipeline requires negative embeddings whenever cfg > 1 (the official pipeline always
        // runs a negative pass at its default cfg=4) — the empty prompt through the same template is the
        // reference uncond.
        Tensor negEmbeds = cfg > 1.0f ? entry.GetOrEncodeNegative(backend, negative) : null;

        TextToImageRequest request = new()
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

        (byte[] rgb, int outW, int outH, int _) = entry.Pipeline.GenerateFromEmbeddings(
            condEmbeds, request,
            cfgScale: cfg,
            negativeCaptionEmbeddings: negEmbeds,
            onProgress: bridge);

        Logs.Verbose($"[HartsyInference][Lumina2] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return [RgbToImage.FromHwcRgb(rgb, outW, outH)];
    }

    /// <summary>Builds the Lumina conditioning text: system prompt + <c>" &lt;Prompt Start&gt; "</c> + prompt as
    /// one plain string (the reference recipe — no role markers, negative uses the same template with its own
    /// text, "" giving the reference uncond).</summary>
    internal static string ApplyTemplate(string prompt) => SystemPrompt + " <Prompt Start> " + prompt;

    /// <summary>Locates the Gemma-2 SentencePiece model: next to the encoder checkpoint, then the canonical
    /// <c>Models/clip/gemma2_tokenizer.model</c>, else downloads it (hash-verified) to Models/clip/Gemma2/.</summary>
    private static string EnsureTokenizer(T2IModel teModel, Action<string> log)
    {
        string encoderDir = Path.GetDirectoryName(teModel.RawFilePath) ?? "";
        foreach (string candidate in new[]
        {
            Path.Combine(encoderDir, "gemma2_tokenizer.model"),
            Path.Combine(encoderDir, "tokenizer.model"),
        })
        {
            if (File.Exists(candidate))
                return candidate;
        }
        string dir = Path.Combine(Program.T2IModelSets["Clip"].DownloadFolderPath, "Gemma2");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "tokenizer.model");
        if (!File.Exists(path))
        {
            log($"Downloading Gemma-2 tokenizer.model from {TokenizerUrl}...");
            Utilities.DownloadFile(TokenizerUrl, path, null, verifyHash: TokenizerSha256).Wait();
        }
        return path;
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            f32[kvp.Key] = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return f32;
    }
}

public sealed class Lumina2CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required Lumina2Pipeline Pipeline { get; init; }
    public required Lumina2Config Config { get; init; }
    public required GemmaTokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder TextEncoder { get; init; }
    public required Lumina2Transformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required List<SafeTensorsLoader> Loaders { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    // Single-slot prompt-embedding caches (cond + negative), keyed on the raw prompt text — repeat prompts
    // skip the Gemma-2 forward. Host-materialized so activation reclaims can't revert them.
    private string _cachedPrompt;
    private Tensor _cachedEmbeds;
    private string _cachedNegPrompt;
    private Tensor _cachedNegEmbeds;

    /// <summary>Returns the [1, T, 2304] <c>hidden_states[-2]</c> conditioning for the prompt (one-slot cache).
    /// Tap index NumLayers-1 in the encoder's HF convention (0 = embeddings) = the input of the LAST layer's
    /// successor, i.e. the second-to-last hidden state, WITHOUT the final RMSNorm — the verified recipe.</summary>
    public unsafe Tensor GetOrEncode(IBackend backend, string prompt)
    {
        if (_cachedEmbeds is not null && _cachedPrompt == prompt)
            return _cachedEmbeds;
        Tensor embeds = EncodeTemplated(backend, prompt);
        _cachedEmbeds?.Dispose();
        _cachedEmbeds = embeds;
        _cachedPrompt = prompt;
        return embeds;
    }

    /// <summary>Negative-prompt twin of <see cref="GetOrEncode"/>.</summary>
    public unsafe Tensor GetOrEncodeNegative(IBackend backend, string negative)
    {
        if (_cachedNegEmbeds is not null && _cachedNegPrompt == negative)
            return _cachedNegEmbeds;
        Tensor embeds = EncodeTemplated(backend, negative);
        _cachedNegEmbeds?.Dispose();
        _cachedNegEmbeds = embeds;
        _cachedNegPrompt = negative;
        return embeds;
    }

    private unsafe Tensor EncodeTemplated(IBackend backend, string prompt)
    {
        int[] tokens = Tokenizer.Encode(Lumina2Loader.ApplyTemplate(prompt));
        int tapIndex = TextEncoder.NumLayers - 1;   // hidden_states[-2]
        Tensor embeds = TextEncoder.EncodeMultiLayer(backend, [tokens], [tapIndex]);
        _ = embeds.DataPointer;   // host-materialize: survives activation reclaims across generations
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
