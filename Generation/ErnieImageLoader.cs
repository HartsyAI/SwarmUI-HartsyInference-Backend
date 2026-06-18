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
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads ERNIE-Image (Baidu, ~8B, Apache-2.0). The picked file is the single-stream DiT transformer; the
/// Mistral3-shaped text encoder (Ministral-3-3B) and the Flux.2-style 128-channel VAE each resolve through the
/// central <see cref="SideModels"/> registry (auto-download, like Anima/Ideogram). The ERNIE tokenizer is a
/// HuggingFace-tokenizers <c>tokenizer.json</c> (Mistral3 byte-level BPE) auto-downloaded from
/// <c>baidu/ERNIE-Image</c> (the Comfy-Org repackage doesn't ship it).
///
/// <para>Mirrors the engine's <c>ErnieImageGenerationTests</c> wiring: <see cref="ErnieImageTransformer"/> +
/// <see cref="VaeDecoder"/>(<c>VaeConfig.Flux2</c>) + <see cref="ErnieImageLlamaTextEncoder"/> over
/// <see cref="LlamaStyleEncoderConfig.Ministral3B"/> → <see cref="ErnieImagePipeline"/>. The pipeline
/// self-manages GPU preload/free per generation, so no preload here. None of the converter's load methods
/// remap keys, so components load extension-side via <see cref="LoadComponent"/> (the converter is
/// folder-layout-only and incompatible with our per-component side-model paths).</para>
/// </summary>
public static class ErnieImageLoader
{
    public const string ErnieImageCompatClassId = "ernie-image";

    /// <summary>ERNIE tokenizer.json (Mistral3 BPE). Public on <c>baidu/ERNIE-Image</c> (~17 MB); the
    /// Comfy-Org repackage that ships the TE/VAE/DiT does NOT include it. Downloaded once to Models/clip/ERNIE/.</summary>
    private const string TokenizerJsonUrl =
        "https://huggingface.co/baidu/ERNIE-Image/resolve/main/tokenizer/tokenizer.json";

    public static ErnieImageCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("ERNIE-Image model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"ERNIE-Image checkpoint not found: {model.RawFilePath}");

        // Companion components auto-resolve via SideModels (download if missing):
        //   - text encoder: Ministral-3-3B (Mistral3-shaped), Models/clip/
        //   - VAE: Flux.2 KL autoencoder, Models/vae/ (shared with Flux.2)
        T2IModel teModel = ModelAutoDownloader.EnsureSideModel(
            userPick: null, entry: SideModels.Ministral_3_3B, log: log);
        T2IModel vaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.VAE), entry: SideModels.Flux2Vae, log: log);

        List<SafeTensorsLoader> loaders = [];
        try
        {
            log($"Loading ERNIE-Image transformer: {Path.GetFileName(model.RawFilePath)}");
            (Dictionary<string, Tensor> transformerWeights, SafeTensorsLoader transformerLoader) =
                LoadComponent(model.RawFilePath, applyFp8Dequant: true);
            loaders.Add(transformerLoader);

            log($"Loading Ministral-3-3B text encoder: {teModel.Name}");
            (Dictionary<string, Tensor> teWeights, SafeTensorsLoader teLoader) =
                LoadComponent(teModel.RawFilePath, applyFp8Dequant: true);
            loaders.Add(teLoader);

            log($"Loading Flux.2 VAE: {vaeModel.Name}");
            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) =
                LoadComponent(vaeModel.RawFilePath, applyFp8Dequant: false);
            loaders.Add(vaeLoader);

            ErnieImageConfig config = ErnieImageConfig.V1;

            log("Building ERNIE-Image transformer...");
            ErnieImageTransformer transformer = new(config);
            transformer.LoadWeights(transformerWeights);

            log("Building Flux.2 VAE decoder...");
            VaeDecoder vae = new(VaeConfig.Flux2);
            vae.LoadWeights(vaeWeights);

            log("Building Ministral-3-3B text encoder (hidden_states[-2] tap)...");
            LlamaStyleEncoder llama = new(LlamaStyleEncoderConfig.Ministral3B);
            llama.LoadWeights(teWeights);
            ErnieImageLlamaTextEncoder textEncoder = new ErnieImageLlamaTextEncoder(llama)
                .WithHiddenSize(LlamaStyleEncoderConfig.Ministral3B.HiddenSize);

            log("Resolving ERNIE tokenizer.json...");
            string tokenizerPath = EnsureTokenizerJson(log);
            ErnieTokenizer tokenizer = new(tokenizerPath);

            // TODO(faithfulness): wire the Flux.2 VAE BN-style latent un-normalization. The diffusers ERNIE
            // pipeline un-normalizes the latent with the VAE's bn.running_mean/running_var before decode;
            // ErnieImagePipeline accepts those as optional ctor args (vaeBnMean/vaeBnVar). Left null here to
            // match the engine's validated wiring test. If decoded images look washed-out/garbled, pass the
            // bn tensors — but first confirm the stage/shape: ApplyBnUnnormalize requires element-count ==
            // latent channels at the call site, which is the 128-ch PACKED latent (before unpatchify), not 32.
            log("Building ERNIE-Image pipeline...");
            ErnieImagePipeline pipeline = new(backend, textEncoder, transformer, vae, config);

            log("ERNIE-Image ready.");
            return new ErnieImageCacheEntry
            {
                ModelName = model.Name,
                CompatClass = ErnieImageCompatClassId,
                Pipeline = pipeline,
                Config = config,
                Tokenizer = tokenizer,
                TextEncoder = textEncoder,
                Llama = llama,
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
        ErnieImageCacheEntry entry,
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
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 25);

        // ERNIE tokenizer: raw prompt, no chat template, BOS prepended, EOS appended, no padding (the pipeline
        // handles sequence assembly). Negative is only used when cfg > 1 but we always tokenize it cheaply.
        int[] promptTokens = entry.Tokenizer.Encode(prompt);
        int[] negTokens = entry.Tokenizer.Encode(negative);

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

        (byte[] rgb, int outW, int outH, int _) = entry.Pipeline.GenerateFromTokens(
            promptTokens, negTokens, promptTokens.Length, negTokens.Length, request, bridge);

        Logs.Verbose($"[HartsyInference][ErnieImage] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return [RgbToImage.FromHwcRgb(rgb, outW, outH)];
    }

    /// <summary>Loads one component from a single safetensors file: drops fp8 <c>scaled_fp8</c> markers and
    /// optionally folds fp8 <c>*.scale_weight</c> companions (harmless no-op on plain bf16/fp16). Mirrors
    /// <c>ErnieImageCheckpointConverter</c> (which does no key remapping), but reads an exact resolved file —
    /// each component is its own registered side-model. The loader owns the tensor memory; keep it alive.</summary>
    private static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadComponent(
        string filePath, bool applyFp8Dequant)
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
                merged[kvp.Key] = kvp.Value;
            }
            return (applyFp8Dequant ? CheckpointConvertUtils.ApplyFp8ScaledDequant(merged) : merged, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Ensures the ERNIE tokenizer.json is present at <c>Models/clip/ERNIE/tokenizer.json</c>,
    /// downloading it from <see cref="TokenizerJsonUrl"/> on first use. Returns the local path.</summary>
    private static string EnsureTokenizerJson(Action<string> log)
    {
        string dir = Path.Combine(Program.T2IModelSets["Clip"].DownloadFolderPath, "ERNIE");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "tokenizer.json");
        if (!File.Exists(path))
        {
            log($"Downloading ERNIE tokenizer.json from {TokenizerJsonUrl}...");
            Utilities.DownloadFile(TokenizerJsonUrl, path, null).Wait();
        }
        return path;
    }
}

public sealed class ErnieImageCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required ErnieImagePipeline Pipeline { get; init; }
    public required ErnieImageConfig Config { get; init; }
    public required ErnieTokenizer Tokenizer { get; init; }
    public required ErnieImageLlamaTextEncoder TextEncoder { get; init; }
    public required LlamaStyleEncoder Llama { get; init; }
    public required ErnieImageTransformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required List<SafeTensorsLoader> Loaders { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        (TextEncoder as IDisposable)?.Dispose();
        Llama?.Dispose();
        Transformer?.Dispose();
        foreach (SafeTensorsLoader l in Loaders) l.Dispose();
    }
}
