using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Anima (Cosmos-Predict2-2B family) models. Anima single-file checkpoints contain BOTH the DiT trunk
/// (<c>net.x_embedder</c>, <c>net.t_embedder</c>, <c>net.blocks.*</c>, <c>net.final_layer</c>) AND the
/// Anima-specific <c>llm_adapter</c> sub-transformer in one file. The Qwen-3 0.6B text encoder and the
/// Qwen-Image VAE come from separate Swarm-registered models picked through the normal parameter system:
/// <list type="bullet">
///   <item><c>T2IParamTypes.QwenModel</c> (Models/clip/) — Qwen-3 0.6B base.</item>
///   <item><c>T2IParamTypes.VAE</c> (Models/vae/) — Qwen-Image VAE.</item>
/// </list>
/// Both auto-download from the central <see cref="SideModels"/> registry if not present.
///
/// <para>Mirrors how Comfy's <c>WorkflowGeneratorModelSupport.cs:1065-1069</c> handles Anima (CLIPLoader type
/// <c>stable_diffusion</c> + VAELoader <c>qwen_image_vae</c>).</para>
/// </summary>
public static class AnimaLoader
{
    public const string AnimaCompatClassId = "anima";

    /// <summary>Qwen3 right-pads sequences with BosTokenId (151643).</summary>
    private const int Qwen3PadTokenId = 151643;

    /// <summary>Anima trains with T5 inputs tokenized to <c>max_length=512</c>. The LlmAdapter main-stream
    /// sequence length equals the T5 token count, and <c>AnimaPipeline</c> pads the adapter output to exactly
    /// 512 tokens (throwing if the input exceeds that), so the T5 token IDs must be capped at 512.</summary>
    private const int T5MaxTokens = 512;

    public static AnimaCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Anima model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Anima checkpoint not found: {model.RawFilePath}");

        // Resolve Qwen-3 0.6B and Qwen-Image VAE via the central SideModels registry.
        T2IModel qwenModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.QwenModel),
            entry: SideModels.Qwen3_0_6B,
            log: log);
        T2IModel vaeModel = ResolveQwenImageVae(input?.Get(T2IParamTypes.VAE), log);

        // 1. Load the single-file Anima checkpoint → DiT trunk + LlmAdapter buckets.
        log($"Loading Anima checkpoint: {model.Name}");
        var (converted, animaLoader) = AnimaCheckpointConverter.LoadAndConvert(model.RawFilePath);
        if (converted.Transformer.Count == 0)
        {
            animaLoader.Dispose();
            throw new InvalidOperationException(
                $"Anima checkpoint '{model.Name}' contains no DiT trunk weights (looked for <c>net.x_embedder.*</c> / <c>net.blocks.*</c>).");
        }
        if (converted.LlmAdapter.Count == 0)
        {
            animaLoader.Dispose();
            throw new InvalidOperationException(
                $"Anima checkpoint '{model.Name}' contains no <c>net.llm_adapter.*</c> weights. " +
                "This isn't a Comfy-format Anima single-file. Expected ~571 transformer tensors + ~118 llm_adapter tensors.");
        }
        log($"Parsed checkpoint: {converted.Transformer.Count} DiT tensors, {converted.LlmAdapter.Count} llm_adapter tensors, fp8_mix={converted.IsFp8Mix}.");

        AnimaConfig animaConfig = AnimaConfig.AnimaPreview3;

        log("Building Anima DiT transformer...");
        AnimaTransformer transformer = new AnimaTransformer(animaConfig);
        transformer.LoadWeights(converted.Transformer);

        log("Building Anima LlmAdapter (6-block self+cross+MLP)...");
        AnimaLlmAdapter llmAdapter = new AnimaLlmAdapter(animaConfig.LlmAdapter);
        llmAdapter.LoadWeights(converted.LlmAdapter);

        // 2. Load Qwen-3 0.6B base text encoder.
        log($"Loading Qwen-3 0.6B encoder weights: {qwenModel.Name}");
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenModel.RawFilePath);
        var qwenWeights = qwenLoader.GetAllTensors();
        if (qwenWeights.Count == 0)
        {
            qwenLoader.Dispose();
            animaLoader.Dispose();
            throw new InvalidOperationException($"Qwen-3 0.6B model file '{qwenModel.Name}' has no tensors.");
        }

        log("Building Qwen-3 0.6B encoder...");
        LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_0_6B);
        qwen.LoadWeights(qwenWeights);

        log("Loading Qwen3 tokenizer (embedded)...");
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 256);

        // Anima's text stack is dual: the Qwen-3 hidden states are the LlmAdapter cross-attention K/V
        // source, while a T5-XXL SentencePiece tokenization of the SAME prompt is the LlmAdapter's main
        // stream lookup (embed[t5_ids]). The T5 tokenizer here only produces token IDs — no T5 encoder
        // model is loaded. maxLength=512 matches Anima training (T5 padded to max_length=512); the pipeline
        // right-pads the LlmAdapter output to 512 itself, so we feed raw (unpadded) IDs capped at 512.
        log("Loading T5 tokenizer (embedded SentencePiece)...");
        T5Tokenizer t5Tokenizer = new T5Tokenizer(maxLength: T5MaxTokens);

        // 3. Load the Qwen-Image VAE.
        log($"Loading Qwen-Image VAE: {vaeModel.Name}");
        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaeModel.RawFilePath);
        // The Qwen-Image VAE is a 3D causal autoencoder (WAN 2.1 family). Keys use upstream WAN
        // naming verbatim (decoder.conv1, decoder.upsamples.N, decoder.middle.N, decoder.head.N),
        // NOT diffusers' AutoencoderKL naming. No key normalization needed — pass through directly.
        Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
        if (vaeWeights.Count == 0)
        {
            vaeLoader.Dispose();
            qwenLoader.Dispose();
            animaLoader.Dispose();
            throw new InvalidOperationException($"VAE file '{vaeModel.Name}' has no tensors.");
        }

        log("Building VAE decoder (Qwen-Image 3D causal, collapsed to 2D for T=1 image mode)...");
        QwenImageVaeDecoder vae = new QwenImageVaeDecoder(VaeConfig.QwenImage);
        vae.LoadWeights(vaeWeights);

        log("Building Anima pipeline...");
        AnimaPipeline pipeline = new AnimaPipeline(backend, transformer, llmAdapter, vae, animaConfig);

        log("Anima ready (scheduler=FlowMatchEuler shift=3.0; ER-SDE pixel-parity TODO).");
        return new AnimaCacheEntry
        {
            ModelName = model.Name,
            CompatClass = AnimaCompatClassId,
            Pipeline = pipeline,
            AnimaConfig = animaConfig,
            Tokenizer = tokenizer,
            T5Tokenizer = t5Tokenizer,
            Qwen = qwen,
            Transformer = transformer,
            LlmAdapter = llmAdapter,
            Vae = vae,
            AnimaCheckpointLoader = animaLoader,
            QwenLoader = qwenLoader,
            VaeLoader = vaeLoader,
        };
    }

    public static Image[] Generate(
        AnimaCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 30);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float cfg = cfgRaw <= 0 ? 1.0f : (float)cfgRaw;

        // Plain (non-chat) tokenization — Anima's reference workflow uses CLIPLoader type="stable_diffusion",
        // which is Comfy's path for raw Qwen-3 text encoding (no chat template).
        int[] tokenIds = entry.Tokenizer.Encode(prompt, appendEos: true);
        int realLen = ComputeRealLength(tokenIds);
        Tensor encodedFull = entry.Qwen.Encode(backend, new[] { tokenIds });
        Tensor positiveEmbeddings = SliceFirstSeqF32(encodedFull, realLen);

        // T5 tokenization of the SAME prompt — the LlmAdapter main stream (embed[t5_ids]).
        int[] positiveT5Ids = EncodeT5(entry.T5Tokenizer, prompt);

        Tensor negativeEmbeddings = null;
        int[] negativeT5Ids = null;
        if (cfg > 1.0f)
        {
            int[] negTokens = entry.Tokenizer.Encode(negative, appendEos: true);
            int negRealLen = ComputeRealLength(negTokens);
            Tensor negEncodedFull = entry.Qwen.Encode(backend, new[] { negTokens });
            negativeEmbeddings = SliceFirstSeqF32(negEncodedFull, negRealLen);
            negativeT5Ids = EncodeT5(entry.T5Tokenizer, negative);
        }

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromEmbeddings(
            positiveEmbeddings,
            positiveT5Ids,
            request,
            cfgScale: cfg,
            negativeTextEmbeddings: negativeEmbeddings,
            negativeT5TokenIds: negativeT5Ids,
            onProgress: bridge);

        Logs.Verbose($"[HartsyInference][Anima] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
    }

    /// <summary>Top-level keys unique to the Qwen-Image / WAN-family VAE that <see cref="QwenImageVaeDecoder"/>
    /// requires verbatim. A Flux/SD-style VAE (diffusers naming) has neither, so probing for these cheaply
    /// distinguishes a compatible pick from an incompatible one.</summary>
    private static readonly string[] QwenImageVaeSignatureKeys = ["conv2.weight", "decoder.conv1.weight"];

    /// <summary>Resolves the Qwen-Image VAE for Anima. Anima's decoder is architecturally fixed to the
    /// Qwen-Image (WAN 2.1 family) VAE, so an incompatible user VAE pick (e.g. a Flux/SD ae left selected
    /// from another model) must NOT be honored — otherwise <see cref="QwenImageVaeDecoder.LoadWeights"/>
    /// throws on the missing <c>conv2.weight</c>. A compatible custom pick is still honored; otherwise we
    /// drop the pick and let <see cref="ModelAutoDownloader.EnsureSideModel"/> auto-resolve (and download
    /// if needed) the canonical <see cref="SideModels.QwenImageVae"/>.</summary>
    private static T2IModel ResolveQwenImageVae(T2IModel userPick, Action<string> log)
    {
        if (userPick is not null && !IsQwenImageVae(userPick))
        {
            log($"Selected VAE '{userPick.Name}' is not a Qwen-Image VAE (missing {string.Join(" / ", QwenImageVaeSignatureKeys)}); " +
                "ignoring it and auto-resolving the Qwen-Image VAE that Anima requires.");
            userPick = null;
        }
        return ModelAutoDownloader.EnsureSideModel(userPick, SideModels.QwenImageVae, log);
    }

    /// <summary>Header-only probe: true iff the file carries every <see cref="QwenImageVaeSignatureKeys"/>
    /// entry. Reads just the safetensors JSON header (no tensor data materialized).</summary>
    private static bool IsQwenImageVae(T2IModel model)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath) || !File.Exists(model.RawFilePath))
        {
            return false;
        }
        try
        {
            using SafeTensorsLoader probe = new SafeTensorsLoader();
            probe.Load(model.RawFilePath);
            foreach (string key in QwenImageVaeSignatureKeys)
            {
                if (!probe.Descriptors.ContainsKey(key)) return false;
            }
            return true;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference][Anima] Could not probe VAE '{model.Name}' ({ex.Message}); treating as incompatible.");
            return false;
        }
    }

    /// <summary>Produces Anima's T5 main-stream token IDs: raw SentencePiece tokens + a single EOS,
    /// with NO padding (the pipeline right-pads the LlmAdapter output to 512 itself). Mirrors the
    /// Python reference <c>sp.encode(prompt, add_eos=True)</c>. Capped at <see cref="T5MaxTokens"/>
    /// (511 tokens + EOS) since the pipeline throws when the adapter sequence exceeds 512.</summary>
    private static int[] EncodeT5(T5Tokenizer t5, string text)
    {
        IReadOnlyList<int> raw = t5.EncodeRaw(text);
        int tokenCount = Math.Min(raw.Count, T5MaxTokens - 1); // reserve one slot for EOS
        int[] result = new int[tokenCount + 1];
        for (int i = 0; i < tokenCount; i++)
        {
            result[i] = raw[i];
        }
        result[tokenCount] = T5Tokenizer.EosTokenId;
        return result;
    }

    private static int ComputeRealLength(int[] tokenIds)
    {
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] == Qwen3PadTokenId) return i;
        }
        return tokenIds.Length;
    }

    private static unsafe Tensor SliceFirstSeqF32(Tensor source, int realLen)
    {
        if (source.Shape.Rank != 3)
            throw new ArgumentException($"Expected 3D tensor, got rank {source.Shape.Rank}.");
        if (source.DType != DType.F32)
            throw new ArgumentException($"SliceFirstSeqF32 expects F32, got {source.DType}.");

        long batch = source.Shape[0];
        long fullLen = source.Shape[1];
        long hidden = source.Shape[2];
        if (realLen <= 0 || realLen > fullLen)
            throw new ArgumentOutOfRangeException(nameof(realLen), $"realLen {realLen} out of range [1..{fullLen}].");

        TensorShape outShape = new TensorShape(batch, realLen, hidden);
        Tensor result = new Tensor(outShape, source.DType);
        long elemSize = source.DType.SizeInBytes;
        long fullRowBytes = fullLen * hidden * elemSize;
        long sliceRowBytes = realLen * hidden * elemSize;

        byte* src = (byte*)source.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        for (long b = 0; b < batch; b++)
        {
            Buffer.MemoryCopy(src + b * fullRowBytes, dst + b * sliceRowBytes, sliceRowBytes, sliceRowBytes);
        }
        return result;
    }
}

public sealed class AnimaCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required AnimaPipeline Pipeline { get; init; }
    public required AnimaConfig AnimaConfig { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required T5Tokenizer T5Tokenizer { get; init; }
    public required LlamaStyleEncoder Qwen { get; init; }
    public required AnimaTransformer Transformer { get; init; }
    public required AnimaLlmAdapter LlmAdapter { get; init; }
    public required QwenImageVaeDecoder Vae { get; init; }
    public required SafeTensorsLoader AnimaCheckpointLoader { get; init; }
    public required SafeTensorsLoader QwenLoader { get; init; }
    public required SafeTensorsLoader VaeLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        Tokenizer?.Dispose();
        T5Tokenizer?.Dispose();
        Qwen?.Dispose();
        Transformer?.Dispose();
        LlmAdapter?.Dispose();
        AnimaCheckpointLoader?.Dispose();
        QwenLoader?.Dispose();
        VaeLoader?.Dispose();
    }
}
