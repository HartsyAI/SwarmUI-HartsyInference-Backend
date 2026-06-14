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
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Ideogram 4 (<c>ideogram-oss/ideogram4</c>, 9.3B, released 2026-06-03).
/// Architecture: Qwen3-VL-8B language tower (13-layer hidden-state tap) → TWO 9.3B
/// single-stream DiTs (conditional + unconditional, asymmetric CFG) → Flux.2 VAE.
///
/// Ships as a folder (no single-file release). Expected layouts (either works —
/// <see cref="Ideogram4CheckpointConverter"/> probes both):
/// <code>
///   Ideogram4/
///   ├── transformer/                ← conditional DiT shards        (diffusers)
///   ├── unconditional_transformer/  ← unconditional DiT shards      (diffusers)
///   ├── text_encoder/               ← Qwen3-VL-8B shards            (diffusers)
///   └── vae/                        ← Flux.2 VAE                    (both layouts)
/// </code>
/// or Comfy-Org: <c>diffusion_models/ideogram4_*.safetensors</c> (+ <c>*unconditional*</c>),
/// <c>text_encoders/qwen3vl_8b_*.safetensors</c>, <c>vae/</c>. The user picks any
/// .safetensors inside the folder; we walk up to the root (F-Lite pattern).
///
/// VRAM: BOTH 9.3B transformers are resident during the denoise loop (each step runs
/// both) — ~19 GB at fp8 for the DiTs alone. We gate on ≥22 GB free (same floor as the
/// upstream E2E test) on CUDA and refuse below it rather than OOM mid-generation.
///
/// Prompting: Ideogram 4 was trained on structured JSON captions; plain text works but
/// the official stack optionally expands prompts via the hosted "magic prompt" API. We
/// pass the user's prompt through the Qwen3 chat template as-is (no server round-trip).
/// Steps/guidance come from the official sampler presets (Turbo12 / Default20 /
/// Quality48) — the Steps param picks the nearest preset; CFG Scale and negative prompt
/// are ignored by design (asymmetric CFG has a baked per-step guidance schedule).
///
/// License note: the released weights are "Ideogram 4 Non-Commercial".
/// </summary>
public static class Ideogram4Loader
{
    public const string Ideogram4CompatClassId = "ideogram-4";

    /// <summary>Minimum free VRAM to attempt a CUDA load — both DiTs + headroom.
    /// Matches the upstream E2E test's floor.</summary>
    private const double MinRequiredVramGb = 22.0;

    public static Ideogram4CacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Ideogram 4 model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Ideogram 4 checkpoint not found: {model.RawFilePath}");

        // Gate BEFORE the multi-minute weight load, not after.
        if (backend is CudaBackend cuda)
        {
            (nuint freeBytes, nuint totalBytes) = cuda.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < MinRequiredVramGb)
            {
                throw new InvalidOperationException(
                    $"Ideogram 4 needs ≥{MinRequiredVramGb:F0} GB free VRAM (both 9.3B transformers stay resident " +
                    $"for asymmetric CFG); this GPU has {freeGb:F1} GB free of {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB total. " +
                    "Use a higher-VRAM GPU or the ComfyUI backend for this model.");
            }
        }
        else
        {
            log("WARNING: Ideogram 4 on a non-CUDA backend — two 9.3B transformers per step will be extremely slow.");
        }

        string folderPath = FLiteLoader.ResolveFolderRoot(model.RawFilePath,
            "transformer", "unconditional_transformer", "diffusion_models");
        log($"Loading Ideogram 4 folder: {Path.GetFileName(folderPath)} (from picked file {Path.GetFileName(model.RawFilePath)})");
        log("Note: Ideogram 4 weights are under a NON-COMMERCIAL license.");

        List<SafeTensorsLoader> loaders = [];
        try
        {
            log("Loading conditional transformer (9.3B)...");
            (Dictionary<string, Tensor> condW, IReadOnlyList<SafeTensorsLoader> condL) =
                Ideogram4CheckpointConverter.LoadTransformer(folderPath, unconditional: false);
            loaders.AddRange(condL);

            log("Loading unconditional transformer (9.3B)...");
            (Dictionary<string, Tensor> uncondW, IReadOnlyList<SafeTensorsLoader> uncondL) =
                Ideogram4CheckpointConverter.LoadTransformer(folderPath, unconditional: true);
            loaders.AddRange(uncondL);

            log("Loading Qwen3-VL-8B text encoder...");
            (Dictionary<string, Tensor> teW, IReadOnlyList<SafeTensorsLoader> teL) =
                Ideogram4CheckpointConverter.LoadTextEncoder(folderPath);
            loaders.AddRange(teL);

            log("Loading Flux.2 VAE...");
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vaeL) =
                Ideogram4CheckpointConverter.LoadVae(folderPath);
            loaders.AddRange(vaeL);

            Ideogram4Config config = Ideogram4Config.V4;
            log($"Building Ideogram 4 models (dim={config.LlmFeaturesDim} LLM features, {config.MaxTextTokens} max text tokens)...");

            Ideogram4Transformer conditional = new(config);
            conditional.LoadWeights(condW);
            Ideogram4Transformer unconditional = new(config);
            unconditional.LoadWeights(uncondW);

            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_8B);
            textEncoder.LoadWeights(teW);

            VaeDecoder vae = new(VaeConfig.Flux2);
            vae.LoadWeights(vaeW);

            log("Loading Qwen3 tokenizer (embedded)...");
            Qwen3Tokenizer tokenizer = new(maxLength: config.MaxTextTokens);

            log("Building Ideogram 4 pipeline...");
            Ideogram4Pipeline pipeline = new(backend, textEncoder, conditional, unconditional, vae, config);

            log("Ideogram 4 ready.");
            return new Ideogram4CacheEntry
            {
                ModelName = model.Name,
                CompatClass = Ideogram4CompatClassId,
                Pipeline = pipeline,
                Config = config,
                Tokenizer = tokenizer,
                TextEncoder = textEncoder,
                Conditional = conditional,
                Unconditional = unconditional,
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
        Ideogram4CacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        string negative = input.Get(T2IParamTypes.NegativePrompt) ?? "";
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);

        // Width/height must be multiples of 16 (2×2 patchify × 8× VAE), range 256–2048.
        int snappedW = Math.Clamp(width / 16 * 16, 256, 2048);
        int snappedH = Math.Clamp(height / 16 * 16, 256, 2048);
        if (snappedW != width || snappedH != height)
        {
            Logs.Info($"[HartsyInference][Ideogram4] Snapped resolution {width}x{height} → {snappedW}x{snappedH} (must be multiple of 16, 256–2048).");
        }

        // Steps → official sampler preset. The presets carry the per-step guidance
        // schedule (gw≈7 main + gw≈3 polish) and logit-normal mu/std, so CFG Scale and
        // negative prompt are intentionally not consumed here.
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: 20);
        Ideogram4SamplerPreset preset = steps <= 14 ? Ideogram4SamplerPreset.Turbo12
            : steps >= 40 ? Ideogram4SamplerPreset.Quality48
            : Ideogram4SamplerPreset.Default20;
        if (steps != preset.NumSteps)
        {
            Logs.Info($"[HartsyInference][Ideogram4] Steps={steps} mapped to official preset {preset.Name} ({preset.NumSteps} steps — Ideogram 4 uses fixed preset schedules).");
        }
        if (!string.IsNullOrWhiteSpace(negative))
        {
            Logs.Info("[HartsyInference][Ideogram4] Negative prompt is ignored — Ideogram 4's asymmetric CFG runs the unconditional branch with zeroed text features.");
        }

        // Chat-template tokenize, then trim the right-pad run (EncodeChat pads to
        // maxLength with BOS; feeding 2048 mostly-pad tokens would both dilute the
        // conditioning and ~3x the attention cost of the unified sequence).
        int[] padded = entry.Tokenizer.EncodeChat(prompt);
        int[] promptTokens = TrimRightPad(padded, Qwen3Tokenizer.BosTokenId);
        Logs.Verbose($"[HartsyInference][Ideogram4] Prompt tokenized to {promptTokens.Length} tokens (chat template, pad trimmed).");

        TextToImageRequest request = new()
        {
            Prompt = prompt,
            NegativePrompt = "",
            Width = snappedW,
            Height = snappedH,
            Steps = preset.NumSteps,
            CfgScale = 7.0f, // informational only; guidance comes from the preset schedule
            Seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
        };

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        var (rgbBytes, outW, outH, _) = entry.Pipeline.GenerateFromTokens(promptTokens, request, preset, bridge);

        Logs.Verbose($"[HartsyInference][Ideogram4] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
        return [RgbToImage.FromHwcRgb(rgbBytes, outW, outH)];
    }

    /// <summary>Strips the trailing run of <paramref name="padId"/> tokens that
    /// <c>Qwen3Tokenizer.EncodeChat</c> right-pads with. Keeps everything up to and
    /// including the last real token (EOS included when present).</summary>
    private static int[] TrimRightPad(int[] tokens, int padId)
    {
        int end = tokens.Length;
        while (end > 1 && tokens[end - 1] == padId) end--;
        return end == tokens.Length ? tokens : tokens[..end];
    }
}

public sealed class Ideogram4CacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required Ideogram4Pipeline Pipeline { get; init; }
    public required Ideogram4Config Config { get; init; }
    public required Qwen3Tokenizer Tokenizer { get; init; }
    public required LlamaStyleEncoder TextEncoder { get; init; }
    public required Ideogram4Transformer Conditional { get; init; }
    public required Ideogram4Transformer Unconditional { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required List<SafeTensorsLoader> Loaders { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        (TextEncoder as IDisposable)?.Dispose();
        (Conditional as IDisposable)?.Dispose();
        (Unconditional as IDisposable)?.Dispose();
        // VaeDecoder holds no unmanaged state — weight tensors are owned by Loaders below.
        Tokenizer?.Dispose();
        foreach (SafeTensorsLoader l in Loaders) l.Dispose();
    }
}
