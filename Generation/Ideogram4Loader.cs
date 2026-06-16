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
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
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
/// (The diagram above is the legacy bundled layout, kept for reference.) Loading no longer walks a
/// bundled folder: the user picks ONLY the conditional DiT file, and every other component is resolved
/// as its own registered model through <see cref="SideModels"/> — auto-downloaded and user-overridable,
/// mirroring ComfyUI/SwarmUI's per-component wiring:
///   - Unconditional DiT — <see cref="SideModels.Ideogram4Unconditional"/> (auto-downloaded).
///   - Text encoder — <see cref="SideModels.Qwen3VL_8B"/>, override via <see cref="T2IParamTypes.QwenModel"/>.
///   - VAE — <see cref="SideModels.Flux2Vae"/> (shared with Flux.2), override via <see cref="T2IParamTypes.VAE"/>.
/// Each is loaded from its exact resolved file via <c>LoadComponent</c> (the NuGet converter is folder-only,
/// so the per-component key transforms are replicated here). Incompatible user picks are rejected by a
/// header probe and fall back to the canonical component.
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

        log($"Loading Ideogram 4 model: {model.Name}");
        log("Note: Ideogram 4 weights are under a NON-COMMERCIAL license.");

        // Resolve the companion components as separate registered side-models (auto-downloaded,
        // user-overridable) — mirrors how ComfyUI/SwarmUI wires Ideogram 4: the picked file is the
        // CONDITIONAL DiT, while the Qwen3-VL-8B text encoder, the Flux.2 VAE, and the UNCONDITIONAL
        // DiT each come from their own model folder via the central SideModels registry. The user only
        // picks one file (the conditional transformer); everything else auto-resolves / auto-downloads.
        T2IModel uncondModel = ModelAutoDownloader.EnsureSideModel(
            userPick: null, entry: SideModels.Ideogram4Unconditional, log: log);
        T2IModel teModel = ResolveQwen3VlTextEncoder(input?.Get(T2IParamTypes.QwenModel), log);
        T2IModel vaeModel = ResolveFlux2Vae(input?.Get(T2IParamTypes.VAE), log);

        List<SafeTensorsLoader> loaders = [];
        try
        {
            log($"Loading conditional transformer (9.3B): {Path.GetFileName(model.RawFilePath)}");
            (Dictionary<string, Tensor> condW, SafeTensorsLoader condL) =
                LoadComponent(model.RawFilePath, StripTransformerPrefix, applyFp8Dequant: true);
            loaders.Add(condL);

            log($"Loading unconditional transformer (9.3B): {uncondModel.Name}");
            (Dictionary<string, Tensor> uncondW, SafeTensorsLoader uncondL) =
                LoadComponent(uncondModel.RawFilePath, StripTransformerPrefix, applyFp8Dequant: true);
            loaders.Add(uncondL);

            log($"Loading Qwen3-VL-8B text encoder: {teModel.Name}");
            (Dictionary<string, Tensor> teW, SafeTensorsLoader teL) =
                LoadComponent(teModel.RawFilePath, RemapQwenKey, applyFp8Dequant: true);
            loaders.Add(teL);

            log($"Loading Flux.2 VAE: {vaeModel.Name}");
            (Dictionary<string, Tensor> vaeW, SafeTensorsLoader vaeL) =
                LoadComponent(vaeModel.RawFilePath, key => key, applyFp8Dequant: false);
            loaders.Add(vaeL);

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

        // Optional magic prompt: rewrite the plain prompt into Ideogram 4's structured JSON caption via a
        // running LLM backend (opt-in). Ideogram 4 also accepts plain text, so when off we tokenize as-is.
        string promptForEncode = prompt;
        if (input.Get(SwarmUIHartsyInference.Ideogram4MagicPromptParam, false))
        {
            if (Ideogram4MagicPrompt.LooksLikeStructuredCaption(prompt))
            {
                // Already a structured caption (e.g. from a prompt-builder UI / pasted JSON) — feed as-is
                // so its user-drawn bboxes survive instead of being re-expanded and stripped by the LLM.
                Logs.Info("[HartsyInference][Ideogram4] Prompt is already a structured caption; skipping magic-prompt expansion.");
            }
            else
            {
                cancel.ThrowIfCancellationRequested();
                string magicModel = input.Get(SwarmUIHartsyInference.Ideogram4MagicPromptModelParam);
                promptForEncode = Ideogram4MagicPrompt.Expand(
                    prompt, snappedW, snappedH, magicModel, input.SourceSession,
                    msg => Logs.Info($"[HartsyInference][Ideogram4] {msg}"));
            }
        }

        // Chat-template tokenize, then trim the right-pad run (EncodeChat pads to
        // maxLength with BOS; feeding 2048 mostly-pad tokens would both dilute the
        // conditioning and ~3x the attention cost of the unified sequence).
        int[] padded = entry.Tokenizer.EncodeChat(promptForEncode);
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

    /// <summary>Qwen3-VL-8B language-tower hidden size — distinguishes the 8B encoder Ideogram 4 needs
    /// from the otherwise key-identical 4B (2560) / 0.6B (1024) Qwen3 variants.</summary>
    private const long Qwen3Vl8BHiddenSize = 4096;

    /// <summary>Loads one component from a single safetensors file: drops the fp8 <c>scaled_fp8</c> marker
    /// tensors, applies <paramref name="keyTransform"/> to each name (return null to drop a key), and
    /// optionally folds fp8 <c>*.scale_weight</c> companions via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>.
    /// Mirrors the per-component logic in the NuGet <c>Ideogram4CheckpointConverter</c>, but reads an exact
    /// resolved file (each component is its own registered model) instead of probing a bundled folder. The
    /// returned <see cref="SafeTensorsLoader"/> owns the tensor memory and must stay alive (and be disposed)
    /// for as long as the weights are used.</summary>
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

    /// <summary>Strips an optional Comfy/diffusers wrapper prefix off a transformer key. Mirrors
    /// <c>Ideogram4CheckpointConverter.StripTransformerPrefix</c> (the NuGet converter is folder-only, so we
    /// replicate the bare key-transform for the single-file path).</summary>
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

    /// <summary>Maps a Qwen3-VL checkpoint key to the <c>LlamaStyleEncoder</c> convention (<c>model.embed_tokens</c>,
    /// <c>model.layers.{i}.*</c>, <c>model.norm</c>), or returns null to drop it (vision tower, <c>lm_head</c>).
    /// The language tower lives under <c>language_model.</c> in HF Qwen3-VL; we re-root it at <c>model.</c>.
    /// Mirrors <c>Ideogram4CheckpointConverter.RemapQwenKey</c>.</summary>
    private static string RemapQwenKey(string key)
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

    /// <summary>Resolves Ideogram 4's text encoder. Ideogram is architecturally fixed to Qwen3-VL-8B, so an
    /// incompatible user pick (a different-size Qwen3, a T5, a VAE left selected, …) is ignored with a warning
    /// and the canonical Qwen3-VL-8B is auto-resolved/downloaded instead. A compatible custom pick is honored.
    /// Same guard pattern as <c>AnimaLoader.ResolveQwenImageVae</c>.</summary>
    private static T2IModel ResolveQwen3VlTextEncoder(T2IModel userPick, Action<string> log)
    {
        if (userPick is not null && !IsQwen3Vl8B(userPick))
        {
            log($"Selected text encoder '{userPick.Name}' is not a Qwen3-VL-8B (no 4096-dim embed_tokens); " +
                "ignoring it and auto-resolving the Qwen3-VL-8B that Ideogram 4 requires.");
            userPick = null;
        }
        return ModelAutoDownloader.EnsureSideModel(userPick, SideModels.Qwen3VL_8B, log);
    }

    /// <summary>Header-only probe: true iff the file looks like an 8B Qwen3 LM — a 4096-wide
    /// <c>embed_tokens.weight</c> plus decoder <c>layers</c>. Rejects 4B/0.6B Qwen3 and non-LLM files.</summary>
    private static bool IsQwen3Vl8B(T2IModel model)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath) || !File.Exists(model.RawFilePath))
            return false;
        try
        {
            using SafeTensorsLoader probe = new();
            probe.Load(model.RawFilePath);
            bool hasEmbed = false, hasLayers = false;
            foreach (KeyValuePair<string, SafeTensorDescriptor> kvp in probe.Descriptors)
            {
                string key = kvp.Key;
                if (key.Contains("embed_tokens.weight"))
                {
                    TensorShape shape = kvp.Value.Shape;
                    long hidden = shape.Rank >= 1 ? shape[shape.Rank - 1] : 0;
                    if (hidden != Qwen3Vl8BHiddenSize) return false;
                    hasEmbed = true;
                }
                if (key.Contains(".layers.")) hasLayers = true;
            }
            return hasEmbed && hasLayers;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference][Ideogram4] Could not probe text encoder '{model.Name}' ({ex.Message}); treating as incompatible.");
            return false;
        }
    }

    /// <summary>Resolves Ideogram 4's VAE (the Flux.2 KL autoencoder). An incompatible user pick (e.g. the
    /// 16-channel Flux.1 ae or a Qwen-Image VAE left selected) is ignored with a warning and the canonical
    /// Flux.2 VAE is auto-resolved/downloaded. A compatible custom pick is honored.</summary>
    private static T2IModel ResolveFlux2Vae(T2IModel userPick, Action<string> log)
    {
        if (userPick is not null && !IsFlux2Vae(userPick))
        {
            log($"Selected VAE '{userPick.Name}' is not a Flux.2 VAE (missing bn.running_mean/var); " +
                "ignoring it and auto-resolving the Flux.2 VAE that Ideogram 4 requires.");
            userPick = null;
        }
        return ModelAutoDownloader.EnsureSideModel(userPick, SideModels.Flux2Vae, log);
    }

    /// <summary>Header-only probe: true iff the file carries the Flux.2 VAE BatchNorm stats
    /// (<c>bn.running_mean</c> / <c>bn.running_var</c>). Same check <c>Flux2Loader</c> uses to reject a
    /// Flux.1 <c>ae.safetensors</c> mistakenly selected as a Flux.2 VAE.</summary>
    private static bool IsFlux2Vae(T2IModel model)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath) || !File.Exists(model.RawFilePath))
            return false;
        try
        {
            using SafeTensorsLoader probe = new();
            probe.Load(model.RawFilePath);
            return probe.Descriptors.ContainsKey("bn.running_mean")
                && probe.Descriptors.ContainsKey("bn.running_var");
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference][Ideogram4] Could not probe VAE '{model.Name}' ({ex.Message}); treating as incompatible.");
            return false;
        }
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
