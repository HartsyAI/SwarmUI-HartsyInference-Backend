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
using HartsyInference.ModelHandler.Lora;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Loads Flux.1 (dev or schnell) using Swarm's built-in model registration:
///   - Diffusion model: T2IParamTypes.Model (the main model dropdown)
///   - CLIP-L: T2IParamTypes.ClipLModel (Models/clip/)
///   - T5-XXL: T2IParamTypes.T5XXLModel (Models/clip/)
///   - VAE: T2IParamTypes.VAE (Models/vae/)
///
/// Each yields a T2IModel; we read .RawFilePath for the actual file. This mirrors how
/// Comfy resolves models in WorkflowGeneratorModelSupport.cs (e.g. line 593:
/// <c>return model.Name</c> — Swarm's model registry already knows the path).
///
/// Falls back to single-file all-in-one loading via FluxCheckpointConverter when the
/// user hasn't picked separate component models — this supports BFL / civitai
/// distributions that bundle everything in one .safetensors. Comfy does not support
/// this fallback (it requires split files); we accept either.
/// </summary>
public static class FluxLoader
{
    public const string Flux1CompatClassId = "flux-1";

    public static FluxCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("Flux model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"Flux checkpoint not found: {model.RawFilePath}");

        // Determine load mode: split-file (Swarm's model browser populated each component)
        // or all-in-one (single safetensors with everything bundled).
        T2IModel clipL = input?.Get(T2IParamTypes.ClipLModel);
        T2IModel t5 = input?.Get(T2IParamTypes.T5XXLModel);
        T2IModel vae = input?.Get(T2IParamTypes.VAE);

        Dictionary<string, Tensor> transformerWeights;
        Dictionary<string, Tensor> clipLWeights;
        Dictionary<string, Tensor> t5Weights;
        Dictionary<string, Tensor> vaeWeights;
        SafeTensorsLoader mainLoader = null;
        IDisposable ggufHandle = null;
        SafeTensorsLoader clipLLoader = null, t5Loader = null, vaeLoader = null;

        // A .gguf main file is a quantized transformer ONLY — CLIP-L / T5 / VAE must be picked
        // separately (Comfy's UnetLoaderGGUF works the same way). Force split mode for GGUF.
        bool isGguf = model.RawFilePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        bool splitMode = (clipL is not null && t5 is not null && vae is not null) || isGguf;
        if (isGguf && !(clipL is not null && t5 is not null && vae is not null))
        {
            throw new InvalidOperationException(
                "Flux GGUF: a .gguf file is a transformer-only quantized model. Pick CLIP-L Model + T5-XXL Model + VAE "
                + "in Swarm's parameters (Models/clip/ and Models/vae/) so HartsyInference can load the rest.");
        }
        if (splitMode)
        {
            transformerWeights = LoadFluxTransformer(model.RawFilePath, isGguf, ref mainLoader, ref ggufHandle, log);
            if (transformerWeights.Count == 0)
            {
                mainLoader?.Dispose();
                ggufHandle?.Dispose();
                throw new InvalidOperationException("Main Flux file contains no transformer weights.");
            }

            log($"  CLIP-L: {clipL.Name}");
            clipLLoader = new SafeTensorsLoader();
            clipLLoader.Load(clipL.RawFilePath);
            clipLWeights = ConvertClipLFromStandalone(clipLLoader.GetAllTensors());

            log($"  T5-XXL: {t5.Name}");
            t5Loader = new SafeTensorsLoader();
            t5Loader.Load(t5.RawFilePath);
            t5Weights = ConvertT5FromStandalone(t5Loader.GetAllTensors());

            log($"  VAE: {vae.Name}");
            vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vae.RawFilePath);
            vaeWeights = ConvertVaeFromStandalone(vaeLoader.GetAllTensors());
        }
        else
        {
            log($"Loading Flux (all-in-one mode) — {Path.GetFileName(model.RawFilePath)}");
            log("  (User didn't pick CLIP-L/T5-XXL/VAE separately — extracting components from this file. " +
                "For Comfy-style split layouts, configure those parameters in Swarm's model browser.)");
            (var converted, mainLoader) = FluxCheckpointConverter.LoadAndConvert(model.RawFilePath);
            transformerWeights = converted.Transformer;
            clipLWeights = converted.ClipL;
            t5Weights = converted.T5;
            vaeWeights = converted.Vae;

            if (transformerWeights.Count == 0)
            {
                mainLoader.Dispose();
                throw new InvalidOperationException(
                    "Flux: this checkpoint contains no transformer weights — not a Flux diffusion model.");
            }

            // Transformer-only checkpoint (Comfy diffusion_models style, e.g. flux1-dev-kontext_fp8_scaled):
            // resolve the missing components from the central SideModels registry (same canonical files the
            // split-file mode and Comfy use; auto-downloaded when absent) instead of refusing. The user's
            // explicit CLIP-L/T5/VAE picks still win — they'd have routed through split mode above.
            if (clipLWeights.Count == 0)
            {
                T2IModel clipLSide = ModelAutoDownloader.EnsureSideModel(userPick: null, entry: SideModels.ClipL, log: log);
                log($"  CLIP-L (side model): {clipLSide.Name}");
                clipLLoader = new SafeTensorsLoader();
                clipLLoader.Load(clipLSide.RawFilePath);
                clipLWeights = ConvertClipLFromStandalone(clipLLoader.GetAllTensors());
            }
            if (t5Weights.Count == 0)
            {
                T2IModel t5Side = ModelAutoDownloader.EnsureSideModel(userPick: null, entry: SideModels.T5XxlEnconly, log: log);
                log($"  T5-XXL (side model): {t5Side.Name}");
                t5Loader = new SafeTensorsLoader();
                t5Loader.Load(t5Side.RawFilePath);
                t5Weights = ConvertT5FromStandalone(t5Loader.GetAllTensors());
            }
            if (vaeWeights.Count == 0)
            {
                T2IModel vaeSide = ModelAutoDownloader.EnsureSideModel(userPick: null, entry: SideModels.FluxAe, log: log);
                log($"  VAE (side model): {vaeSide.Name}");
                vaeLoader = new SafeTensorsLoader();
                vaeLoader.Load(vaeSide.RawFilePath);
                vaeWeights = ConvertVaeFromStandalone(vaeLoader.GetAllTensors());
            }
        }

        var (doubleBlocks, singleBlocks, hasGuidance) = FluxCheckpointConverter.DetectArchitecture(transformerWeights);

        // FLUX.1 Tools detection: vanilla Flux has x_embedder input dim 64 (16 latent
        // channels × 2×2 packing). Canny / Depth have 128 (64 noise + 64 packed control);
        // Fill has 384 (64 noise + 64 masked latent + 256 packed mask). Canny and Fill are
        // fully wired; Depth runs the in-engine Depth-Anything-V2 annotator at gen time.
        FluxToolsMode toolsMode = DetectToolsMode(transformerWeights, model.Name);
        // Kontext is architecturally identical to Dev (x_embedder stays 64-wide — the reference image is
        // sequence-appended at runtime, not channel-concat like Tools), so it can only be identified from
        // Swarm's model class or the filename. Kontext models accept an Init Image as the EDIT REFERENCE.
        bool isKontext = (model.ModelClass?.ID?.Contains("kontext", StringComparison.OrdinalIgnoreCase) ?? false)
            || model.Name.Contains("kontext", StringComparison.OrdinalIgnoreCase);
        log($"Architecture: {doubleBlocks} double, {singleBlocks} single, guidance={hasGuidance} ({(hasGuidance ? "Dev" : "Schnell")}), tools={toolsMode}, kontext={isKontext}");
        FluxConfig fluxConfig = toolsMode switch
        {
            FluxToolsMode.Fill => FluxConfig.Flux1Fill,
            FluxToolsMode.Canny or FluxToolsMode.Depth => FluxConfig.Flux1Tools,
            _ => hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell,
        };

        // fp8 checkpoints on hardware without native FP8 GEMM (Ampere and below) fall back to
        // casting each fp8 weight to F16/BF16 on first use. With CacheWeightCasts=true (default)
        // that cast is kept resident forever — roughly doubling VRAM for every fp8 tensor touched
        // (transformer + CLIP-L + T5-XXL here) and OOMs a 12 GB 3060 partway through text encoding.
        // Keep weights fp8-resident with a transient per-GEMM dequant instead (same fix already
        // applied to LtxVideoLoader's 13B fp8 path).
        if (backend is HartsyInference.Cuda.CudaBackend cudaBackend && !cudaBackend.EnableNativeFp8Gemm
            && (HasFp8Weights(transformerWeights) || HasFp8Weights(clipLWeights) || HasFp8Weights(t5Weights)))
        {
            cudaBackend.CacheWeightCasts = false;
            log("  fp8 checkpoint on non-native-FP8 hardware: CacheWeightCasts disabled (fp8-resident, transient per-GEMM dequant).");
        }

        log("Building transformer...");
        FluxTransformer transformer = new FluxTransformer(fluxConfig);
        transformer.LoadWeights(transformerWeights);

        log("Building CLIP-L encoder...");
        ClipTextEncoder clipLEnc = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        clipLEnc.LoadWeights(clipLWeights, prefix: "text_model");

        log("Building T5-XXL encoder...");
        T5TextEncoder t5Enc = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5Enc.LoadWeights(t5Weights);

        log("Building VAE decoder...");
        VaeDecoder vaeDec = new VaeDecoder(VaeConfig.Flux);
        vaeDec.LoadWeights(vaeWeights);

        log("Building VAE encoder (img2img)...");
        VaeEncoder vaeEnc = new VaeEncoder(VaeConfig.Flux);
        vaeEnc.LoadWeights(vaeWeights);

        log("Building pipeline...");
        FluxPipeline pipeline = new FluxPipeline(backend, clipLEnc, t5Enc, transformer, vaeDec, vaeEnc, fluxConfig);

        log("Loading tokenizers (embedded in HartsyInference.Tokenizers)...");
        ClipTokenizer clipTok = new ClipTokenizer();
        int t5MaxLength = hasGuidance ? 512 : 256;
        T5Tokenizer t5Tok = new T5Tokenizer(maxLength: t5MaxLength);

        log($"Flux ready ({(splitMode ? "split-file" : "all-in-one")} mode, Dev={hasGuidance}, tools={toolsMode}).");
        return new FluxCacheEntry
        {
            ModelName = model.Name,
            CompatClass = Flux1CompatClassId,
            Backend = backend,
            IsDev = hasGuidance,
            IsKontext = isKontext,
            ToolsMode = toolsMode,
            Pipeline = pipeline,
            FluxConfig = fluxConfig,
            ClipTokenizer = clipTok,
            T5Tokenizer = t5Tok,
            ClipEncoder = clipLEnc,
            T5Encoder = t5Enc,
            Transformer = transformer,
            Vae = vaeDec,
            VaeEncoder = vaeEnc,
            CheckpointLoader = mainLoader,
            GgufHandle = ggufHandle,
            ClipLLoader = clipLLoader,
            T5Loader = t5Loader,
            VaeLoader = vaeLoader,
            // Retained for the LoRA path. T5 isn't a LoraTarget in upstream
            // HartsyInference (no Kohya/Diffusers Flux LoRA targets it), and the VAE
            // never is, so we only retain Transformer + ClipL.
            TransformerWeights = transformerWeights,
            ClipLWeights = clipLWeights,
        };
    }

    /// <summary>LoRA-merged generation path for Flux. Builds fresh CLIP-L + Transformer
    /// from shallow-cloned weight dicts with the LoraStack merged in. T5 and VAE are
    /// reused from the cached entry (not LoRA targets).</summary>
    public static Image[] GenerateWithLoras(
        FluxCacheEntry entry,
        IReadOnlyList<LoraResolver.LoraSpec> loras,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        Dictionary<string, Tensor> transformerWeights = LoraApplier.ShallowClone(entry.TransformerWeights);
        Dictionary<string, Tensor> clipLWeights = LoraApplier.ShallowClone(entry.ClipLWeights);

        LoraStack stack = LoraApplier.BuildAndApply(
            loras, backend,
            transformerWeights: transformerWeights,
            clipLWeights: clipLWeights);

        ClipTextEncoder clipLEnc = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        FluxTransformer transformer = new FluxTransformer(entry.FluxConfig);
        try
        {
            clipLEnc.LoadWeights(clipLWeights, prefix: "text_model");
            transformer.LoadWeights(transformerWeights);

            using FluxPipeline pipeline = new FluxPipeline(
                backend, clipLEnc, entry.T5Encoder, transformer, entry.Vae, entry.VaeEncoder, entry.FluxConfig);

            return RunFluxPipeline(pipeline, entry, input, onProgress, cancel);
        }
        finally
        {
            // FluxTransformer is IDisposable (owns native pointers); ClipTextEncoder
            // isn't (relies on tensor finalizers). Stack last so merged tensors
            // outlive the components that referenced them via LoadWeights.
            transformer?.Dispose();
            stack?.Dispose();
        }
    }

    public static Image[] Generate(
        FluxCacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        return RunFluxPipeline(entry.Pipeline, entry, input, onProgress, cancel);
    }

    /// <summary>Shared per-pipeline driver — same logic whether the pipeline is the
    /// cached one (no-LoRA) or a freshly-built per-gen one (LoRA).</summary>
    private static Image[] RunFluxPipeline(
        FluxPipeline pipeline,
        FluxCacheEntry entry,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string prompt = PromptConditioningResolver.BaseText(input.Get(T2IParamTypes.Prompt));
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.IsDev ? 20 : 4);
        int width = input.Get(T2IParamTypes.Width);
        int height = input.Get(T2IParamTypes.Height);
        long seedLong = input.Get(T2IParamTypes.Seed);

        // Read FluxGuidanceScale via the typed parameter Swarm registers.
        // Defaults to 3.5 (BFL distillation target). Schnell ignores guidance entirely.
        float guidance = entry.IsDev ? ResolveFluxGuidance(input, defaultValue: 3.5f) : 0f;

        int[] clipTokens = entry.ClipTokenizer.Encode(prompt);
        int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
        int[] t5Tokens = entry.T5Tokenizer.Encode(prompt);
        int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);
        Logs.Verbose($"[HartsyInference][Flux] Tokenized — CLIP-L: {clipTokens.Length} tokens (eos@{eosPos}), T5: {t5Tokens.Length} tokens. " +
            $"Mode={(entry.IsDev ? "Dev" : "Schnell")}, steps={steps}, guidance={guidance:F2}.");

        // FLUX.1 Canny: build the control-image tensor from the user's reference. We
        // accept the reference from Controlnets[0].Image (the natural place — user wants
        // to drive structure from an image), falling back to InitImage. The canny
        // preprocessor extracts edges; we then convert the [0, 1] RGB output to the
        // [-1, 1] range Flux's VAE expects, and pass that to the pipeline which handles
        // VAE encoding + packing internally.
        Tensor fluxCannyControl = null;
        if (entry.ToolsMode is FluxToolsMode.Canny or FluxToolsMode.Depth)
        {
            Image cnImage = null;
            T2IParamTypes.ControlNetParamHolder[] cnHolders = T2IParamTypes.Controlnets;
            if (cnHolders is not null && cnHolders.Length > 0 && cnHolders[0]?.Image is not null)
            {
                cnImage = input.Get(cnHolders[0].Image);
            }
            cnImage ??= input.Get(T2IParamTypes.InitImage);
            if (cnImage is null)
            {
                throw new InvalidOperationException(
                    $"FLUX.1 {entry.ToolsMode} requires a reference image. Set ControlNet Image Input or Init Image — " +
                    $"the {(entry.ToolsMode == FluxToolsMode.Canny ? "canny edges" : "depth map")} will be extracted from it and used as the control conditioning.");
            }
            Logs.Verbose($"[HartsyInference][Flux {entry.ToolsMode}] Building control map from reference image at {width}x{height}.");
            Tensor controlZeroOne = entry.ToolsMode == FluxToolsMode.Canny
                ? CannyPreprocessor.Process(cnImage, width, height)
                : DepthPreprocessor.Process(cnImage, width, height, entry.Backend,
                    msg => Logs.Verbose($"[HartsyInference][Flux Depth] {msg}"), fluxScaling: true);
            try
            {
                string dumpDir = Environment.GetEnvironmentVariable("HARTSY_DUMP_CONTROL");
                if (!string.IsNullOrEmpty(dumpDir))
                {
                    DumpControlPng(controlZeroOne, Path.Combine(dumpDir, $"flux_control_{entry.ToolsMode}.png".ToLowerInvariant()));
                }
                // Flux VAE expects RGB in [-1, 1]; both preprocessors return [0, 1].
                fluxCannyControl = ScaleZeroOneToMinusOneOne(controlZeroOne);
            }
            finally
            {
                controlZeroOne.Dispose();
            }
        }

        Img2ImgResolver.Img2ImgSpec img2img = Img2ImgResolver.Resolve(input, width, height);
        // FLUX.1 Fill: the engine conditions the wide x_embedder on Init Image + Mask directly
        // (diffusers FluxFillPipeline). Both are required; denoise strength defaults to 1.0
        // (full regeneration of the masked region) unless the user explicitly set creativity.
        float img2imgStrength = img2img?.Strength ?? 0f;
        if (entry.ToolsMode == FluxToolsMode.Fill)
        {
            if (img2img is null || img2img.MaskTensor is null)
            {
                img2img?.Dispose();
                throw new InvalidOperationException(
                    "FLUX.1 Fill requires both an Init Image and a Mask Image — the masked region is regenerated from the prompt.");
            }
            if (!input.TryGet(T2IParamTypes.InitImageCreativity, out _))
            {
                img2imgStrength = 1.0f;
                Logs.Verbose("[HartsyInference][Flux Fill] No explicit Init Image Creativity; defaulting fill strength to 1.0.");
            }
        }
        // Kontext editing: on a Kontext model the Init Image is the EDIT REFERENCE, not an img2img source —
        // the pipeline VAE-encodes it and sequence-appends its tokens every step (diffusers FluxKontext flow),
        // while denoising still starts from pure noise. Consume the resolved spec's tensor as the reference
        // and drop the img2img request (strength/creativity doesn't apply to Kontext).
        Tensor kontextRef = null;
        if (entry.IsKontext && img2img is not null)
        {
            kontextRef = img2img.SourceTensor;
            Logs.Verbose("[HartsyInference][Flux Kontext] Init Image routed as the Kontext edit reference.");
            img2img.MaskTensor?.Dispose();
            img2img = null;
        }
        // FLUX.1 Redux (style model): SigLIP + projector run on the calling thread; the pipeline
        // appends the resulting tokens after the T5 stream. Works on any Flux variant (Comfy applies
        // style models to the positive conditioning the same way).
        ReduxResolver.ReduxSpec redux = ReduxResolver.Resolve(
            input, entry.Backend, msg => Logs.Verbose($"[HartsyInference][Flux Redux] {msg}"));

        int? seed = seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF);
        // Variation seed: Flux injects unpacked [1,16,H/8,W/8] noise (FluxLatentChannels=16).
        Tensor variationNoise = VariationSeedResolver.Resolve(input, width, height, seed, VariationSeedResolver.FluxLatentChannels);
        TextToImageRequest request;
        if (img2img is not null)
        {
            request = new ImageToImageRequest
            {
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = seed,
                InitialNoise = variationNoise,
                SourceImage = img2img.SourceTensor,
                Strength = img2imgStrength,
                Mask = img2img.MaskTensor,
            };
        }
        else
        {
            request = new TextToImageRequest
            {
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = seed,
                InitialNoise = variationNoise,
            };
        }

        try
        {
            long start = Environment.TickCount64;
            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                onProgress(p);
            };

            var (rgbBytes, outW, outH, _) = pipeline.GenerateFromTokens(
                clipTokens, eosPos, t5Tokens, t5Mask, request,
                guidanceScale: guidance,
                onProgress: bridge,
                controlImage: fluxCannyControl,
                kontextRefImage: kontextRef,
                reduxImageEmbeds: redux?.Embeds,
                reduxApplyStartFraction: redux?.ApplyStart ?? 0f);

            Logs.Verbose($"[HartsyInference][Flux] Pipeline returned {outW}x{outH} in {Environment.TickCount64 - start}ms.");
            return new[] { RgbToImage.FromHwcRgb(rgbBytes, outW, outH) };
        }
        finally
        {
            img2img?.Dispose();
            kontextRef?.Dispose();
            fluxCannyControl?.Dispose();
            redux?.Dispose();
        }
    }

    /// <summary>Loads + converts the Flux transformer from either a .safetensors or a .gguf file.
    /// GGUF goes through <see cref="GgufConverterBridge"/> (dequantize to F16 → same FluxCheckpointConverter),
    /// keeping the dequant handle alive via <paramref name="ggufHandle"/>; safetensors uses the normal
    /// mmap loader. Exactly one of <paramref name="mainLoader"/> / <paramref name="ggufHandle"/> is set.</summary>
    private static Dictionary<string, Tensor> LoadFluxTransformer(
        string path, bool isGguf, ref SafeTensorsLoader mainLoader, ref IDisposable ggufHandle, Action<string> log)
    {
        if (isGguf)
        {
            log($"Loading Flux (GGUF transformer + split CLIP/T5/VAE) — main: {Path.GetFileName(path)}");
            (FluxCheckpointConverter.ConvertedWeights converted, GgufModelLoader.LoadedGgufModel handle) =
                GgufConverterBridge.LoadGguf(path, DType.F16, FluxCheckpointConverter.Convert);
            ggufHandle = handle;
            return converted.Transformer;
        }
        log($"Loading Flux (split-file mode) — main: {Path.GetFileName(path)}");
        (FluxCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader loader) = FluxCheckpointConverter.LoadAndConvert(path);
        mainLoader = loader;
        return conv.Transformer;
    }

    /// <summary>Read FluxGuidanceScale (registered by the Comfy extension under "flux-dev"
    /// feature flag) without taking a hard reference to that assembly.</summary>
    private static float ResolveFluxGuidance(T2IParamInput input, float defaultValue)
    {
        if (T2IParamTypes.TryGetType("fluxguidancescale", out T2IParamType type, input)
            && input.TryGetRaw(type, out object raw))
        {
            if (raw is double d) return (float)d;
            if (raw is float f) return f;
            if (raw is string s && double.TryParse(s, System.Globalization.CultureInfo.InvariantCulture, out double parsed))
                return (float)parsed;
        }
        return defaultValue;
    }

    /// <summary>Standalone CLIP-L safetensors files have a few possible top-level layouts.
    /// FluxCheckpointConverter handles diffusers + ComfyUI prefixes for in-bundle CLIP-L,
    /// but standalone files often store keys without the wrapping prefix. Strip them so
    /// ClipTextEncoder.LoadWeights(prefix: "text_model") finds what it expects.</summary>
    private static Dictionary<string, Tensor> ConvertClipLFromStandalone(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var kv in raw)
        {
            string k = kv.Key;
            // Strip Comfy wrapping if present.
            if (k.StartsWith("text_encoders.clip_l.transformer.", StringComparison.Ordinal))
                k = k["text_encoders.clip_l.".Length..]; // keep "transformer." → no wait, we want "text_model.*"
            // Actually keep the Comfy stripping consistent with how FluxCheckpointConverter does it:
            // ConvertedWeights.ClipL strips down to "text_model.*". Re-apply that here.
            if (kv.Key.StartsWith("text_encoders.clip_l.transformer.", StringComparison.Ordinal))
            {
                string rest = kv.Key["text_encoders.clip_l.transformer.".Length..];
                if (!rest.EndsWith("position_ids", StringComparison.Ordinal))
                    result[rest] = kv.Value;
            }
            else if (kv.Key.StartsWith("conditioner.embedders.0.transformer.", StringComparison.Ordinal))
            {
                string rest = kv.Key["conditioner.embedders.0.transformer.".Length..];
                if (!rest.EndsWith("position_ids", StringComparison.Ordinal))
                    result[rest] = kv.Value;
            }
            else
            {
                // Already in expected layout (text_model.*) — pass through.
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }

    /// <summary>Standalone T5-XXL safetensors typically store keys directly (no prefix).
    /// HartsyInference's T5TextEncoder.LoadWeights expects "encoder.embed_tokens.weight"
    /// etc. Strip Comfy's "text_encoders.t5xxl.transformer." wrapper if present.</summary>
    private static Dictionary<string, Tensor> ConvertT5FromStandalone(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (var kv in raw)
        {
            if (kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal))
                result[kv.Key[ComfyPrefix.Length..]] = kv.Value;
            else
                result[kv.Key] = kv.Value;
        }
        return result;
    }

    /// <summary>Convert a <c>[1, 3, H, W]</c> F32 tensor in <c>[0, 1]</c> to <c>[-1, 1]</c>
    /// (the range the Flux VAE expects on its input). The CLIP-Vision preprocessor and
    /// the canny preprocessor both produce <c>[0, 1]</c>; the Flux VAE wants <c>[-1, 1]</c>;
    /// the scale is just <c>x * 2 - 1</c>.</summary>
    private static unsafe Tensor ScaleZeroOneToMinusOneOne(Tensor input)
    {
        Tensor output = new Tensor(input.Shape, DType.F32);
        long count = input.ElementCount;
        float* sp = (float*)input.DataPointer;
        float* dp = (float*)output.DataPointer;
        for (long i = 0; i < count; i++) dp[i] = sp[i] * 2.0f - 1.0f;
        return output;
    }

    /// <summary>Detects FLUX.1 Tools variant from the loaded transformer weights and
    /// filename. Tools variants share a single architectural shape (32-channel
    /// <c>x_embedder</c>); they differ only in how the control image is preprocessed,
    /// which we resolve from the filename keyword. <c>Vanilla</c> when the x_embedder
    /// input dim is 64; <c>Canny / Depth / Fill</c> when it's 128 (filename keyword
    /// disambiguates). Defaults to Canny when the input dim is 128 but the filename
    /// gives no hint — Canny is the most common Tools release.</summary>
    private static FluxToolsMode DetectToolsMode(Dictionary<string, Tensor> transformerWeights, string modelName)
    {
        if (!transformerWeights.TryGetValue("x_embedder.weight", out Tensor xEmbed)) return FluxToolsMode.Vanilla;
        long inputDim = xEmbed.Shape.Rank >= 2 ? xEmbed.Shape[1] : 0;
        // Fill is architecturally distinct: 384-wide x_embedder (64 noise + 64 masked latent + 256 packed
        // mask) vs the 128-wide Canny/Depth control concat — no filename hint needed.
        if (inputDim >= 384) return FluxToolsMode.Fill;
        if (inputDim != 128) return FluxToolsMode.Vanilla;
        string lowered = modelName.ToLowerInvariant();
        if (lowered.Contains("canny")) return FluxToolsMode.Canny;
        if (lowered.Contains("depth")) return FluxToolsMode.Depth;
        return FluxToolsMode.Canny;
    }

    /// <summary>Debug aid (HARTSY_DUMP_CONTROL=dir): writes a [1,3,H,W] [0,1] control tensor as a PNG so the conditioning map can be inspected.</summary>
    private static unsafe void DumpControlPng(Tensor control, string path)
    {
        try
        {
            int h = (int)control.Shape[2];
            int w = (int)control.Shape[3];
            float* p = (float*)control.DataPointer;
            byte[] rgb = new byte[h * w * 3];
            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < h * w; i++)
                {
                    rgb[i * 3 + c] = (byte)Math.Clamp((int)(p[(long)c * h * w + i] * 255f), 0, 255);
                }
            }
            using var img = SixLabors.ImageSharp.Image.LoadPixelData<SixLabors.ImageSharp.PixelFormats.Rgb24>(rgb, w, h);
            using FileStream fs = File.OpenWrite(path);
            SixLabors.ImageSharp.ImageExtensions.SaveAsPng(img, fs);
            Logs.Info($"[HartsyInference] Control map dumped to {path}");
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] Control map dump failed: {ex.Message}");
        }
    }

    /// <summary>Normalizes the keys of a standalone Flux VAE safetensors file into the
    /// diffusers naming <see cref="VaeDecoder.LoadWeights"/> expects. Three flavours:
    /// <list type="bullet">
    /// <item>Comfy-prefixed: <c>vae.*</c> or <c>first_stage_model.*</c></item>
    /// <item>BFL-native LDM bare keys: <c>decoder.mid.block_1.norm1.weight</c></item>
    /// <item>Already diffusers: <c>decoder.mid_block.resnets.0.norm1.weight</c></item>
    /// </list>
    /// All three route through <see cref="CheckpointConvertUtils.ConvertVaeKey"/>, which
    /// remaps LDM → diffusers and passes diffusers-form keys through unchanged. The previous
    /// pass-through path silently dropped LDM-native files (matching the auto-downloaded
    /// mcmonkey <c>flux_ae.safetensors</c>) into VaeDecoder.LoadWeights with the wrong key
    /// names, which threw on the first missing key.</summary>
    private static Dictionary<string, Tensor> ConvertVaeFromStandalone(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach (var (key, tensor) in raw)
        {
            string ldmKey = key;
            if (ldmKey.StartsWith("first_stage_model.", StringComparison.Ordinal))
                ldmKey = ldmKey["first_stage_model.".Length..];
            else if (ldmKey.StartsWith("vae.", StringComparison.Ordinal))
                ldmKey = ldmKey["vae.".Length..];

            var diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
            if (diffusersKey is not null)
            {
                result[diffusersKey] = tensor;
            }
        }
        return result;
    }

    /// <summary>True if any tensor in the dict is still in raw FP8 (E4M3/E5M2) — i.e. not
    /// pre-dequantized by <c>FluxCheckpointConverter</c>'s ComfyUI-scaled-fp8 handling.</summary>
    private static bool HasFp8Weights(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values)
        {
            if (tensor.DType.IsFp8)
                return true;
        }
        return false;
    }

}

/// <summary>Identifies a FLUX.1 Tools variant by the user-facing conditioning behavior. Detected from the checkpoint's <c>x_embedder</c> shape + filename keyword. <c>Vanilla</c> = standard text-to-image Flux; <c>Canny</c> / <c>Depth</c> / <c>Fill</c> = Tools variants that take a control image alongside the prompt.</summary>
public enum FluxToolsMode
{
    Vanilla,
    Canny,
    Depth,
    Fill,
}

public sealed class FluxCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    /// <summary>The backend the pipeline was built on — used by gen-time resolvers (Redux SigLIP encode, Depth-Anything) that run device work outside the pipeline.</summary>
    public required IBackend Backend { get; init; }
    public required bool IsDev { get; init; }

    /// <summary>True for FLUX.1 Kontext models — the Init Image routes as the sequence-appended edit
    /// reference instead of an img2img source.</summary>
    public bool IsKontext { get; init; }
    public required FluxToolsMode ToolsMode { get; init; }
    public required FluxPipeline Pipeline { get; init; }
    public required FluxConfig FluxConfig { get; init; }
    public required ClipTokenizer ClipTokenizer { get; init; }
    public required T5Tokenizer T5Tokenizer { get; init; }
    public required ClipTextEncoder ClipEncoder { get; init; }
    public required T5TextEncoder T5Encoder { get; init; }
    public required FluxTransformer Transformer { get; init; }
    public required VaeDecoder Vae { get; init; }
    public required VaeEncoder VaeEncoder { get; init; }
    /// <summary>Holds the main transformer file open for the safetensors path. Null when the
    /// transformer came from a .gguf — see <see cref="GgufHandle"/>.</summary>
    public SafeTensorsLoader CheckpointLoader { get; init; }
    /// <summary>Keeps the dequantized GGUF transformer weights alive (null for the safetensors path).</summary>
    public IDisposable GgufHandle { get; init; }
    public SafeTensorsLoader ClipLLoader { get; init; }
    public SafeTensorsLoader T5Loader { get; init; }
    public SafeTensorsLoader VaeLoader { get; init; }

    /// <summary>Transformer weights from the converter, retained for the LoRA path.</summary>
    public required Dictionary<string, Tensor> TransformerWeights { get; init; }
    /// <summary>CLIP-L weights retained for the LoRA path.</summary>
    public required Dictionary<string, Tensor> ClipLWeights { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        ClipTokenizer?.Dispose();
        T5Tokenizer?.Dispose();
        CheckpointLoader?.Dispose();
        GgufHandle?.Dispose();
        ClipLLoader?.Dispose();
        T5Loader?.Dispose();
        VaeLoader?.Dispose();
    }
}
