using System.IO;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelHandler.SafeTensors;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's IP-Adapter params (registered by the ComfyUI extension under the
/// <c>ipadapter</c> feature flag) into a list of
/// <see cref="IpAdapterConditioning"/>s ready to hand to a HartsyInference SDXL
/// pipeline. Loads each selected IPA checkpoint, runs CLIP-Vision on the user's
/// prompt image, projects the result into image-prompt tokens, and returns the
/// per-cross-attn-layer wiring.
///
/// <para><b>v1 scope:</b> SDXL standard + Plus + Plus-Face IPA (Plus-Face is
/// architecturally identical to Plus, just different training). Single adapter
/// only — Comfy lets users stack multiple IPA models, which would sum the
/// per-cross-attn image-attention outputs; deferred. Other base models (SD 1.5,
/// Flux) and FaceID variants (which use InsightFace ArcFace embeddings instead
/// of CLIP-Vision) refused upstream by <see cref="IpAdapter"/>'s constructor.</para>
///
/// <para>The IPA model file is located under <c>&lt;ModelRoot&gt;/ipadapter/&lt;filename&gt;</c>
/// — the standard Comfy path. CLIP-Vision is resolved from
/// <see cref="T2IParamTypes.ClipVisionModel"/>, falling back to auto-download of
/// the canonical CLIP-ViT-H/14 (the encoder every common SDXL IPA was trained
/// against; see <see cref="SideModels.ClipVisionH14"/>). The loaded entries are
/// returned in a disposable spec so the caller can clean up image tokens after
/// the generation.</para>
/// </summary>
public static class IpAdapterResolver
{
    /// <summary>One generation's worth of resolved IPA state. Owns the image-prompt token
    /// tensor produced by <see cref="IpAdapter.ProjectImage"/>; the loaded IPA + CLIP-Vision
    /// live in <see cref="IpAdapterCacheEntry"/>s that the cache holds across gens.</summary>
    public sealed class ResolvedSpec : IDisposable
    {
        public required List<IpAdapterConditioning> Conditionings { get; init; }
        public required List<Tensor> ImageTokens { get; init; }

        public void Dispose()
        {
            foreach (Tensor t in ImageTokens) t.Dispose();
        }
    }

    /// <summary>Resolve IPA for this generation. Returns null when no IPA is configured.
    /// <paramref name="baseModel"/> selects which UNet config the IPA must match — pass
    /// <c>IpAdapterBaseModel.Sdxl</c> for SDXL pipelines, <c>Sd15</c> for SD 1.5 pipelines.
    /// The detected variant from the checkpoint must match this base — mismatches throw.</summary>
    public static ResolvedSpec Resolve(
        T2IParamInput input,
        IBackend backend,
        IpAdapterBaseModel baseModel,
        Action<string> log,
        Func<string, IpAdapterCacheEntry> cacheLookup,
        Action<IpAdapterCacheEntry> cachePut)
    {
        if (input is null) return null;

        // Read Comfy-extension params via string ID (avoids hard dependency on Comfy assembly).
        string ipaModelName = ReadStringParam(input, "useipadapter");
        if (string.IsNullOrEmpty(ipaModelName) || ipaModelName == "None") return null;

        // Reference image(s) — IPA reuses Swarm's PromptImages list (same source as ReVision).
        // Multi-image averaging: when N>1 images are supplied, average their CLIP-Vision
        // outputs before projection. Matches diffusers / Cubiq IPAdapterPlus default behavior
        // for "multiple reference images" — produces a single set of image-prompt tokens
        // representing the centroid of the references' visual style.
        List<Image> promptImages = input.Get(T2IParamTypes.PromptImages);
        if (promptImages is null || promptImages.Count == 0)
        {
            throw new InvalidOperationException(
                "IP-Adapter is enabled (Use IP-Adapter is set) but no Prompt Image was provided. " +
                "Add an image via the prompt-image input or disable IP-Adapter.");
        }

        double weightRaw = ReadDoubleParam(input, "ipadapterweight", defaultValue: 1.0);
        double startRaw = Math.Clamp(ReadDoubleParam(input, "ipadapterstart", defaultValue: 0.0), 0.0, 1.0);
        double endRaw = Math.Clamp(ReadDoubleParam(input, "ipadapterend", defaultValue: 1.0), 0.0, 1.0);
        if (endRaw < startRaw) endRaw = startRaw;
        string weightType = ReadStringParam(input, "ipadapterweighttype") ?? "standard";

        // Locate the IPA file under <ModelRoot>/ipadapter/<filename>. Comfy convention.
        string ipaPath = ResolveIpaModelPath(ipaModelName)
            ?? throw new InvalidOperationException(
                $"IP-Adapter model '{ipaModelName}' not found in any <ModelRoot>/ipadapter/ subfolder. " +
                $"Place the .safetensors file in '{Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, "ipadapter")}' or pick a different model.");

        // Cache key: the IPA file's full path (different files = different entries).
        IpAdapterCacheEntry entry = cacheLookup(ipaPath);
        if (entry is null)
        {
            log($"Loading IP-Adapter: {ipaModelName}");
            entry = LoadIpaEntry(input, ipaPath, baseModel, log);
            cachePut(entry);
        }
        else
        {
            log($"IP-Adapter '{ipaModelName}' (cached).");
        }

        // Validate base model match.
        if (entry.IpAdapter.Config.BaseModel != baseModel)
        {
            throw new InvalidOperationException(
                $"IP-Adapter '{ipaModelName}' is for base={entry.IpAdapter.Config.BaseModel}, but the current pipeline expects {baseModel}.");
        }

        // Run CLIP-Vision over all prompt images and average the outputs (pre-projection).
        // Image-token projection runs ONCE on the averaged vision output.
        Tensor averagedVisionOut;
        if (promptImages.Count == 1)
        {
            Tensor pixelValues = ClipImagePreprocessor.Process(promptImages[0], entry.ClipVision.Config.ImageSize);
            try
            {
                averagedVisionOut = entry.IpAdapter.Config.IsPlus
                    ? entry.ClipVision.EncodeHiddenStates(backend, pixelValues)
                    : entry.ClipVision.EncodeImageEmbeds(backend, pixelValues);
            }
            finally
            {
                pixelValues.Dispose();
            }
        }
        else
        {
            log($"  averaging {promptImages.Count} reference images (vision-output mean before projection)");
            averagedVisionOut = AverageVisionOutputs(backend, entry, promptImages);
        }

        Tensor imageTokens;
        try
        {
            imageTokens = entry.IpAdapter.ProjectImage(backend, averagedVisionOut);
        }
        finally
        {
            averagedVisionOut.Dispose();
        }

        List<IpAdapterConditioning> conditionings = new()
        {
            new IpAdapterConditioning
            {
                Adapter = entry.IpAdapter,
                ImageTokens = imageTokens,
                Scale = (float)weightRaw,
                WeightType = weightType,
                StartFraction = (float)startRaw,
                EndFraction = (float)endRaw,
            }
        };
        List<Tensor> imageTokenList = new() { imageTokens };

        log($"IP-Adapter ready: variant={(entry.IpAdapter.Config.IsPlus ? "Plus" : "Standard")}, base={baseModel}, weight={weightRaw:F2}, weightType={weightType}, window=[{startRaw:F2}, {endRaw:F2}], tokens={entry.IpAdapter.NumImageTokens}.");
        return new ResolvedSpec { Conditionings = conditionings, ImageTokens = imageTokenList };
    }

    /// <summary>Run CLIP-Vision on each prompt image, average the outputs along the batch
    /// dimension. All inputs share the same shape (after the 224×224 preprocess), so the
    /// averaged tensor has the same shape as a single one. Used when the user supplies
    /// multiple reference images and wants their styles merged into one IP conditioning.</summary>
    private static unsafe Tensor AverageVisionOutputs(IBackend backend, IpAdapterCacheEntry entry, List<Image> images)
    {
        // Encode the first image to determine shape.
        Tensor firstPixels = ClipImagePreprocessor.Process(images[0], entry.ClipVision.Config.ImageSize);
        Tensor accumulator;
        try
        {
            accumulator = entry.IpAdapter.Config.IsPlus
                ? entry.ClipVision.EncodeHiddenStates(backend, firstPixels)
                : entry.ClipVision.EncodeImageEmbeds(backend, firstPixels);
        }
        finally
        {
            firstPixels.Dispose();
        }

        long count = accumulator.ElementCount;
        // Encode remaining images and accumulate.
        for (int i = 1; i < images.Count; i++)
        {
            Tensor pixels = ClipImagePreprocessor.Process(images[i], entry.ClipVision.Config.ImageSize);
            Tensor next;
            try
            {
                next = entry.IpAdapter.Config.IsPlus
                    ? entry.ClipVision.EncodeHiddenStates(backend, pixels)
                    : entry.ClipVision.EncodeImageEmbeds(backend, pixels);
            }
            finally
            {
                pixels.Dispose();
            }
            try
            {
                if (next.Shape != accumulator.Shape || next.DType != accumulator.DType)
                {
                    throw new InvalidOperationException(
                        $"CLIP-Vision output shape mismatch across reference images: {accumulator.Shape} vs {next.Shape}.");
                }
                float* ap = (float*)accumulator.DataPointer;
                float* np = (float*)next.DataPointer;
                for (long e = 0; e < count; e++) ap[e] += np[e];
            }
            finally
            {
                next.Dispose();
            }
        }

        // Divide by image count.
        float invN = 1.0f / images.Count;
        float* aPtr = (float*)accumulator.DataPointer;
        for (long e = 0; e < count; e++) aPtr[e] *= invN;
        return accumulator;
    }

    /// <summary>Look up the IPA file under <c>&lt;ModelRoot&gt;/ipadapter/&lt;filename&gt;</c>
    /// across all configured model roots. Falls back to the standard Comfy convention if
    /// none of Swarm's <see cref="T2IModelHandler.FolderPaths"/> register an "IPAdapter" key.</summary>
    private static string ResolveIpaModelPath(string filename)
    {
        // Try the file as given, plus common .safetensors/.bin variants.
        string[] candidateNames = filename.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            || filename.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)
            || filename.EndsWith(".pth", StringComparison.OrdinalIgnoreCase)
            ? new[] { filename }
            : new[] { filename + ".safetensors", filename + ".bin", filename + ".pth", filename };

        foreach (string root in Program.ServerSettings.Paths.ActualModelRoots)
        {
            // Comfy's standard location is <ModelRoot>/ipadapter. Also try Capitalized + plural variants.
            foreach (string sub in new[] { "ipadapter", "IpAdapter", "IPAdapter", "ip_adapter" })
            {
                foreach (string name in candidateNames)
                {
                    string candidate = Path.Combine(root, sub, name);
                    if (File.Exists(candidate)) return candidate;
                }
            }
        }
        return null;
    }

    /// <summary>Loads + constructs the IPA + CLIP-Vision pair, then returns a cache entry.
    /// Accepts both SD 1.5 and SDXL IPA checkpoints (Flux IPA refused upstream by
    /// <see cref="IpAdapter"/>'s ctor; FaceID variants refused for the same reason —
    /// they need an InsightFace runtime we don't link).</summary>
    private static IpAdapterCacheEntry LoadIpaEntry(T2IParamInput input, string ipaPath, IpAdapterBaseModel expectedBase, Action<string> log)
    {
        // 1. Load and detect the IPA file (sets variant + base model from key signatures).
        IpAdapterFile file = IpAdapterLoader.Load(ipaPath);
        SafeTensorsLoader clipVisionLoader = null;
        try
        {
            if (file.BaseModel != IpAdapterBaseModel.Sdxl && file.BaseModel != IpAdapterBaseModel.Sd15)
            {
                throw new InvalidOperationException(
                    $"IP-Adapter '{Path.GetFileName(ipaPath)}' detected as base={file.BaseModel}. " +
                    $"This extension currently supports SDXL and SD 1.5 IP-Adapters. Flux IPA uses a DiT cross-attention layout that needs a separate adapter implementation.");
            }
            if (file.BaseModel != expectedBase)
            {
                throw new InvalidOperationException(
                    $"IP-Adapter '{Path.GetFileName(ipaPath)}' is for base={file.BaseModel}, but the current generation is using base={expectedBase}. " +
                    $"Pick an IP-Adapter trained for {expectedBase}, or switch the base model.");
            }
            if (file.Config.IsFaceId)
            {
                throw new InvalidOperationException(
                    $"IP-Adapter FaceID '{Path.GetFileName(ipaPath)}' uses InsightFace ArcFace embeddings instead of CLIP-Vision. " +
                    $"This extension doesn't link an InsightFace runtime; FaceID is refused. Use a non-FaceID variant (standard / Plus / Plus-Face).");
            }
            log($"  variant: {(file.Config.IsPlus ? "Plus" : "Standard")}, base={file.BaseModel}, tokens={file.Config.NumImageTokens}");

            // 2. Build the IPA adapter and load its weights (image projection + per-layer K_ip/V_ip).
            IpAdapter adapter = new IpAdapter(file.Config);
            adapter.LoadWeights(file.Weights);
            log($"  loaded {adapter.CrossAttentionLayerCount} per-cross-attn projections.");

            // 3. Resolve CLIP-Vision: user-selected ClipVisionModel takes priority; otherwise
            //    auto-download CLIP-ViT-H/14 (the encoder all current SDXL IPAs were trained on).
            T2IModel cvModel = ModelAutoDownloader.EnsureSideModel(
                userPick: input?.Get(T2IParamTypes.ClipVisionModel),
                entry: SideModels.ClipVisionH14,
                log: log);
            log($"  CLIP-Vision: {cvModel.Name}");

            clipVisionLoader = new SafeTensorsLoader();
            clipVisionLoader.Load(cvModel.RawFilePath);
            Dictionary<string, Tensor> cvWeights = clipVisionLoader.GetAllTensors();
            // Some image-encoder safetensors ship under "vision_model." prefix already; others
            // ship rooted (e.g. just "embeddings.patch_embedding.weight"). Detect by probing for
            // the patch_embedding weight under either naming.
            string cvPrefix = cvWeights.ContainsKey("vision_model.embeddings.patch_embedding.weight")
                ? "vision_model"
                : (cvWeights.ContainsKey("embeddings.patch_embedding.weight") ? "" : "vision_model");
            ClipVisionEncoder clipVision = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
            clipVision.LoadWeights(cvWeights, prefix: cvPrefix);

            return new IpAdapterCacheEntry
            {
                FilePath = ipaPath,
                File = file,
                IpAdapter = adapter,
                ClipVision = clipVision,
                ClipVisionLoader = clipVisionLoader,
            };
        }
        catch
        {
            clipVisionLoader?.Dispose();
            file.Dispose();
            throw;
        }
    }

    private static string ReadStringParam(T2IParamInput input, string id)
    {
        if (T2IParamTypes.TryGetType(id, out T2IParamType type, input)
            && input.TryGetRaw(type, out object raw) && raw is string s)
        {
            return s;
        }
        return null;
    }

    private static double ReadDoubleParam(T2IParamInput input, string id, double defaultValue)
    {
        if (T2IParamTypes.TryGetType(id, out T2IParamType type, input)
            && input.TryGetRaw(type, out object raw))
        {
            return raw switch
            {
                double d => d,
                float f => f,
                int i => i,
                string s when double.TryParse(s, out double parsed) => parsed,
                _ => defaultValue,
            };
        }
        return defaultValue;
    }
}

/// <summary>Loaded IP-Adapter + its CLIP-Vision encoder, kept around across generations
/// (the weights are identical for repeat gens and CLIP-Vision-H is a 600 MB upload — don't
/// thrash). The cache is keyed by IPA file path; CLIP-Vision is auto-downloaded once and
/// reused. <see cref="Dispose"/> drops both halves; the safetensors loaders' mmap
/// invalidates the underlying tensors.</summary>
public sealed class IpAdapterCacheEntry : IDisposable
{
    public required string FilePath { get; init; }
    public required IpAdapterFile File { get; init; }
    public required IpAdapter IpAdapter { get; init; }
    public required ClipVisionEncoder ClipVision { get; init; }
    public required SafeTensorsLoader ClipVisionLoader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        IpAdapter.Dispose();
        File.Dispose();
        ClipVisionLoader.Dispose();
    }
}
