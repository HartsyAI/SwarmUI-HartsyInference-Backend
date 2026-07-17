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
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Face;
using HartsyInference.Vision.FaceDetection;
using EngineClipPreprocessor = HartsyInference.Vision.Clip.ClipImagePreprocessor;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's IP-Adapter params (registered by the ComfyUI extension under the
/// <c>ipadapter</c> feature flag) into a list of
/// <see cref="IpAdapterConditioning"/>s ready to hand to a HartsyInference SDXL
/// pipeline. Loads each selected IPA checkpoint, runs the variant's image encoder on the
/// user's prompt image (CLIP-Vision for standard/Plus, ArcFace face embedding for FaceID),
/// projects the result into image-prompt tokens, and returns the per-cross-attn-layer wiring.
///
/// <para><b>Scope:</b> SDXL + SD 1.5; standard + Plus + Plus-Face + FaceID + FaceID-Plus/Plus-v2.
/// For FaceID the face is detected with the engine's YOLO11-pose detector (largest/strongest
/// face), aligned to the ArcFace 112×112 template from the eyes+nose keypoints (see
/// <see cref="FaceAlignment"/> for the documented deviation from insightface's 5-point SCRFD
/// alignment), and embedded with the in-engine ArcFace IR-50. FaceID-Plus additionally renders
/// the same alignment at 224×224 and feeds its CLIP-Vision-H penultimate hidden states to the
/// two-input projection; the Plus-v2 shortcut strength comes from the extension's "FaceID V2
/// Weight" param (default 1.0, official pipeline default). All FaceID checkpoints also ship a
/// rank-128 UNet LoRA — the companion kohya <c>*_lora.safetensors</c> is auto-downloaded and its
/// path surfaced via <see cref="ResolvedSpec.FaceIdLoraPath"/> so the backend merges it through
/// the normal LoRA path. Single adapter only — Comfy lets users stack multiple IPA models, which
/// would sum the per-cross-attn image-attention outputs; deferred. Flux refused upstream.</para>
///
/// <para>The IPA model file is located under <c>&lt;ModelRoot&gt;/ipadapter/&lt;filename&gt;</c>
/// — the standard Comfy path (the known h94 FaceID checkpoints auto-download there). CLIP-Vision
/// is resolved from <see cref="T2IParamTypes.ClipVisionModel"/>, falling back to auto-download of
/// the canonical CLIP-ViT-H/14; the ArcFace weights are the buffalo_l <c>w600k_r50.onnx</c>
/// converted with <c>tests/python-reference/convert_arcface_onnx.py</c>. The loaded entries are
/// returned in a disposable spec so the caller can clean up image tokens after the generation.</para>
/// </summary>
public static class IpAdapterResolver
{
    /// <summary>Converted ArcFace recognition backbone expected under <c>&lt;ModelRoot&gt;/ipadapter/</c>.
    /// Source: buffalo_l w600k_r50.onnx (SHA-256 4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43)
    /// → convert_arcface_onnx.py → SHA-256 cc4023376c340b7fe9d6c192b2ed121c5af34cfccfddaaf9b1fb93bd109fe0df.</summary>
    public const string ArcFaceWeightsFile = "arcface_w600k_r50.safetensors";

    /// <summary>A known FaceID checkpoint + its companion UNet-LoRA half (both from h94/IP-Adapter-FaceID).</summary>
    private sealed record FaceIdDownload(string BinFile, string BinSha, string LoraFile, string LoraSha)
    {
        public string BinUrl => $"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/{BinFile}";
        public string LoraUrl => $"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/{LoraFile}";
    }

    private static readonly FaceIdDownload[] KnownFaceIdDownloads =
    [
        new("ip-adapter-faceid_sdxl.bin", "f455fed24e207c878ec1e0466b34a969d37bab857c5faa4e8d259a0b4ff63d7e",
            "ip-adapter-faceid_sdxl_lora.safetensors", "4fcf93d6e8dc8dd18f5f9e51c8306f369486ed0aa0780ade9961308aff7f0d64"),
        new("ip-adapter-faceid_sd15.bin", "201344e22e6f55849cf07ca7a6e53d8c3b001327c66cb9710d69fd5da48a8da7",
            "ip-adapter-faceid_sd15_lora.safetensors", "70699f0dbfadd47de1f81d263cf4c86bd4b7271d841304af9b340b3a7f38e86a"),
        new("ip-adapter-faceid-plusv2_sdxl.bin", "c6945d82b543700cc3ccbb98d363b837e9c596281607857c74b713a876daf5fb",
            "ip-adapter-faceid-plusv2_sdxl_lora.safetensors", "f24b4bb2dad6638a09c00f151cde84991baf374409385bcbab53c1871a30cb7b"),
        new("ip-adapter-faceid-plusv2_sd15.bin", "26d0d86a1d60d6cc811d3b8862178b461e1eeb651e6fe2b72ba17aa95411e313",
            "ip-adapter-faceid-plusv2_sd15_lora.safetensors", "8abff87a15a049f3e0186c2e82c1c8e77783baf2cfb63f34c412656052eb57b0"),
        new("ip-adapter-faceid-plus_sd15.bin", "252fb53e0d018489d9e7f9b9e2001a52ff700e491894011ada7cfb471e0fadf2",
            "ip-adapter-faceid-plus_sd15_lora.safetensors", "3f00341d11e5e7b5aadf63cbdead09ef82eb28669156161cf1bfc2105d4ff1cd"),
    ];

    /// <summary>Side length of the CLIP-Vision face crop FaceID-Plus consumes (insightface
    /// <c>norm_crop(image_size=224)</c>, matching the official IPAdapterFaceIDPlus pipeline).</summary>
    private const int ClipFaceCropSize = 224;

    /// <summary>One generation's worth of resolved IPA state. Owns the image-prompt token
    /// tensor produced by <see cref="IpAdapter.ProjectImage"/>; the loaded IPA + image encoder
    /// live in <see cref="IpAdapterCacheEntry"/>s that the cache holds across gens.</summary>
    public sealed class ResolvedSpec : IDisposable
    {
        public required List<IpAdapterConditioning> Conditionings { get; init; }
        public required List<Tensor> ImageTokens { get; init; }

        /// <summary>Path of the FaceID companion UNet LoRA to merge (kohya format), or null for
        /// non-FaceID adapters / when the companion file couldn't be located.</summary>
        public string FaceIdLoraPath { get; init; }

        /// <summary>Merge strength for <see cref="FaceIdLoraPath"/>. 1.0 matches the official
        /// IPAdapterFaceID pipeline default (lora_scale=1.0, alpha==rank in the kohya release).</summary>
        public float FaceIdLoraStrength { get; init; } = 1.0f;

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
        // Multi-image handling: standard/Plus average CLIP-Vision outputs before projection;
        // FaceID averages the L2-normalized ArcFace embeddings (renormalized) before projection.
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

        // Locate the IPA file under <ModelRoot>/ipadapter/<filename>. Comfy convention. Known
        // h94 FaceID checkpoints are auto-downloaded there when the user picks them by name.
        string ipaPath = ResolveIpaModelPath(ipaModelName);
        if (ipaPath is null && TryGetKnownFaceIdDownload(ipaModelName, out FaceIdDownload dl))
        {
            ipaPath = AnnotatorDownloader.EnsureFileInFolder("ipadapter", dl.BinFile, dl.BinUrl, dl.BinSha, log);
        }
        if (ipaPath is null)
        {
            throw new InvalidOperationException(
                $"IP-Adapter model '{ipaModelName}' not found in any <ModelRoot>/ipadapter/ subfolder. " +
                $"Place the file in '{Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, "ipadapter")}' or pick a different model.");
        }

        // Cache key: the IPA file's full path (different files = different entries).
        IpAdapterCacheEntry entry = cacheLookup(ipaPath);
        if (entry is null)
        {
            log($"Loading IP-Adapter: {ipaModelName}");
            entry = LoadIpaEntry(input, backend, ipaPath, baseModel, log);
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

        // Run the variant's image encoder over all prompt images (averaging), then project ONCE.
        Tensor imageTokens;
        if (entry.IpAdapter.Config.IsFaceId && entry.IpAdapter.Config.IsPlus)
        {
            // FaceID-Plus / Plus-v2: ArcFace embedding + CLIP-Vision hidden states of the SAME aligned face,
            // mixed by the two-input projection. The v2 shortcut weight follows the official default of 1.0.
            double v2Weight = ReadDoubleParam(input, "faceidv2weight", defaultValue: 1.0);
            (Tensor faceEmbeds, Tensor clipHidden) = EmbedFacesPlus(backend, entry, promptImages, log);
            try
            {
                imageTokens = entry.IpAdapter.ProjectImage(backend, faceEmbeds, clipHidden, (float)v2Weight);
            }
            finally
            {
                faceEmbeds.Dispose();
                clipHidden.Dispose();
            }
            if (entry.IpAdapter.Config.IsFaceIdV2)
            {
                log($"  FaceID V2 weight: {v2Weight:F2}");
            }
        }
        else
        {
            Tensor encoderOut = entry.IpAdapter.Config.IsFaceId
                ? EmbedFaces(backend, entry, promptImages, log)
                : EncodeClipVision(backend, entry, promptImages, log);
            try
            {
                imageTokens = entry.IpAdapter.ProjectImage(backend, encoderOut);
            }
            finally
            {
                encoderOut.Dispose();
            }
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

        string variant = VariantName(entry.IpAdapter.Config);
        log($"IP-Adapter ready: variant={variant}, base={baseModel}, weight={weightRaw:F2}, weightType={weightType}, window=[{startRaw:F2}, {endRaw:F2}], tokens={entry.IpAdapter.NumImageTokens}.");
        return new ResolvedSpec
        {
            Conditionings = conditionings,
            ImageTokens = imageTokenList,
            FaceIdLoraPath = entry.FaceIdLoraPath,
        };
    }

    /// <summary>Standard/Plus image encoding: run CLIP-Vision on each prompt image, average the outputs.</summary>
    private static Tensor EncodeClipVision(IBackend backend, IpAdapterCacheEntry entry, List<Image> images, Action<string> log)
    {
        if (images.Count > 1)
        {
            log($"  averaging {images.Count} reference images (vision-output mean before projection)");
        }
        return AverageVisionOutputs(backend, entry, images);
    }

    /// <summary>FaceID image encoding: for each prompt image detect the strongest face (YOLO11-pose keypoints),
    /// align it to the ArcFace 112×112 template, embed with ArcFace IR-50, then average the L2-normalized
    /// embeddings and renormalize. Returns <c>[1, 512]</c>.</summary>
    private static Tensor EmbedFaces(IBackend backend, IpAdapterCacheEntry entry, List<Image> images, Action<string> log)
    {
        (Tensor faceEmbeds, Tensor clipHidden) = EmbedFacesCore(backend, entry, images, wantClipCrop: false, log);
        clipHidden?.Dispose();
        return faceEmbeds;
    }

    /// <summary>FaceID-Plus image encoding: like <see cref="EmbedFaces"/>, but ALSO runs CLIP-Vision over the
    /// same alignment rendered at 224×224 (the official pipeline's <c>norm_crop(image_size=224)</c> input) and
    /// averages the penultimate hidden states across images. Returns (<c>[1, 512]</c>, <c>[1, 257, 1280]</c>).</summary>
    private static (Tensor faceEmbeds, Tensor clipHidden) EmbedFacesPlus(IBackend backend, IpAdapterCacheEntry entry, List<Image> images, Action<string> log)
    {
        return EmbedFacesCore(backend, entry, images, wantClipCrop: true, log);
    }

    private static unsafe (Tensor faceEmbeds, Tensor clipHidden) EmbedFacesCore(
        IBackend backend, IpAdapterCacheEntry entry, List<Image> images, bool wantClipCrop, Action<string> log)
    {
        ArcFaceModel arcFace = entry.ArcFace
            ?? throw new InvalidOperationException("FaceID cache entry has no ArcFace model (loader bug).");
        YoloPosePipeline pose = entry.PosePipeline
            ?? throw new InvalidOperationException("FaceID cache entry has no pose pipeline (loader bug).");
        if (wantClipCrop && entry.ClipVision is null)
        {
            throw new InvalidOperationException("FaceID-Plus cache entry has no CLIP-Vision encoder (loader bug).");
        }

        float[] accumulator = new float[ArcFaceModel.EmbeddingDim];
        Tensor clipAccumulator = null;
        long clipCount = 0;
        backend.PreloadWeights(arcFace.EnumerateWeights());
        try
        {
            foreach (Image image in images)
            {
                (byte[] rgb, int width, int height) = RgbToImage.ToHwcRgb(image);
                (byte[] crop, byte[] clipCrop) = DetectAndAlignFace(pose, rgb, width, height, wantClipCrop, log);
                Tensor inputTensor = ArcFaceModel.PreprocessAligned(crop);
                Tensor embed;
                try
                {
                    embed = arcFace.EmbedNormalized(backend, inputTensor);
                }
                finally
                {
                    inputTensor.Dispose();
                }
                try
                {
                    float* ep = (float*)embed.DataPointer;
                    for (int d = 0; d < ArcFaceModel.EmbeddingDim; d++) accumulator[d] += ep[d];
                }
                finally
                {
                    embed.Dispose();
                }

                if (!wantClipCrop)
                {
                    continue;
                }
                EngineClipPreprocessor preprocess = new EngineClipPreprocessor(ClipFaceCropSize);
                Tensor pixels = preprocess.Preprocess(clipCrop, ClipFaceCropSize, ClipFaceCropSize);
                Tensor hidden;
                try
                {
                    hidden = entry.ClipVision.EncodeHiddenStates(backend, pixels);
                }
                finally
                {
                    pixels.Dispose();
                }
                if (clipAccumulator is null)
                {
                    clipAccumulator = hidden;
                    clipCount = hidden.ElementCount;
                }
                else
                {
                    try
                    {
                        if (hidden.Shape != clipAccumulator.Shape)
                        {
                            throw new InvalidOperationException(
                                $"CLIP-Vision face-crop output shape mismatch across reference images: {clipAccumulator.Shape} vs {hidden.Shape}.");
                        }
                        float* ap = (float*)clipAccumulator.DataPointer;
                        float* hp = (float*)hidden.DataPointer;
                        for (long e = 0; e < clipCount; e++) ap[e] += hp[e];
                    }
                    finally
                    {
                        hidden.Dispose();
                    }
                }
            }
        }
        catch
        {
            clipAccumulator?.Dispose();
            throw;
        }
        finally
        {
            backend.FreeWeights(arcFace.EnumerateWeights());
        }

        if (images.Count > 1)
        {
            log($"  averaged {images.Count} face embeddings (renormalized identity centroid)"
                + (wantClipCrop ? " + CLIP face-crop hidden states (mean)" : ""));
        }
        double norm = 0;
        for (int d = 0; d < ArcFaceModel.EmbeddingDim; d++) norm += (double)accumulator[d] * accumulator[d];
        float inv = (float)(1.0 / Math.Max(Math.Sqrt(norm), 1e-12));
        Tensor result = new Tensor(new TensorShape(1, ArcFaceModel.EmbeddingDim), DType.F32);
        float* rp = (float*)result.DataPointer;
        for (int d = 0; d < ArcFaceModel.EmbeddingDim; d++) rp[d] = accumulator[d] * inv;

        if (wantClipCrop && images.Count > 1)
        {
            float invN = 1.0f / images.Count;
            float* cp = (float*)clipAccumulator.DataPointer;
            for (long e = 0; e < clipCount; e++) cp[e] *= invN;
        }
        return (result, clipAccumulator);
    }

    private static string VariantName(IpAdapterConfig config) => config.IsFaceId
        ? config.IsPlus ? (config.IsFaceIdV2 ? "FaceID-PlusV2" : "FaceID-Plus") : "FaceID"
        : config.IsPlus ? "Plus" : "Standard";

    /// <summary>Detects people, picks the best face (largest inter-eye distance among detections with usable
    /// eye+nose keypoints, else the highest-confidence person), and returns an ArcFace-aligned 112×112 RGB crop
    /// plus (when <paramref name="wantClipCrop"/>) the SAME alignment rendered at 224×224 for CLIP-Vision.
    /// Falls back to a square face-region crop (no rotation) when keypoints are missing, and to a center crop
    /// when no person is detected at all.</summary>
    private static (byte[] arcCrop, byte[] clipCrop) DetectAndAlignFace(YoloPosePipeline pose, byte[] rgb, int width, int height, bool wantClipCrop, Action<string> log)
    {
        IReadOnlyList<PoseDetection> people = pose.Detect(rgb, width, height, confidenceThreshold: 0.25f, iouThreshold: 0.45f);

        PoseDetection bestAligned = null;
        float[] bestPoints = null;
        float bestEyeDist = -1f;
        PoseDetection bestAny = null;
        foreach (PoseDetection person in people)
        {
            if (bestAny is null || person.Confidence > bestAny.Confidence) bestAny = person;
            if (FaceAlignment.TryGetAlignmentPoints(person, visThreshold: 0.3f, out float[] pts))
            {
                float dx = pts[2] - pts[0], dy = pts[3] - pts[1];
                float eyeDist = MathF.Sqrt(dx * dx + dy * dy);
                if (eyeDist > bestEyeDist)
                {
                    bestEyeDist = eyeDist;
                    bestAligned = person;
                    bestPoints = pts;
                }
            }
        }

        if (bestAligned is not null)
        {
            return (FaceAlignment.AlignToTemplate(rgb, width, height, bestPoints),
                wantClipCrop ? FaceAlignment.AlignToTemplate(rgb, width, height, bestPoints, outputSize: ClipFaceCropSize) : null);
        }

        if (bestAny is not null)
        {
            log("  FaceID: face keypoints not visible — falling back to unrotated square face crop.");
            PoseFaceCrop.Rect rect = PoseFaceCrop.ComputeSquareCrop(bestAny, width, height, expand: 1.6f);
            return (SquareCropTo(rgb, width, height, rect.X, rect.Y, rect.Size, FaceAlignment.CropSize),
                wantClipCrop ? SquareCropTo(rgb, width, height, rect.X, rect.Y, rect.Size, ClipFaceCropSize) : null);
        }

        log("  FaceID: WARNING — no person detected in the prompt image; using a center crop. Identity transfer will be weak.");
        float side = Math.Min(width, height);
        float cx = (width - side) * 0.5f, cy = (height - side) * 0.5f;
        return (SquareCropTo(rgb, width, height, cx, cy, side, FaceAlignment.CropSize),
            wantClipCrop ? SquareCropTo(rgb, width, height, cx, cy, side, ClipFaceCropSize) : null);
    }

    /// <summary>Scales a square source region to an <paramref name="outSize"/>² crop via the shared affine warp
    /// (112 for ArcFace, 224 for the FaceID-Plus CLIP-Vision input).</summary>
    private static byte[] SquareCropTo(byte[] rgb, int width, int height, float x, float y, float side, int outSize)
    {
        float s = outSize / Math.Max(side, 1f);
        FaceAlignment.Affine2x3 srcToDst = new(s, 0f, -x * s, 0f, s, -y * s);
        return FaceAlignment.WarpAffine(rgb, width, height, srcToDst, outSize, outSize);
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

    private static bool TryGetKnownFaceIdDownload(string requestedName, out FaceIdDownload download)
    {
        string bare = Path.GetFileNameWithoutExtension(requestedName).ToLowerInvariant();
        foreach (FaceIdDownload dl in KnownFaceIdDownloads)
        {
            if (Path.GetFileNameWithoutExtension(dl.BinFile).ToLowerInvariant() == bare)
            {
                download = dl;
                return true;
            }
        }
        download = null;
        return false;
    }

    /// <summary>Loads + constructs the IPA + its image encoder, then returns a cache entry.
    /// Standard/Plus get CLIP-Vision (auto-downloaded ViT-H/14 unless the user picked one);
    /// FaceID gets the ArcFace IR-50 + YOLO11-pose pair and its companion UNet LoRA path.
    /// Accepts both SD 1.5 and SDXL checkpoints (Flux + FaceID-Plus refused upstream by
    /// <see cref="IpAdapter"/>'s ctor).</summary>
    private static IpAdapterCacheEntry LoadIpaEntry(T2IParamInput input, IBackend backend, string ipaPath, IpAdapterBaseModel expectedBase, Action<string> log)
    {
        // 1. Load and detect the IPA file (sets variant + base model from key signatures).
        IpAdapterFile file = IpAdapterLoader.Load(ipaPath);
        SafeTensorsLoader clipVisionLoader = null;
        SafeTensorsLoader arcFaceLoader = null;
        YoloPosePipeline posePipeline = null;
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
            log($"  variant: {VariantName(file.Config)}, base={file.BaseModel}, tokens={file.Config.NumImageTokens}");

            // 2. Build the IPA adapter and load its weights (image projection + per-layer K_ip/V_ip).
            IpAdapter adapter = new IpAdapter(file.Config);
            adapter.LoadWeights(file.Weights);
            log($"  loaded {adapter.CrossAttentionLayerCount} per-cross-attn projections.");

            if (file.Config.IsFaceId)
            {
                // 3a. FaceID: ArcFace face embedder + YOLO11-pose face detector + companion UNet LoRA.
                //     FaceID-Plus/Plus-v2 additionally need CLIP-Vision for the aligned face crop.
                string arcFacePath = ResolveArcFaceWeights();
                arcFaceLoader = new SafeTensorsLoader();
                arcFaceLoader.Load(arcFacePath);
                ArcFaceModel arcFace = new ArcFaceModel();
                arcFace.LoadWeights(arcFaceLoader.GetAllTensors());
                log($"  ArcFace: {Path.GetFileName(arcFacePath)}");

                posePipeline = new YoloPosePipeline(backend, YoloConfig.YoloV11nPose, WanAnimateLoader.ResolvePoseWeights(), inputSize: 640);

                string loraPath = ResolveFaceIdLora(ipaPath, log);
                if (loraPath is null)
                {
                    log("  WARNING: FaceID companion LoRA not found — identity likeness will be much weaker. " +
                        "Place the matching *_lora.safetensors next to the FaceID checkpoint.");
                }

                ClipVisionEncoder faceClipVision = null;
                if (file.Config.IsPlus)
                {
                    (faceClipVision, clipVisionLoader) = LoadClipVision(input, log);
                }

                return new IpAdapterCacheEntry
                {
                    FilePath = ipaPath,
                    File = file,
                    IpAdapter = adapter,
                    ClipVision = faceClipVision,
                    ClipVisionLoader = clipVisionLoader,
                    ArcFace = arcFace,
                    ArcFaceLoader = arcFaceLoader,
                    PosePipeline = posePipeline,
                    FaceIdLoraPath = loraPath,
                };
            }

            // 3b. Standard/Plus: CLIP-Vision over the full reference image.
            ClipVisionEncoder clipVision;
            (clipVision, clipVisionLoader) = LoadClipVision(input, log);

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
            posePipeline?.Dispose();
            arcFaceLoader?.Dispose();
            clipVisionLoader?.Dispose();
            file.Dispose();
            throw;
        }
    }

    /// <summary>Resolves and loads the CLIP-Vision encoder: the user-selected ClipVisionModel takes priority;
    /// otherwise the canonical CLIP-ViT-H/14 is auto-downloaded (the encoder every supported IPA — including
    /// FaceID-Plus — was trained against). Returns the encoder plus the loader that owns its mmap.</summary>
    private static (ClipVisionEncoder encoder, SafeTensorsLoader loader) LoadClipVision(T2IParamInput input, Action<string> log)
    {
        T2IModel cvModel = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.ClipVisionModel),
            entry: SideModels.ClipVisionH14,
            log: log);
        log($"  CLIP-Vision: {cvModel.Name}");

        SafeTensorsLoader clipVisionLoader = new SafeTensorsLoader();
        try
        {
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
            return (clipVision, clipVisionLoader);
        }
        catch
        {
            clipVisionLoader.Dispose();
            throw;
        }
    }

    /// <summary>Locates the converted ArcFace safetensors under any <c>&lt;ModelRoot&gt;/ipadapter/</c>.
    /// The file is a local conversion of the official ONNX (same precedent as the YOLO11n-pose weights) —
    /// there is no hosted copy to auto-download yet.</summary>
    private static string ResolveArcFaceWeights()
    {
        foreach (string root in Program.ServerSettings.Paths.ActualModelRoots)
        {
            foreach (string sub in new[] { "ipadapter", "IpAdapter", "IPAdapter", "ip_adapter" })
            {
                string candidate = Path.Combine(root, sub, ArcFaceWeightsFile);
                if (File.Exists(candidate)) return candidate;
            }
        }
        throw new SwarmUserErrorException(
            $"HartsyInference: IP-Adapter FaceID needs the ArcFace face-embedding weights at " +
            $"'{Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, "ipadapter", ArcFaceWeightsFile)}'. " +
            "Convert insightface buffalo_l's w600k_r50.onnx with tests/python-reference/convert_arcface_onnx.py " +
            "(pip install onnx safetensors; the script also emits a parity reference).");
    }

    /// <summary>Finds the FaceID companion UNet LoRA: a sibling <c>&lt;name&gt;_lora.safetensors</c> next to
    /// the checkpoint, else the known h94 companion (auto-downloaded). Returns null when unavailable.</summary>
    private static string ResolveFaceIdLora(string ipaPath, Action<string> log)
    {
        string sibling = Path.Combine(
            Path.GetDirectoryName(ipaPath) ?? "",
            Path.GetFileNameWithoutExtension(ipaPath) + "_lora.safetensors");
        if (File.Exists(sibling)) return sibling;

        if (TryGetKnownFaceIdDownload(Path.GetFileName(ipaPath), out FaceIdDownload dl))
        {
            try
            {
                return AnnotatorDownloader.EnsureFileInFolder("ipadapter", dl.LoraFile, dl.LoraUrl, dl.LoraSha, log);
            }
            catch (Exception ex)
            {
                Logs.Error($"[HartsyInference] FaceID companion LoRA download failed: {ex.Message}");
                return null;
            }
        }
        return null;
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

/// <summary>Loaded IP-Adapter + its image encoder, kept around across generations
/// (the weights are identical for repeat gens and CLIP-Vision-H is a 600 MB upload — don't
/// thrash). The cache is keyed by IPA file path. Standard/Plus entries hold CLIP-Vision;
/// FaceID entries hold the ArcFace embedder + YOLO11-pose detector + companion LoRA path
/// instead. <see cref="Dispose"/> drops whichever halves are present; the safetensors
/// loaders' mmap invalidates the underlying tensors.</summary>
public sealed class IpAdapterCacheEntry : IDisposable
{
    public required string FilePath { get; init; }
    public required IpAdapterFile File { get; init; }
    public required IpAdapter IpAdapter { get; init; }
    public required ClipVisionEncoder ClipVision { get; init; }
    public required SafeTensorsLoader ClipVisionLoader { get; init; }

    /// <summary>ArcFace IR-50 face embedder (FaceID entries only).</summary>
    public ArcFaceModel ArcFace { get; init; }

    /// <summary>Loader owning the ArcFace weights' mmap (FaceID entries only).</summary>
    public SafeTensorsLoader ArcFaceLoader { get; init; }

    /// <summary>YOLO11-pose face/keypoint detector (FaceID entries only).</summary>
    public YoloPosePipeline PosePipeline { get; init; }

    /// <summary>Path of the FaceID companion UNet LoRA (kohya), or null.</summary>
    public string FaceIdLoraPath { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        IpAdapter.Dispose();
        File.Dispose();
        ClipVisionLoader?.Dispose();
        PosePipeline?.Dispose();
        ArcFaceLoader?.Dispose();
    }
}
