using System.IO;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Siglip;
using ISImage = SixLabors.ImageSharp.Image;
using ISSize = SixLabors.ImageSharp.Size;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's style-model params (Comfy's <c>usestylemodel</c> + the three
/// <c>stylemodel*strength</c> knobs) into FLUX.1 Redux conditioning tokens for the engine:
/// SigLIP so400m/14-384 patch tokens → Redux projector → <c>[1, 729, 4096]</c> tokens appended
/// after the T5 stream by <c>FluxPipeline</c>.
///
/// <para>Strength semantics match Comfy/SwarmUI's workflow generation: <c>stylemodelmultiplystrength</c>
/// scales the redux tokens (StyleModelApply strength_type=multiply), and <c>stylemodelmergestrength</c>
/// — a ConditioningAverage between styled and original conditioning — reduces to ANOTHER scalar on
/// the redux tokens (the text tokens are identical on both sides; the original's missing redux
/// positions are zero-padded, so the lerp leaves text unchanged and scales redux by the merge
/// fraction). <c>stylemodelapplystart</c> maps to the pipeline's per-step conditioning switch.</para>
/// </summary>
public static class ReduxResolver
{
    /// <summary>Resolved Redux conditioning: projected style tokens (strength pre-applied) + the apply-start fraction.</summary>
    public sealed class ReduxSpec : IDisposable
    {
        public required Tensor Embeds { get; init; }
        public required float ApplyStart { get; init; }

        public void Dispose() => Embeds?.Dispose();
    }

    private static readonly object s_cacheLock = new();
    private static string s_cachedKey;
    private static SiglipVisionEncoder s_cachedSiglip;
    private static ReduxImageEncoder s_cachedProjector;
    // The encoders' weight tensors are (partly) views over these loaders' mmaps — the loaders
    // must stay alive as long as the cached encoders do (the FluxCacheEntry pattern).
    private static SafeTensorsLoader s_cachedVisionLoader;
    private static SafeTensorsLoader s_cachedStyleLoader;

    /// <summary>Returns null when no style model is selected. Throws with a clear message when a style
    /// model is selected but unusable (missing file / missing prompt image).</summary>
    public static unsafe ReduxSpec Resolve(T2IParamInput input, IBackend backend, Action<string> log)
    {
        if (input is null) return null;
        string styleModelName = ReadStringParam(input, "usestylemodel");
        if (string.IsNullOrEmpty(styleModelName) || styleModelName == "None") return null;

        List<Image> promptImages = input.Get(T2IParamTypes.PromptImages);
        if (promptImages is null || promptImages.Count == 0)
        {
            throw new InvalidOperationException(
                "A Style Model (FLUX.1 Redux) is selected but no Prompt Image was provided. " +
                "Add the style reference via the prompt-image input or set Use Style Model to None.");
        }

        float multiply = (float)ReadDoubleParam(input, "stylemodelmultiplystrength", defaultValue: 1.0);
        float merge = (float)Math.Clamp(ReadDoubleParam(input, "stylemodelmergestrength", defaultValue: 1.0), 0.0, 1.0);
        float applyStart = (float)Math.Clamp(ReadDoubleParam(input, "stylemodelapplystart", defaultValue: 0.0), 0.0, 1.0);
        float scale = multiply * merge;

        string styleModelPath = ResolveStyleModelPath(styleModelName)
            ?? throw new InvalidOperationException(
                $"Style model '{styleModelName}' not found under '{Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, "style_models")}'.");

        T2IModel visionPick = input.TryGet(T2IParamTypes.ClipVisionModel, out T2IModel userVision) ? userVision : null;
        T2IModel visionModel = ModelAutoDownloader.EnsureSideModel(visionPick, SideModels.SigclipVision384, log);

        (SiglipVisionEncoder siglip, ReduxImageEncoder projector) = GetOrLoad(styleModelPath, visionModel.RawFilePath, log);

        // SigLIP encode of the FIRST prompt image (multi-image redux averaging can come later —
        // Comfy's StyleModelApply is also single-image). Straight stretch to 384×384 (crop:"none",
        // matching Swarm's CLIPVisionEncode wiring for redux) + mean/std 0.5 normalize.
        Tensor pixels = PreprocessSiglip(promptImages[0], siglip.Preset.ImageSize);
        Tensor hidden;
        try
        {
            hidden = siglip.EncodeHiddenStates(backend, pixels);
        }
        finally
        {
            pixels.Dispose();
        }

        Tensor embeds;
        try
        {
            embeds = projector.Project(backend, hidden);
        }
        finally
        {
            hidden.Dispose();
        }

        // Free the encoder weights from the device — the DiT needs the VRAM; the next Resolve
        // re-uploads on demand (SigLIP-so400m is ~0.8 GB, a one-off cost per style gen).
        backend.FreeWeights(siglip.EnumerateWeights());

        if (scale != 1.0f)
        {
            ScaleInPlace(embeds, scale);
        }
        // Host-materialize so the tokens survive the pipeline's pre-loop activation sweep.
        _ = embeds.DataPointer;
        log($"Redux style tokens ready: {embeds.Shape[1]} tokens, scale={scale:F2} (multiply {multiply:F2} × merge {merge:F2}), applyStart={applyStart:F2}.");
        return new ReduxSpec { Embeds = embeds, ApplyStart = applyStart };
    }

    private static (SiglipVisionEncoder, ReduxImageEncoder) GetOrLoad(string styleModelPath, string visionPath, Action<string> log)
    {
        string key = styleModelPath + "|" + visionPath;
        lock (s_cacheLock)
        {
            if (s_cachedKey == key)
            {
                log("Redux encoders (cached).");
                return (s_cachedSiglip, s_cachedProjector);
            }
            s_cachedProjector?.Dispose();
            s_cachedVisionLoader?.Dispose();
            s_cachedStyleLoader?.Dispose();
            s_cachedSiglip = null;
            s_cachedProjector = null;
            s_cachedVisionLoader = null;
            s_cachedStyleLoader = null;

            log($"Loading SigLIP vision encoder: {Path.GetFileName(visionPath)}");
            SafeTensorsLoader visionLoader = new SafeTensorsLoader();
            visionLoader.Load(visionPath);
            SiglipVisionEncoder siglip = new SiglipVisionEncoder(SiglipPreset.So400m14_384);
            siglip.LoadWeights(visionLoader.GetAllTensors());

            log($"Loading Redux projector: {Path.GetFileName(styleModelPath)}");
            SafeTensorsLoader styleLoader = new SafeTensorsLoader();
            styleLoader.Load(styleModelPath);
            ReduxImageEncoder projector = new ReduxImageEncoder();
            projector.LoadWeights(styleLoader.GetAllTensors());

            s_cachedKey = key;
            s_cachedSiglip = siglip;
            s_cachedProjector = projector;
            s_cachedVisionLoader = visionLoader;
            s_cachedStyleLoader = styleLoader;
            return (siglip, projector);
        }
    }

    /// <summary>Stretch-resize to <paramref name="imageSize"/>² (no crop — diffusers SiglipImageProcessor
    /// and Swarm's crop:"none" CLIPVisionEncode both stretch) + SigLIP normalize (mean = std = 0.5).</summary>
    private static unsafe Tensor PreprocessSiglip(Image input, int imageSize)
    {
        using var src = ISImage.Load<Rgb24>(input.RawData);
        src.Mutate(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new ISSize(imageSize, imageSize),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Bicubic,
        }));
        byte[] rgb = new byte[imageSize * imageSize * 3];
        src.CopyPixelDataTo(rgb);

        Tensor output = new Tensor(new TensorShape(1, 3, imageSize, imageSize), DType.F32);
        float* dp = (float*)output.DataPointer;
        int spatial = imageSize * imageSize;
        const float inv127_5 = 1f / 127.5f;
        for (int c = 0; c < 3; c++)
        {
            int chOff = c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                dp[chOff + i] = rgb[i * 3 + c] * inv127_5 - 1f;
            }
        }
        return output;
    }

    private static unsafe void ScaleInPlace(Tensor t, float scale)
    {
        float* p = (float*)t.DataPointer;
        long count = t.ElementCount;
        for (long i = 0; i < count; i++) p[i] *= scale;
    }

    private static string ResolveStyleModelPath(string name)
    {
        string root = Program.ServerSettings.Paths.ActualModelRoot;
        string folder = Path.Combine(root, "style_models");
        if (!Directory.Exists(folder)) return null;
        string direct = Path.Combine(folder, name);
        if (File.Exists(direct)) return direct;
        if (File.Exists(direct + ".safetensors")) return direct + ".safetensors";
        foreach (string file in Directory.EnumerateFiles(folder, "*.safetensors", SearchOption.AllDirectories))
        {
            if (Path.GetFileNameWithoutExtension(file).Equals(Path.GetFileNameWithoutExtension(name), StringComparison.OrdinalIgnoreCase))
            {
                return file;
            }
        }
        return null;
    }

    private static string ReadStringParam(T2IParamInput input, string id)
    {
        if (T2IParamTypes.TryGetType(id, out T2IParamType type, input) && input.TryGetRaw(type, out object val) && val is not null)
        {
            return val.ToString();
        }
        return null;
    }

    private static double ReadDoubleParam(T2IParamInput input, string id, double defaultValue)
    {
        if (T2IParamTypes.TryGetType(id, out T2IParamType type, input) && input.TryGetRaw(type, out object val) && val is not null
            && double.TryParse(val.ToString(), out double parsed))
        {
            return parsed;
        }
        return defaultValue;
    }
}
