using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.Vision.Annotators;
using ISImage = SixLabors.ImageSharp.Image;
using ISSize = SixLabors.ImageSharp.Size;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// ControlNet conditioning wrappers for the in-engine annotator models (HED soft-edge, Lineart,
/// NormalBAE, UperNet segmentation) — the same auto-download + static-cache + [1,3,H,W]-[0,1] contract as
/// <see cref="DepthPreprocessor"/>. Each model runs at its architecture-valid resolution
/// (rounded from the generation size) and the map is bilinearly rescaled to the exact target
/// (nearest-neighbor for the segmentation palette map, which must stay on exact class colors).
/// Parity vs controlnet_aux is pinned by the engine's Hed/Lineart/NormalBae/UperNetSeg parity tests.
/// </summary>
public static class AnnotatorControlPreprocessors
{
    private static readonly object s_lock = new();
    private static HedPreprocessor s_hed;
    private static LineartPreprocessor s_lineart;
    private static NormalBaePreprocessor s_normal;
    private static UperNetSegPreprocessor s_segment;
    // Model weights are (partly) views owned by their loaders — loaders live with the cached models.
    private static readonly List<PytorchPickleLoader> s_loaders = [];

    /// <summary>HED soft edge map (softedge_hed form) for <paramref name="input"/> at the generation resolution.</summary>
    public static Tensor ProcessSoftEdge(Image input, int targetWidth, int targetHeight, IBackend backend, Action<string> log)
    {
        HedPreprocessor pre = GetHed(log);
        (byte[] rgb, int w, int h) = DecodeAtValid(input, targetWidth, targetHeight, multiple: 16);
        float[] unit = pre.Process(backend, rgb, w, h, safe: true);
        return UnitMapToConditioning(unit, w, h, targetWidth, targetHeight);
    }

    /// <summary>Scribble form of the HED map (blur + NMS + binarize), for scribble-mode ControlNets.</summary>
    public static Tensor ProcessScribble(Image input, int targetWidth, int targetHeight, IBackend backend, Action<string> log)
    {
        HedPreprocessor pre = GetHed(log);
        (byte[] rgb, int w, int h) = DecodeAtValid(input, targetWidth, targetHeight, multiple: 16);
        float[] unit = pre.ProcessScribble(backend, rgb, w, h);
        return UnitMapToConditioning(unit, w, h, targetWidth, targetHeight);
    }

    /// <summary>Lineart map (white lines on black — the CN conditioning form) at the generation resolution.</summary>
    public static Tensor ProcessLineart(Image input, int targetWidth, int targetHeight, IBackend backend, Action<string> log)
    {
        LineartPreprocessor pre = GetLineart(log);
        (byte[] rgb, int w, int h) = DecodeAtValid(input, targetWidth, targetHeight, multiple: 4);
        float[] unit = pre.Process(backend, rgb, w, h);
        return UnitMapToConditioning(unit, w, h, targetWidth, targetHeight);
    }

    /// <summary>NormalBAE surface-normal RGB map at the generation resolution.</summary>
    public static unsafe Tensor ProcessNormal(Image input, int targetWidth, int targetHeight, IBackend backend, Action<string> log)
    {
        NormalBaePreprocessor pre = GetNormal(log);
        (byte[] rgb, int w, int h) = DecodeAtValid(input, targetWidth, targetHeight, multiple: 32);
        byte[] normalRgb = pre.Process(backend, rgb, w, h);
        // RGB24 → [1,3,h,w] [0,1], then rescale each channel to the exact target dims.
        float[][] channels = new float[3][];
        for (int c = 0; c < 3; c++)
        {
            channels[c] = new float[w * h];
            for (int i = 0; i < w * h; i++) channels[c][i] = normalRgb[i * 3 + c] / 255f;
        }
        Tensor output = new Tensor(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        float* dp = (float*)output.DataPointer;
        long plane = (long)targetWidth * targetHeight;
        for (int c = 0; c < 3; c++)
        {
            float[] resized = BilinearResize(channels[c], w, h, targetWidth, targetHeight);
            fixed (float* rp = resized)
            {
                Buffer.MemoryCopy(rp, dp + c * plane, plane * sizeof(float), plane * sizeof(float));
            }
        }
        log($"NormalBAE map ready: {targetWidth}x{targetHeight}.");
        return output;
    }

    /// <summary>ADE20K-palette segmentation map (UperNet-ConvNeXt-Small, the diffusers reference for
    /// control_v11p_sd15_seg) at the generation resolution. Detection runs at the reference 512×512;
    /// the class map is nearest-neighbor rescaled so conditioning colors stay on exact palette values.</summary>
    public static unsafe Tensor ProcessSegment(Image input, int targetWidth, int targetHeight, IBackend backend, Action<string> log)
    {
        UperNetSegPreprocessor pre = GetSegment(log);
        const int detect = UperNetSegPreprocessor.ReferenceSize;
        (byte[] rgb, int w, int h) = DecodeAtValid(input, detect, detect, multiple: 32);
        byte[] classMap = pre.Process(backend, rgb, w, h);
        Tensor output = new Tensor(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        float* dp = (float*)output.DataPointer;
        long plane = (long)targetWidth * targetHeight;
        for (int y = 0; y < targetHeight; y++)
        {
            int sy = Math.Min((int)((y + 0.5f) * h / targetHeight), h - 1);
            for (int x = 0; x < targetWidth; x++)
            {
                int sx = Math.Min((int)((x + 0.5f) * w / targetWidth), w - 1);
                uint color = Ade20kPalette.Color(classMap[sy * w + sx]);
                long i = (long)y * targetWidth + x;
                dp[i] = ((color >> 16) & 0xFF) / 255f;
                dp[plane + i] = ((color >> 8) & 0xFF) / 255f;
                dp[2 * plane + i] = (color & 0xFF) / 255f;
            }
        }
        log($"Segmentation map ready: {targetWidth}x{targetHeight}.");
        return output;
    }

    private static HedPreprocessor GetHed(Action<string> log)
    {
        lock (s_lock)
        {
            if (s_hed is not null) return s_hed;
            HedPreset preset = HedPreset.Default;
            string path = AnnotatorDownloader.EnsureAnnotator("ControlNetHED.pth",
                $"https://huggingface.co/{preset.HuggingFaceRepo}/resolve/main/ControlNetHED.pth",
                "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa", log);
            log($"Loading HED annotator: {path}");
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            HedModel model = new HedModel();
            model.LoadWeights(loader.GetAllTensors());
            s_loaders.Add(loader);
            s_hed = new HedPreprocessor(model);
            return s_hed;
        }
    }

    private static LineartPreprocessor GetLineart(Action<string> log)
    {
        lock (s_lock)
        {
            if (s_lineart is not null) return s_lineart;
            LineartPreset preset = LineartPreset.Realistic;
            string path = AnnotatorDownloader.EnsureAnnotator("sk_model.pth",
                $"https://huggingface.co/{preset.HuggingFaceRepo}/resolve/main/sk_model.pth",
                "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6", log);
            log($"Loading Lineart annotator: {path}");
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            LineartGenerator model = new LineartGenerator(preset);
            model.LoadWeights(loader.GetAllTensors());
            s_loaders.Add(loader);
            s_lineart = new LineartPreprocessor(model);
            return s_lineart;
        }
    }

    private static NormalBaePreprocessor GetNormal(Action<string> log)
    {
        lock (s_lock)
        {
            if (s_normal is not null) return s_normal;
            NormalBaePreset preset = NormalBaePreset.Default;
            string path = AnnotatorDownloader.EnsureAnnotator("scannet.pt",
                $"https://huggingface.co/{preset.HuggingFaceRepo}/resolve/main/scannet.pt",
                "03dbf1600c51ee3d45c29f77b77bf1a3b7a24c3452dba62a4ae658f37330c209", log);
            log($"Loading NormalBAE annotator: {path}");
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            NormalBaeModel model = new NormalBaeModel(preset);
            model.LoadWeights(loader.GetAllTensors());
            s_loaders.Add(loader);
            s_normal = new NormalBaePreprocessor(model);
            return s_normal;
        }
    }

    private static UperNetSegPreprocessor GetSegment(Action<string> log)
    {
        lock (s_lock)
        {
            if (s_segment is not null) return s_segment;
            UperNetSegPreset preset = UperNetSegPreset.ConvNextSmall;
            string path = AnnotatorDownloader.EnsureAnnotator(preset.LocalFileName,
                $"https://huggingface.co/{preset.HuggingFaceRepo}/resolve/main/{preset.CheckpointFile}",
                preset.Sha256, log);
            log($"Loading UperNet segmentation annotator: {path}");
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            UperNetSegModel model = new UperNetSegModel();
            model.LoadWeights(loader.GetAllTensors());
            s_loaders.Add(loader);
            s_segment = new UperNetSegPreprocessor(model);
            return s_segment;
        }
    }

    /// <summary>Decodes and stretch-resizes the reference to the nearest architecture-valid size at or below the target (min one multiple).</summary>
    private static (byte[] Rgb, int W, int H) DecodeAtValid(Image input, int targetWidth, int targetHeight, int multiple)
    {
        int w = Math.Max(multiple, targetWidth / multiple * multiple);
        int h = Math.Max(multiple, targetHeight / multiple * multiple);
        using var src = ISImage.Load<Rgb24>(input.RawData);
        src.Mutate(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new ISSize(w, h),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Bicubic,
        }));
        byte[] rgb = new byte[w * h * 3];
        src.CopyPixelDataTo(rgb);
        return (rgb, w, h);
    }

    private static unsafe Tensor UnitMapToConditioning(float[] unit, int w, int h, int targetWidth, int targetHeight)
    {
        float[] resized = w == targetWidth && h == targetHeight ? unit : BilinearResize(unit, w, h, targetWidth, targetHeight);
        Tensor output = new Tensor(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        float* dp = (float*)output.DataPointer;
        long plane = (long)targetWidth * targetHeight;
        fixed (float* rp = resized)
        {
            for (int c = 0; c < 3; c++)
            {
                Buffer.MemoryCopy(rp, dp + c * plane, plane * sizeof(float), plane * sizeof(float));
            }
        }
        return output;
    }

    private static float[] BilinearResize(float[] src, int srcW, int srcH, int dstW, int dstH)
    {
        float[] dst = new float[(long)dstW * dstH];
        for (int y = 0; y < dstH; y++)
        {
            float sy = dstH == 1 ? 0f : y * (float)(srcH - 1) / (dstH - 1);
            int y0 = (int)MathF.Floor(sy);
            float fy = sy - y0;
            int y1 = Math.Min(y0 + 1, srcH - 1);
            for (int x = 0; x < dstW; x++)
            {
                float sx = dstW == 1 ? 0f : x * (float)(srcW - 1) / (dstW - 1);
                int x0 = (int)MathF.Floor(sx);
                float fx = sx - x0;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float v0 = src[y0 * srcW + x0] * (1 - fx) + src[y0 * srcW + x1] * fx;
                float v1 = src[y1 * srcW + x0] * (1 - fx) + src[y1 * srcW + x1] * fx;
                dst[(long)y * dstW + x] = v0 * (1 - fy) + v1 * fy;
            }
        }
        return dst;
    }
}
