using System.IO;
using FreneticUtilities.FreneticExtensions;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Segmentation;
using HartsyInference.Vision.Segmentation.Sam2;
using ISImage = SixLabors.ImageSharp.Image;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves a Swarm <c>&lt;segment:yolo-...&gt;</c> prompt part into a grayscale mask Image
/// (white = region to refine) by running the pure-C# YOLO detector in
/// <c>HartsyInference.Vision</c> on the just-generated base image.
///
/// <para>Syntax (matches Comfy's SwarmYoloDetection): <c>&lt;segment:yolo-MODEL[-INDEX][:CLASS],STRENGTH,CREATIVITY&gt; prompt&gt;</c>.
/// The strength becomes the YOLO confidence threshold, INDEX selects the Nth detection in
/// <see cref="T2IParamTypes.SegmentSortOrder"/> order, and CLASS filters by COCO label substring.
/// A negative strength inverts the mask.</para>
///
/// <para>This handles the <c>yolo-</c> target only; free-text targets are routed to
/// <see cref="ClipSegResolver"/> (CLIPSeg) by <see cref="SegmentRefiner"/>.</para>
///
/// <para>YOLO weights are resolved as <c>.safetensors</c> from a <c>yolov8/</c> models folder
/// (engine loads safetensors, not Ultralytics <c>.pt</c>). The model architecture variant
/// (v8 n/s/m/l/x, v11) is inferred from the filename.</para>
/// </summary>
public static class SegmentResolver
{
    /// <summary>Returns true if the segment part targets YOLO (the only supported detector).</summary>
    public static bool IsYoloTarget(PromptRegion.Part part) =>
        part.DataText is not null && part.DataText.StartsWith("yolo-", StringComparison.OrdinalIgnoreCase);

    /// <summary>Builds a grayscale mask Image for a YOLO segment part, or null when nothing was
    /// detected (caller skips the refine for that segment). Mask resolution matches the base image.</summary>
    public static Image BuildYoloMask(IBackend backend, Image baseImage, PromptRegion.Part part, T2IParamInput input, Action<string> log)
    {
        // Parse "yolo-<modelname>[-index][:classfilter]".
        string fullname = part.DataText["yolo-".Length..];
        string[] modelAndClass = fullname.Split(':');
        fullname = modelAndClass[0];
        string classFilter = modelAndClass.Length > 1 ? modelAndClass[1].Trim() : "";
        (string mname, string indexText) = fullname.BeforeAndAfterLast('-');
        int index = 0;
        if (!string.IsNullOrWhiteSpace(indexText) && int.TryParse(indexText, out int parsedIdx))
        {
            fullname = mname;
            index = parsedIdx;
        }

        // Strength == threshold. The old "1.0 means default" convention is no longer valid.
        float threshold = (float)Math.Abs(part.Strength);
        if (threshold > 0.999f) threshold = 0.25f;
        if (threshold <= 0f) threshold = 0.25f;

        string modelPath = ResolveYoloModelPath(fullname);
        if (modelPath is null)
        {
            throw new InvalidOperationException(
                $"YOLO model '{fullname}' not found. Place a .safetensors YOLO model (e.g. converted from " +
                "face_yolov8m.pt) in a 'yolov8' folder under your model root. HartsyInference loads safetensors, " +
                "not Ultralytics .pt files.");
        }

        var (rgb, w, h) = RgbToImage.ToHwcRgb(baseImage);

        YoloConfig config = InferConfig(fullname);
        bool isV11 = fullname.Contains("yolo11", StringComparison.OrdinalIgnoreCase) || fullname.Contains("yolov11", StringComparison.OrdinalIgnoreCase);
        using YoloPipeline pipeline = isV11
            ? YoloPipeline.LoadV11(backend, config, modelPath)
            : new YoloPipeline(backend, config, modelPath);

        IReadOnlyList<YoloDetection> detections = pipeline.Detect(rgb, w, h, confidenceThreshold: threshold);
        if (detections.Count == 0)
        {
            log($"[Segment] yolo-{fullname}: no detections above threshold {threshold:F2} — skipping this segment.");
            return null;
        }

        // Class filter (COCO label substring), then sort + pick the Nth.
        List<YoloDetection> filtered = detections.ToList();
        if (!string.IsNullOrWhiteSpace(classFilter))
        {
            filtered = filtered.Where(d => pipeline.GetLabel(d.ClassId).Contains(classFilter, StringComparison.OrdinalIgnoreCase)).ToList();
            if (filtered.Count == 0)
            {
                log($"[Segment] yolo-{fullname}: no detections matched class '{classFilter}' — skipping.");
                return null;
            }
        }
        string sortOrder = input.Get(T2IParamTypes.SegmentSortOrder, "left-right");
        filtered = SortDetections(filtered, sortOrder);
        if (index < 0 || index >= filtered.Count)
        {
            log($"[Segment] yolo-{fullname}: index {index} out of range (have {filtered.Count}) — using detection 0.");
            index = 0;
        }
        YoloDetection chosen = filtered[index];

        // Build the H×W mask buffer (white inside), grow via separable max-filter dilation, then build the L8
        // image and Gaussian-blur the edges. A negative part strength inverts the mask (refine everything BUT
        // the region). When a SAM 2 checkpoint is installed we feed the chosen YOLO box as a box-prompt to SAM 2
        // for a PIXEL-ACCURATE object mask (a big quality upgrade over the crude bounding-box rasterization);
        // otherwise we fall back to rasterizing the box. The grow/blur post-step is identical either way.
        bool invert = part.Strength < 0;
        int grow = input.Get(T2IParamTypes.SegmentMaskGrow, 16);
        int blur = input.Get(T2IParamTypes.SegmentMaskBlur, 10);
        byte[] maskBytes = TryBuildSam2Mask(backend, baseImage, chosen, w, h, invert, log)
            ?? RasterizeBox(chosen, w, h, invert);
        if (grow > 0) DilateInPlaceSeparable(maskBytes, w, h, grow);
        var maskImg = SixLabors.ImageSharp.Image.LoadPixelData<L8>(maskBytes, w, h);
        try
        {
            if (blur > 0) maskImg.Mutate(ctx => ctx.GaussianBlur(blur / 2.0f));
            log($"[Segment] yolo-{fullname}: matched '{pipeline.GetLabel(chosen.ClassId)}' " +
                $"conf={chosen.Confidence:F2} box=({chosen.X1:F0},{chosen.Y1:F0})-({chosen.X2:F0},{chosen.Y2:F0}) " +
                $"[{index + 1}/{filtered.Count}, sort={sortOrder}].");
            return new Image(maskImg);
        }
        finally
        {
            maskImg.Dispose();
        }
    }

    // TODO(B7): wire RT-DETR + Grounding DINO detectors into this segment path once their deformable attention
    // is GPU-routed. They are real-weight verified but currently host-loop (too slow for interactive gen), so
    // only the YOLO detector feeds <segment:> for now.

    // SAM 2 uses ImageNet normalization on a 1024² square (HF Sam2ImageProcessor: resize to exactly 1024×1024,
    // then (x/255 - mean) / std). Box/point coordinates and the returned mask share that square frame.
    private static readonly float[] s_sam2Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] s_sam2Std = { 0.229f, 0.224f, 0.225f };
    private const int Sam2InputSize = 1024;

    // SAM 2 pipelines are expensive to construct (the Hiera encoder copies every weight), so cache one per
    // checkpoint path. The backend it was built against is stored so a backend swap rebuilds cleanly.
    private static readonly Dictionary<string, (IBackend backend, SamPipeline pipe)> s_samCache = new();
    private static readonly object s_samLock = new();

    /// <summary>Refines the chosen YOLO detection box into a pixel-accurate mask with SAM 2 (box prompt), or
    /// returns null when no SAM 2 checkpoint is installed / the refine fails — in which case the caller falls
    /// back to rasterizing the bounding box. Output is an H×W L8 buffer (255 inside, honoring <paramref name="invert"/>).</summary>
    private static unsafe byte[] TryBuildSam2Mask(IBackend backend, Image baseImage, YoloDetection box, int w, int h, bool invert, Action<string> log)
    {
        string ckpt = ResolveSam2ModelPath();
        if (ckpt is null)
        {
            log("[Segment] SAM2 checkpoint not installed — using the YOLO bounding-box mask. " +
                "Place a converted sam2_hiera_*.safetensors in a 'sam2' folder under your model root for pixel-accurate masks.");
            return null;
        }
        try
        {
            SamPipeline pipeline = GetOrLoadSam2(backend, ckpt);
            Tensor image = BuildSam2Input(baseImage, Sam2InputSize);
            SamMaskResult best;
            try
            {
                float sx = (float)Sam2InputSize / w, sy = (float)Sam2InputSize / h;
                SamBoxPrompt bp = new SamBoxPrompt(
                    Math.Clamp(box.X1 * sx, 0f, Sam2InputSize - 1f), Math.Clamp(box.Y1 * sy, 0f, Sam2InputSize - 1f),
                    Math.Clamp(box.X2 * sx, 0f, Sam2InputSize - 1f), Math.Clamp(box.Y2 * sy, 0f, Sam2InputSize - 1f));
                SamMaskResult[] results = pipeline.Segment(new SamPrompt { Box = bp }, image, Sam2InputSize, Sam2InputSize);
                if (results is null || results.Length == 0)
                {
                    return null;
                }
                best = results[0]; // best-first by predicted IoU
            }
            finally
            {
                image.Dispose();
            }

            // Downscale the model-resolution binary mask to the base image size (nearest), applying invert.
            byte inside = invert ? (byte)0 : (byte)255;
            byte outside = invert ? (byte)255 : (byte)0;
            byte[] outMask = new byte[w * h];
            byte[] src = best.Mask;
            long inside_count = 0;
            for (int y = 0; y < h; y++)
            {
                int syi = Math.Clamp((int)((y + 0.5f) * best.Height / h), 0, best.Height - 1);
                int rowSrc = syi * best.Width;
                int rowDst = y * w;
                for (int x = 0; x < w; x++)
                {
                    int sxi = Math.Clamp((int)((x + 0.5f) * best.Width / w), 0, best.Width - 1);
                    bool on = src[rowSrc + sxi] != 0;
                    outMask[rowDst + x] = on ? inside : outside;
                    if (on) inside_count++;
                }
            }
            log($"[Segment] SAM2 refine: pixel-accurate mask from box prompt (score={best.IoUScore:F3}, " +
                $"{inside_count} px object, checkpoint '{Path.GetFileName(ckpt)}').");
            return outMask;
        }
        catch (Exception ex)
        {
            log($"[Segment] SAM2 refine failed ({ex.Message}) — falling back to the YOLO bounding-box mask.");
            return null;
        }
    }

    /// <summary>Builds the SAM 2 input tensor <c>[1,3,1024,1024]</c> (ImageNet-normalized) from a base image.</summary>
    private static unsafe Tensor BuildSam2Input(Image baseImage, int size)
    {
        byte[] rgb = RgbToImage.ToHwcRgbResized(baseImage, size, size); // HWC RGB u8, size*size*3
        Tensor t = new Tensor(new TensorShape(1, 3, size, size), DType.F32);
        float* dp = (float*)t.DataPointer;
        int spatial = size * size;
        const float inv255 = 1f / 255f;
        for (int c = 0; c < 3; c++)
        {
            float mean = s_sam2Mean[c], invStd = 1f / s_sam2Std[c];
            int chOff = c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                dp[chOff + i] = (rgb[i * 3 + c] * inv255 - mean) * invStd;
            }
        }
        return t;
    }

    private static SamPipeline GetOrLoadSam2(IBackend backend, string ckpt)
    {
        lock (s_samLock)
        {
            if (s_samCache.TryGetValue(ckpt, out (IBackend backend, SamPipeline pipe) entry))
            {
                if (ReferenceEquals(entry.backend, backend)) return entry.pipe;
                entry.pipe.Dispose();
                s_samCache.Remove(ckpt);
            }
            Sam2Config cfg = InferSam2Config(ckpt);
            SamPipeline pipe = SamPipeline.Load(backend, cfg, ckpt);
            s_samCache[ckpt] = (backend, pipe);
            return pipe;
        }
    }

    /// <summary>Infers the SAM 2 Hiera variant from the checkpoint filename (tiny/small/large/base+).</summary>
    private static Sam2Config InferSam2Config(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        if (n.Contains("tiny") || n.Contains("hiera_t")) return Sam2Config.HieraTiny;
        if (n.Contains("small") || n.Contains("hiera_s")) return Sam2Config.HieraSmall;
        if (n.Contains("large") || n.Contains("hiera_l")) return Sam2Config.HieraLarge;
        return Sam2Config.HieraBasePlus; // base_plus / base / unknown
    }

    /// <summary>Locates a SAM 2 <c>.safetensors</c> under a conventional <c>sam2</c> folder (sibling of the SD
    /// model roots, plus a top-level <c>&lt;ModelRoot&gt;/sam2</c>). Returns null when none is installed — the
    /// caller then falls back to the bounding-box mask, so SAM 2 is a pure quality upgrade, never a hard dep.</summary>
    private static string ResolveSam2ModelPath()
    {
        List<string> roots = [];
        if (Program.T2IModelSets.TryGetValue("Stable-Diffusion", out T2IModelHandler sd))
        {
            foreach (string fp in sd.FolderPaths)
            {
                roots.Add(Path.Combine(fp, "sam2"));
                string parent = Path.GetDirectoryName(fp.TrimEnd('/', '\\'));
                if (!string.IsNullOrEmpty(parent)) roots.Add(Path.Combine(parent, "sam2"));
            }
        }
        foreach (string root in Program.ServerSettings.Paths.ActualModelRoots)
        {
            roots.Add(Path.Combine(root, "sam2"));
        }
        foreach (string root in roots.Distinct())
        {
            if (!Directory.Exists(root)) continue;
            foreach (string f in Directory.EnumerateFiles(root, "*.safetensors").OrderBy(x => x, StringComparer.Ordinal))
            {
                return f;
            }
        }
        return null;
    }

    private static byte[] RasterizeBox(YoloDetection box, int w, int h, bool invert)
    {
        byte inside = invert ? (byte)0 : (byte)255;
        byte outside = invert ? (byte)255 : (byte)0;
        byte[] bytes = new byte[w * h];
        if (outside != 0) Array.Fill(bytes, outside);
        int x1 = Math.Clamp((int)MathF.Floor(box.X1), 0, w);
        int x2 = Math.Clamp((int)MathF.Ceiling(box.X2), 0, w);
        int y1 = Math.Clamp((int)MathF.Floor(box.Y1), 0, h);
        int y2 = Math.Clamp((int)MathF.Ceiling(box.Y2), 0, h);
        for (int y = y1; y < y2; y++)
        {
            int rowOff = y * w;
            for (int x = x1; x < x2; x++)
            {
                bytes[rowOff + x] = inside;
            }
        }
        return bytes;
    }

    /// <summary>Separable max-filter dilation (grows the white region by ~radius px, Chebyshev),
    /// matching Comfy's GrowMask intent. Two 1-D passes, same approach as MaskResolver.</summary>
    internal static void DilateInPlaceSeparable(byte[] mask, int w, int h, int radius)
    {
        byte[] tmp = new byte[mask.Length];
        for (int y = 0; y < h; y++)
        {
            int rowOff = y * w;
            for (int x = 0; x < w; x++)
            {
                int x0 = Math.Max(0, x - radius), x1 = Math.Min(w - 1, x + radius);
                byte mx = 0;
                for (int xi = x0; xi <= x1; xi++) { byte v = mask[rowOff + xi]; if (v > mx) mx = v; }
                tmp[rowOff + x] = mx;
            }
        }
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                int y0 = Math.Max(0, y - radius), y1 = Math.Min(h - 1, y + radius);
                byte mx = 0;
                for (int yi = y0; yi <= y1; yi++) { byte v = tmp[yi * w + x]; if (v > mx) mx = v; }
                mask[y * w + x] = mx;
            }
        }
    }

    private static List<YoloDetection> SortDetections(List<YoloDetection> dets, string order) => order switch
    {
        "right-left" => dets.OrderByDescending(d => d.X1).ToList(),
        "top-bottom" => dets.OrderBy(d => d.Y1).ToList(),
        "bottom-top" => dets.OrderByDescending(d => d.Y1).ToList(),
        "largest-smallest" => dets.OrderByDescending(d => d.Area).ToList(),
        "smallest-largest" => dets.OrderBy(d => d.Area).ToList(),
        _ => dets.OrderBy(d => d.X1).ToList(), // left-right (default)
    };

    /// <summary>Infers the YOLO architecture variant from the filename so the right backbone width
    /// is built. Defaults to v8m / v11m when the size letter is absent.</summary>
    private static YoloConfig InferConfig(string name)
    {
        string n = name.ToLowerInvariant();
        bool v11 = n.Contains("yolo11") || n.Contains("yolov11");
        char size = 'm';
        foreach (char c in new[] { 'n', 's', 'm', 'l', 'x' })
        {
            if (n.Contains($"yolov8{c}") || n.Contains($"yolo11{c}") || n.Contains($"yolov11{c}")) { size = c; break; }
        }
        return (v11, size) switch
        {
            (true, 'n') => YoloConfig.YoloV11n,
            (true, 's') => YoloConfig.YoloV11s,
            (true, 'l') => YoloConfig.YoloV11l,
            (true, 'x') => YoloConfig.YoloV11x,
            (true, _) => YoloConfig.YoloV11m,
            (false, 'n') => YoloConfig.YoloV8n,
            (false, 's') => YoloConfig.YoloV8s,
            (false, 'l') => YoloConfig.YoloV8l,
            (false, 'x') => YoloConfig.YoloV8x,
            _ => YoloConfig.YoloV8m,
        };
    }

    /// <summary>Locates a YOLO <c>.safetensors</c> by name across the conventional <c>yolov8</c>
    /// folders (sibling of the SD model roots, mirroring Comfy's <c>folder_paths("yolov8")</c>).
    /// Accepts the name with or without extension.</summary>
    private static string ResolveYoloModelPath(string name)
    {
        List<string> roots = [];
        if (Program.T2IModelSets.TryGetValue("Stable-Diffusion", out T2IModelHandler sd))
        {
            foreach (string fp in sd.FolderPaths)
            {
                roots.Add(Path.Combine(fp, "yolov8"));
                string parent = Path.GetDirectoryName(fp.TrimEnd('/', '\\'));
                if (!string.IsNullOrEmpty(parent)) roots.Add(Path.Combine(parent, "yolov8"));
            }
        }
        if (Program.T2IModelSets.TryGetValue("Clip", out T2IModelHandler clip) && clip.DownloadFolderPath is not null)
        {
            roots.Add(Path.Combine(clip.DownloadFolderPath, "Yolo"));
        }
        string[] candidates = name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            ? [name]
            : [name + ".safetensors", name];
        foreach (string root in roots.Distinct())
        {
            foreach (string cand in candidates)
            {
                string path = Path.Combine(root, cand);
                if (File.Exists(path)) return path;
            }
        }
        return null;
    }
}
