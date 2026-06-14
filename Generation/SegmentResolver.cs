using System.IO;
using FreneticUtilities.FreneticExtensions;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Vision.Detection;
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
/// <para>Only the <c>yolo-</c> target is supported. Text/CLIP-Seg targets need a segmentation
/// head the engine doesn't ship, so they're refused upfront in the backend validator.</para>
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

        // Rasterize the chosen bbox into an H×W mask buffer (white inside), grow via separable
        // max-filter dilation, then build the L8 image and Gaussian-blur the edges. A negative
        // part strength inverts the mask (refine everything BUT the box).
        bool invert = part.Strength < 0;
        int grow = input.Get(T2IParamTypes.SegmentMaskGrow, 16);
        int blur = input.Get(T2IParamTypes.SegmentMaskBlur, 10);
        byte[] maskBytes = RasterizeBox(chosen, w, h, invert);
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
    private static void DilateInPlaceSeparable(byte[] mask, int w, int h, int radius)
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
