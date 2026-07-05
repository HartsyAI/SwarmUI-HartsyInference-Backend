using System.IO;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation;
using ISImage = SixLabors.ImageSharp.Image;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves a Swarm text <c>&lt;segment:SOME TEXT,threshold,creativity&gt;</c> part into a grayscale
/// mask Image by running the pure-C# CLIPSeg pipeline (<see cref="ClipSegPipeline"/>) on the
/// just-generated base image. This is the text/CLIP-Seg counterpart to <see cref="SegmentResolver"/>'s
/// YOLO path — the missing half of Swarm's <c>&lt;segment:&gt;</c> story (YOLO is class-prompted,
/// CLIPSeg is free-text-prompted).
///
/// <para>CLIPSeg produces a soft [0,1] mask at 224² highlighting the region matching the prompt text.
/// We upsample it to the base-image resolution, binarize at the match threshold (the segment part's
/// first number), then grow + feather-blur exactly like the YOLO path so it feeds the same img2img +
/// inpaint-blend re-denoise.</para>
///
/// <para>Weights: a <c>clipseg-rd64-refined</c> folder containing <c>model.safetensors</c> is resolved
/// under a <c>clipseg</c> models folder (sibling of the SD model roots). The loaded pipeline is cached
/// per-path because construction copies every decoder + CLIP weight to F32.</para>
/// </summary>
public static class ClipSegResolver
{
    // A CLIPSeg pipeline is stateless after load and expensive to construct, so cache by model path.
    private static readonly Dictionary<string, ClipSegPipeline> s_cache = new();
    private static readonly object s_lock = new();

    /// <summary>Builds a grayscale mask Image for a text segment part, or null when the CLIPSeg model
    /// isn't installed (caller falls back / skips). Mask resolution matches the base image.</summary>
    public static Image BuildTextMask(IBackend backend, Image baseImage, PromptRegion.Part part, T2IParamInput input, Action<string> log)
    {
        string target = (part.DataText ?? "").Trim();
        if (string.IsNullOrEmpty(target))
        {
            log("[Segment] CLIPSeg: empty segment target text — skipping.");
            return null;
        }

        string modelDir = ResolveClipSegModelDir();
        if (modelDir is null)
        {
            throw new InvalidOperationException(
                "CLIPSeg model not found. Place the 'clipseg-rd64-refined' folder (with model.safetensors) "
                + "inside a 'clipseg' folder under your model root to enable text '<segment:>' targets.");
        }

        ClipSegPipeline pipeline = GetOrLoad(modelDir);

        Tensor pixels = ToChwRgb01(baseImage, out int w, out int h);
        float[] soft;
        int mw, mh;
        try
        {
            soft = pipeline.Segment(backend, pixels, target, out mw, out mh); // [mh*mw] in [0,1] at 224²
        }
        finally
        {
            pixels.Dispose();
        }

        // Match threshold: the segment part's first number binarizes the soft mask. Swarm's default
        // (no number given → Strength ~= 1.0) maps to a sensible 0.5 midpoint.
        float threshold = MathF.Abs((float)part.Strength);
        if (threshold <= 0f || threshold > 0.999f) threshold = 0.5f;
        bool invert = part.Strength < 0;

        // Upsample the coarse 224² mask to base resolution (bilinear), then binarize at the threshold.
        byte[] maskBytes = new byte[w * h];
        int matched = 0;
        for (int y = 0; y < h; y++)
        {
            float fy = (y + 0.5f) * mh / h - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(fy), 0, mh - 1);
            int y1 = Math.Min(y0 + 1, mh - 1);
            float wy = MathF.Max(0f, fy - y0);
            for (int x = 0; x < w; x++)
            {
                float fx = (x + 0.5f) * mw / w - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(fx), 0, mw - 1);
                int x1 = Math.Min(x0 + 1, mw - 1);
                float wx = MathF.Max(0f, fx - x0);
                float top = soft[y0 * mw + x0] + (soft[y0 * mw + x1] - soft[y0 * mw + x0]) * wx;
                float bot = soft[y1 * mw + x0] + (soft[y1 * mw + x1] - soft[y1 * mw + x0]) * wx;
                float v = top + (bot - top) * wy;
                bool on = v >= threshold;
                if (invert) on = !on;
                if (on) { maskBytes[y * w + x] = 255; matched++; }
            }
        }

        if (matched == 0)
        {
            log($"[Segment] clipseg '{target}': nothing matched above threshold {threshold:F2} — skipping this segment.");
            return null;
        }

        int grow = input.Get(T2IParamTypes.SegmentMaskGrow, 16);
        int blur = input.Get(T2IParamTypes.SegmentMaskBlur, 10);
        if (grow > 0) SegmentResolver.DilateInPlaceSeparable(maskBytes, w, h, grow);
        var maskImg = SixLabors.ImageSharp.Image.LoadPixelData<L8>(maskBytes, w, h);
        try
        {
            if (blur > 0) maskImg.Mutate(ctx => ctx.GaussianBlur(blur / 2.0f));
            log($"[Segment] clipseg '{target}': matched {matched} px ({100.0 * matched / (w * h):F1}% of image) "
                + $"at threshold {threshold:F2}.");
            return new Image(maskImg);
        }
        finally
        {
            maskImg.Dispose();
        }
    }

    private static ClipSegPipeline GetOrLoad(string modelDir)
    {
        lock (s_lock)
        {
            if (!s_cache.TryGetValue(modelDir, out ClipSegPipeline pipe))
            {
                pipe = new ClipSegPipeline(modelDir);
                s_cache[modelDir] = pipe;
            }
            return pipe;
        }
    }

    /// <summary>Converts a Swarm <see cref="Image"/> to a <c>[1, 3, H, W]</c> F32 tensor in raw [0,1]
    /// RGB (CLIPSeg's <c>Segment</c> applies its own ImageNet normalization + resize internally).</summary>
    private static unsafe Tensor ToChwRgb01(Image image, out int width, out int height)
    {
        using SixLabors.ImageSharp.Image<Rgb24> src = ISImage.Load<Rgb24>(image.RawData);
        width = src.Width;
        height = src.Height;
        byte[] rgb = new byte[width * height * 3];
        src.CopyPixelDataTo(rgb);
        Tensor t = new Tensor(new TensorShape(1, 3, height, width), DType.F32);
        float* dp = (float*)t.DataPointer;
        int spatial = width * height;
        const float inv255 = 1f / 255f;
        for (int c = 0; c < 3; c++)
        {
            int chOff = c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                dp[chOff + i] = rgb[i * 3 + c] * inv255;
            }
        }
        return t;
    }

    /// <summary>Locates a CLIPSeg model directory (containing <c>model.safetensors</c>) under a
    /// conventional <c>clipseg</c> folder near the SD model roots. Returns null if none installed.</summary>
    private static string ResolveClipSegModelDir()
    {
        List<string> roots = [];
        if (Program.T2IModelSets.TryGetValue("Stable-Diffusion", out T2IModelHandler sd))
        {
            foreach (string fp in sd.FolderPaths)
            {
                roots.Add(Path.Combine(fp, "clipseg"));
                string parent = Path.GetDirectoryName(fp.TrimEnd('/', '\\'));
                if (!string.IsNullOrEmpty(parent)) roots.Add(Path.Combine(parent, "clipseg"));
            }
        }
        foreach (string root in roots.Distinct())
        {
            if (!Directory.Exists(root)) continue;
            // Direct: root/model.safetensors
            if (File.Exists(Path.Combine(root, "model.safetensors"))) return root;
            // Nested one level: root/<any>/model.safetensors (e.g. clipseg-rd64-refined-fp16-safetensors).
            foreach (string sub in Directory.EnumerateDirectories(root))
            {
                if (File.Exists(Path.Combine(sub, "model.safetensors"))) return sub;
            }
        }
        return null;
    }
}
