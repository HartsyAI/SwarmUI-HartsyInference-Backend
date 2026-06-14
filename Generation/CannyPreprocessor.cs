using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Utils;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using ISImage = SixLabors.ImageSharp.Image;
using ISSize = SixLabors.ImageSharp.Size;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Pure-C# Canny edge preprocessor for ControlNet conditioning.
/// Implements the standard 5-stage Canny pipeline:
/// (1) grayscale conversion + Gaussian noise reduction,
/// (2) Sobel gradient computation,
/// (3) non-maximum suppression along the gradient direction,
/// (4) double thresholding,
/// (5) hysteresis edge tracking.
///
/// <para>Output is a 3-channel RGB image where detected edges are white
/// (<c>255,255,255</c>) on a black background — the format SDXL/SD1.5 Canny
/// ControlNet checkpoints (<c>diffusers/controlnet-canny-sdxl-1.0</c>,
/// <c>lllyasviel/sd-controlnet-canny</c>) were trained to consume.
/// Tensor form is <c>[1, 3, H, W]</c> F32, normalized to <c>[0, 1]</c> —
/// the convention diffusers' ControlNet expects on its conditioning input
/// (the hint encoder bakes any further normalization into its weights).</para>
///
/// <para>Defaults <c>low=100, high=200</c> match Comfy's <c>CannyEdgePreprocessor</c>
/// for parity with the reference workflow. Higher thresholds → fewer / cleaner
/// edges; lower → more / noisier. v1 keeps these hardcoded; expose as advanced
/// params in a follow-up if users ask.</para>
/// </summary>
public static unsafe class CannyPreprocessor
{
    /// <summary>Run Canny on a Swarm <see cref="Image"/>, resize to (<paramref name="targetW"/>, <paramref name="targetH"/>) before processing, and return a <c>[1, 3, H, W]</c> F32 tensor in <c>[0, 1]</c> ready to hand to <see cref="HartsyInference.Diffusion.Adapters.ControlNet"/>. Caller owns the returned tensor.</summary>
    public static Tensor Process(Image inputImage, int targetW, int targetH, int lowThreshold = 100, int highThreshold = 200)
    {
        if (lowThreshold < 0 || highThreshold <= lowThreshold || highThreshold > 255)
        {
            throw new ArgumentException(
                $"Canny thresholds must satisfy 0 <= low < high <= 255; got low={lowThreshold}, high={highThreshold}.");
        }

        byte[] gray = LoadResizedGrayscale(inputImage, targetW, targetH);
        GaussianBlurInPlace(gray, targetW, targetH, sigma: 1.4f);

        // Sobel: separate Gx, Gy; magnitude + quantized direction (0, 45, 90, 135 degrees).
        float[] mag = new float[targetW * targetH];
        byte[] dir = new byte[targetW * targetH];
        ComputeGradients(gray, targetW, targetH, mag, dir);

        // Non-max suppression: zero anything that isn't a local maximum along its gradient direction.
        NonMaxSuppressionInPlace(mag, dir, targetW, targetH);

        // Double threshold + hysteresis: connect weak edges to strong ones via 8-connectivity flood.
        byte[] edges = HysteresisThreshold(mag, targetW, targetH, lowThreshold, highThreshold);

        Logs.Verbose($"[HartsyInference][Canny] {targetW}x{targetH}, low={lowThreshold}, high={highThreshold} -> tensor [1, 3, {targetH}, {targetW}].");
        return EdgesToRgbTensor(edges, targetW, targetH);
    }

    /// <summary>Decode a Swarm <see cref="Image"/> as L8 (luminance), resize bicubic to the target dims, and return the row-major byte buffer.</summary>
    private static byte[] LoadResizedGrayscale(Image inputImage, int w, int h)
    {
        using var frame = ISImage.Load<L8>(inputImage.RawData);
        if (frame.Width != w || frame.Height != h)
        {
            frame.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new ISSize(w, h),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Bicubic,
            }));
        }
        byte[] bytes = new byte[w * h];
        frame.CopyPixelDataTo(bytes);
        return bytes;
    }

    /// <summary>Gaussian blur via ImageSharp's L8 filter. Sigma 1.4 is the textbook Canny default — strong enough to suppress sensor noise without erasing the fine edges Canny is meant to find.</summary>
    private static void GaussianBlurInPlace(byte[] gray, int w, int h, float sigma)
    {
        if (sigma < 1e-3f) return;
        using var img = ISImage.LoadPixelData<L8>(gray, w, h);
        img.Mutate(ctx => ctx.GaussianBlur(sigma));
        img.CopyPixelDataTo(gray);
    }

    /// <summary>3×3 Sobel: <c>Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]</c>, <c>Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]</c>. Computes magnitude (Euclidean) and quantizes the gradient direction into 4 bins (0, 45, 90, 135 degrees) for the NMS step. Boundary pixels left at zero — Canny edges along the image border are uncommon and the simpler boundary handling is worth it.</summary>
    private static void ComputeGradients(byte[] gray, int w, int h, float[] mag, byte[] dir)
    {
        for (int y = 1; y < h - 1; y++)
        {
            int rowOff = y * w;
            for (int x = 1; x < w - 1; x++)
            {
                int idx = rowOff + x;
                // 3×3 window
                int p00 = gray[(y - 1) * w + (x - 1)], p01 = gray[(y - 1) * w + x], p02 = gray[(y - 1) * w + (x + 1)];
                int p10 = gray[y * w + (x - 1)], /* center */                       p12 = gray[y * w + (x + 1)];
                int p20 = gray[(y + 1) * w + (x - 1)], p21 = gray[(y + 1) * w + x], p22 = gray[(y + 1) * w + (x + 1)];
                int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
                int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
                mag[idx] = MathF.Sqrt(gx * gx + gy * gy);

                // Quantize angle into 4 bins. atan2 result in (-π, π]; mirror across origin
                // (gradient direction is unsigned for Canny purposes).
                float angle = MathF.Atan2(gy, gx);
                if (angle < 0) angle += MathF.PI; // now in [0, π]
                float deg = angle * (180f / MathF.PI);
                if (deg < 22.5f || deg >= 157.5f) dir[idx] = 0;        // horizontal edge → check E/W neighbors
                else if (deg < 67.5f) dir[idx] = 1;                    // 45° edge → check NE/SW
                else if (deg < 112.5f) dir[idx] = 2;                   // vertical edge → check N/S
                else dir[idx] = 3;                                     // 135° edge → check NW/SE
            }
        }
    }

    /// <summary>For each pixel, zero its magnitude unless it strictly exceeds the two neighbors along its quantized gradient direction. Standard Canny NMS: thins ridge plateaus down to single-pixel-wide edges.</summary>
    private static void NonMaxSuppressionInPlace(float[] mag, byte[] dir, int w, int h)
    {
        // Use a separate output buffer to avoid in-place suppression reading already-zeroed neighbors.
        float[] tmp = new float[mag.Length];
        for (int y = 1; y < h - 1; y++)
        {
            int rowOff = y * w;
            for (int x = 1; x < w - 1; x++)
            {
                int idx = rowOff + x;
                float m = mag[idx];
                float n1, n2;
                switch (dir[idx])
                {
                    case 0: n1 = mag[idx - 1]; n2 = mag[idx + 1]; break;          // E/W
                    case 1: n1 = mag[idx - w + 1]; n2 = mag[idx + w - 1]; break;  // NE/SW
                    case 2: n1 = mag[idx - w]; n2 = mag[idx + w]; break;          // N/S
                    default: n1 = mag[idx - w - 1]; n2 = mag[idx + w + 1]; break; // NW/SE
                }
                tmp[idx] = (m >= n1 && m >= n2) ? m : 0f;
            }
        }
        Array.Copy(tmp, mag, mag.Length);
    }

    /// <summary>Double-threshold + hysteresis. Pixels above <paramref name="high"/> are unconditional edges; pixels in [<paramref name="low"/>, <paramref name="high"/>) are kept only if 8-connected (transitively) to a strong edge. BFS from each strong-edge seed marks the connected weak component.</summary>
    private static byte[] HysteresisThreshold(float[] mag, int w, int h, int low, int high)
    {
        byte[] result = new byte[mag.Length];
        // Marks: 0=non-edge, 1=weak (above low, below high), 2=strong (above high).
        for (int i = 0; i < mag.Length; i++)
        {
            float m = mag[i];
            if (m >= high) result[i] = 2;
            else if (m >= low) result[i] = 1;
        }

        // BFS from every strong pixel; promote any reachable weak pixels to strong.
        // 8-connectivity flood — diffusers' canny preprocessor uses the same connectivity
        // via skimage.feature.canny defaults.
        Queue<int> queue = new();
        for (int i = 0; i < result.Length; i++) if (result[i] == 2) queue.Enqueue(i);
        while (queue.Count > 0)
        {
            int idx = queue.Dequeue();
            int y = idx / w;
            int x = idx % w;
            for (int dy = -1; dy <= 1; dy++)
            {
                int ny = y + dy;
                if (ny < 0 || ny >= h) continue;
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    if (nx < 0 || nx >= w) continue;
                    int nIdx = ny * w + nx;
                    if (result[nIdx] == 1)
                    {
                        result[nIdx] = 2;
                        queue.Enqueue(nIdx);
                    }
                }
            }
        }

        // Final pass: only strong pixels survive. Convert mark to {0, 255}.
        byte[] edges = new byte[mag.Length];
        for (int i = 0; i < edges.Length; i++)
        {
            edges[i] = result[i] == 2 ? (byte)255 : (byte)0;
        }
        return edges;
    }

    /// <summary>Convert a single-channel edge map (0/255 bytes) into a <c>[1, 3, H, W]</c> F32 tensor with values in <c>[0, 1]</c>. The 3 channels are duplicated (canny is grayscale; ControlNet's hint encoder takes 3-channel input regardless).</summary>
    private static Tensor EdgesToRgbTensor(byte[] edges, int w, int h)
    {
        Tensor tensor = new Tensor(new TensorShape(1, 3, h, w), DType.F32);
        float* dp = (float*)tensor.DataPointer;
        int spatial = w * h;
        const float inv255 = 1f / 255f;
        for (int i = 0; i < spatial; i++)
        {
            float v = edges[i] * inv255;
            dp[i] = v;                  // R channel
            dp[spatial + i] = v;        // G channel
            dp[2 * spatial + i] = v;    // B channel
        }
        return tensor;
    }
}
