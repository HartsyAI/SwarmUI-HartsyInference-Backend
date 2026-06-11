using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Core.Tensors;
using ISImage = SixLabors.ImageSharp.Image;
using ISSize = SixLabors.ImageSharp.Size;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's mask params (<see cref="T2IParamTypes.MaskImage"/>,
/// <see cref="T2IParamTypes.MaskGrow"/>, <see cref="T2IParamTypes.MaskBlur"/>)
/// into a pixel-space mask Tensor <c>[1, 1, H, W]</c> F32 in <c>[0, 1]</c>
/// that SharpInference's inpaint pipelines consume. Mask convention matches
/// Swarm's UX: white pixels (1.0) get inpainted, black pixels (0.0) are
/// preserved, gray pixels blend proportionally.
///
/// <para>Caller owns the returned Tensor and must dispose it. Returns null
/// when no mask is selected.</para>
///
/// <para><b>Not implemented (v1):</b> <see cref="T2IParamTypes.MaskShrinkGrow"/>
/// (Swarm's "inpaint only masked", crop-to-bbox-of-mask + grow + run on the
/// crop + composite back). It's an optimization for refining tiny details on
/// large canvases — the common full-image inpaint path doesn't need it. To add
/// later, crop the init image and mask to the mask's bbox + grow before
/// passing both to <see cref="Img2ImgResolver"/>, then composite the
/// generated crop back into the original canvas in the loader. The param is
/// silently ignored if set today.</para>
/// </summary>
public static unsafe class MaskResolver
{
    /// <summary>Build a mask tensor from the input, or return null if
    /// <see cref="T2IParamTypes.MaskImage"/> isn't selected. Resizes the
    /// mask to (<paramref name="targetWidth"/>, <paramref name="targetHeight"/>),
    /// applies grow + blur, and returns a <c>[1, 1, H, W]</c> F32 tensor.
    /// Caller is responsible for disposal.</summary>
    public static Tensor Resolve(T2IParamInput input, int targetWidth, int targetHeight)
    {
        if (input is null) return null;
        Image maskImage = input.Get(T2IParamTypes.MaskImage);
        if (maskImage is null) return null;

        byte[] maskBytes = LoadResizedGrayscale(maskImage, targetWidth, targetHeight);

        int grow = input.Get(T2IParamTypes.MaskGrow);
        if (grow > 0)
        {
            DilateInPlaceSeparable(maskBytes, targetWidth, targetHeight, grow);
        }

        int blur = input.Get(T2IParamTypes.MaskBlur);
        if (blur > 0)
        {
            GaussianBlurInPlace(maskBytes, targetWidth, targetHeight, blur);
        }

        Tensor mask = MaskBytesToTensor(maskBytes, targetWidth, targetHeight);
        Logs.Verbose($"[SharpInference][Mask] enabled: {targetWidth}x{targetHeight}, grow={grow}px, blur={blur}px");
        return mask;
    }

    /// <summary>Decode a Swarm <see cref="Image"/> as L8 (single-channel grayscale),
    /// resize bicubic to (w, h), and emit the row-major byte array. Loading
    /// a non-grayscale source as L8 collapses RGB → luminance via ImageSharp's
    /// standard ITU-R BT.709 weights, which is the right semantic when a user
    /// uploads a colored mask layer (e.g. red strokes from a painting tool).</summary>
    private static byte[] LoadResizedGrayscale(Image maskImage, int w, int h)
    {
        using var frame = ISImage.Load<L8>(maskImage.RawData);
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

    /// <summary>Convert an <c>H × W</c> grayscale byte buffer into a
    /// <c>[1, 1, H, W]</c> F32 tensor with values normalized to <c>[0, 1]</c>.</summary>
    private static Tensor MaskBytesToTensor(byte[] bytes, int w, int h)
    {
        Tensor tensor = new Tensor(new TensorShape(1, 1, h, w), DType.F32);
        float* dp = (float*)tensor.DataPointer;
        const float inv255 = 1f / 255f;
        for (int i = 0; i < bytes.Length; i++)
        {
            dp[i] = bytes[i] * inv255;
        }
        return tensor;
    }

    /// <summary>Separable max-filter dilation: each output pixel = max over
    /// a Chebyshev (square) window of radius <paramref name="radius"/> in the
    /// input. Two 1-D passes (horizontal then vertical) instead of a single
    /// 2-D scan — O(W·H·R) vs O(W·H·R²). Approximates a square morphological
    /// dilation kernel; close enough to Comfy's behavior for inpaint-mask
    /// edges where exactness doesn't matter.</summary>
    private static void DilateInPlaceSeparable(byte[] mask, int w, int h, int radius)
    {
        byte[] tmp = new byte[mask.Length];
        // Horizontal pass: tmp[y, x] = max(mask[y, x-r..x+r])
        for (int y = 0; y < h; y++)
        {
            int rowOff = y * w;
            for (int x = 0; x < w; x++)
            {
                int x0 = Math.Max(0, x - radius);
                int x1 = Math.Min(w - 1, x + radius);
                byte mx = 0;
                for (int xi = x0; xi <= x1; xi++)
                {
                    byte v = mask[rowOff + xi];
                    if (v > mx) mx = v;
                }
                tmp[rowOff + x] = mx;
            }
        }
        // Vertical pass: mask[y, x] = max(tmp[y-r..y+r, x])
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                int y0 = Math.Max(0, y - radius);
                int y1 = Math.Min(h - 1, y + radius);
                byte mx = 0;
                for (int yi = y0; yi <= y1; yi++)
                {
                    byte v = tmp[yi * w + x];
                    if (v > mx) mx = v;
                }
                mask[y * w + x] = mx;
            }
        }
    }

    /// <summary>Gaussian blur via ImageSharp's built-in filter on a
    /// temporary L8 image. Swarm's "Mask Blur" param is loosely "kernel
    /// size in pixels"; we map that to <c>sigma = blurFactor / 2</c>, which
    /// gives a kernel whose half-power radius matches the user's intent
    /// (matches the legacy Swarm behavior).</summary>
    private static void GaussianBlurInPlace(byte[] mask, int w, int h, int blurFactor)
    {
        float sigma = blurFactor / 2.0f;
        if (sigma < 1e-3f) return;
        using var img = ISImage.LoadPixelData<L8>(mask, w, h);
        img.Mutate(ctx => ctx.GaussianBlur(sigma));
        img.CopyPixelDataTo(mask);
    }
}
