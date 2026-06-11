using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Utils;
using SharpInference.Core.Tensors;
using ISImage = SixLabors.ImageSharp.Image;
using ISSize = SixLabors.ImageSharp.Size;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Preprocesses a Swarm <see cref="Image"/> into the <c>[1, 3, H, W]</c> F32 tensor
/// the upstream <see cref="SharpInference.Diffusion.Models.TextEncoders.ClipVisionEncoder"/>
/// expects: bicubic resize so the shorter edge equals the target image size, then
/// center-crop to a square <c>imageSize × imageSize</c>, then per-channel normalize
/// with CLIP's standard mean / std (the OpenAI values that OpenCLIP also uses).
///
/// <para>This matches HuggingFace's <c>CLIPImageProcessor</c> default behavior for
/// <c>do_resize=true, do_center_crop=true, do_normalize=true, do_rescale=true</c>,
/// which is what every common IP-Adapter checkpoint was trained against. Bicubic
/// resampling matches the <c>Image.BICUBIC</c> default in PIL.</para>
///
/// <para>Mean/std values are <c>[0.48145466, 0.4578275, 0.40821073]</c> and
/// <c>[0.26862954, 0.26130258, 0.27577711]</c> respectively — embedded as constants
/// rather than read from a config because they're invariant across CLIP-ViT-H/14,
/// CLIP-ViT-L/14, and OpenCLIP variants used by the supported IP-Adapter family.</para>
/// </summary>
public static unsafe class ClipImagePreprocessor
{
    // OpenAI / OpenCLIP normalization constants — same for all CLIP image encoders
    // we currently support (ViT-H/14, ViT-L/14).
    private static readonly float[] s_mean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] s_std = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>Convert a Swarm <see cref="Image"/> to the CLIP-Vision input tensor at the
    /// requested edge size (224 standard for ViT-H/14, 336 for some larger variants).</summary>
    public static Tensor Process(Image input, int imageSize = 224)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        using var src = ISImage.Load<Rgb24>(input.RawData);

        // Resize so the shorter edge = imageSize, preserving aspect, then center-crop.
        // ImageSharp's ResizeMode.Crop resizes the larger dim and then crops centered —
        // exactly what CLIPImageProcessor.center_crop does after the initial resize.
        src.Mutate(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new ISSize(imageSize, imageSize),
            Mode = ResizeMode.Crop,
            Sampler = KnownResamplers.Bicubic,
            Position = AnchorPositionMode.Center,
        }));

        byte[] rgb = new byte[imageSize * imageSize * 3];
        src.CopyPixelDataTo(rgb);

        // Convert to NCHW F32 [1, 3, H, W] and normalize per-channel: (x/255 - mean) / std.
        Tensor output = new Tensor(new TensorShape(1, 3, imageSize, imageSize), DType.F32);
        float* dp = (float*)output.DataPointer;
        int spatial = imageSize * imageSize;
        const float inv255 = 1f / 255f;
        for (int c = 0; c < 3; c++)
        {
            float invStd = 1f / s_std[c];
            float mean = s_mean[c];
            int chOff = c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                // RGB byte buffer is HWC, so element i in spatial maps to pixel i, channel c.
                byte b = rgb[i * 3 + c];
                dp[chOff + i] = (b * inv255 - mean) * invStd;
            }
        }
        Logs.Verbose($"[SharpInference][ClipImage] Preprocessed Swarm image -> [1, 3, {imageSize}, {imageSize}] F32 (CLIP normalized).");
        return output;
    }
}
