using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using ISImage = SixLabors.ImageSharp.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's <see cref="T2IParamTypes.InitImage"/> +
/// <see cref="T2IParamTypes.InitImageCreativity"/> (and, when present, the
/// mask params via <see cref="MaskResolver"/>) into the
/// <see cref="Tensor"/> + strength + optional mask inputs that
/// HartsyInference's pipelines need to build an
/// <c>ImageToImageRequest</c>. Also resizes the init image to match the
/// request resolution (the upstream pipelines require source dims == request dims).
///
/// <para>Mask handling: when <see cref="T2IParamTypes.MaskImage"/> is set,
/// the spec carries a <c>[1, 1, H, W]</c> F32 mask in the convention
/// <c>1 = inpaint, 0 = preserve</c> (matches Swarm UX). The pipeline then
/// runs blend-on-vanilla inpaint — works with any vanilla SDXL/Flux/SD3
/// checkpoint, no dedicated 9-channel inpaint UNet required.</para>
///
/// <para><c>InitImageNoise</c> and <c>InitImageResetToNorm</c> aren't honored
/// yet — both are advanced knobs that would require additional pre-processing
/// in the pipeline. For the first img2img pass we wire the core
/// <c>(image, strength, mask)</c> path only.</para>
/// </summary>
public static class Img2ImgResolver
{
    public sealed class Img2ImgSpec : IDisposable
    {
        /// <summary>Source tensor [1,3,H,W] in [-1,1]. Caller must dispose
        /// when finished (use <see cref="Dispose"/> for one-shot cleanup).</summary>
        public required Tensor SourceTensor { get; init; }
        public required float Strength { get; init; }

        /// <summary>Optional inpaint mask <c>[1, 1, H, W]</c> F32 in <c>[0, 1]</c>
        /// (1 = inpaint, 0 = preserve). Null when no mask is selected — caller
        /// then issues a plain img2img request without a mask.</summary>
        public Tensor MaskTensor { get; init; }

        public void Dispose()
        {
            SourceTensor?.Dispose();
            MaskTensor?.Dispose();
        }
    }

    /// <summary>Build an img2img spec from the input, or return null if no
    /// init image is selected. The caller is responsible for disposing the
    /// returned spec (which disposes both source and mask tensors).</summary>
    public static Img2ImgSpec Resolve(T2IParamInput input, int targetWidth, int targetHeight)
    {
        if (input is null) return null;
        Image initImage = input.Get(T2IParamTypes.InitImage);
        if (initImage is null) return null;

        double creativity = input.Get(T2IParamTypes.InitImageCreativity);
        // Strength==0 short-circuits to byte-identical pass-through upstream — a valid
        // (if degenerate) state. Don't filter it out here; let the user see the result.
        float strength = (float)Math.Clamp(creativity, 0.0, 1.0);

        // Pipelines require source dims == request dims. Resize the init image if needed.
        byte[] rgb = LoadResizedRgb(initImage, targetWidth, targetHeight);
        Tensor tensor = ImagePostProcessor.RgbBytesToTensor(rgb, targetWidth, targetHeight);

        // Mask is optional. When the user has selected MaskImage, attach it; the pipeline
        // promotes the request to the inpaint blend path. Without a mask, plain img2img.
        Tensor mask = MaskResolver.Resolve(input, targetWidth, targetHeight);

        string label = mask is null ? "img2img" : "inpaint";
        Logs.Verbose($"[HartsyInference][Img2img] Enabled: target={targetWidth}x{targetHeight}, strength={strength:F2}, mode={label}.");
        return new Img2ImgSpec
        {
            SourceTensor = tensor,
            Strength = strength,
            MaskTensor = mask,
        };
    }

    /// <summary>Decode a Swarm Image, resize to (w, h) with high-quality bicubic, and
    /// emit the HWC RGB byte array that <see cref="ImagePostProcessor.RgbBytesToTensor"/>
    /// consumes. Bicubic matches what SwarmUI's own resize helpers default to.</summary>
    private static byte[] LoadResizedRgb(Image initImage, int w, int h)
    {
        using var frame = ISImage.Load<Rgb24>(initImage.RawData);
        if (frame.Width != w || frame.Height != h)
        {
            frame.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new SixLabors.ImageSharp.Size(w, h),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Bicubic,
            }));
        }
        byte[] rgb = new byte[w * h * 3];
        frame.CopyPixelDataTo(rgb);
        return rgb;
    }
}
