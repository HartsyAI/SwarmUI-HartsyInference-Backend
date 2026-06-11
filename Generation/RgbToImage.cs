using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using ISImage = SixLabors.ImageSharp.Image;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Convert SharpInference's HWC RGB byte output (the format
/// <see cref="SharpInference.Diffusion.Pipelines.FluxPipeline.GenerateFromTokens"/>
/// returns) into a SwarmUI <see cref="Image"/> as PNG bytes.
/// </summary>
public static class RgbToImage
{
    /// <summary>
    /// Wrap an HWC-interleaved RGB byte array as a SwarmUI Image (PNG-encoded).
    /// </summary>
    /// <param name="rgbData">Pixel bytes, layout <c>rgb[y*width*3 + x*3 + c]</c>, range 0..255.</param>
    /// <param name="width">Pixel width.</param>
    /// <param name="height">Pixel height.</param>
    public static Image FromHwcRgb(byte[] rgbData, int width, int height)
    {
        if (rgbData is null || rgbData.Length != width * height * 3)
        {
            throw new ArgumentException(
                $"RGB data length {rgbData?.Length ?? 0} does not match expected {width * height * 3} for {width}x{height}.",
                nameof(rgbData));
        }

        // ImageSharp's Image<Rgb24>.LoadPixelData expects the same HWC layout.
        using var image = ISImage.LoadPixelData<Rgb24>(rgbData, width, height);
        return new Image(image);
    }

    /// <summary>Decodes only the image header to get pixel dimensions (no full decode) — used to
    /// compute a video target resolution from an init image's aspect before resizing.</summary>
    public static (int width, int height) GetDimensions(Image image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var info = ISImage.Identify(image.RawData);
        return (info.Width, info.Height);
    }

    /// <summary>Like <see cref="ToHwcRgb"/> but resampled to an exact target size first — used to fit
    /// an init image to a video model's latent-grid-aligned resolution before VAE encoding.</summary>
    public static byte[] ToHwcRgbResized(Image image, int width, int height)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        using var frame = ISImage.Load<Rgb24>(image.RawData);
        frame.Mutate(x => x.Resize(width, height));
        byte[] rgb = new byte[width * height * 3];
        frame.CopyPixelDataTo(rgb);
        return rgb;
    }

    /// <summary>Inverse of <see cref="FromHwcRgb"/>: extract HWC-interleaved RGB bytes
    /// from a SwarmUI Image. Used to feed a base-stage image back into the refiner's
    /// VAE encoder which expects raw pixel data.</summary>
    /// <returns>(rgbData, width, height) in the same layout <see cref="FromHwcRgb"/> consumes.</returns>
    public static (byte[] rgbData, int width, int height) ToHwcRgb(Image image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));

        // Decode whatever the image is encoded as (PNG, JPEG, etc.) to an Rgb24 frame.
        using var frame = SixLabors.ImageSharp.Image.Load<Rgb24>(image.RawData);
        int w = frame.Width;
        int h = frame.Height;
        byte[] rgb = new byte[w * h * 3];
        // CopyPixelDataTo writes contiguous HWC bytes — same layout SharpInference's
        // ImagePostProcessor.RgbBytesToTensor expects.
        frame.CopyPixelDataTo(rgb);
        return (rgb, w, h);
    }
}
