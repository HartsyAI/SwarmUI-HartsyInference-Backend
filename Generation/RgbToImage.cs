using ImageMagick;
using ImageMagick.Formats;
using SwarmUI.Media;
using Image = SwarmUI.Utils.Image;
using EngineFluxPipeline = HartsyInference.Diffusion.Pipelines.FluxPipeline;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Converts HartsyInference's HWC RGB byte output (the format <see cref="EngineFluxPipeline.GenerateFromTokens"/> returns) into a SwarmUI <see cref="Image"/> as PNG bytes.</summary>
public static class RgbToImage
{
    /// <summary>Wraps an HWC-interleaved RGB byte array as a SwarmUI Image (PNG-encoded).</summary>
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

        using MagickImage image = HwcRgbToMagick(rgbData, width, height);
        return new Image(image.ToByteArray(MagickFormat.Png), MediaType.ImagePng);
    }

    /// <summary>Wraps HWC RGB bytes plus a straight 8-bit alpha plane (255 = opaque) as an explicit RGBA PNG (colour type 6). A cutout needs its alpha to survive the encode, so the encoder's default colour type is not relied on.</summary>
    public static Image FromHwcRgba(byte[] rgbData, byte[] alpha, int width, int height)
    {
        if (rgbData is null || rgbData.Length != width * height * 3)
        {
            throw new ArgumentException(
                $"RGB data length {rgbData?.Length ?? 0} does not match expected {width * height * 3} for {width}x{height}.",
                nameof(rgbData));
        }
        if (alpha is null || alpha.Length != width * height)
        {
            throw new ArgumentException(
                $"Alpha length {alpha?.Length ?? 0} does not match expected {width * height} for {width}x{height}.",
                nameof(alpha));
        }
        byte[] rgba = new byte[width * height * 4];
        for (int p = 0, s = 0, d = 0; p < width * height; p++, s += 3, d += 4)
        {
            rgba[d] = rgbData[s];
            rgba[d + 1] = rgbData[s + 1];
            rgba[d + 2] = rgbData[s + 2];
            rgba[d + 3] = alpha[p];
        }
        using MagickImage image = new();
        image.ReadPixels(rgba, new PixelReadSettings((uint)width, (uint)height, StorageType.Char, PixelMapping.RGBA));
        image.Settings.SetDefines(new PngWriteDefines() { ColorType = ColorType.TrueColorAlpha, CompressionLevel = 1 });
        return new Image(image.ToByteArray(MagickFormat.Png), MediaType.ImagePng);
    }

    /// <summary>Plain resample of HWC RGB bytes to a new size (bicubic Catmull-Rom), used for a Refiner Upscale below 1, where there is nothing to super-resolve.</summary>
    public static byte[] ResizeHwcRgb(byte[] rgbData, int width, int height, int newWidth, int newHeight)
    {
        if (rgbData is null || rgbData.Length != width * height * 3)
        {
            throw new ArgumentException(
                $"RGB data length {rgbData?.Length ?? 0} does not match expected {width * height * 3} for {width}x{height}.",
                nameof(rgbData));
        }
        if (newWidth == width && newHeight == height)
        {
            return rgbData;
        }
        using MagickImage image = HwcRgbToMagick(rgbData, width, height);
        image.FilterType = FilterType.Catrom;
        image.Resize(new MagickGeometry((uint)newWidth, (uint)newHeight) { IgnoreAspectRatio = true });
        using IPixelCollection<ushort> pixels = image.GetPixels();
        return pixels.ToByteArray(PixelMapping.RGB);
    }

    /// <summary>Reads only the image header for pixel dimensions, e.g. to size a video target from an init image's aspect.</summary>
    public static (int width, int height) GetDimensions(Image image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        MagickImageInfo info = new(image.RawData);
        return ((int)info.Width, (int)info.Height);
    }

    /// <summary>Like <see cref="ToHwcRgb"/> but resampled to an exact size first, to fit an init image to a video model's latent grid before VAE encoding.</summary>
    public static byte[] ToHwcRgbResized(Image image, int width, int height)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        (byte[] rgb, int w, int h) = ToHwcRgb(image);
        return ResizeHwcRgb(rgb, w, h, width, height);
    }

    /// <summary>Inverse of <see cref="FromHwcRgb"/>: extracts HWC-interleaved RGB bytes from a SwarmUI Image.</summary>
    /// <returns>(rgbData, width, height) in the same layout <see cref="FromHwcRgb"/> consumes.</returns>
    public static (byte[] rgbData, int width, int height) ToHwcRgb(Image image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));

        // Decode whatever the image is encoded as (PNG, JPEG, etc.) to 8-bit RGB, first frame only.
        using MagickImage frame = DecodeSrgb(image.RawData);
        int w = (int)frame.Width;
        int h = (int)frame.Height;
        // ToByteArray writes contiguous HWC bytes, the layout HartsyInference's
        // ImagePostProcessor.RgbBytesToTensor expects.
        using IPixelCollection<ushort> pixels = frame.GetPixels();
        return (pixels.ToByteArray(PixelMapping.RGB), w, h);
    }

    /// <summary>Decodes encoded image bytes to an sRGB Magick image, so CMYK sources export as RGB pixels.</summary>
    public static MagickImage DecodeSrgb(byte[] encoded)
    {
        MagickImage frame = new(encoded);
        frame.ColorSpace = ColorSpace.sRGB;
        return frame;
    }

    /// <summary>Builds a Magick image from HWC-interleaved RGB bytes, for the encoders and resamplers that take Magick images.</summary>
    public static MagickImage HwcRgbToMagick(byte[] rgbData, int width, int height)
    {
        MagickImage image = new();
        image.ReadPixels(rgbData, new PixelReadSettings((uint)width, (uint)height, StorageType.Char, PixelMapping.RGB));
        return image;
    }
}
