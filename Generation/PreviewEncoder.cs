#nullable enable
using System;
using Newtonsoft.Json.Linq;
using HartsyInference.Engine.Services;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Jpeg;
using ISImage = SixLabors.ImageSharp.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Turns the RGB preview the Engine attaches to a <see cref="StepPreview"/> tick into a base64 JPEG data URI
/// that the SwarmUI frontend renders through its existing <c>gen_progress.preview</c> handler.
/// <para>Latent decoding (latent2rgb / TAESD) now belongs to the Engine — it owns the latent, and only it knows the
/// architecture's factor table. This class is pure marshalling: pixels in, wire JSON out.</para>
/// <para>Throttled internally to <see cref="MinIntervalMs"/> between encoded previews so a fast sampler can't flood
/// the WebSocket.</para></summary>
public sealed class PreviewEncoder
{
    /// <summary>Minimum milliseconds between encoded previews (~4/sec) — the UI can't render faster, and JPEG
    /// encoding every tick would be pure waste.</summary>
    private const int MinIntervalMs = 250;

    private readonly bool _enabled;
    private long _lastEmitMs;

    /// <summary>Creates an encoder; <paramref name="enabled"/> false makes every <see cref="TryEncode"/> return null.</summary>
    public PreviewEncoder(bool enabled)
    {
        _enabled = enabled;
        _lastEmitMs = 0;
    }

    /// <summary>Whether previews are turned on for this generation.</summary>
    public bool Enabled => _enabled;

    /// <summary>Encodes the preview carried by <paramref name="preview"/>, or returns null when previews are off, the
    /// throttle hasn't elapsed, or this engine build produced no preview pixels for the tick. The returned JObject
    /// matches the schema the frontend's <c>data.gen_progress.preview</c> handler expects.</summary>
    public JObject? TryEncode(StepPreview preview, string batchId, double overallPercent)
    {
        if (!_enabled)
        {
            return null;
        }
        byte[]? rgb = preview.PreviewRgb;
        int w = preview.PreviewWidth;
        int h = preview.PreviewHeight;
        if (rgb is null || w <= 0 || h <= 0 || rgb.Length != w * h * 3)
        {
            return null;
        }
        long now = Environment.TickCount64;
        if (now - _lastEmitMs < MinIntervalMs)
        {
            return null;
        }
        _lastEmitMs = now;
        string dataUri = "data:image/jpeg;base64," + Convert.ToBase64String(EncodeJpeg(rgb, w, h));
        return new JObject
        {
            ["batch_index"] = batchId,
            ["preview"] = dataUri,
            ["overall_percent"] = overallPercent,
            ["current_percent"] = overallPercent,
        };
    }

    /// <summary>Encodes raw HWC RGB bytes to JPEG via ImageSharp. Quality 70 — plenty for a preview, and the
    /// resulting frames are typically under 30 KB on the wire.</summary>
    private static byte[] EncodeJpeg(byte[] rgb, int width, int height)
    {
        using SixLabors.ImageSharp.Image<Rgb24> img = ISImage.LoadPixelData<Rgb24>(rgb, width, height);
        using System.IO.MemoryStream ms = new();
        img.Save(ms, new JpegEncoder { Quality = 70 });
        return ms.ToArray();
    }
}
