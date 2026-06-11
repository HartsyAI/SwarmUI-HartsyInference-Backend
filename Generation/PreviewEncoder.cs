#nullable enable
using System;
using Newtonsoft.Json.Linq;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Jpeg;
using ISImage = SixLabors.ImageSharp.Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>Turns an in-flight diffusion latent into a base64-encoded JPEG that the
/// SwarmUI frontend can render directly via its existing <c>gen_progress.preview</c>
/// handler. Two pluggable decode paths:
/// <list type="bullet">
/// <item><b>Latent2Rgb</b> — fast (&lt;5 ms), no extra model files, blurry/color-shifted output.</item>
/// <item><b>Taesd</b> — slower (~30–80 ms), high fidelity. Requires a per-architecture TAESD
/// safetensors file resolved + downloaded via <see cref="TaesdResolver"/>. Falls back to
/// Latent2Rgb if the file can't be obtained or the arch has no published TAESD.</item>
/// </list>
/// <para>Throttled internally to <see cref="MinIntervalMs"/> ms between encoded previews —
/// keeps the WebSocket from flooding the UI on fast samplers and keeps preview cost
/// &lt; 1% of generation time even on the slow (TAESD) path.</para></summary>
public sealed class PreviewEncoder
{
    /// <summary>How previews are decoded from the latent.</summary>
    public enum Method
    {
        /// <summary>No previews emitted at all.</summary>
        Off,
        /// <summary>Latent-factor approximation. Fast, model-free.</summary>
        Latent2Rgb,
        /// <summary>Tiny autoencoder (~10 MB per arch). Slower but much better-looking previews.</summary>
        Taesd,
    }

    /// <summary>Minimum milliseconds between encoded previews. The pipeline can fire
    /// onProgress 20+ times per second on fast samplers; encoding every callback would
    /// be wasteful (UI can't render that fast anyway). 250 ms ≈ 4 previews/sec is a
    /// good UX/cost balance.</summary>
    private const int MinIntervalMs = 250;

    private readonly Method _method;
    private readonly IBackend? _backend;
    private readonly Func<LatentArchitecture, TaesdDecoder?>? _taesdResolver;
    private long _lastEmitMs;

    /// <summary>Construct an encoder. Pass <paramref name="backend"/> + <paramref name="taesdResolver"/>
    /// to enable the TAESD path; pass <c>null</c> for both to disable it (Latent2Rgb still works).
    /// The resolver is invoked lazily — it's only called when a TAESD-eligible preview is about
    /// to be encoded, so no download happens until previews actually start firing.</summary>
    public PreviewEncoder(Method method, IBackend? backend = null, Func<LatentArchitecture, TaesdDecoder?>? taesdResolver = null)
    {
        _method = method;
        _backend = backend;
        _taesdResolver = taesdResolver;
        _lastEmitMs = 0;
    }

    public bool Enabled => _method != Method.Off;

    /// <summary>Attempts to encode a preview from the current generation progress. Returns
    /// null when previews are disabled, the throttle hasn't elapsed, the latent is missing,
    /// or the architecture has no factor table. The returned JObject matches the schema the
    /// SwarmUI frontend's <c>data.gen_progress.preview</c> handler expects.</summary>
    public JObject? TryEncode(GenerationProgress progress, string batchId, double overallPercent)
    {
        if (_method == Method.Off) return null;
        if (progress.Latent is null) return null;
        if (progress.LatentArch == LatentArchitecture.Unknown) return null;

        long now = Environment.TickCount64;
        if (now - _lastEmitMs < MinIntervalMs) return null;

        byte[]? rgb;
        int w, h;
        if (_method == Method.Taesd)
        {
            rgb = TryEncodeTaesd(progress.Latent, progress.LatentArch, out w, out h);
            if (rgb is null)
            {
                // TAESD unavailable / failed — fall through to Latent2Rgb so the user sees
                // *something* rather than a black frame for the whole generation.
                rgb = LatentPreview.DecodeLatent2Rgb(progress.Latent, progress.LatentArch, out w, out h);
            }
        }
        else
        {
            rgb = LatentPreview.DecodeLatent2Rgb(progress.Latent, progress.LatentArch, out w, out h);
        }

        if (rgb is null || w <= 0 || h <= 0) return null;

        byte[] jpeg = EncodeJpeg(rgb, w, h);
        string dataUri = "data:image/jpeg;base64," + Convert.ToBase64String(jpeg);
        _lastEmitMs = now;

        return new JObject
        {
            ["batch_index"] = batchId,
            ["preview"] = dataUri,
            ["overall_percent"] = overallPercent,
            ["current_percent"] = overallPercent,
        };
    }

    /// <summary>Encodes raw HWC RGB bytes to JPEG via ImageSharp. Quality 70 — plenty for a
    /// preview, and the resulting frames are typically &lt; 30 KB on the wire.</summary>
    private static byte[] EncodeJpeg(byte[] rgb, int width, int height)
    {
        using SixLabors.ImageSharp.Image<Rgb24> img = ISImage.LoadPixelData<Rgb24>(rgb, width, height);
        using System.IO.MemoryStream ms = new();
        img.Save(ms, new JpegEncoder { Quality = 70 });
        return ms.ToArray();
    }

    /// <summary>Runs the TAESD decoder over the in-flight latent and returns HWC RGB bytes at
    /// the final image's resolution (8× the latent's H×W). Returns null if no decoder is
    /// available for the architecture, the backend isn't wired in, or the forward pass throws.</summary>
    private byte[]? TryEncodeTaesd(Tensor latent, LatentArchitecture arch, out int width, out int height)
    {
        width = 0;
        height = 0;
        if (_backend is null || _taesdResolver is null) return null;
        TaesdDecoder? decoder;
        try { decoder = _taesdResolver(arch); }
        catch (Exception ex)
        {
            Logs.Warning($"[SharpInference][TAESD] resolver threw: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
        if (decoder is null) return null;

        Tensor? rgb01 = null;
        try
        {
            rgb01 = decoder.Forward(_backend, latent);
            return TaesdDecoder.ToHwcRgbBytes(rgb01, out width, out height);
        }
        catch (Exception ex)
        {
            Logs.Warning($"[SharpInference][TAESD] forward pass threw: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
        finally
        {
            rgb01?.Dispose();
        }
    }
}
