using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Text.Json;
using SwarmUI.Text2Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Variant discrimination for the Wan-Video family. Every Wan conditioning variant (plain T2V/I2V, VACE
/// control, Animate character-animation, S2V speech-to-video) is built on the same backbone and SwarmUI maps
/// them onto the same CompatClass (<c>wan-21-14b</c> / <c>wan-21-1_3b</c> / <c>wan-22-5b</c>), so the backend
/// can't dispatch off the compat class alone. <see cref="Detect"/> recovers the variant so the dispatcher can
/// route to the right loader (each variant drives a different engine pipeline):
/// <list type="bullet">
/// <item><b>VACE</b> — from the SwarmUI model-class ID (<c>wan-2_1-vace-*</c>; SwarmUI core already classifies it).</item>
/// <item><b>Animate</b> — from a <c>pose_patch_embedding</c> / <c>motion_encoder</c> weight in the checkpoint header.</item>
/// <item><b>S2V</b> — from an <c>audio_encoder</c> / <c>audio_injector</c> weight in the checkpoint header.</item>
/// </list>
/// These signature tensors are unique to their variant (absent from base Wan and from each other), so detection
/// can't false-positive on a plain Wan checkpoint. The header (a small JSON key map) is read directly off the
/// safetensors file and cached per (path, mtime) — far cheaper than loading + converting the whole checkpoint.
/// </summary>
public static class WanModelVariants
{
    public enum Variant { Base, Vace, Animate, S2V }

    /// <summary>SwarmUI model-class ID for the Wan2.1 VACE-14B checkpoint.</summary>
    public const string Vace14BModelClassId = "wan-2_1-vace-14b";

    /// <summary>SwarmUI model-class ID for the Wan2.1 VACE-1.3B checkpoint.</summary>
    public const string Vace1_3BModelClassId = "wan-2_1-vace-1_3b";

    /// <summary>True when the given model-class ID is one of the Wan VACE variants.</summary>
    public static bool IsVace(string modelClassId) =>
        modelClassId is Vace14BModelClassId or Vace1_3BModelClassId;

    /// <summary>Classifies a Wan checkpoint into its conditioning variant. VACE is taken from the model-class ID
    /// (SwarmUI core detects it); Animate/S2V are sniffed from the safetensors header keys.</summary>
    public static Variant Detect(T2IModel model)
    {
        if (IsVace(model?.ModelClass?.ID)) return Variant.Vace;
        string path = model?.RawFilePath;
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return Variant.Base;
        IReadOnlySet<string> keys = PeekKeys(path);
        if (keys.Any(k => k.Contains("pose_patch_embedding", StringComparison.Ordinal)
            || k.Contains("motion_encoder", StringComparison.Ordinal)
            || k.Contains("face_adapter", StringComparison.Ordinal)))
            return Variant.Animate;
        if (keys.Any(k => k.Contains("audio_encoder", StringComparison.Ordinal)
            || k.Contains("audio_injector", StringComparison.Ordinal)))
            return Variant.S2V;
        return Variant.Base;
    }

    private static readonly ConcurrentDictionary<string, (DateTime Stamp, IReadOnlySet<string> Keys)> _peekCache = new();

    /// <summary>Reads the tensor-name set from a safetensors header (8-byte little-endian length + JSON map),
    /// without loading any tensor data. Cached per (path, last-write-time); returns empty on any read error.</summary>
    public static IReadOnlySet<string> PeekKeys(string path)
    {
        try
        {
            DateTime mtime = File.GetLastWriteTimeUtc(path);
            if (_peekCache.TryGetValue(path, out var cached) && cached.Stamp == mtime) return cached.Keys;

            HashSet<string> keys = new();
            using (FileStream fs = File.OpenRead(path))
            {
                Span<byte> lenBuf = stackalloc byte[8];
                fs.ReadExactly(lenBuf);
                long headerLen = BinaryPrimitives.ReadInt64LittleEndian(lenBuf);
                if (headerLen is <= 0 or > 64 * 1024 * 1024) return keys;   // sanity bound
                byte[] json = new byte[headerLen];
                fs.ReadExactly(json, 0, (int)headerLen);
                using JsonDocument doc = JsonDocument.Parse(json);
                foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
                {
                    if (prop.Name != "__metadata__") keys.Add(prop.Name);
                }
            }
            _peekCache[path] = (mtime, keys);
            return keys;
        }
        catch (Exception ex)
        {
            SwarmUI.Utils.Logs.Verbose($"[HartsyInference][Wan] Header peek failed for '{path}': {ex.Message}");
            return new HashSet<string>();
        }
    }
}
