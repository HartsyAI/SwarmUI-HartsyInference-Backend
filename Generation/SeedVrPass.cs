using System.Buffers.Binary;
using System.IO;
using System.Text;
using System.Text.Json;
using HartsyInference.Video.Pipelines;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Reads SwarmUI's core SeedVR group (its ComfyUI-backend params) and turns it into the settings this backend runs
/// on the HartsyInference SeedVR2 restore, so a request set up in core restores the same way here.</summary>
/// <remarks>The group is core's, not ours: its params are registered with the ComfyUI feature flag, so this backend
/// reads them by ID and refuses none of them. Core's SeedVR model picks from files installed in its model folders, and a
/// bare SeedVR2 file cannot load on the engine (the VAE and frozen embeddings come with the engine's catalog entry), so the
/// file selects a catalog variant (3B or 7B) and the engine fetches its own weights on first use.</remarks>
public static class SeedVrPass
{
    /// <summary>Core's <c>SeedVR Model</c> param ID, as the param system cleans it.</summary>
    public const string ModelParamId = "seedvrmodel";

    /// <summary>The engine catalog entry for each SeedVR2 variant.</summary>
    public const string Variant3B = "seedvr2-3b", Variant7B = "seedvr2-7b";

    /// <summary>Width of the DiT's text-input projection on the 7B checkpoint; the 3B's is 2560 (see the engine's SeedVr2Config).</summary>
    private const long Vid7BDim = 3072;

    /// <summary>Whether core's SeedVR group is switched on for this request (a SeedVR model is picked).</summary>
    public static bool IsActive(T2IParamInput input) =>
        input.TryGet(ComfyUIBackendExtension.SeedVRModel, out T2IModel model) && model is not null;

    /// <summary>Reads core's SeedVR settings for a request. Call only when <see cref="IsActive"/> is true.</summary>
    /// <param name="input">The request.</param>
    /// <param name="variantPick">Maps the picked model file to its engine catalog id.</param>
    public static SeedVrSettings Read(T2IParamInput input, Func<T2IModel, string> variantPick)
    {
        input.TryGet(ComfyUIBackendExtension.SeedVRModel, out T2IModel model);
        double downscale = input.Get(ComfyUIBackendExtension.SeedVRPreDownscale, 1);
        if (downscale <= 0)
        {
            throw new SwarmReadableErrorException($"Invalid SeedVR pre-downscale value {downscale}. Must be greater than 0.");
        }
        double upscale = input.Get(ComfyUIBackendExtension.SeedVRUpscale, 1);
        string method = input.Get(ComfyUIBackendExtension.SeedVRUpscaleMethod, "pixel-lanczos");
        string color = input.Get(ComfyUIBackendExtension.SeedVRColorCorrectionBehavior, "lab");
        // Core seeds SeedVR's own section (base seed + 9 unless set), as its workflow does.
        long seed = input.Get(T2IParamTypes.Seed, input.Get(T2IParamTypes.Seed) + 9, sectionId: T2IParamInput.SectionID_SeedVR, includeBase: false);
        return new SeedVrSettings(
            RestoreModelId: variantPick(model),
            RestoreModelPath: model.RawFilePath,
            Downscale: downscale,
            Upscale: upscale,
            UpscaleMethod: method,
            ColorTransfer: ColorTransferFor(color),
            Strength: color == "wavelet" ? 0f : 1f,
            SplitLatent: input.Get(ComfyUIBackendExtension.SeedVRSplitLatent, false),
            TemporalOverlap: input.Get(ComfyUIBackendExtension.SeedVRTemporalVideoOverlap, 0),
            Seed: (int)(seed & 0x7FFFFFFF));
    }

    /// <summary>Core's colour modes, as the engine runs them. Wavelet is the engine's <c>Strength</c> lerp at 0, so it has no transfer of its own.</summary>
    public static SeedVr2ColorTransfer ColorTransferFor(string behavior) => behavior switch
    {
        "lab" => SeedVr2ColorTransfer.Lab,
        "adain" => SeedVr2ColorTransfer.AdaIn,
        _ => SeedVr2ColorTransfer.None,
    };

    /// <summary>Picks the engine variant for a SeedVR2 model file, reading the DiT's text-input width from its safetensors header.</summary>
    /// <remarks>Falls back to 3B when the file cannot be read, and says so in the log.</remarks>
    public static string VariantFor(T2IModel model)
    {
        if (model is null)
        {
            return Variant3B;
        }
        try
        {
            long[] shape = ReadTensorShape(model.RawFilePath, "txt_in.weight");
            return shape.Length > 0 && shape[0] == Vid7BDim ? Variant7B : Variant3B;
        }
        catch (Exception ex) when (ex is IOException or JsonException or InvalidDataException or UnauthorizedAccessException)
        {
            HartsyLogs.Warning($"SeedVR model '{model.Name}' header could not be read ({ex.Message}); using {Variant3B}.");
            return Variant3B;
        }
    }

    /// <summary>Reads one tensor's shape from a safetensors file's JSON header, returning an empty array when the tensor is absent.</summary>
    private static long[] ReadTensorShape(string path, string tensor)
    {
        using FileStream stream = File.OpenRead(path);
        Span<byte> lengthBytes = stackalloc byte[8];
        stream.ReadExactly(lengthBytes);
        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(lengthBytes);
        if (headerLength == 0 || headerLength > 100 * 1024 * 1024)
        {
            throw new InvalidDataException($"implausible safetensors header length {headerLength}");
        }
        byte[] header = new byte[headerLength];
        stream.ReadExactly(header);
        using JsonDocument doc = JsonDocument.Parse(Encoding.UTF8.GetString(header));
        if (!doc.RootElement.TryGetProperty(tensor, out JsonElement entry) || !entry.TryGetProperty("shape", out JsonElement shape))
        {
            return [];
        }
        return [.. shape.EnumerateArray().Select(d => d.GetInt64())];
    }
}

/// <summary>The settings of one core-SeedVR pass, as the engine runs them.</summary>
/// <param name="RestoreModelId">The engine catalog id. It supplies the VAE and embedding every variant shares, and runs its own DiT when no file is picked.</param>
/// <param name="RestoreModelPath">The checkpoint file core picked: the DiT the engine runs. Empty runs the catalog variant's own file.</param>
/// <param name="Downscale">Core's pre-downscale: the image is resized by this factor before the upscale.</param>
/// <param name="Upscale">Core's upscale factor, applied after the pre-downscale is undone.</param>
/// <param name="UpscaleMethod">Core's upscale method. <c>pixel-*</c> leaves the resize to the engine's own area resize, which is the model's contract.</param>
/// <param name="ColorTransfer">The colour mode the engine applies to each restored frame.</param>
/// <param name="Strength">The engine's wavelet strength: 0 for core's wavelet mode, 1 otherwise.</param>
/// <param name="SplitLatent">Core's split-latent switch. Off restores the whole clip in one chunk.</param>
/// <param name="TemporalOverlap">Core's overlap, in latent frames. Four pixel frames make one latent frame.</param>
/// <param name="Seed">The SeedVR section's seed.</param>
public sealed record SeedVrSettings(string RestoreModelId, string RestoreModelPath, double Downscale, double Upscale, string UpscaleMethod,
    SeedVr2ColorTransfer ColorTransfer, float Strength, bool SplitLatent,
    int TemporalOverlap, int Seed);
