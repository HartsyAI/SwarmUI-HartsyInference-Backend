using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// MovieRoll (Tencent, distilled text to video). Its DiT has the same tensor names and shapes in the base, distilled T2V and distilled I2V
/// releases, and core classes the file as Hunyuan Image, so it is matched and told apart by folder: a .safetensors file under a folder whose
/// name contains "movieroll", laid out as the engine's catalog (<c>MovieRoll/distilled-t2v/model.safetensors</c>) or as the upstream
/// repository (<c>MovieRoll/distilled/t2v/model.safetensors</c>).
/// </summary>
/// <remarks>No SwarmUI or engine types: <c>tests/SwarmUI-HartsyInference.Tests</c> compiles this file alone.</remarks>
public static class MovieRollResolver
{
    /// <summary>The compat class and model class ID, matching the engine's published identity for the family.</summary>
    public const string CompatClassId = "movieroll";

    /// <summary>The engine family id (<c>MovieRollRecipe</c>, catalog row <c>movieroll</c>).</summary>
    public const string FamilyId = "movieroll";

    private const string FolderToken = "movieroll";
    private const string DefaultFileName = "model.safetensors";
    private const int FolderDepth = 3;
    private static readonly string[] VariantTokens = ["base", "distilled", "t2v", "i2v"];

    /// <summary>Which MovieRoll release a file is, read from its folders.</summary>
    public enum Variant
    {
        /// <summary>Distilled text to video: the one the engine runs.</summary>
        DistilledT2V,
        /// <summary>Distilled image to video.</summary>
        DistilledI2V,
        /// <summary>The base (non-distilled) model.</summary>
        Base,
        /// <summary>The folders do not say which release this is.</summary>
        Unknown,
    }

    /// <summary>What the validator knows about a request that picked a MovieRoll file.</summary>
    /// <param name="ModelFile">The selected model's path.</param>
    /// <param name="AssignedCompatClass">The compat class SwarmUI gave the file, before any path routing.</param>
    /// <param name="NegativePrompt">The request's negative prompt.</param>
    /// <param name="VaeSelected">True when the request picks a VAE.</param>
    /// <param name="QwenSelected">True when the request picks a Qwen text encoder.</param>
    public sealed record Request(string ModelFile, string AssignedCompatClass, string NegativePrompt, bool VaeSelected, bool QwenSelected);

    /// <summary>True when <paramref name="path"/> is a MovieRoll DiT by its layout: a .safetensors file at most three folders below a folder
    /// whose name contains "movieroll", whose folders name a release or whose file name is <c>model.safetensors</c>.</summary>
    public static bool IsMovieRollFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !string.Equals(Path.GetExtension(path), ".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        HashSet<string> tokens = FolderTokens(path);
        return tokens is not null
            && (tokens.Overlaps(VariantTokens) || string.Equals(Path.GetFileName(path), DefaultFileName, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>The release <paramref name="path"/> holds, by the folder names from the "movieroll" folder down.</summary>
    public static Variant VariantOf(string path)
    {
        HashSet<string> tokens = string.IsNullOrWhiteSpace(path) ? null : FolderTokens(path);
        if (tokens is null)
        {
            return Variant.Unknown;
        }
        if (tokens.Contains("i2v"))
        {
            return Variant.DistilledI2V;
        }
        if (tokens.Contains("base"))
        {
            return Variant.Base;
        }
        return tokens.Contains("distilled") && tokens.Contains("t2v") ? Variant.DistilledT2V : Variant.Unknown;
    }

    /// <summary>Why the file itself cannot run, or null for a distilled text-to-video file.</summary>
    public static string CheckpointRefusal(string path) => VariantOf(path) switch
    {
        Variant.DistilledT2V => null,
        Variant.DistilledI2V => "MovieRoll distilled image-to-video isn't supported: the engine runs the distilled text-to-video model only. "
            + "Select the file in MovieRoll/distilled-t2v.",
        Variant.Base => "MovieRoll Base isn't supported: the engine runs the distilled text-to-video model only. "
            + "Select the file in MovieRoll/distilled-t2v.",
        _ => "The MovieRoll releases look identical inside, so the folder has to say which one this is. Put the distilled text-to-video file "
            + $"at MovieRoll/distilled-t2v/model.safetensors and refresh the model list (selected: '{path}').",
    };

    /// <summary>Every reason to refuse the request, empty when it can run. Never throws.</summary>
    public static IReadOnlyList<string> Refusals(Request request)
    {
        ArgumentNullException.ThrowIfNull(request);
        List<string> reasons = [];
        string checkpoint = CheckpointRefusal(request.ModelFile);
        if (checkpoint is not null)
        {
            reasons.Add(checkpoint);
        }
        else if (request.AssignedCompatClass != CompatClassId)
        {
            // Left as core classed it, the gen page treats the file as an image model and sends its image resolution.
            string sidecar = Path.ChangeExtension(request.ModelFile, ".swarm.json");
            // The resolution is named too: a file SwarmUI has already written its own header into carries the image resolution there.
            reasons.Add($"SwarmUI classes this MovieRoll file as '{request.AssignedCompatClass ?? "unknown"}', not as a video model. "
                + $"Create '{sidecar}' containing {{\"modelspec.architecture\": \"{CompatClassId}\", \"modelspec.resolution\": \"864x480\"}}, "
                + "then restart SwarmUI.");
        }
        if (!string.IsNullOrWhiteSpace(request.NegativePrompt))
        {
            reasons.Add("MovieRoll runs without classifier-free guidance, so it has no negative prompt. Clear the Negative Prompt.");
        }
        if (request.VaeSelected)
        {
            reasons.Add("MovieRoll always uses the HunyuanVideo-1.5 VAE. Unset the VAE.");
        }
        if (request.QwenSelected)
        {
            reasons.Add("MovieRoll always uses its own Qwen2.5-VL text encoder. Unset Qwen Model.");
        }
        return reasons;
    }

    /// <summary>The log line for steps or CFG that differ from the model's fixed schedule, or null when both match or are unset.</summary>
    /// <param name="requestedSteps">The request's step count.</param>
    /// <param name="requestedCfg">The request's CFG scale.</param>
    /// <param name="steps">The engine descriptor's step count.</param>
    /// <param name="cfg">The engine descriptor's CFG scale.</param>
    public static string FixedScheduleNote(int? requestedSteps, double? requestedCfg, int steps, double cfg)
    {
        List<string> ignored = [];
        if (requestedSteps is int wantedSteps && wantedSteps != steps)
        {
            ignored.Add($"Steps {wantedSteps}");
        }
        if (requestedCfg is double wantedCfg && Math.Abs(wantedCfg - cfg) > 1e-6)
        {
            ignored.Add(FormattableString.Invariant($"CFG Scale {wantedCfg}"));
        }
        return ignored.Count == 0
            ? null
            : FormattableString.Invariant($"MovieRoll runs its fixed {steps}-step schedule at CFG {cfg}; ignoring {string.Join(" and ", ignored)}.");
    }

    /// <summary>The name tokens of the folders from the nearest "movieroll" folder down to the file's folder, or null when none of the file's
    /// three closest folders is a "movieroll" folder.</summary>
    private static HashSet<string> FolderTokens(string path)
    {
        // Split on both separators, so a Windows path reads the same on Linux.
        string[] parts = path.Split(['/', '\\'], StringSplitOptions.RemoveEmptyEntries);
        int fileIndex = parts.Length - 1;
        for (int index = fileIndex - 1; index >= 0 && index >= fileIndex - FolderDepth; index--)
        {
            if (parts[index].Contains(FolderToken, StringComparison.OrdinalIgnoreCase))
            {
                return [.. parts[index..fileIndex].SelectMany(part => part.ToLowerInvariant().Split(['-', '_', ' ', '.']))];
            }
        }
        return null;
    }
}
