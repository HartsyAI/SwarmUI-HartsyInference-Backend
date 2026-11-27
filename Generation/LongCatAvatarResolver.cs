using System;
using System.Collections.Generic;
using System.IO;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// LongCat-Video-Avatar 1.5 and 1.6 (speech and image to video). The model is matched and resolved from its path alone: the DiT
/// shards sit in a folder named <c>base_model</c> (1.5) or <c>base_model_int8</c> (1.6 INT8), inside a checkpoint folder whose name
/// contains "avatar". Core SwarmUI may class these shards as Wan, and the release is a folder rather than one file, so the match is
/// by folder and the engine is handed the DiT folder.
/// </summary>
/// <remarks>
/// This file has no SwarmUI or engine types on purpose: <c>tests/SwarmUI-HartsyInference.Tests</c> compiles it alone, so the
/// resolver is tested without a SwarmUI build. The engine's own rules are in <c>LongCatRecipeSupport</c> and
/// <c>LongCatAvatarRecipe</c>: the 1.6 layout is <c>loraA/lora.safetensors</c> and <c>loraB/lora.safetensors</c> beside the
/// INT8 folder, and the 1.5 distillation LoRA is <c>lora/dmd_lora.safetensors</c> beside the base folder.
/// </remarks>
public static class LongCatAvatarResolver
{
    /// <summary>The compat class core-side registration uses for LongCat avatar DiT shards.</summary>
    public const string CompatClassId = "longcat-avatar-video";

    /// <summary>The engine family id for 1.5 (distilled, base folder and DMD LoRA).</summary>
    public const string Family15 = "longcat-avatar";

    /// <summary>The engine family id for 1.6 (INT8 folder, loraA and loraB mixed per step).</summary>
    public const string Family16 = "longcat-avatar-1.6";

    /// <summary>The engine's output frame rate (<c>LongCatAvatarRecipe.OutputFps</c>). The request's FPS is overridden to this.</summary>
    public const int OutputFps = 25;

    /// <summary>The engine's segment length (<c>LongCatAvatarRecipe.SegmentFrames</c>); frames are 1 + 4k up to this.</summary>
    public const int SegmentFrames = 93;

    private const string Folder15 = "base_model";
    private const string Folder16 = "base_model_int8";

    /// <summary>True when <paramref name="path"/> is a LongCat avatar DiT shard: a .safetensors file whose own folder is
    /// <c>base_model</c> or <c>base_model_int8</c>, inside a folder whose name contains "avatar". Only the file's own folder and the
    /// one above it are looked at; a deeper folder named "avatar" does not count.</summary>
    public static bool IsAvatarDitFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return false;
        }
        if (!string.Equals(Path.GetExtension(path), ".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        string ditFolder = Path.GetDirectoryName(path);
        if (string.IsNullOrEmpty(ditFolder))
        {
            return false;
        }
        string folderName = Path.GetFileName(ditFolder);
        if (!string.Equals(folderName, Folder15, StringComparison.OrdinalIgnoreCase)
            && !string.Equals(folderName, Folder16, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        string checkpointRoot = Path.GetDirectoryName(ditFolder);
        return !string.IsNullOrEmpty(checkpointRoot)
            && Path.GetFileName(checkpointRoot).Contains("avatar", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>What a LongCat avatar model file resolves to.</summary>
    /// <param name="FamilyId">The engine family: <see cref="Family15"/> or <see cref="Family16"/>.</param>
    /// <param name="DitFolder">The folder holding the DiT shards. This is the path handed to the engine as the model's local path.</param>
    /// <param name="CheckpointRoot">The folder beside the DiT folder, where the LoRA folders live.</param>
    /// <param name="MissingFiles">Required files that are not there (empty when the checkpoint is complete).</param>
    public sealed record Resolution(string FamilyId, string DitFolder, string CheckpointRoot, IReadOnlyList<string> MissingFiles);

    /// <summary>Resolves a LongCat avatar DiT shard to its family, DiT folder and required files.</summary>
    /// <param name="modelFile">A DiT shard path, as <see cref="IsAvatarDitFile"/> accepts it.</param>
    /// <param name="fileExists">Tests whether a file exists. Injected so the tests need no disk layout.</param>
    /// <exception cref="ArgumentException">The path is not a LongCat avatar DiT shard.</exception>
    /// <exception cref="InvalidOperationException">The folder has the 1.6 layout (loraA) but is not the INT8 folder. The engine's
    /// 1.5 build refuses that layout, so the refusal is raised here, with the name of the folder to load instead.</exception>
    public static Resolution Resolve(string modelFile, Func<string, bool> fileExists)
    {
        if (!IsAvatarDitFile(modelFile))
        {
            throw new ArgumentException($"Not a LongCat-Video-Avatar DiT shard: '{modelFile}'.", nameof(modelFile));
        }
        ArgumentNullException.ThrowIfNull(fileExists);
        string ditFolder = Path.GetDirectoryName(modelFile);
        string root = Path.GetDirectoryName(ditFolder);
        bool is16 = string.Equals(Path.GetFileName(ditFolder), Folder16, StringComparison.OrdinalIgnoreCase);
        string loraA = Path.Combine(root, "loraA", "lora.safetensors");
        string loraB = Path.Combine(root, "loraB", "lora.safetensors");
        string dmd = Path.Combine(root, "lora", "dmd_lora.safetensors");

        if (!is16 && fileExists(loraA))
        {
            throw new InvalidOperationException(
                $"LongCat-Video-Avatar: '{root}' has the 1.6 layout (loraA and loraB). Load its {Folder16} folder as LongCat-Video-Avatar 1.6, not the {Folder15} folder.");
        }

        List<string> missing = [];
        if (is16)
        {
            if (!fileExists(loraA)) missing.Add(loraA);
            if (!fileExists(loraB)) missing.Add(loraB);
        }
        else if (!fileExists(dmd))
        {
            // The engine runs the base at 8 steps only with the distillation LoRA; without it the request is refused.
            missing.Add(dmd);
        }
        return new Resolution(is16 ? Family16 : Family15, ditFolder, root, missing);
    }

    /// <summary>The refusal to show for an incomplete checkpoint, or null when every required file is there.</summary>
    public static string RefusalFor(Resolution resolution)
    {
        ArgumentNullException.ThrowIfNull(resolution);
        if (resolution.MissingFiles.Count == 0)
        {
            return null;
        }
        string name = resolution.FamilyId == Family16 ? "1.6" : "1.5";
        return $"LongCat-Video-Avatar {name} needs these files, which are missing: {string.Join(", ", resolution.MissingFiles)}.";
    }
}
