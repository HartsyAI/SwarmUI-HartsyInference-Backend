using System;
using System.Collections.Generic;
using System.IO;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Base LongCat-Video (text and image to video). The DiT shards sit in a folder named <c>dit</c> inside a checkpoint folder whose name
/// contains "longcat-video" and not "avatar". Matched and resolved from the path alone, like <see cref="LongCatAvatarResolver"/>, so core's
/// Wan class cannot take the shards and the avatar rule never matches them.
/// </summary>
/// <remarks>
/// No SwarmUI or engine types: <c>tests/SwarmUI-HartsyInference.Tests</c> compiles this file alone. The engine reads the DiT folder as
/// its checkpoint directory (<c>LongCatRecipeSupport.CheckpointDirectory</c>). The engine can derive the geometry when <c>config.json</c> is absent;
/// this extension requires it and the shard index anyway, because a missing file means an incomplete download, and refuses the request before any
/// weight is read.
/// </remarks>
public static class LongCatVideoResolver
{
    /// <summary>The compat class core-side registration uses for base LongCat-Video DiT shards.</summary>
    public const string CompatClassId = "longcat-video-video";

    /// <summary>The engine family id (<c>LongCatVideoRecipe</c>, catalog row <c>longcat-video</c>).</summary>
    public const string FamilyId = "longcat-video";

    /// <summary>The engine's default frame rate for the base family (<c>LongCatVideoRecipe.Defaults</c>). Informational: the request's FPS is not overridden.</summary>
    public const int DefaultFps = 15;

    private const string DitFolder = "dit";
    private const string ConfigFile = "config.json";
    private const string IndexFile = "diffusion_pytorch_model.safetensors.index.json";

    /// <summary>True when <paramref name="path"/> is a base LongCat-Video DiT shard: a .safetensors file in a folder named <c>dit</c>, inside a
    /// folder whose name contains "longcat-video" and not "avatar". Only the file's own folder and the one above it are looked at.</summary>
    public static bool IsDitFile(string path)
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
        if (string.IsNullOrEmpty(ditFolder) || !string.Equals(Path.GetFileName(ditFolder), DitFolder, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        string checkpointRoot = Path.GetFileName(Path.GetDirectoryName(ditFolder) ?? "");
        return checkpointRoot.Contains("longcat-video", StringComparison.OrdinalIgnoreCase)
            && !checkpointRoot.Contains("avatar", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>What a base LongCat-Video DiT shard resolves to.</summary>
    /// <param name="FamilyId">Always <see cref="FamilyId"/>.</param>
    /// <param name="DitFolder">The DiT folder: the path handed to the engine as the model's local path.</param>
    /// <param name="CheckpointRoot">The folder above the DiT folder.</param>
    /// <param name="MissingFiles">Required files that are not there (empty when the folder is complete).</param>
    public sealed record Resolution(string FamilyId, string DitFolder, string CheckpointRoot, IReadOnlyList<string> MissingFiles);

    /// <summary>Resolves a base LongCat-Video DiT shard to its DiT folder and the files the engine needs there.</summary>
    /// <param name="modelFile">A DiT shard path, as <see cref="IsDitFile"/> accepts it.</param>
    /// <param name="fileExists">Tests whether a file exists. Injected so the tests need no disk layout.</param>
    /// <exception cref="ArgumentException">The path is not a base LongCat-Video DiT shard.</exception>
    public static Resolution Resolve(string modelFile, Func<string, bool> fileExists)
    {
        if (!IsDitFile(modelFile))
        {
            throw new ArgumentException($"Not a base LongCat-Video DiT shard: '{modelFile}'.", nameof(modelFile));
        }
        ArgumentNullException.ThrowIfNull(fileExists);
        string ditFolder = Path.GetDirectoryName(modelFile);
        string root = Path.GetDirectoryName(ditFolder);
        List<string> missing = [];
        string config = Path.Combine(ditFolder, ConfigFile);
        string index = Path.Combine(ditFolder, IndexFile);
        if (!fileExists(config)) missing.Add(config);
        if (!fileExists(index)) missing.Add(index);
        return new Resolution(FamilyId, ditFolder, root, missing);
    }

    /// <summary>The refusal to show for an incomplete DiT folder, or null when the files are there.</summary>
    public static string RefusalFor(Resolution resolution)
    {
        ArgumentNullException.ThrowIfNull(resolution);
        if (resolution.MissingFiles.Count == 0)
        {
            return null;
        }
        return $"LongCat-Video's DiT folder is incomplete; these files are missing: {string.Join(", ", resolution.MissingFiles)}. "
            + "The download did not finish: fetch them again into that folder.";
    }
}
