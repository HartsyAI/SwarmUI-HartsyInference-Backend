using System;
using System.IO;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Picks the FastFrames generator file for a selected FastFrames model file.
/// <para>SwarmUI lists the companion files of a FastFrames folder (<c>kv_refiner.safetensors</c>, <c>pixel_decoder.safetensors</c>) as
/// FastFrames models, but the generator is the file the engine reads as the transformer: <c>framewise-2step.pt</c> for S and
/// <c>dit.safetensors</c> for L. The engine finds the companions beside the generator, so the generator is the path that gets passed on.</para>
/// <para>No SwarmUI types here, so the test project compiles this file alone.</para></summary>
public static class FastFramesGenerator
{
    /// <summary>The S generator, a Causal Forcing++ PyTorch pickle.</summary>
    public const string SGenerator = "framewise-2step.pt";

    /// <summary>The L generator, a safetensors file.</summary>
    public const string LGenerator = "dit.safetensors";

    /// <summary>True when the file sits in a folder whose own name contains "fastframes" (FastFrames/fastframes-s, fastframes-l). Only the
    /// file's own folder counts, not an ancestor.</summary>
    public static bool IsFastFramesFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return false;
        }
        string folder = Path.GetDirectoryName(path);
        if (string.IsNullOrEmpty(folder))
        {
            return false;
        }
        return Path.GetFileName(folder).Contains("fastframes", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Returns the generator path for the selected FastFrames file. A selected generator is returned as it is; any other
    /// file (a companion) resolves to the S generator, then the L generator, in the same folder.</summary>
    /// <exception cref="InvalidOperationException">Neither generator is in the folder.</exception>
    public static string Resolve(string selectedPath)
    {
        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            throw new InvalidOperationException("FastFrames: no model file was selected.");
        }
        string name = Path.GetFileName(selectedPath);
        if (name.Equals(SGenerator, StringComparison.OrdinalIgnoreCase) || name.Equals(LGenerator, StringComparison.OrdinalIgnoreCase))
        {
            return selectedPath;
        }
        string folder = Path.GetDirectoryName(selectedPath) ?? "";
        foreach (string candidate in new[] { SGenerator, LGenerator })
        {
            string path = Path.Combine(folder, candidate);
            if (File.Exists(path))
            {
                return path;
            }
        }
        throw new InvalidOperationException(
            $"FastFrames: no generator in '{folder}'. Put {SGenerator} (S) or {LGenerator} (L) beside the kv_refiner and pixel_decoder files.");
    }
}
