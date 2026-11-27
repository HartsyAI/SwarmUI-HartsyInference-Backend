using System;
using System.IO;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using Xunit;

namespace SwarmUI_HartsyInference.Tests;

/// <summary>The FastFrames companion files resolve to the generator beside them.</summary>
public sealed class FastFramesGeneratorTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "ff-gen-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(root))
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private string Folder(string name, params string[] files)
    {
        string dir = Path.Combine(root, name);
        Directory.CreateDirectory(dir);
        foreach (string file in files)
        {
            File.WriteAllText(Path.Combine(dir, file), "stub");
        }
        return dir;
    }

    [Fact]
    public void ACompanionInAnSFolderResolvesToTheSGenerator()
    {
        string dir = Folder("fastframes-s", "framewise-2step.pt", "kv_refiner.safetensors", "pixel_decoder.safetensors");

        Assert.Equal(Path.Combine(dir, "framewise-2step.pt"), FastFramesGenerator.Resolve(Path.Combine(dir, "kv_refiner.safetensors")));
        Assert.Equal(Path.Combine(dir, "framewise-2step.pt"), FastFramesGenerator.Resolve(Path.Combine(dir, "pixel_decoder.safetensors")));
    }

    [Fact]
    public void ACompanionInAnLFolderResolvesToTheLGenerator()
    {
        string dir = Folder("fastframes-l", "dit.safetensors", "kv_refiner.safetensors", "pixel_decoder.safetensors");

        Assert.Equal(Path.Combine(dir, "dit.safetensors"), FastFramesGenerator.Resolve(Path.Combine(dir, "kv_refiner.safetensors")));
    }

    [Fact]
    public void ASelectedGeneratorIsReturnedAsItIs()
    {
        string dir = Folder("fastframes-s", "framewise-2step.pt", "kv_refiner.safetensors");

        Assert.Equal(Path.Combine(dir, "framewise-2step.pt"), FastFramesGenerator.Resolve(Path.Combine(dir, "framewise-2step.pt")));
    }

    [Fact]
    public void ASFolderNeverFallsBackToAnLGenerator()
    {
        // The S generator is preferred when both are present, so the choice does not depend on the folder listing.
        string dir = Folder("fastframes-x", "framewise-2step.pt", "dit.safetensors", "kv_refiner.safetensors");

        Assert.Equal(Path.Combine(dir, "framewise-2step.pt"), FastFramesGenerator.Resolve(Path.Combine(dir, "kv_refiner.safetensors")));
    }

    [Fact]
    public void AFolderWithNoGeneratorRefusesByName()
    {
        string dir = Folder("fastframes-s", "kv_refiner.safetensors", "pixel_decoder.safetensors");

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => FastFramesGenerator.Resolve(Path.Combine(dir, "kv_refiner.safetensors")));
        Assert.Contains("framewise-2step.pt", ex.Message);
        Assert.Contains("dit.safetensors", ex.Message);
    }

    [Fact]
    public void AnLGeneratorInAnLFolderIsAFastFramesFile()
    {
        // Core classes this file as Wan 2.1 14B; the routing in ModelSupport.CompatIdOf sends it to FastFrames because of its folder.
        string dir = Folder("fastframes-l", "dit.safetensors");

        Assert.True(FastFramesGenerator.IsFastFramesFile(Path.Combine(dir, "dit.safetensors")));
    }

    [Fact]
    public void AWanFileOutsideAFastFramesFolderIsNotAFastFramesFile()
    {
        string dir = Folder("wan21-14b", "dit.safetensors");

        Assert.False(FastFramesGenerator.IsFastFramesFile(Path.Combine(dir, "dit.safetensors")));
    }

    [Fact]
    public void AFastFramesAncestorDoesNotCountForAFileInASubfolder()
    {
        string dir = Folder("fastframes-l/other", "dit.safetensors");

        Assert.False(FastFramesGenerator.IsFastFramesFile(Path.Combine(dir, "dit.safetensors")));
    }

    [Fact]
    public void AnEmptySelectionRefuses()
    {
        Assert.Throws<InvalidOperationException>(() => FastFramesGenerator.Resolve(""));
    }
}
