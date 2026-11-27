using System;
using System.Collections.Generic;
using System.IO;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using Xunit;

namespace Hartsy.Extensions.HartsyInferenceBackend.Tests;

public sealed class LongCatVideoResolverTests
{
    private static string Dit(string checkpointFolder) => Path.Combine("models", "LongCat", checkpointFolder, "dit");

    private static Func<string, bool> Files(params string[] present)
    {
        HashSet<string> set = new(present, StringComparer.OrdinalIgnoreCase);
        return path => set.Contains(path);
    }

    [Fact]
    public void ABaseShardInTheDitFolderIsALongCatVideoFile()
    {
        Assert.True(LongCatVideoResolver.IsDitFile(Path.Combine(Dit("LongCat-Video"), "diffusion_pytorch_model-00001-of-00006.safetensors")));
    }

    [Fact]
    public void AnAvatarShardIsNotABaseLongCatVideoFile()
    {
        // An avatar folder has base_model, not dit; and the checkpoint name contains "avatar". Neither rule may take it.
        Assert.False(LongCatVideoResolver.IsDitFile(Path.Combine("models", "LongCat", "avatar-1.5", "base_model", "x.safetensors")));
        Assert.False(LongCatVideoResolver.IsDitFile(Path.Combine("models", "LongCat", "avatar-1.5", "dit", "x.safetensors")));
    }

    [Fact]
    public void AWanOrFastFramesShardIsNotABaseLongCatVideoFile()
    {
        Assert.False(LongCatVideoResolver.IsDitFile(Path.Combine("models", "Wan", "wan21-14b", "dit", "x.safetensors")));
        Assert.False(LongCatVideoResolver.IsDitFile(Path.Combine("models", "FastFrames", "fastframes-l", "dit.safetensors")));
    }

    [Fact]
    public void ADitAncestorDoesNotCountForAFileInASubfolder()
    {
        Assert.False(LongCatVideoResolver.IsDitFile(Path.Combine(Dit("LongCat-Video"), "extra", "x.safetensors")));
    }

    [Fact]
    public void NonShardsAreNotDitFiles()
    {
        Assert.False(LongCatVideoResolver.IsDitFile(Path.Combine(Dit("LongCat-Video"), "config.json")));
        Assert.False(LongCatVideoResolver.IsDitFile(Path.Combine(Dit("LongCat-Video"), "diffusion_pytorch_model.safetensors.index.json")));
        Assert.False(LongCatVideoResolver.IsDitFile(""));
        Assert.False(LongCatVideoResolver.IsDitFile(null));
    }

    [Fact]
    public void ACompleteDitFolderResolvesToTheBaseFamilyAndHasNoRefusal()
    {
        string shard = Path.Combine(Dit("LongCat-Video"), "diffusion_pytorch_model-00001-of-00006.safetensors");
        string config = Path.Combine(Dit("LongCat-Video"), "config.json");
        string index = Path.Combine(Dit("LongCat-Video"), "diffusion_pytorch_model.safetensors.index.json");
        LongCatVideoResolver.Resolution r = LongCatVideoResolver.Resolve(shard, Files(config, index));
        Assert.Equal(LongCatVideoResolver.FamilyId, r.FamilyId);
        Assert.Equal(Dit("LongCat-Video"), r.DitFolder);
        Assert.Equal(Path.Combine("models", "LongCat", "LongCat-Video"), r.CheckpointRoot);
        Assert.Empty(r.MissingFiles);
        Assert.Null(LongCatVideoResolver.RefusalFor(r));
    }

    [Fact]
    public void AMissingIndexOrConfigIsRefusedByName()
    {
        string shard = Path.Combine(Dit("LongCat-Video"), "diffusion_pytorch_model-00001-of-00006.safetensors");
        LongCatVideoResolver.Resolution r = LongCatVideoResolver.Resolve(shard, Files());
        Assert.Equal(2, r.MissingFiles.Count);
        Assert.Contains("config.json", LongCatVideoResolver.RefusalFor(r));
        Assert.Contains("diffusion_pytorch_model.safetensors.index.json", LongCatVideoResolver.RefusalFor(r));
    }

    [Fact]
    public void ResolvingANonBaseFileIsAnArgumentError()
    {
        Assert.Throws<ArgumentException>(() => LongCatVideoResolver.Resolve(Path.Combine("models", "Wan", "x.safetensors"), Files()));
    }

    [Fact]
    public void TheFamilyAndClassIdsMatchTheEngine()
    {
        Assert.Equal("longcat-video", LongCatVideoResolver.FamilyId);
        Assert.Equal(15, LongCatVideoResolver.DefaultFps);
    }
}
