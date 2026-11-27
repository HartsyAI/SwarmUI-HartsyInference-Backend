using System;
using System.Collections.Generic;
using System.IO;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using Xunit;

namespace Hartsy.Extensions.HartsyInferenceBackend.Tests;

public sealed class LongCatAvatarResolverTests
{
    private static string Root(string checkpointFolder) => Path.Combine("models", "LongCat", checkpointFolder);

    private static Func<string, bool> Files(params string[] present)
    {
        HashSet<string> set = new(present, StringComparer.OrdinalIgnoreCase);
        return path => set.Contains(path);
    }

    [Fact]
    public void A15ShardInItsBaseFolderIsAnAvatarDitFile()
    {
        string shard = Path.Combine(Root("avatar-1.5"), "base_model", "diffusion_pytorch_model-00001-of-00006.safetensors");
        Assert.True(LongCatAvatarResolver.IsAvatarDitFile(shard));
    }

    [Fact]
    public void A16ShardInItsInt8FolderIsAnAvatarDitFile()
    {
        string shard = Path.Combine(Root("avatar-1.6"), "base_model_int8", "quantized_model-00001-of-00004.safetensors");
        Assert.True(LongCatAvatarResolver.IsAvatarDitFile(shard));
    }

    [Fact]
    public void AWanOrFastFramesFileOutsideAnAvatarFolderIsNot()
    {
        // Core may class these as Wan; they are not LongCat, so the route must not claim them.
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(Path.Combine("models", "Wan", "wan21-14b", "base_model", "x.safetensors")));
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(Path.Combine("models", "FastFrames", "fastframes-l", "dit.safetensors")));
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(Path.Combine("models", "LongCat", "video", "base_model", "x.safetensors")));
    }

    [Fact]
    public void AnAvatarAncestorDoesNotCountForAFileInASubfolder()
    {
        string nested = Path.Combine(Root("avatar-1.5"), "extra", "base_model", "x.safetensors");
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(nested));
    }

    [Fact]
    public void NonShardFilesAreNotAvatarDitFiles()
    {
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(Path.Combine(Root("avatar-1.5"), "base_model", "config.json")));
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(Path.Combine(Root("avatar-1.5"), "base_model", "diffusion_pytorch_model.safetensors.index.json")));
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(""));
        Assert.False(LongCatAvatarResolver.IsAvatarDitFile(null));
    }

    [Fact]
    public void A15WithItsDistillationLoraResolvesCompletely()
    {
        string shard = Path.Combine(Root("avatar-1.5"), "base_model", "diffusion_pytorch_model-00001-of-00006.safetensors");
        string dmd = Path.Combine(Root("avatar-1.5"), "lora", "dmd_lora.safetensors");
        LongCatAvatarResolver.Resolution r = LongCatAvatarResolver.Resolve(shard, Files(dmd));
        Assert.Equal(LongCatAvatarResolver.Family15, r.FamilyId);
        Assert.Equal(Path.Combine(Root("avatar-1.5"), "base_model"), r.DitFolder);
        Assert.Equal(Root("avatar-1.5"), r.CheckpointRoot);
        Assert.Empty(r.MissingFiles);
        Assert.Null(LongCatAvatarResolver.RefusalFor(r));
    }

    [Fact]
    public void A15WithoutItsDmdLoraIsRefusedByName()
    {
        string shard = Path.Combine(Root("avatar-1.5"), "base_model", "diffusion_pytorch_model-00001-of-00006.safetensors");
        LongCatAvatarResolver.Resolution r = LongCatAvatarResolver.Resolve(shard, Files());
        string dmd = Path.Combine(Root("avatar-1.5"), "lora", "dmd_lora.safetensors");
        Assert.Equal(new[] { dmd }, r.MissingFiles);
        Assert.Contains("dmd_lora.safetensors", LongCatAvatarResolver.RefusalFor(r));
    }

    [Fact]
    public void A16WithBothLorasResolvesToTheInt8FolderAndTheSixteenFamily()
    {
        string shard = Path.Combine(Root("avatar-1.6"), "base_model_int8", "quantized_model-00001-of-00004.safetensors");
        string a = Path.Combine(Root("avatar-1.6"), "loraA", "lora.safetensors");
        string b = Path.Combine(Root("avatar-1.6"), "loraB", "lora.safetensors");
        LongCatAvatarResolver.Resolution r = LongCatAvatarResolver.Resolve(shard, Files(a, b));
        Assert.Equal(LongCatAvatarResolver.Family16, r.FamilyId);
        Assert.Equal(Path.Combine(Root("avatar-1.6"), "base_model_int8"), r.DitFolder);
        Assert.Empty(r.MissingFiles);
    }

    [Fact]
    public void A16MissingLoraBIsRefusedByName()
    {
        string shard = Path.Combine(Root("avatar-1.6"), "base_model_int8", "quantized_model-00001-of-00004.safetensors");
        string a = Path.Combine(Root("avatar-1.6"), "loraA", "lora.safetensors");
        LongCatAvatarResolver.Resolution r = LongCatAvatarResolver.Resolve(shard, Files(a));
        Assert.Single(r.MissingFiles);
        Assert.Contains("loraB", r.MissingFiles[0]);
        Assert.Contains("1.6", LongCatAvatarResolver.RefusalFor(r));
    }

    [Fact]
    public void A15FolderHoldingTheA16LayoutIsRefusedNotRunAsA15()
    {
        string shard = Path.Combine(Root("avatar-1.5"), "base_model", "diffusion_pytorch_model-00001-of-00006.safetensors");
        string a = Path.Combine(Root("avatar-1.5"), "loraA", "lora.safetensors");
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => LongCatAvatarResolver.Resolve(shard, Files(a)));
        Assert.Contains("base_model_int8", ex.Message);
    }

    [Fact]
    public void ResolvingANonAvatarFileIsAnArgumentError()
    {
        Assert.Throws<ArgumentException>(() => LongCatAvatarResolver.Resolve(Path.Combine("models", "Wan", "x.safetensors"), Files()));
    }

    [Fact]
    public void TheFixedOutputRateAndSegmentMatchTheEngine()
    {
        Assert.Equal(25, LongCatAvatarResolver.OutputFps);
        Assert.Equal(93, LongCatAvatarResolver.SegmentFrames);
    }
}
