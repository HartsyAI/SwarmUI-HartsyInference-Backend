using System.Collections.Generic;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using Xunit;

namespace Hartsy.Extensions.HartsyInferenceBackend.Tests;

public sealed class MovieRollResolverTests
{
    private const string Good = "models/Stable-Diffusion/MovieRoll/distilled-t2v/model.safetensors";

    private static MovieRollResolver.Request Request(string file = Good, string compat = "movieroll", string negative = "",
        bool vae = false, bool qwen = false) => new(file, compat, negative, vae, qwen);

    [Theory]
    [InlineData(Good)]
    [InlineData("models/diffusion_models/MovieRoll/distilled/t2v/model.safetensors")]
    [InlineData(@"E:\Models\Stable-Diffusion\movieroll\Distilled-T2V\model.safetensors")]
    [InlineData("models/Stable-Diffusion/MovieRoll-distilled-t2v/movieroll.safetensors")]
    public void ADistilledTextToVideoFileIsMatchedAndAccepted(string path)
    {
        Assert.True(MovieRollResolver.IsMovieRollFile(path));
        Assert.Equal(MovieRollResolver.Variant.DistilledT2V, MovieRollResolver.VariantOf(path));
        Assert.Null(MovieRollResolver.CheckpointRefusal(path));
    }

    [Theory]
    [InlineData("models/MovieRoll/distilled-i2v/model.safetensors", MovieRollResolver.Variant.DistilledI2V, "image-to-video")]
    [InlineData("models/MovieRoll/distilled/i2v/model.safetensors", MovieRollResolver.Variant.DistilledI2V, "image-to-video")]
    [InlineData("models/MovieRoll/base/model.safetensors", MovieRollResolver.Variant.Base, "MovieRoll Base")]
    [InlineData("models/MovieRoll/model.safetensors", MovieRollResolver.Variant.Unknown, "MovieRoll/distilled-t2v/model.safetensors")]
    [InlineData("models/MovieRoll/t2v/model.safetensors", MovieRollResolver.Variant.Unknown, "folder has to say")]
    public void EveryOtherReleaseIsMatchedAndRefusedByName(string path, MovieRollResolver.Variant variant, string named)
    {
        Assert.True(MovieRollResolver.IsMovieRollFile(path));
        Assert.Equal(variant, MovieRollResolver.VariantOf(path));
        Assert.Contains(named, Assert.Single(MovieRollResolver.Refusals(Request(path))));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("models/Stable-Diffusion/hunyuanimage2.1.safetensors")]
    [InlineData("models/HunyuanVideo-1.5/vae/diffusion_pytorch_model.safetensors")]
    // A folder that only mentions MovieRoll does not claim another checkpoint in it.
    [InlineData("models/movieroll-tests/sdxl.safetensors")]
    // More than three folders above the file.
    [InlineData("MovieRoll/a/b/c/model.safetensors")]
    [InlineData("models/MovieRoll/distilled-t2v/model.gguf")]
    // The file name alone does not count: the folder has to be a MovieRoll folder.
    [InlineData("models/Stable-Diffusion/movieroll-distilled-t2v.safetensors")]
    public void OtherFilesAreNotMovieRoll(string path)
    {
        Assert.False(MovieRollResolver.IsMovieRollFile(path));
    }

    [Fact]
    public void ARequestWithNothingExtraIsAccepted()
    {
        Assert.Empty(MovieRollResolver.Refusals(Request()));
        Assert.Empty(MovieRollResolver.Refusals(Request(negative: "   ")));
        Assert.Empty(MovieRollResolver.Refusals(Request(negative: null)));
    }

    [Theory]
    [InlineData("hunyuan-image-2_1")]
    [InlineData(null)]
    public void AFileLeftOnAnotherClassIsRefusedWithTheSidecarToCreate(string compat)
    {
        string reason = Assert.Single(MovieRollResolver.Refusals(Request(compat: compat)));
        Assert.Contains("model.swarm.json", reason);
        Assert.Contains("\"modelspec.architecture\": \"movieroll\", \"modelspec.resolution\": \"864x480\"", reason);
    }

    [Fact]
    public void TheNegativePromptAndComponentPicksAreEachRefusedByName()
    {
        IReadOnlyList<string> reasons = MovieRollResolver.Refusals(Request(negative: "blurry", vae: true, qwen: true));
        Assert.Collection(reasons,
            reason => Assert.Contains("Negative Prompt", reason),
            reason => Assert.Contains("VAE", reason),
            reason => Assert.Contains("Qwen Model", reason));
    }

    [Fact]
    public void AMatchingOrUnsetScheduleSaysNothing()
    {
        Assert.Null(MovieRollResolver.FixedScheduleNote(5, 1.0, 5, 1.0));
        Assert.Null(MovieRollResolver.FixedScheduleNote(null, null, 5, 1.0));
    }

    [Fact]
    public void SwarmsDefaultStepsAndCfgAreReportedAsIgnored()
    {
        Assert.Equal("MovieRoll runs its fixed 5-step schedule at CFG 1; ignoring Steps 20 and CFG Scale 7.",
            MovieRollResolver.FixedScheduleNote(20, 7.0, 5, 1.0));
        Assert.Equal("MovieRoll runs its fixed 5-step schedule at CFG 1; ignoring CFG Scale 3.5.",
            MovieRollResolver.FixedScheduleNote(5, 3.5, 5, 1.0));
    }
}
