using System.Collections.Generic;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using Xunit;

namespace Hartsy.Extensions.HartsyInferenceBackend.Tests;

public sealed class MovieRollResolverTests
{
    private const string Good = "models/Stable-Diffusion/MovieRoll/distilled-t2v/model.safetensors";

    private static MovieRollResolver.Request Request(string file = Good, string compat = "movieroll", string negative = "",
        bool vae = false, bool qwen = false, MovieRollResolver.Schedule schedule = null) =>
        new(file, compat, negative, vae, qwen, schedule ?? Schedule(20, 7));

    // The model runs 5 steps at CFG 1; the parameter defaults here stand for the ones SwarmUI declares.
    private static MovieRollResolver.Schedule Schedule(double? steps = null, double? cfg = null, double? videoSteps = null,
        double? videoCfg = null) => new(5, 1,
        [
            new("Steps", false, steps, 20),
            new("CFG Scale", true, cfg, 7),
            new("Video Steps", false, videoSteps, 20),
            new("Video CFG", true, videoCfg, 7),
        ]);

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

    [Theory]
    // Unset.
    [InlineData(null, null, null, null)]
    // The model's own values.
    [InlineData(5.0, 1.0, 5.0, 1.0)]
    // SwarmUI's defaults, alone and mixed with the model's values.
    [InlineData(20.0, 7.0, null, null)]
    [InlineData(20.0, 7.0, 20.0, 7.0)]
    [InlineData(5.0, 7.0, 20.0, 1.0)]
    public void AnUnsetModelOrSwarmDefaultScheduleIsAccepted(double? steps, double? cfg, double? videoSteps, double? videoCfg)
    {
        Assert.Empty(MovieRollResolver.Refusals(Request(schedule: Schedule(steps, cfg, videoSteps, videoCfg))));
    }

    [Theory]
    [InlineData(30.0, null, null, null, "Steps 30")]
    // One step off the model's own, and a value that is another parameter's default.
    [InlineData(6.0, null, null, null, "Steps 6")]
    [InlineData(7.0, null, null, null, "Steps 7")]
    [InlineData(null, 3.5, null, null, "CFG Scale 3.5")]
    [InlineData(null, 20.0, null, null, "CFG Scale 20")]
    [InlineData(null, null, 8.0, null, "Video Steps 8")]
    [InlineData(null, null, null, 6.0, "Video CFG 6")]
    // A chosen core value is refused even when the video group's value would take precedence.
    [InlineData(40.0, null, 5.0, null, "Steps 40")]
    public void AChosenStepsOrCfgIsRefusedByName(double? steps, double? cfg, double? videoSteps, double? videoCfg, string named)
    {
        string reason = Assert.Single(MovieRollResolver.Refusals(Request(schedule: Schedule(steps, cfg, videoSteps, videoCfg))));
        Assert.StartsWith($"MovieRoll runs 5 steps at CFG 1, so it cannot use {named}.", reason);
    }

    [Fact]
    public void TheScheduleRefusalSaysWhatToSet()
    {
        Assert.Equal("MovieRoll runs 5 steps at CFG 1, so it cannot use Steps 30. Set Steps to 5, or leave it unset or at SwarmUI's default 20.",
            Assert.Single(MovieRollResolver.Refusals(Request(schedule: Schedule(30)))));
    }

    [Fact]
    public void EveryChosenValueIsNamed()
    {
        Assert.Equal(2, MovieRollResolver.Refusals(Request(schedule: Schedule(30, 4))).Count);
    }

    [Fact]
    public void AModelOrUnsetScheduleSaysNothing()
    {
        Assert.Null(MovieRollResolver.FixedScheduleNote(Schedule(5, 1)));
        Assert.Null(MovieRollResolver.FixedScheduleNote(Schedule()));
    }

    [Fact]
    public void ReplacedSwarmDefaultsAreNamedInTheNote()
    {
        Assert.Equal("MovieRoll runs 5 steps at CFG 1, replacing SwarmUI's default Steps 20 and CFG Scale 7.",
            MovieRollResolver.FixedScheduleNote(Schedule(20, 7)));
        Assert.Equal("MovieRoll runs 5 steps at CFG 1, replacing SwarmUI's default CFG Scale 7.",
            MovieRollResolver.FixedScheduleNote(Schedule(5, 7)));
    }
}
