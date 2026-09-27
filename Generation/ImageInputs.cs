using Newtonsoft.Json.Linq;
using SwarmUI.Text2Image;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>What images a model takes as input: how many, in which modes, whether one is required, and whether a mask
/// or end frame applies. Feeds the <c>HartsyInferenceGetImageInputs</c> route and this backend's image refusals.</summary>
/// <remarks>Every number comes from the engine's per-variant answers (<see cref="ModelCapabilities"/>), so a Qwen-Image
/// base and Edit build, or a Wan I2V and Animate checkpoint, are described as what they resolve to.</remarks>
public static class ImageInputs
{
    /// <summary>Most prompt images a reference-conditioned video family is handed (the MiniMax-H3 reference node's cap).</summary>
    public const int MaxReferenceImages = 9;

    /// <summary>The engine's input-image limits for a concrete model's resolved variant.</summary>
    public static ImageInputLimits LimitsFor(T2IModel model, ModelSupport.Family family) =>
        ModelCapabilities.ImageInputLimitsFor(ModelSupport.BuildSpec(model, family, stageBundles: false));

    /// <summary>The route's <c>inputs</c> object for a concrete model; null when its compat class is not drivable here.</summary>
    public static JObject Describe(T2IModel model)
    {
        string compat = model?.ModelClass?.CompatClass?.ID;
        if (!ModelSupport.IsArchitectureSupported(compat))
        {
            return null;
        }
        ModelSupport.Family family = ModelSupport.Resolve(compat);
        return family.Kind switch
        {
            ModelSupport.Kind.Image => DescribeImage(ModelSupport.SupportedFeatures(model), LimitsFor(model, family)),
            ModelSupport.Kind.Video => DescribeVideo(ModelSupport.SupportedVideoFeatures(model), compat, model.ModelClass?.ID),
            _ => None(),
        };
    }

    /// <summary>The route's <c>inputs</c> object for a Swarm model class with no file in hand: the class id is passed
    /// as the variant hint, which is what separates Qwen-Image base from Edit. Null when not drivable here.</summary>
    public static JObject Describe(string compat, string modelClassId)
    {
        if (!ModelSupport.IsArchitectureSupported(compat))
        {
            return null;
        }
        ModelSupport.Family family = ModelSupport.Resolve(compat);
        ModelSpec spec = SpecFor(family, modelClassId);
        return family.Kind switch
        {
            ModelSupport.Kind.Image =>
                DescribeImage(ModelCapabilities.ImageFeaturesFor(spec), ModelCapabilities.ImageInputLimitsFor(spec)),
            ModelSupport.Kind.Video => DescribeVideo(ModelCapabilities.VideoFeaturesFor(spec), compat, modelClassId),
            _ => None(),
        };
    }

    private static ModelSpec SpecFor(ModelSupport.Family family, string modelClassId)
    {
        Modality modality = family.Kind == ModelSupport.Kind.Video ? Modality.Video : Modality.Image;
        return new ModelSpec
        {
            Requested = family.Id,
            Modality = modality,
            Variant = modelClassId,
            Catalog = new CatalogEntry
            {
                Id = family.Id,
                Modality = modality,
                DisplayName = modelClassId ?? family.Id,
                Architecture = family.Id,
                Status = ModelStatus.Verified,
            },
        };
    }

    private static JObject DescribeImage(ImageFeatures supported, ImageInputLimits limits)
    {
        List<string> modes = [];
        if ((supported & ImageFeatures.Img2Img) != 0)
        {
            modes.Add("denoise");
        }
        if ((supported & ImageFeatures.RefEdit) != 0)
        {
            modes.Add("reference");
        }
        // What initimagemode=auto does with an Init Image alone: families offering both prefer denoise, so a client
        // wanting an edit on a Qwen Edit build must send initimagemode=reference.
        string autoMode = modes.Count == 0 ? null : modes[0];
        int maxImages = modes.Count == 0 ? 0 : limits.MaxImages;
        bool mask = (supported & ImageFeatures.Inpaint) != 0;
        return Shape(maxImages == 0 ? "none" : "optional", modes, autoMode, maxImages, mask, endFrame: false);
    }

    private static JObject DescribeVideo(VideoFeatures supported, string compat, string modelClassId)
    {
        // Wan-Animate's Init Image slot carries the driving video; its one picture is the character to animate,
        // which it cannot run without.
        if ((supported & VideoFeatures.DrivingVideo) != 0)
        {
            return Shape("required", ["reference"], null, 1, mask: false, endFrame: false);
        }
        bool init = (supported & VideoFeatures.InitImage) != 0;
        bool refs = (supported & VideoFeatures.ReferenceImages) != 0;
        List<string> modes = [];
        if (init)
        {
            modes.Add("init");
        }
        if (refs)
        {
            modes.Add("reference");
        }
        string need = !init && !refs ? "none" : init && IsWanImageToVideoClass(compat, modelClassId) ? "required" : "optional";
        // An init frame and references are separate tasks (MiniMax-H3 refuses the two together), so a request carries
        // one or the other: the larger of the two, not their sum.
        int maxImages = Math.Max(init ? 1 : 0, refs ? MaxReferenceImages : 0);
        return Shape(need, modes, init ? "init" : null, maxImages, mask: false, (supported & VideoFeatures.EndFrame) != 0);
    }

    /// <summary>Wan model classes whose checkpoints are concat-I2V and refuse to run without an init image.</summary>
    private static bool IsWanImageToVideoClass(string compat, string modelClassId) =>
        compat == "wan-21-14b" && modelClassId is not null
        && (modelClassId.Contains("image2video", StringComparison.Ordinal) || modelClassId.Contains("flf2v", StringComparison.Ordinal));

    private static JObject None() => Shape("none", [], null, 0, mask: false, endFrame: false);

    private static JObject Shape(string need, List<string> modes, string autoMode, int maxImages, bool mask, bool endFrame) => new()
    {
        ["need"] = need,
        ["modes"] = new JArray(modes),
        ["autoMode"] = autoMode,
        ["maxImages"] = maxImages,
        ["mask"] = mask,
        ["endFrame"] = endFrame,
    };
}
