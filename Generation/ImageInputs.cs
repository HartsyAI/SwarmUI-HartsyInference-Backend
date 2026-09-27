using Newtonsoft.Json.Linq;
using SwarmUI.Text2Image;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>What images a model can take as input, per Swarm model class rather than per compat class: how many, in which
/// modes, whether an init image is needed, and whether a mask or end frame applies. One answer feeds both the
/// <c>HartsyInferenceGetImageInputs</c> route (so a client such as the Discord bot can size its upload flow before it
/// sends anything) and the refusals in <c>IsValidForThisBackend</c>, so the two cannot disagree.</summary>
/// <remarks>Per model class because the compat class is too coarse for Qwen-Image: core files <c>qwen-image</c>,
/// <c>qwen-image-edit</c> and <c>qwen-image-edit-plus</c> under one compat class, the engine recipe cannot tell them
/// apart from the weights, and reference editing on a base checkpoint runs and returns a wrong image.
/// <para>The image limits below mirror the engine's per-recipe <c>IArchitectureRecipe.InputLimits</c>, which this
/// extension's pinned engine (alpha.181) predates. When the pin moves past the version that adds it, replace
/// <see cref="MaxImages"/> and <see cref="ReferencesRequireInitImage"/> with reads of
/// <c>RecipeRegistry.Resolve(family.Id).InputLimits</c>; the Qwen base/edit split stays here, because only Swarm's
/// model class knows it.</para></remarks>
public static class ImageInputs
{
    /// <summary>Swarm model classes for Qwen-Image checkpoints that are Edit builds. Core never auto-detects
    /// <c>qwen-image-edit</c> (its <c>IsThisModelOfClass</c> is always false), so an Edit 2509 file only lands there
    /// when its metadata says so; <c>qwen-image-edit-plus</c> is header-detected for 2511.</summary>
    private static readonly HashSet<string> QwenEditClasses = new(StringComparer.OrdinalIgnoreCase)
    {
        "qwen-image-edit",
        "qwen-image-edit-plus",
    };

    /// <summary>Families that edit the init image alone and never read prompt images: extra images were discarded and
    /// prompt images with no init image produced plain text-to-image.</summary>
    private static readonly HashSet<string> InitOnlyEditFamilies = new(StringComparer.OrdinalIgnoreCase)
    {
        "boogu",
        "omnigen2",
        "mage-flow",
    };

    /// <summary>Most prompt images any reference-conditioned family is handed (the MiniMax-H3 reference node's cap); the backend truncates to this too.</summary>
    public const int MaxReferenceImages = 9;

    /// <summary>True for a Qwen-Image checkpoint whose model class is not one of the Edit classes.</summary>
    public static bool IsBaseQwen(ModelSupport.Family family, string modelClassId) =>
        family?.Id == "qwen-image" && !QwenEditClasses.Contains(modelClassId ?? "");

    /// <summary>True when prompt images become in-context edit references on this family rather than IP-Adapter input.</summary>
    public static bool PromptImagesAreReferences(ImageFeatures supported) => (supported & ImageFeatures.RefEdit) != 0;

    /// <summary>Most images (Init Image + prompt images) one request can hand this image family.</summary>
    public static int MaxImages(ModelSupport.Family family, string modelClassId, ImageFeatures supported)
    {
        if (family.Id == "qwen-image")
        {
            // Base Qwen-Image denoises from one init image; only the Edit builds read the Picture 1..3 template.
            return IsBaseQwen(family, modelClassId) ? 1 : QwenImageEditConditioning.MaxReferences;
        }
        return (supported & (ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.RefEdit)) != 0 ? 1 : 0;
    }

    /// <summary>True when prompt images mean nothing on this family without an Init Image.</summary>
    public static bool ReferencesRequireInitImage(ModelSupport.Family family) => InitOnlyEditFamilies.Contains(family.Id);

    /// <summary>Wan model classes whose checkpoints are concat-I2V and refuse to run without an init image.</summary>
    private static bool IsWanImageToVideoClass(string compat, string modelClassId) =>
        compat == "wan-21-14b" && modelClassId is not null
        && (modelClassId.Contains("image2video", StringComparison.Ordinal) || modelClassId.Contains("flf2v", StringComparison.Ordinal));

    /// <summary>Describes the image inputs for one model class as the route returns them:
    /// <c>{ need, modes, autoMode, maxImages, mask, endFrame }</c>. Null when the compat class is not drivable here.</summary>
    /// <param name="compat">The model's compat class id.</param>
    /// <param name="modelClassId">The model's Swarm model class id (its architecture), when known.</param>
    /// <param name="checkpointPath">The checkpoint file, when a concrete model is known; lets Wan and MiniMax-H3 narrow per file.</param>
    public static JObject Describe(string compat, string modelClassId, string checkpointPath)
    {
        if (!ModelSupport.IsArchitectureSupported(compat))
        {
            return null;
        }
        ModelSupport.Family family = ModelSupport.Resolve(compat);
        return family.Kind switch
        {
            ModelSupport.Kind.Image => DescribeImage(family, compat, modelClassId),
            ModelSupport.Kind.Video => DescribeVideo(compat, modelClassId, checkpointPath),
            _ => Shape("none", [], null, 0, false, false),
        };
    }

    private static JObject DescribeImage(ModelSupport.Family family, string compat, string modelClassId)
    {
        ImageFeatures supported = ModelSupport.SupportedFeatures(compat);
        List<string> modes = [];
        if ((supported & ImageFeatures.Img2Img) != 0)
        {
            modes.Add("denoise");
        }
        if ((supported & ImageFeatures.RefEdit) != 0 && !IsBaseQwen(family, modelClassId))
        {
            modes.Add("reference");
        }
        int maxImages = MaxImages(family, modelClassId, supported);
        // What initimagemode=auto does with an Init Image alone: families offering both prefer denoise (engine
        // Img2ImgBit), so a client that wants an edit on a Qwen Edit checkpoint must send initimagemode=reference.
        string autoMode = modes.Count == 0 ? null : modes[0];
        return Shape(maxImages == 0 ? "none" : "optional", modes, autoMode, maxImages,
            (supported & ImageFeatures.Inpaint) != 0, endFrame: false);
    }

    private static JObject DescribeVideo(string compat, string modelClassId, string checkpointPath)
    {
        VideoFeatures supported = ModelSupport.SupportedVideoFeatures(compat, checkpointPath);
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
        int maxImages = (init ? 1 : 0) + (refs ? MaxReferenceImages : 0);
        return Shape(need, modes, init ? "init" : null, maxImages, mask: false, (supported & VideoFeatures.EndFrame) != 0);
    }

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
