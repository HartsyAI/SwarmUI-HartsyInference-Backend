using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// The single translation table between SwarmUI's <c>T2IModel.ModelClass.CompatClass.ID</c> and the
/// <c>HartsyInference.Engine</c> family id its recipe is registered under, plus the modality that family runs in.
/// <para>This is the whole of the extension's "architecture knowledge" now: the Engine owns detection, recipe
/// construction, pipeline caching and generation, so all this layer has to do is name the family. Whether a family is
/// actually <i>drivable</i> is never hard-coded here — it is asked of the Engine's
/// <see cref="RecipeRegistry"/> / <see cref="VideoRecipeRegistry"/> at call time, so the answer can't drift from what
/// the Engine will really do.</para>
/// </summary>
public static class ModelSupport
{
    /// <summary>What a mapped compat class generates.</summary>
    public enum Kind
    {
        /// <summary>Still image, via <c>InferenceEngine.Images</c>.</summary>
        Image,

        /// <summary>Video clip, via <c>InferenceEngine.Video</c>.</summary>
        Video,

        /// <summary>Music / audio, via <c>InferenceEngine.Music</c>.</summary>
        Music,
    }

    /// <summary>A compat class's Engine family id plus the service that drives it.</summary>
    public sealed record Family(string Id, Kind Kind);

    /// <summary>SwarmUI compat class → Engine family. Every entry here is a family the Engine has (or is expected to
    /// have) a registered recipe for; the registry lookup decides whether it is live today.</summary>
    private static readonly Dictionary<string, Family> _families = new(StringComparer.OrdinalIgnoreCase)
    {
        // ── Image (Engine RecipeRegistry family ids) ──
        ["stable-diffusion-v1"] = new("sd15", Kind.Image),
        ["stable-diffusion-xl-v1"] = new("sdxl", Kind.Image),
        ["stable-diffusion-xl-v1-refiner"] = new("sdxl-refiner", Kind.Image),
        ["stable-diffusion-v3-medium"] = new("sd3", Kind.Image),
        ["stable-diffusion-v3.5-medium"] = new("sd3", Kind.Image),
        ["stable-diffusion-v3.5-large"] = new("sd3", Kind.Image),
        ["flux-1"] = new("flux1", Kind.Image),
        ["flux-2"] = new("flux2", Kind.Image),
        ["flux-2-klein-4b"] = new("flux2", Kind.Image),
        ["flux-2-klein-9b"] = new("flux2", Kind.Image),
        ["chroma"] = new("chroma", Kind.Image),
        ["chroma-radiance"] = new("chroma-radiance", Kind.Image),
        ["zeta-chroma"] = new("zeta-chroma", Kind.Image),
        ["auraflow-v1"] = new("auraflow", Kind.Image),
        [ModelClassRegistrations.FLiteCompatClassId] = new("f-lite", Kind.Image),
        ["ideogram-4"] = new("ideogram4", Kind.Image),
        ["boogu"] = new("boogu", Kind.Image),
        ["ernie-image"] = new("ernie-image", Kind.Image),
        ["lumina-2"] = new("lumina2", Kind.Image),
        ["hunyuan-image-2_1"] = new("hunyuan-image", Kind.Image),
        ["omnigen-2"] = new("omnigen2", Kind.Image),
        ["z-image"] = new("zimage", Kind.Image),
        ["anima"] = new("anima", Kind.Image),
        ["hidream-i1"] = new("hidream", Kind.Image),
        ["qwen-image"] = new("qwen-image", Kind.Image),
        ["kandinsky5-imglite"] = new("kandinsky5", Kind.Image),
        [ModelClassRegistrations.LanceCompatClassId] = new("lance-image", Kind.Image),
        ["lens"] = new("lens", Kind.Image),
        ["krea-2"] = new("krea2", Kind.Image),
        ["mage-flow"] = new("mage-flow", Kind.Image),

        // ── Video (Engine VideoRecipeRegistry family ids; the Wan compat classes are registered verbatim,
        //    and WanVideoRecipe sniffs the checkpoint header to route the VACE / Animate / S2V variants) ──
        ["wan-22-5b"] = new("wan-22-5b", Kind.Video),
        ["wan-21-1_3b"] = new("wan-21-1_3b", Kind.Video),
        ["wan-21-14b"] = new("wan-21-14b", Kind.Video),
        // The hunyuan-video compat class also covers the SkyReels / I2V variants; the Engine recipe drives the
        // classic 13B text-to-video checkpoint (I2V conditioning is a recipe TODO, not a mapping concern).
        ["hunyuan-video"] = new("hunyuan-video", Kind.Video),
        ["lightricks-ltx-video"] = new("ltx-video", Kind.Video),
        ["lightricks-ltx-video-2"] = new("ltx-video-2", Kind.Video),
        [ModelClassRegistrations.LanceVideoCompatClassId] = new("lance-video", Kind.Video),
        // Core owns the minimax-h3 compat class (T2IModelClassSorter, "MiniMax H3 support" #1469) and shares it
        // with the video/audio VAE classes, exactly as LTX-2 does — this maps it, it must not re-register it.
        ["minimax-h3"] = new("minimax-h3", Kind.Video),

        // ── Music (Engine MusicCatalog descriptor ids) ──
        ["ace-step-1_5"] = new("acestep", Kind.Music),
        [ModelClassRegistrations.MusicGenCompatClassId] = new("musicgen", Kind.Music),
        [ModelClassRegistrations.YueCompatClassId] = new("yue", Kind.Music),
    };

    /// <summary>The Engine family for <paramref name="compatClass"/>, or null when this compat class has no mapping
    /// (i.e. the Engine has no such family at all).</summary>
    public static Family Resolve(string compatClass)
    {
        if (string.IsNullOrEmpty(compatClass))
        {
            return null;
        }
        return _families.TryGetValue(compatClass, out Family family) ? family : null;
    }

    /// <summary>True when the Engine can actually drive <paramref name="compatClass"/> today: it is mapped to a
    /// family AND that family has a registered recipe (music descriptors are always registered).</summary>
    public static bool IsArchitectureSupported(string compatClass)
    {
        Family family = Resolve(compatClass);
        if (family is null)
        {
            return false;
        }
        return family.Kind switch
        {
            Kind.Image => RecipeRegistry.Resolve(family.Id) is not null,
            Kind.Video => VideoRecipeRegistry.Resolve(family.Id) is not null,
            Kind.Music => true,
            _ => false,
        };
    }

    /// <summary>The composition features the Engine's recipe for <paramref name="compatClass"/> declares it can
    /// apply. <see cref="ImageFeatures.None"/> for video/music/unmapped families.</summary>
    public static ImageFeatures SupportedFeatures(string compatClass)
    {
        Family family = Resolve(compatClass);
        if (family is null || family.Kind != Kind.Image)
        {
            return ImageFeatures.None;
        }
        return RecipeRegistry.Resolve(family.Id)?.Supports ?? ImageFeatures.None;
    }

    /// <summary>The conditioning the Engine's video recipe for <paramref name="compatClass"/> declares it can apply.
    /// <see cref="VideoFeatures.None"/> for image/music/unmapped families. Asked of the Engine's registry at call time,
    /// exactly like <see cref="SupportedFeatures"/>, so it cannot drift from what the pipeline will really do.</summary>
    public static VideoFeatures SupportedVideoFeatures(string compatClass) => SupportedVideoFeatures(compatClass, null);

    /// <summary>Checkpoint-aware overload: Wan's VACE/Animate/S2V variants share the family compat classes and are
    /// only detectable from the checkpoint header, so pass <paramref name="checkpointPath"/> when the model is known
    /// (a driving video on an Animate checkpoint under <c>wan-21-14b</c> would otherwise be refused).</summary>
    public static VideoFeatures SupportedVideoFeatures(string compatClass, string checkpointPath)
    {
        Family family = Resolve(compatClass);
        if (family is null || family.Kind != Kind.Video)
        {
            return VideoFeatures.None;
        }
        IVideoRecipe recipe = VideoRecipeRegistry.Resolve(family.Id);
        return recipe switch
        {
            null => VideoFeatures.None,
            HartsyInference.Engine.Recipes.Video.WanVideoRecipe wan => wan.SupportsFor(checkpointPath),
            _ => recipe.Supports,
        };
    }

    /// <summary>Human-readable explanation of why a compat class isn't drivable. Distinguishes "the Engine knows this
    /// family but hasn't lifted its recipe yet" from "the Engine has nothing for this architecture".</summary>
    public static string WhyNotSupported(string compatClass)
    {
        if (string.IsNullOrEmpty(compatClass))
        {
            return "Model has no architecture compat class set — HartsyInference can't dispatch.";
        }
        Family family = Resolve(compatClass);
        if (family is null)
        {
            return $"Architecture '{compatClass}' has no HartsyInference family mapping. "
                + $"Supported today: {string.Join(", ", SupportedArchitectures)}.";
        }
        string drivable = family.Kind == Kind.Video
            ? string.Join(", ", VideoRecipeRegistry.RegisteredNames)
            : string.Join(", ", RecipeRegistry.RegisteredNames);
        return $"Architecture '{compatClass}' maps to HartsyInference family '{family.Id}', but no "
            + $"{family.Kind.ToString().ToLowerInvariant()} recipe is registered for it in this engine build. "
            + $"Currently drivable: {drivable}. Use the ComfyUI backend for this architecture in the meantime.";
    }

    /// <summary>Compat classes the Engine can drive right now.</summary>
    public static IReadOnlyCollection<string> SupportedArchitectures =>
        [.. _families.Keys.Where(IsArchitectureSupported).Order(StringComparer.Ordinal)];

    /// <summary>Compat classes that are mapped to an Engine family whose recipe isn't registered yet, mapped to the
    /// human-readable blocker. Surfaced by the WebAPI for admin UX.</summary>
    public static IReadOnlyDictionary<string, string> PendingArchitectures =>
        _families.Keys.Where(k => !IsArchitectureSupported(k))
            .Order(StringComparer.Ordinal)
            .ToDictionary(k => k, WhyNotSupported, StringComparer.Ordinal);

    /// <summary>Builds the Engine's load request for a Swarm model: the family id travels as the catalog id (that is
    /// what the Engine keys its recipe registry on), and the checkpoint path as the resolved local path.</summary>
    public static ModelSpec BuildSpec(SwarmUI.Text2Image.T2IModel model, Family family)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(family);
        Modality modality = family.Kind switch
        {
            Kind.Video => Modality.Video,
            Kind.Music => Modality.Music,
            _ => Modality.Image,
        };
        return new ModelSpec
        {
            Requested = family.Id,
            Modality = modality,
            LocalPath = model.RawFilePath,
            Catalog = new CatalogEntry
            {
                Id = family.Id,
                Modality = modality,
                DisplayName = model.Name,
                Architecture = family.Id,
                Status = ModelStatus.Verified,
            },
        };
    }

    /// <summary>Kept for the extension entry point's call order; the Engine's registries are self-registering, so
    /// there is nothing left to pre-register here.</summary>
    public static void RegisterBuiltins()
    {
    }
}
