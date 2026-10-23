using HartsyInference.Core.Configuration;
using System.IO;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Engine;
using HartsyInference.Engine.Catalog;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Translates SwarmUI's <c>T2IModel.ModelClass.CompatClass.ID</c> into the <c>HartsyInference.Engine</c> family id and modality.
/// <para>The Engine owns detection, recipe construction, caching and generation. Whether a family is drivable is asked of <see cref="RecipeRegistry"/> and <see cref="VideoRecipeRegistry"/> at call time, never hard-coded here.</para></summary>
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

    /// <summary>SwarmUI compat class → Engine family. Every entry is a family the Engine may have a recipe for; the registry lookup decides whether it is live.</summary>
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
        // Core owns the FLUX.3 Klein and MOSS-Image classes: mapped here, never re-registered.
        [T2IModelClassSorter.CompatFlux3ImageKlein.ID] = new("flux3", Kind.Image),
        [T2IModelClassSorter.CompatMossImage.ID] = new("moss-image", Kind.Image),
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
        // Core owns the qwen-image-2.1 compat, model and VAE classes. Map it, do not re-register it; key off core's constant.
        [T2IModelClassSorter.CompatQwenImage21.ID] = new("qwen-image-2.1", Kind.Image),
        ["kandinsky5-imglite"] = new("kandinsky5", Kind.Image),
        [ModelClassRegistrations.LanceCompatClassId] = new("lance-image", Kind.Image),
        ["lens"] = new("lens", Kind.Image),
        ["krea-2"] = new("krea2", Kind.Image),
        ["mage-flow"] = new("mage-flow", Kind.Image),
        // Core owns the fibo-2 compat class (Bria Fibo 2.0, CC-BY-NC-4.0). Text-to-image only; the Engine refuses references, a negative prompt and non-Euler samplers.
        ["fibo-2"] = new("fibo2", Kind.Image),

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
        // Both compat classes route to Kandinsky5VideoRecipe, which derives Lite vs. Pro from the loaded transformer's model_dim (see Kandinsky5VideoRecipe.DetectConfig).
        ["kandinsky5-vidlite"] = new("kandinsky5-video", Kind.Video),
        ["kandinsky5-vidpro"] = new("kandinsky5-video", Kind.Video),
        [ModelClassRegistrations.LanceVideoCompatClassId] = new("lance-video", Kind.Video),
        // Core owns the minimax-h3 compat class. Map it, do not re-register it; keying off core's constant means a rename cannot silently match nothing.
        [T2IModelClassSorter.CompatMiniMaxH3.ID] = new("minimax-h3", Kind.Video),

        // ── Music (Engine MusicCatalog descriptor ids) ──
        // SwarmUI-native music only: core registers ace-step-1_5 and minimax-music-3. Other families (YuE, MusicGen, …) belong to AudioLab, which maps them through its own model classes.
        ["ace-step-1_5"] = new("acestep", Kind.Music),
        [T2IModelClassSorter.CompatMiniMaxMusic3.ID] = new("minimaxmusic3", Kind.Music),
    };

    /// <summary>The Engine family for <paramref name="compatClass"/>, or null when it has no mapping.</summary>
    public static Family Resolve(string compatClass)
    {
        if (string.IsNullOrEmpty(compatClass))
        {
            return null;
        }
        return _families.TryGetValue(compatClass, out Family family) ? family : null;
    }

    /// <summary>True when the Engine can drive <paramref name="compatClass"/>: it is mapped to a family with a registered recipe (music descriptors are always registered).</summary>
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

    /// <summary>The composition features the Engine's recipe for <paramref name="compatClass"/> declares. <see cref="ImageFeatures.None"/> for video, music and unmapped families.</summary>
    public static ImageFeatures SupportedFeatures(string compatClass)
    {
        Family family = Resolve(compatClass);
        if (family is null || family.Kind != Kind.Image)
        {
            return ImageFeatures.None;
        }
        return RecipeRegistry.Resolve(family.Id)?.Supports ?? ImageFeatures.None;
    }

    /// <summary>The composition features the Engine applies to <paramref name="model"/>'s resolved variant, narrowed from <see cref="SupportedFeatures(string)"/> (a base Qwen-Image has no reference editing; its Edit builds do). <see cref="ImageFeatures.None"/> for non-image models.</summary>
    public static ImageFeatures SupportedFeatures(SwarmUI.Text2Image.T2IModel model)
    {
        Family family = Resolve(model?.ModelClass?.CompatClass?.ID);
        return family is null || family.Kind != Kind.Image
            ? ImageFeatures.None
            : (Describe(model, family) as ImageModelDescriptor)?.Features ?? ImageFeatures.None;
    }

    /// <summary>The conditioning the Engine's video recipe for <paramref name="compatClass"/> declares. <see cref="VideoFeatures.None"/> for non-video families. Asked of the registry at call time, like <see cref="SupportedFeatures(string)"/>.</summary>
    public static VideoFeatures SupportedVideoFeatures(string compatClass)
    {
        Family family = Resolve(compatClass);
        return family is null || family.Kind != Kind.Video
            ? VideoFeatures.None
            : VideoRecipeRegistry.Resolve(family.Id)?.Supports ?? VideoFeatures.None;
    }

    /// <summary>The conditioning the Engine applies to <paramref name="model"/>'s resolved variant. Wan's VACE, Animate and S2V builds share the family compat classes and are told apart engine-side, so a driving video is accepted on an Animate checkpoint under <c>wan-21-14b</c>. <see cref="VideoFeatures.None"/> for non-video models.</summary>
    public static VideoFeatures SupportedVideoFeatures(SwarmUI.Text2Image.T2IModel model)
    {
        Family family = Resolve(model?.ModelClass?.CompatClass?.ID);
        if (family is null || family.Kind != Kind.Video)
        {
            return VideoFeatures.None;
        }
        VideoFeatures features = (Describe(model, family) as VideoModelDescriptor)?.Features ?? VideoFeatures.None;
        return family.Id == "minimax-h3" ? MiniMaxH3TaskFeatures(features, model.RawFilePath) : features;
    }

    /// <summary>The Engine's recipe descriptor for <paramref name="model"/>'s resolved family and variant: its defaults, features,
    /// input limits and the variant it resolved to. Null when the family has no recipe for its modality, so callers keep the
    /// answer they gave before the catalog existed (no features, text-only, no variant).</summary>
    public static ModelDescriptor Describe(SwarmUI.Text2Image.T2IModel model, Family family)
    {
        // The catalog describes image and video recipes; music has none.
        if (family.Kind == Kind.Music)
        {
            return null;
        }
        return DescribeSpec(BuildSpec(model, family, stageBundles: false));
    }

    /// <summary>The Engine's recipe descriptor for <paramref name="spec"/>, or null when the family it routes to (variant
    /// routing can move a spec to another family) has no recipe.</summary>
    public static ModelDescriptor DescribeSpec(ModelSpec spec)
    {
        try
        {
            return Recipes.Describe(spec);
        }
        catch (RecipeNotRegisteredException)
        {
            return null;
        }
    }

    /// <summary>Narrows the recipe's MiniMax-H3 task union to this checkpoint's task, so an fl2va model does not advertise reference features. MiniMax-H3 ships fl2va (first/last frame) and ref2va (references) as separate checkpoints.
    /// <para>Matched by FILENAME: the two builds have byte-identical key sets and shapes, so no header sniff separates them. A name matching neither keeps the full union rather than refusing renamed checkpoints.</para></summary>
    private static VideoFeatures MiniMaxH3TaskFeatures(VideoFeatures declared, string checkpointPath)
    {
        string name = Path.GetFileNameWithoutExtension(checkpointPath ?? "");
        const VideoFeatures refs = VideoFeatures.ReferenceImages | VideoFeatures.ReferenceVideos | VideoFeatures.ReferenceAudios;
        const VideoFeatures frames = VideoFeatures.InitImage | VideoFeatures.EndFrame;
        if (name.Contains("fl2va", StringComparison.OrdinalIgnoreCase))
        {
            return declared & ~refs;
        }
        if (name.Contains("ref2va", StringComparison.OrdinalIgnoreCase))
        {
            return declared & ~frames;
        }
        return declared;
    }

    /// <summary>Human-readable reason a compat class is not drivable: its family has no registered recipe, or the Engine has no family for it.</summary>
    public static string WhyNotSupported(string compatClass)
    {
        if (string.IsNullOrEmpty(compatClass))
        {
            return "Model has no architecture compat class set. HartsyInference can't dispatch.";
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
            + $"Currently drivable: {drivable}.";
    }

    /// <summary>The Engine's recipe catalog, shared by every caller in this extension. Stateless, so one static instance serves
    /// these static callers before any backend has loaded an engine.</summary>
    public static IRecipeCatalog Recipes { get; } = new RecipeCatalog();

    /// <summary>Compat classes the Engine can drive right now.</summary>
    public static IReadOnlyCollection<string> SupportedArchitectures =>
        [.. _families.Keys.Where(IsArchitectureSupported).Order(StringComparer.Ordinal)];

    /// <summary>Compat classes mapped to an Engine family with no registered recipe, with the human-readable blocker. Surfaced by the WebAPI for admin UX.</summary>
    public static IReadOnlyDictionary<string, string> PendingArchitectures =>
        _families.Keys.Where(k => !IsArchitectureSupported(k))
            .Order(StringComparer.Ordinal)
            .ToDictionary(k => k, WhyNotSupported, StringComparer.Ordinal);

    /// <summary>Core's model class id for LTX-2.5, read directly because the compat class cannot tell it apart: core header-sniffs LTX-2, 2.3 and 2.5 into distinct model classes that share <c>lightricks-ltx-video-2</c>.</summary>
    public const string Ltx25ModelClassId = "lightricks-ltx-video-2-5";

    /// <summary>Whether the selected model is LTX-2.5, by core's own header detection rather than a name guess.</summary>
    private static bool IsLtx25(T2IModel model) => model?.ModelClass?.ID == Ltx25ModelClassId;

    /// <summary>The Gemma-4 tower's discriminator: its per-layer <c>layer_scalar</c> is the key the Engine's converter keys on, not a missing <c>v_proj</c> (the global layers also lack that in Gemma 3).</summary>
    private const string Gemma4SignatureKey = "model.layers.0.layer_scalar";

    /// <summary>The convolutional video decoder's marker. Shared with LTX-2.3's VAE and absent from the 2.5 diffusion decoder, which makes it usable as the "is the conv VAE here" test.</summary>
    private const string ConvVideoVaeSignatureKey = "decoder.conv_in.conv.bias";

    /// <summary>Cached bundle scans keyed by directory. Invalidated on the file-set signature, so a file staged mid-session is picked up without a header rescan.</summary>
    private static readonly ConcurrentDictionary<string, (string Signature, string Problem)> _ltx25Scans = new(StringComparer.Ordinal);

    /// <summary>Header-scans <paramref name="directory"/>, memoised on its file-set signature.</summary>
    private static string ScanCached(string directory)
    {
        string[] files = Directory.GetFiles(directory, "*.safetensors", SearchOption.AllDirectories);
        // Cheap signature first: a stat per file, where a rescan costs a header parse per file.
        string signature = $"{files.Length}|{files.Select(f => new FileInfo(f).LastWriteTimeUtc.Ticks).DefaultIfEmpty(0L).Max()}";
        if (_ltx25Scans.TryGetValue(directory, out (string Signature, string Problem) cached) && cached.Signature == signature)
        {
            return cached.Problem;
        }
        string problem = ScanLtx25Bundle(directory, files);
        _ltx25Scans[directory] = (signature, problem);
        return problem;
    }

    // ────────────────────────────────────────────────────────────────────────────────────────────────────────────
    // LTX-2.5 side models, resolved the way Swarm resolves them for every other backend.
    //
    // The user picks the DiT alone. The video and audio VAEs come from the VAE folder and the Gemma-4 tower from the
    // text-encoder folder, each resolved through core's registry and downloaded into the folder core assigns when absent.
    // The Engine's LtxVideo2Recipe takes exactly ONE path and does not honor RecipeContext.Components, so the resolved
    // parts are linked into a staging directory under Data/ (never a model root, so Swarm does not index the links).
    // ────────────────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Core's known-model ids for LTX-2.5's two VAEs (registered in <c>CommonModels.RegisterCoreSet</c>), so the URL, hash and folder live in core's registry for both backends.</summary>
    private const string Ltx25VideoVaeKnownId = "ltx2-5-video-vae";

    /// <summary>Companion of <see cref="Ltx25VideoVaeKnownId"/> for the audio VAE. No user param selects it, so this is always the registry's answer.</summary>
    private const string Ltx25AudioVaeKnownId = "ltx2-5-audio-vae";

    /// <summary>The Gemma-4 tower. Not read from <c>CommonModels</c>, since core does not register it there. Its ComfyUI backend fetches it inline via <c>RequireClipModel</c>, so the URL and hash are mirrored from <c>WorkflowGeneratorModelSupport.GetLTX25Gemma4Model</c>.</summary>
    private const string Ltx25GemmaFileName = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot-v2.safetensors";

    private const string Ltx25GemmaUrl = "https://huggingface.co/mcmonkey/swarm-models/resolve/main/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot-v2.safetensors";

    private const string Ltx25GemmaHash = "6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f";

    /// <summary>Serialises staging so two concurrent generations can't half-build the same link set.</summary>
    private static readonly object _ltx25StageLock = new();

    /// <summary>The four files an LTX-2.5 run needs, as absolute paths.</summary>
    private sealed record Ltx25Parts(string Dit, string VideoVae, string AudioVae, string Gemma);

    /// <summary>Matches any LTX-2.5 Gemma-4 build in the text-encoder folder, whatever its repack: the Engine's <c>int8_lean_convrot</c> and core's <c>comfy-int8-convrot</c> builds are both valid.</summary>
    private static bool IsLtx25GemmaName(string name) =>
        name.Contains("gemma4", StringComparison.OrdinalIgnoreCase)
        && name.Contains("ltx-2.5", StringComparison.OrdinalIgnoreCase);

    /// <summary>A user's component pick, or null when there is none. "None" arrives as a model of that name and means "use the default", as core's <c>DoVaeLoader</c> also treats it. Without this check it would read as a pick and stage a bundle for nothing.</summary>
    private static string PickedPath(T2IModel picked)
    {
        if (picked?.RawFilePath is null || picked.Name == "None" || !File.Exists(picked.RawFilePath))
        {
            return null;
        }
        return picked.RawFilePath;
    }

    /// <summary>Whether the Engine decodes with the diffusion video VAE. Read through the Engine's own knob so the staged VAE matches the one the recipe requests; staging both corrupts the selection.</summary>
    private static bool WantDiffusionVae() => EngineKnobs.Ltx2DiffusionVae.Value;

    /// <summary>Matching models in <paramref name="setName"/>, ordered by file name. The model set is a concurrent dictionary, so an unordered first match would vary between runs when several builds are staged.</summary>
    private static List<T2IModel> FindInSet(string setName, Func<string, bool> matchesFileName)
    {
        if (!Program.T2IModelSets.TryGetValue(setName, out T2IModelHandler handler))
        {
            return [];
        }
        return [.. handler.Models.Values
            .Where(m => m?.RawFilePath is not null && matchesFileName(Path.GetFileName(m.RawFilePath)) && File.Exists(m.RawFilePath))
            .OrderBy(m => Path.GetFileName(m.RawFilePath), StringComparer.Ordinal)];
    }

    /// <summary>Resolves a core known model to its on-disk path, downloading it into the folder core assigns when missing. With <paramref name="download"/> false (validation time) nothing is fetched; the fetch belongs at load, as in the ComfyUI backend.</summary>
    private static string EnsureKnownModel(string knownId, string setName, bool download, T2IParamInput input, out string problem)
    {
        problem = null;
        if (!CommonModels.Known.TryGetValue(knownId, out CommonModels.ModelInfo known))
        {
            problem = $"SwarmUI has no known-model registration '{knownId}', so it can't be downloaded automatically. "
                + "Update SwarmUI, or download the file into your VAE folder and pass it as the VAE parameter.";
            return null;
        }
        if (!Program.T2IModelSets.TryGetValue(setName, out T2IModelHandler handler))
        {
            problem = $"SwarmUI has no '{setName}' model folder configured.";
            return null;
        }
        string path = $"{handler.DownloadFolderPath}/{known.FileName}";
        if (File.Exists(path))
        {
            return path;
        }
        // May be under a different model root, or listed under a name without the extension.
        T2IModel listed = handler.GetModel(known.FileName);
        if (listed?.RawFilePath is not null && File.Exists(listed.RawFilePath))
        {
            return listed.RawFilePath;
        }
        if (!download)
        {
            return null;
        }
        HartsyLogs.Info($"LTX-2.5: '{known.DisplayName}' is missing, so downloading it to '{path}' through "
            + "SwarmUI's known-model registry.");
        known.DownloadNow(session: input?.SourceSession).Wait();
        Program.RefreshAllModelSets();
        if (!File.Exists(path))
        {
            problem = $"the download of '{known.DisplayName}' did not produce '{path}'.";
            return null;
        }
        return path;
    }

    /// <summary>Resolves the Gemma-4 tower: the user's pick, else any LTX-2.5 Gemma-4 already in the text-encoder folder, else core's downloadable build. A present build wins over a download, since an existing install usually carries the Engine's own repack.</summary>
    private static string ResolveLtx25Gemma(T2IParamInput input, bool download, out string problem)
    {
        problem = null;
        if (PickedPath(input?.Get(T2IParamTypes.GemmaModel)) is string picked)
        {
            return picked;
        }
        List<T2IModel> present = FindInSet("Clip", IsLtx25GemmaName);
        if (present.Count > 0)
        {
            // Prefer the Engine's repack over core's when both are staged: an existing install generates with it.
            T2IModel chosen = present.FirstOrDefault(m =>
                Path.GetFileName(m.RawFilePath).Contains("lean", StringComparison.OrdinalIgnoreCase)) ?? present[0];
            return chosen.RawFilePath;
        }
        if (!Program.T2IModelSets.TryGetValue("Clip", out T2IModelHandler handler))
        {
            problem = "SwarmUI has no text-encoder ('Clip') model folder configured.";
            return null;
        }
        string path = $"{handler.DownloadFolderPath}/{Ltx25GemmaFileName}";
        if (File.Exists(path))
        {
            return path;
        }
        if (!download)
        {
            return null;
        }
        HartsyLogs.Info($"LTX-2.5: the Gemma-4 text encoder is missing, so downloading it to '{path}'.");
        Utilities.DownloadFile(Ltx25GemmaUrl, path, null, verifyHash: Ltx25GemmaHash, session: input?.SourceSession).Wait();
        Program.RefreshAllModelSets();
        if (!File.Exists(path))
        {
            problem = $"the Gemma-4 text-encoder download did not produce '{path}'.";
            return null;
        }
        return path;
    }

    /// <summary>Resolves the video VAE: the user's <c>vae</c> parameter wins outright, else the registry's build for whichever decoder the Engine uses.</summary>
    private static string ResolveLtx25VideoVae(T2IParamInput input, bool download, out string problem)
    {
        problem = null;
        if (PickedPath(input?.Get(T2IParamTypes.VAE)) is string picked)
        {
            return picked;
        }
        if (WantDiffusionVae())
        {
            // Core registers only the conv build, so the diffusion one is resolved by name from the VAE folder.
            List<T2IModel> diffusion = FindInSet("VAE", n => n.Contains("ltx-2.5-video-vae", StringComparison.OrdinalIgnoreCase)
                && !n.Contains("conv", StringComparison.OrdinalIgnoreCase));
            if (diffusion.Count > 0)
            {
                return diffusion[0].RawFilePath;
            }
            problem = "HARTSY_LTX2_DIFFUSION_VAE is set but 'ltx-2.5-video-vae-bf16.safetensors' isn't in your VAE "
                + "folder. Download it there, or unset the variable to use the convolutional VAE (which SwarmUI can "
                + "fetch automatically).";
            return null;
        }
        return EnsureKnownModel(Ltx25VideoVaeKnownId, "VAE", download, input, out problem);
    }

    /// <summary>Resolves all four parts. With <paramref name="download"/> false this checks whether each part is present or fetchable; with it true, missing parts are fetched.</summary>
    private static bool TryPlanLtx25Parts(T2IModel model, T2IParamInput input, bool download, out Ltx25Parts parts, out string problem)
    {
        parts = null;
        string videoVae = ResolveLtx25VideoVae(input, download, out problem);
        if (problem is not null)
        {
            return false;
        }
        string audioVae = EnsureKnownModel(Ltx25AudioVaeKnownId, "VAE", download, input, out problem);
        if (problem is not null)
        {
            return false;
        }
        string gemma = ResolveLtx25Gemma(input, download, out problem);
        if (problem is not null)
        {
            return false;
        }
        List<string> missing = [];
        if (videoVae is null)
        {
            missing.Add("the video VAE");
        }
        if (audioVae is null)
        {
            missing.Add("the audio VAE");
        }
        if (gemma is null)
        {
            missing.Add("the Gemma-4 text encoder");
        }
        if (missing.Count > 0)
        {
            problem = $"SwarmUI could not resolve {string.Join(", ", missing)} for LTX-2.5.";
            return false;
        }
        parts = new Ltx25Parts(model.RawFilePath, videoVae, audioVae, gemma);
        return true;
    }

    /// <summary>Links the resolved parts into a private directory under <c>Data/</c> and returns it. The directory is named by a hash of the resolved paths, so the Engine's path-keyed pipeline cache cannot return a pipeline built for other files.</summary>
    private static string StageLtx25(Ltx25Parts parts)
    {
        string joined = string.Join('|', parts.Dit, parts.VideoVae, parts.AudioVae, parts.Gemma);
        string key = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(joined)))[..16].ToLowerInvariant();
        string dir = Path.Combine(Program.DataDir, "HartsyInference", "ltx2.5", key);
        lock (_ltx25StageLock)
        {
            Directory.CreateDirectory(dir);
            foreach (string target in new[] { parts.Dit, parts.VideoVae, parts.AudioVae, parts.Gemma })
            {
                string full = Path.GetFullPath(target);
                string link = Path.Combine(dir, Path.GetFileName(full));
                FileInfo info = new FileInfo(link);
                // LinkTarget catches a dangling link, which Exists reports as absent but CreateSymbolicLink still
                // refuses to overwrite.
                if (info.Exists || info.LinkTarget is not null)
                {
                    info.Delete();
                }
                File.CreateSymbolicLink(link, full);
            }
        }
        return dir;
    }

    /// <summary>Validation-time check: can this LTX-2.5 model run? True when a complete bundle sits beside the DiT, or every side model resolves through SwarmUI. Nothing is downloaded here.</summary>
    /// <remarks>The Engine's <c>LtxVideo2Recipe</c> takes exactly ONE path and re-buckets the merged keys under it. Given the bare DiT it finds no side-model keys and silently falls back to LTX-<b>2.3</b>'s Gemma 3 and VAEs, producing a video rather than an error. That is why this check and <see cref="ResolveLtx25LoadPath"/> refuse up front.</remarks>
    public static bool TryResolveLtx25Bundle(T2IModel model, out string directory, out string problem)
    {
        directory = null;
        problem = null;
        string checkpoint = model?.RawFilePath;
        if (string.IsNullOrEmpty(checkpoint))
        {
            problem = "the model has no resolved file path.";
            return false;
        }
        string dir = Path.GetDirectoryName(checkpoint);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            problem = $"'{checkpoint}' has no containing directory.";
            return false;
        }
        if (LooksLikeBundleDir(model, dir) && ScanCached(dir) is null)
        {
            directory = dir;
            return true;
        }
        return TryPlanLtx25Parts(model, null, download: false, out _, out problem);
    }

    /// <summary>Whether a bundle could sit beside the DiT, i.e. the DiT is in a subfolder of its model root. A checkpoint directly in a model root is never bundle-scanned, since the scan header-probes every neighbouring safetensors.</summary>
    private static bool LooksLikeBundleDir(T2IModel model, string directory)
    {
        string root = model?.OriginatingFolderPath;
        if (string.IsNullOrEmpty(root) || string.IsNullOrEmpty(directory))
        {
            return false;
        }
        string normRoot = Path.GetFullPath(root).Replace('\\', '/').TrimEnd('/');
        string normDir = Path.GetFullPath(directory).Replace('\\', '/').TrimEnd('/');
        return !string.Equals(normRoot, normDir, StringComparison.Ordinal);
    }

    /// <summary>The single path the Engine's LTX-2.5 recipe is handed: an existing complete bundle beside the DiT if there is one, else a staging directory linking the parts SwarmUI resolved and downloaded.</summary>
    private static string ResolveLtx25LoadPath(T2IModel model, T2IParamInput input)
    {
        string checkpoint = model?.RawFilePath
            ?? throw new SwarmUserErrorException("HartsyInference: the LTX-2.5 model has no resolved file path.");
        string dir = Path.GetDirectoryName(checkpoint);
        bool userPickedVae = PickedPath(input?.Get(T2IParamTypes.VAE)) is not null;
        // A complete bundle beside the DiT still wins, unless the user picked a VAE, which a bundle would silently ignore.
        if (!userPickedVae && !string.IsNullOrEmpty(dir) && Directory.Exists(dir) && LooksLikeBundleDir(model, dir)
            && ScanCached(dir) is null)
        {
            HartsyLogs.Debug($"LTX-2.5: loading the complete bundle already staged at '{dir}'.");
            return dir;
        }
        if (!TryPlanLtx25Parts(model, input, download: true, out Ltx25Parts parts, out string problem))
        {
            throw new SwarmUserErrorException($"HartsyInference: LTX-2.5 side models can't be resolved: {problem}");
        }
        string staged = StageLtx25(parts);
        // Re-scanned over what the Engine will read, so a staging bug fails here rather than reaching the recipe's LTX-2.3 fallback.
        if (ScanCached(staged) is string staleProblem)
        {
            throw new SwarmUserErrorException("HartsyInference: LTX-2.5 side models resolved but the staged set is "
                + $"not loadable: {staleProblem}");
        }
        HartsyLogs.Debug($"LTX-2.5 resolved through SwarmUI: video VAE '{Path.GetFileName(parts.VideoVae)}', "
            + $"audio VAE '{Path.GetFileName(parts.AudioVae)}', text encoder '{Path.GetFileName(parts.Gemma)}' "
            + $"(linked into '{staged}').");
        return staged;
    }

    /// <summary>Header-probes every safetensors file beside the checkpoint and reports what is missing. Headers are probed rather than filenames, since repacks rename freely.</summary>
    private static string ScanLtx25Bundle(string directory, string[] files)
    {
        List<string> allKeys = [];
        bool gemma4 = false, convVae = false, audioVae = false;
        foreach (string file in files)
        {
            IReadOnlyCollection<string> keys;
            try
            {
                using SafeTensorsLoader loader = new SafeTensorsLoader();
                loader.Load(file);
                keys = [.. loader.Descriptors.Keys];
            }
            catch (Exception ex)
            {
                // An unparseable neighbour is not fatal, since the Engine skips it too.
                HartsyLogs.Verbose($"LTX-2.5 bundle scan skipped '{Path.GetFileName(file)}': {ex.Message}");
                continue;
            }
            allKeys.AddRange(keys);
            foreach (string key in keys)
            {
                gemma4 |= key == Gemma4SignatureKey;
                // The conv decoder's marker, shared with 2.3's VAE; the diffusion decoder does NOT carry it.
                // Matched exactly (bare or under the Engine's "vae." prefix) rather than by suffix: the AUDIO VAE
                // carries the same tail under "audio_vae.", so a suffix match would call a bundle complete when only
                // the audio VAE is staged.
                convVae |= key == ConvVideoVaeSignatureKey || key == $"vae.{ConvVideoVaeSignatureKey}";
                audioVae |= key.StartsWith("audio_vae.", StringComparison.Ordinal)
                    || key.Contains("vocoder.mel_stft.mel_basis", StringComparison.Ordinal);
            }
        }
        // Asked of the Engine's whole-file rule rather than re-derived: both decoders share decoder.conv_in, so this cannot be answered per key.
        // The Engine decodes with either decoder; only the ambiguity is fatal.
        bool diffusionVae = LtxVideo2CheckpointConverter.IsDiffusionVideoVae(allKeys);
        if (convVae && diffusionVae)
        {
            return $"'{directory}' stages both LTX-2.5 video VAEs. The diffusion/conv question is one boolean over the "
                + "merged key set, so staging both corrupts the selection for either; keep exactly one of "
                + "'ltx-2.5-video-vae-conv-bf16.safetensors' or 'ltx-2.5-video-vae-bf16.safetensors'.";
        }
        List<string> missing = [];
        if (!gemma4)
        {
            missing.Add("the Gemma-4 text encoder (gemma4-12b-with-proj-ltx-2.5-*.safetensors)");
        }
        if (!convVae && !diffusionVae)
        {
            missing.Add("a video VAE (ltx-2.5-video-vae-conv-bf16.safetensors, or "
                + "ltx-2.5-video-vae-bf16.safetensors with HARTSY_LTX2_DIFFUSION_VAE=1)");
        }
        if (!audioVae)
        {
            missing.Add("the audio VAE (ltx-2.5-audio-vae-bf16.safetensors)");
        }
        if (missing.Count > 0)
        {
            return $"'{directory}' is missing {string.Join(", ", missing)}. LTX-2.5 ships split across four files and "
                + "must be loaded as a folder: put the DiT, text encoder and both VAEs (or symlinks to them) in one "
                + "directory. Without them the engine would silently fall back to LTX-2.3's Gemma 3 and 2.3 VAEs.";
        }
        return null;
    }

    /// <summary>Builds the Engine's load request for a Swarm model: the family id travels as the catalog id (what the Engine keys its recipe registry on), and the checkpoint path as the resolved local path.</summary>
    /// <param name="input">The request, when there is one, so a user-picked VAE / text encoder is honoured. Optional
    /// only so callers that merely describe a model (rather than run it) need not supply one.</param>
    /// <param name="stageBundles">False skips staging split checkpoints (LTX-2.5) into a bundle directory. A complete bundle beside the DiT is still used, and otherwise the spec carries no path. For callers that only inspect the checkpoint, such as memory routing, which must not rewrite links a running generation reads or size the bare DiT with LTX-2.3's side models.</param>
    public static ModelSpec BuildSpec(SwarmUI.Text2Image.T2IModel model, Family family, T2IParamInput input = null,
        bool stageBundles = true)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(family);
        Modality modality = family.Kind switch
        {
            Kind.Video => Modality.Video,
            Kind.Music => Modality.Music,
            _ => Modality.Image,
        };
        string familyId = family.Id;
        if (familyId == "minimaxmusic3")
        {
            familyId = MiniMaxMusicFamily(model, input);
        }
        string localPath = model.RawFilePath;
        if (!stageBundles && IsLtx25(model))
        {
            localPath = TryResolveLtx25Bundle(model, out string bundle, out _) && bundle is not null ? bundle : null;
        }
        else if (IsLtx25(model))
        {
            // LTX-2.5 ships split, and the Engine takes ONE path. This resolves the side models through SwarmUI and passes a directory holding exactly the right set. It throws rather than let the recipe load 2.3 components under a 2.5 name.
            localPath = ResolveLtx25LoadPath(model, input);
        }
        return new ModelSpec
        {
            Requested = familyId,
            Modality = modality,
            LocalPath = localPath,
            // The model class carries what the checkpoint cannot: a user-picked or modelspec-stamped variant
            // (qwen-image-edit). The engine ignores a class that names no variant of the family.
            Variant = model.ModelClass?.ID,
            Catalog = new CatalogEntry
            {
                Id = familyId,
                Modality = modality,
                DisplayName = model.Name,
                Architecture = familyId,
                Status = ModelStatus.Verified,
            },
        };
    }

    /// <summary>Picks MiniMax Music 3's language-model precision: the family id carries it as the selector variant (<c>minimaxmusic3:q8</c>), which the engine's descriptor maps to its GGUF-quantized language-model path.</summary>
    /// <remarks>An explicit <c>q8</c> or <c>q4</c> on the LM-precision param wins. Otherwise the picked model decides: AudioLab offers one checkpoint as <c>base</c>, <c>q8</c> and <c>q4</c> entries. Those entries have no file, so the name's last segment is matched (<c>Audio Models/MiniMaxMusic3/q4</c>); a real file's stem is read the same way. The param defaults to <c>bf16</c>, so the default must not override the picked entry.</remarks>
    private static string MiniMaxMusicFamily(SwarmUI.Text2Image.T2IModel model, T2IParamInput input)
    {
        if (input is not null
            && input.TryGet(HartsyInference.MiniMaxMusicLmPrecisionParam, out string lmPrecision)
            && lmPrecision is "q8" or "q4")
        {
            return $"minimaxmusic3:{lmPrecision}";
        }
        string stem = Path.GetFileNameWithoutExtension(model.RawFilePath ?? model.Name ?? "").ToLowerInvariant();
        foreach (string precision in new[] { "q8", "q4" })
        {
            if (stem == precision
                || stem.EndsWith($"-{precision}", StringComparison.Ordinal)
                || stem.EndsWith($"_{precision}", StringComparison.Ordinal)
                || stem.Contains(precision == "q8" ? "q8_0" : "q4_k", StringComparison.Ordinal))
            {
                return $"minimaxmusic3:{precision}";
            }
        }
        return "minimaxmusic3";
    }

    /// <summary>No-op: the Engine's registries self-register. Retained for the extension entry point's call order.</summary>
    public static void RegisterBuiltins()
    {
    }
}
