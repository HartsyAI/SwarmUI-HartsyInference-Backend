using System.IO;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

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
        // Both compat classes route to Kandinsky5VideoRecipe, which now weight-derives Lite vs. Pro from the
        // loaded transformer's model_dim (see Kandinsky5VideoRecipe.DetectConfig) — mapping "vidpro" here was
        // unsafe before that detection existed (it would have silently constructed the Lite-2B architecture).
        ["kandinsky5-vidlite"] = new("kandinsky5-video", Kind.Video),
        ["kandinsky5-vidpro"] = new("kandinsky5-video", Kind.Video),
        [ModelClassRegistrations.LanceVideoCompatClassId] = new("lance-video", Kind.Video),
        // Core owns the minimax-h3 compat class (T2IModelClassSorter, "MiniMax H3 support" #1469) and shares it
        // with the video/audio VAE classes, exactly as LTX-2 does — this maps it, it must not re-register it.
        // Keyed off core's own registration rather than a repeated literal, so a rename there cannot leave this
        // table silently matching nothing (which reads as "architecture not supported", not as a typo).
        [T2IModelClassSorter.CompatMiniMaxH3.ID] = new("minimax-h3", Kind.Video),

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
            HartsyInference.Engine.Recipes.Video.LtxVideoRecipe ltx => ltx.SupportsFor(checkpointPath),
            _ when family.Id == "minimax-h3" => MiniMaxH3TaskFeatures(recipe.Supports, checkpointPath),
            _ => recipe.Supports,
        };
    }

    /// <summary>MiniMax-H3 ships fl2va (first/last-frame) and ref2va (references) as SEPARATE checkpoints, and the
    /// recipe declares the union because it serves both. Narrow that to the task this checkpoint actually does, or a
    /// reference attached to an fl2va model is accepted and quietly ignored — which reads as the model disregarding
    /// the reference rather than as the wrong checkpoint being loaded.
    /// <para>By FILENAME, unavoidably: the two builds have byte-identical key sets and tensor shapes (verified
    /// 2026-08-08 across all 1082 tensors of both pruned fp8 builds), so no header sniff can tell them apart the way
    /// <c>WanVideoRecipe.SupportsFor</c> can for Wan's variants. A name matching neither keeps the full union rather
    /// than guessing — refusing on an unrecognized name would break renamed-but-valid checkpoints.</para></summary>
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

    /// <summary>Core's model class id for LTX-2.5. Read directly — and uniquely in this extension — because the
    /// compat class deliberately cannot answer this: core header-sniffs LTX-2 / 2.3 / 2.5 into three distinct model
    /// classes that all share <c>lightricks-ltx-video-2</c>, and only 2.5 ships as a split bundle.</summary>
    public const string Ltx25ModelClassId = "lightricks-ltx-video-2-5";

    /// <summary>Whether the selected model is LTX-2.5, by core's own header detection rather than a name guess.</summary>
    private static bool IsLtx25(T2IModel model) => model?.ModelClass?.ID == Ltx25ModelClassId;

    /// <summary>The Gemma-4 tower's discriminator. Its per-layer <c>layer_scalar</c> is the key the Engine's own
    /// converter keys on — NOT a missing <c>v_proj</c>, which the global layers also lack in Gemma 3.</summary>
    private const string Gemma4SignatureKey = "model.layers.0.layer_scalar";

    /// <summary>The convolutional video decoder's marker. Shared with LTX-2.3's VAE, and absent from the 2.5
    /// diffusion decoder — which is what makes it usable as the "is the conv VAE here" test.</summary>
    private const string ConvVideoVaeSignatureKey = "decoder.conv_in.conv.bias";

    /// <summary>Cached bundle scans, keyed by directory. Invalidated on the directory's file-set signature so a
    /// staged file appearing mid-session is picked up without re-reading a 21 GB header on every request.</summary>
    private static readonly ConcurrentDictionary<string, (string Signature, string Problem)> _ltx25Scans = new(StringComparer.Ordinal);

    /// <summary>Header-scans <paramref name="directory"/>, memoised on its file-set signature.</summary>
    private static string ScanCached(string directory)
    {
        string[] files = Directory.GetFiles(directory, "*.safetensors", SearchOption.AllDirectories);
        // Cheap signature first — computing it costs a stat per file, where a rescan costs a header parse per file.
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
    // The model the user picks in the UI is the DiT alone. The video VAE and audio VAE come from the VAE folder and
    // the Gemma-4 tower from the text-encoder folder, each resolved through core's own registry and downloaded into
    // the folder core assigns when absent — exactly what Builtin_ComfyUIBackend does. The Engine's LtxVideo2Recipe
    // takes exactly ONE path though (it globs a directory and re-buckets the merged keys, and honours no
    // RecipeContext.Components override yet), so the resolved parts are linked into a staging directory and THAT is
    // what the Engine is handed. The staging directory lives under Data/ rather than a model root, so Swarm never
    // indexes the links as models of their own.
    // ────────────────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Core's known-model ids for LTX-2.5's two VAEs (registered in <c>CommonModels.RegisterCoreSet</c>), so
    /// the URL, hash and destination folder live in one place — core's registry — for both backends.</summary>
    private const string Ltx25VideoVaeKnownId = "ltx2-5-video-vae";

    /// <summary>Companion of <see cref="Ltx25VideoVaeKnownId"/> for the audio VAE. No user param selects an audio VAE,
    /// so this is always the registry's answer.</summary>
    private const string Ltx25AudioVaeKnownId = "ltx2-5-audio-vae";

    /// <summary>The Gemma-4 tower. Deliberately NOT read from <c>CommonModels</c>: core doesn't register it there —
    /// its ComfyUI backend fetches it inline via <c>RequireClipModel</c> — so the URL and hash are mirrored from that
    /// call (<c>WorkflowGeneratorModelSupport.GetLTX25Gemma4Model</c>) rather than invented here.</summary>
    private const string Ltx25GemmaFileName = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";

    private const string Ltx25GemmaUrl = "https://huggingface.co/mcmonkey/swarm-models/resolve/main/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";

    private const string Ltx25GemmaHash = "09a89e084de1a149c3de60cfe9dfd3e5161967eb09eea39e806fcdeffdd568de";

    /// <summary>Serialises staging so two concurrent generations can't half-build the same link set.</summary>
    private static readonly object _ltx25StageLock = new();

    /// <summary>The four files an LTX-2.5 run needs, as absolute paths.</summary>
    private sealed record Ltx25Parts(string Dit, string VideoVae, string AudioVae, string Gemma);

    /// <summary>Matches any LTX-2.5 Gemma-4 build already in the text-encoder folder, whatever it was repacked as —
    /// the Engine's own <c>int8_lean_convrot</c> build and core's <c>comfy-int8-convrot</c> build are both valid.</summary>
    private static bool IsLtx25GemmaName(string name) =>
        name.Contains("gemma4", StringComparison.OrdinalIgnoreCase)
        && name.Contains("ltx-2.5", StringComparison.OrdinalIgnoreCase);

    /// <summary>A user's component pick, or null when there isn't one. "None" is a real selectable entry that arrives
    /// as a model of that name and means "use the default" — core's <c>DoVaeLoader</c> nulls it the same way, and
    /// without this it would read as a pick and push a complete bundle onto the staging path for nothing.</summary>
    private static string PickedPath(T2IModel picked)
    {
        if (picked?.RawFilePath is null || picked.Name == "None" || !File.Exists(picked.RawFilePath))
        {
            return null;
        }
        return picked.RawFilePath;
    }

    /// <summary>Whether the Engine will decode with the diffusion video VAE. Read here as well as in the Engine so the
    /// staged VAE is the one the recipe will actually ask for — staging both is what corrupts the selection.</summary>
    private static bool WantDiffusionVae()
    {
        string raw = Environment.GetEnvironmentVariable("HARTSY_LTX2_DIFFUSION_VAE");
        return raw is not null && (raw == "1" || raw.Equals("true", StringComparison.OrdinalIgnoreCase)
            || raw.Equals("yes", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>Matching models in <paramref name="setName"/>, ordered by file name. Ordered because the model set is
    /// a concurrent dictionary: taking its first match would pick a different file run to run whenever a user has
    /// more than one valid build of the same component staged.</summary>
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

    /// <summary>Resolves a core known model to its on-disk path, downloading it into the folder core assigns when it
    /// isn't there. <paramref name="download"/> is false at validation time, where the question is only whether the
    /// registration exists at all — the fetch itself belongs at load, same as the ComfyUI backend.</summary>
    private static string EnsureKnownModel(string knownId, string setName, bool download, out string problem)
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
            // Registered, so it IS resolvable — validation is satisfied without paying for the download.
            return null;
        }
        Logs.Info($"[HartsyInference] LTX-2.5: '{known.DisplayName}' is missing — downloading it to '{path}' through "
            + "SwarmUI's known-model registry.");
        known.DownloadNow().Wait();
        Program.RefreshAllModelSets();
        if (!File.Exists(path))
        {
            problem = $"the download of '{known.DisplayName}' did not produce '{path}'.";
            return null;
        }
        return path;
    }

    /// <summary>Resolves the Gemma-4 tower: the user's pick, else any LTX-2.5 Gemma-4 already in the text-encoder
    /// folder, else core's downloadable build. An already-present build wins over a download deliberately — an
    /// existing install usually carries the Engine's own repack, and swapping it would change the text encoder
    /// silently.</summary>
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
            // The Engine's own repack is preferred over core's when both are staged — it is what an existing install
            // is already generating with, so picking the other one would change the text encoder without saying so.
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
        Logs.Info($"[HartsyInference] LTX-2.5: the Gemma-4 text encoder is missing — downloading it to '{path}'.");
        Utilities.DownloadFile(Ltx25GemmaUrl, path, null, verifyHash: Ltx25GemmaHash).Wait();
        Program.RefreshAllModelSets();
        if (!File.Exists(path))
        {
            problem = $"the Gemma-4 text-encoder download did not produce '{path}'.";
            return null;
        }
        return path;
    }

    /// <summary>Resolves the video VAE: the user's <c>vae</c> parameter wins outright, else the registry's build for
    /// whichever decoder the Engine is set to use.</summary>
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
        return EnsureKnownModel(Ltx25VideoVaeKnownId, "VAE", download, out problem);
    }

    /// <summary>Resolves all four parts. With <paramref name="download"/> false this is the validation-time question
    /// ("is each part present or fetchable?"); with it true, the missing ones are actually fetched.</summary>
    private static bool TryPlanLtx25Parts(T2IModel model, T2IParamInput input, bool download, out Ltx25Parts parts, out string problem)
    {
        parts = null;
        string videoVae = ResolveLtx25VideoVae(input, download, out problem);
        if (problem is not null)
        {
            return false;
        }
        string audioVae = EnsureKnownModel(Ltx25AudioVaeKnownId, "VAE", download, out problem);
        if (problem is not null)
        {
            return false;
        }
        string gemma = ResolveLtx25Gemma(input, download, out problem);
        if (problem is not null)
        {
            return false;
        }
        if (!download)
        {
            // Nulls here mean "registered but not downloaded yet", which is fine — nothing left to check.
            return true;
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

    /// <summary>Links the resolved parts into a private directory under <c>Data/</c> and returns it. Named by a hash
    /// of the resolved paths so switching the <c>vae</c> parameter yields a different directory — which is also what
    /// keeps the Engine's path-keyed pipeline cache from serving a pipeline built with the previous pick.</summary>
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

    /// <summary>Validation-time check: can this LTX-2.5 model be run? True when either a complete hand-assembled
    /// bundle sits beside the DiT, or every side model resolves through SwarmUI. Nothing is downloaded here.</summary>
    /// <remarks>The Engine's <c>LtxVideo2Recipe</c> takes exactly ONE path and re-buckets the merged keys of
    /// everything under it. Handing it the bare DiT is not a soft failure: the recipe finds no video-VAE and no
    /// text-encoder keys and falls through to its LTX-<b>2.3</b> side-model resolver, silently fetching Gemma 3 and
    /// the 2.3 VAEs and generating with them. That produces a video rather than an error, which is why both this and
    /// <see cref="ResolveLtx25LoadPath"/> refuse up front instead.</remarks>
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

    /// <summary>Whether a hand-assembled bundle could plausibly live beside the DiT — i.e. the DiT is in a SUBFOLDER
    /// of its model root, not sitting directly in the root itself. A bare checkpoint in a model root (where
    /// <c>diffusion_models</c> keeps them) is never bundle-scanned: that scan header-probes every neighbouring
    /// safetensors, which over a model root means opening every checkpoint the user owns.</summary>
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

    /// <summary>The single path the Engine's LTX-2.5 recipe is handed: a pre-existing complete bundle beside the DiT
    /// if there is one, else a staging directory linking the parts SwarmUI resolved and downloaded.</summary>
    private static string ResolveLtx25LoadPath(T2IModel model, T2IParamInput input)
    {
        string checkpoint = model?.RawFilePath
            ?? throw new SwarmUserErrorException("HartsyInference: the LTX-2.5 model has no resolved file path.");
        string dir = Path.GetDirectoryName(checkpoint);
        bool userPickedVae = PickedPath(input?.Get(T2IParamTypes.VAE)) is not null;
        // A complete bundle beside the DiT still wins, so installs that already have one keep working unchanged —
        // except when the user explicitly picked a VAE, which a bundle would silently ignore.
        if (!userPickedVae && !string.IsNullOrEmpty(dir) && Directory.Exists(dir) && LooksLikeBundleDir(model, dir)
            && ScanCached(dir) is null)
        {
            Logs.Verbose($"[HartsyInference] LTX-2.5: loading the complete bundle already staged at '{dir}'.");
            return dir;
        }
        if (!TryPlanLtx25Parts(model, input, download: true, out Ltx25Parts parts, out string problem))
        {
            throw new SwarmUserErrorException($"HartsyInference: LTX-2.5 side models can't be resolved — {problem}");
        }
        string staged = StageLtx25(parts);
        // Re-scanned over what the Engine will actually read. A staging bug fails loudly HERE rather than reaching
        // the recipe's LTX-2.3 fallback, which would generate a plausible video with the wrong model.
        if (ScanCached(staged) is string staleProblem)
        {
            throw new SwarmUserErrorException("HartsyInference: LTX-2.5 side models resolved but the staged set is "
                + $"not loadable — {staleProblem}");
        }
        Logs.Info($"[HartsyInference] LTX-2.5 resolved through SwarmUI: video VAE '{Path.GetFileName(parts.VideoVae)}', "
            + $"audio VAE '{Path.GetFileName(parts.AudioVae)}', text encoder '{Path.GetFileName(parts.Gemma)}' "
            + $"(linked into '{staged}').");
        return staged;
    }

    /// <summary>Header-probes every safetensors file beside the checkpoint and reports the first missing companion.
    /// Probes headers rather than filenames because repacks rename freely — the same reason the Engine refuses to
    /// infer the distilled schedule from a filename.</summary>
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
                // A neighbouring file we can't parse isn't fatal — the Engine would skip it too. Say so and move on.
                Logs.Verbose($"[HartsyInference] LTX-2.5 bundle scan skipped '{Path.GetFileName(file)}': {ex.Message}");
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
        // Asked of the Engine's own whole-file rule rather than re-derived: both decoders share decoder.conv_in,
        // so this cannot be answered per-key. The Engine decodes with EITHER decoder now; only the ambiguity is fatal.
        bool diffusionVae = LtxVideo2CheckpointConverter.IsDiffusionVideoVae(allKeys);
        if (convVae && diffusionVae)
        {
            return $"'{directory}' stages both LTX-2.5 video VAEs. The diffusion/conv question is one boolean over the "
                + "merged key set, so staging both corrupts the selection for either — keep exactly one of "
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
                + "must be loaded as a folder — put the DiT, text encoder and both VAEs (or symlinks to them) in one "
                + "directory. Without them the engine would silently fall back to LTX-2.3's Gemma 3 and 2.3 VAEs.";
        }
        return null;
    }

    /// <summary>Builds the Engine's load request for a Swarm model: the family id travels as the catalog id (that is
    /// what the Engine keys its recipe registry on), and the checkpoint path as the resolved local path.</summary>
    /// <param name="input">The request, when there is one, so a user-picked VAE / text encoder is honoured. Optional
    /// only so callers that merely describe a model (rather than run it) need not supply one.</param>
    public static ModelSpec BuildSpec(SwarmUI.Text2Image.T2IModel model, Family family, T2IParamInput input = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(family);
        Modality modality = family.Kind switch
        {
            Kind.Video => Modality.Video,
            Kind.Music => Modality.Music,
            _ => Modality.Image,
        };
        string localPath = model.RawFilePath;
        if (IsLtx25(model))
        {
            // LTX-2.5 ships split, and the Engine takes ONE path. This resolves the side models through SwarmUI and
            // hands over a directory holding exactly the right set — throwing rather than letting the recipe load
            // 2.3 components under a 2.5 name.
            localPath = ResolveLtx25LoadPath(model, input);
        }
        return new ModelSpec
        {
            Requested = family.Id,
            Modality = modality,
            LocalPath = localPath,
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
