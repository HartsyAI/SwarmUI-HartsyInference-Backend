using System.IO;
using System.Collections.Concurrent;
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

    /// <summary>Resolves the directory an LTX-2.5 bundle must be loaded from, or explains why it can't be.</summary>
    /// <remarks>LTX-2.5 ships split across four files and the Engine's <c>LtxVideo2Recipe</c> takes exactly ONE path,
    /// recursively globbing a directory and re-bucketing the merged keys. Handing it the bare DiT instead is not a
    /// soft failure: the recipe finds no video-VAE and no text-encoder keys and falls through to its LTX-<b>2.3</b>
    /// side-model resolver, silently downloading Gemma 3 and the 2.3 VAEs and generating with them. That produces a
    /// video rather than an error, so this refuses up front instead.</remarks>
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
        directory = Path.GetDirectoryName(checkpoint);
        if (string.IsNullOrEmpty(directory) || !Directory.Exists(directory))
        {
            problem = $"'{checkpoint}' has no containing directory to load the bundle from.";
            return false;
        }
        string[] files = Directory.GetFiles(directory, "*.safetensors", SearchOption.AllDirectories);
        // Cheap signature first — computing it costs a stat per file, where a rescan costs a header parse per file.
        string signature = $"{files.Length}|{files.Select(f => new FileInfo(f).LastWriteTimeUtc.Ticks).DefaultIfEmpty(0L).Max()}";
        if (_ltx25Scans.TryGetValue(directory, out (string Signature, string Problem) cached) && cached.Signature == signature)
        {
            problem = cached.Problem;
            return problem is null;
        }
        problem = ScanLtx25Bundle(directory, files);
        _ltx25Scans[directory] = (signature, problem);
        return problem is null;
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
        // so this cannot be answered per-key.
        if (LtxVideo2CheckpointConverter.IsDiffusionVideoVae(allKeys))
        {
            return $"'{directory}' contains the LTX-2.5 diffusion video VAE, which the engine's LTX-2 pipeline does "
                + "not decode with yet. Stage 'ltx-2.5-video-vae-conv-bf16.safetensors' instead and remove "
                + "'ltx-2.5-video-vae-bf16.safetensors' from the folder.";
        }
        List<string> missing = [];
        if (!gemma4)
        {
            missing.Add("the Gemma-4 text encoder (gemma4-12b-with-proj-ltx-2.5-*.safetensors)");
        }
        if (!convVae)
        {
            missing.Add("the conv video VAE (ltx-2.5-video-vae-conv-bf16.safetensors)");
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
        string localPath = model.RawFilePath;
        if (IsLtx25(model))
        {
            // Backstop: ValidateVideo already refused this case with the same message, so reaching here means the
            // request bypassed validation. Throwing beats loading 2.3 components under a 2.5 name.
            if (!TryResolveLtx25Bundle(model, out string bundleDir, out string problem))
            {
                throw new SwarmUserErrorException($"HartsyInference: LTX-2.5 bundle can't be loaded — {problem}");
            }
            localPath = bundleDir;
            Logs.Verbose($"[HartsyInference] LTX-2.5 split bundle — loading the folder '{bundleDir}' rather than the "
                + "bare checkpoint, so the Gemma-4 tower and both VAEs are merged in.");
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
