using System;
using System.IO;
using System.Linq;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using SwarmUI.Text2Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// SwarmUI <see cref="T2IModelClass"/> / <see cref="T2IModelCompatClass"/> registrations for architectures core
/// doesn't classify on its own. This is pure SwarmUI glue — model-file <i>detection</i>, so a checkpoint lands in the
/// right compat class and lights up the right param groups. It used to live inside the per-architecture
/// <c>*Loader.cs</c> files; the loaders themselves were lifted into <c>HartsyInference.Engine</c>'s recipes, but the
/// engine has no concept of a Swarm model class, so the registrations stay here.
/// <para>Called once from <c>SwarmUIHartsyInference.OnPreInit</c>, BEFORE model folders are scanned.</para>
/// </summary>
public static class ModelClassRegistrations
{
    /// <summary>ACE-Step v1 model class ID (v1 checkpoints share core's <c>ace-step-1_5</c> compat class).</summary>
    public const string AceStepV1ClassId = "ace-step-v1";

    /// <summary>Lance image compat class ID.</summary>
    public const string LanceCompatClassId = "lance";

    /// <summary>Lance video compat class ID.</summary>
    public const string LanceVideoCompatClassId = "lance-video";

    /// <summary>Lance text-to-image model class ID.</summary>
    public const string LanceT2IClassId = "lance-t2i";

    /// <summary>Lance text-to-video model class ID.</summary>
    public const string LanceT2VClassId = "lance-t2v";

    /// <summary>F-Lite compat + model class ID.</summary>
    public const string FLiteCompatClassId = "f-lite";

    /// <summary>Qwen-Image 2.1 compat + model class ID. Deliberately NOT core's <c>qwen-image</c> compat class:
    /// 2.1 shares neither the architecture nor the VAE family, and core's own <c>isQwenImage</c> requires
    /// <c>img_in.bias</c> and <c>add_k_proj.bias</c>, which a bias-free single-stream 2.1 checkpoint does not
    /// have — so core leaves it unclassified rather than mislabelling it.</summary>
    public const string QwenImage21CompatClassId = "qwen-image-2.1";

    /// <summary>Registers every extension-owned model class. Call EXACTLY once, at pre-init, and only for classes
    /// Swarm core does not already define: <c>T2IModelClassSorter.Register</c>/<c>RegisterCompat</c> are backed by
    /// <c>Dictionary.Add</c>, so re-registering an existing ID throws rather than overwriting.</summary>
    public static void RegisterAll()
    {
        RegisterAceStepV1();
        RegisterLance();
        // MusicGen and YuE moved to AudioLab ownership (they are not SwarmUI-native music classes); their
        // compat/model class registrations left with them. Core itself owns ace-step-1_5 and minimax-music-3.
        RegisterFLite();
        RegisterQwenImage21();
        // Mage-Flow: core registers the compat class and model class itself, and its class carries this same
        // compat id, so checkpoints it classifies already land on the ModelSupport row. Nothing to add here.
    }

    /// <summary>ACE-Step v1: core only registers the v1.5 class, so v1 checkpoints would otherwise be unclassified.
    /// Registered under core's <c>ace-step-1_5</c> compat class so the Text2Audio params light up.</summary>
    private static void RegisterAceStepV1()
    {
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = AceStepV1ClassId,
            CompatClass = T2IModelClassSorter.CompatAceStep15,
            Name = "ACE-Step v1",
            IsThisModelOfClass = (model, header) =>
                header.ContainsKey("lyric_embs.weight") || header.ContainsKey("model.lyric_embs.weight")
                // The artifact that actually ships: Comfy-Org/ACE-Step_ComfyUI_repackaged all_in_one.
                || header.ContainsKey("model.diffusion_model.lyric_embs.weight"),
        });
    }

    /// <summary>Lance (ByteDance 3B): core has no Lance classes at all. The checkpoints are folder models
    /// (sharded safetensors + <c>llm_config.json</c>); image and video variants ship byte-identical configs and are
    /// told apart by folder name.</summary>
    private static void RegisterLance()
    {
        T2IModelCompatClass compatImage = T2IModelClassSorter.RegisterCompat(new() { ID = LanceCompatClassId, ShortCode = "Lance", LorasTargetTextEnc = false });
        T2IModelCompatClass compatVideo = T2IModelClassSorter.RegisterCompat(new() { ID = LanceVideoCompatClassId, ShortCode = "LanceV", LorasTargetTextEnc = false, IsText2Video = true });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = LanceT2IClassId,
            CompatClass = compatImage,
            Name = "Lance 3B (image)",
            StandardWidth = 768,
            StandardHeight = 768,
            IsThisModelOfClass = (model, header) => IsLanceFolder(model, header, video: false),
        });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = LanceT2VClassId,
            CompatClass = compatVideo,
            Name = "Lance 3B (video)",
            StandardWidth = 832,
            StandardHeight = 480,
            IsThisModelOfClass = (model, header) => IsLanceFolder(model, header, video: true),
        });
    }



    /// <summary>F-Lite (Freepik 10B): detected by its cross-attention context projection.</summary>
    private static void RegisterFLite()
    {
        T2IModelCompatClass compat = T2IModelClassSorter.RegisterCompat(new() { ID = FLiteCompatClassId, ShortCode = "FLite" });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = FLiteCompatClassId,
            CompatClass = compat,
            Name = "F-Lite",
            StandardWidth = 1024,
            StandardHeight = 1024,
            IsThisModelOfClass = (model, header) =>
                header is not null && header.ContainsKey("blocks.0.cross_attn.context_kv.weight"),
        });
    }

    /// <summary>Qwen-Image 2.1 (single-stream DiT). ComfyUI's own signature from <c>model_detection.py</c>: the
    /// shared <c>modulation.1</c>, the zero-centered <c>txt_in.text_norm</c>, per-head QK norms, and the patch-1
    /// <c>img_in</c>/<c>proj_out</c> pair, plus an <c>img_mlp</c> in either the fused or split form. Checked with
    /// and without the <c>model.diffusion_model.</c> wrapper a repack may add.</summary>
    private static void RegisterQwenImage21()
    {
        T2IModelCompatClass compat = T2IModelClassSorter.RegisterCompat(
            new() { ID = QwenImage21CompatClassId, ShortCode = "Qwen2.1", LorasTargetTextEnc = false });
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = QwenImage21CompatClassId,
            CompatClass = compat,
            Name = "Qwen Image 2.1",
            StandardWidth = 1024,
            StandardHeight = 1024,
            IsThisModelOfClass = (model, header) => IsQwenImage21(header),
        });
    }

    private static bool IsQwenImage21(JObject header)
    {
        if (header is null)
        {
            return false;
        }
        bool has(string key) => header.ContainsKey(key) || header.ContainsKey("model.diffusion_model." + key);
        return has("txt_in.text_norm.weight")
            && has("modulation.1.weight")
            && has("transformer_blocks.0.attn.norm_q.weight")
            && has("img_in.weight")
            && has("proj_out.weight")
            && (has("transformer_blocks.0.img_mlp.gate_up.weight") || has("transformer_blocks.0.img_mlp.proj.weight"));
    }

    /// <summary>A Lance checkpoint is a folder with <c>llm_config.json</c> declaring the Qwen2.5-VL backbone and
    /// <c>language_model.*</c> transformer keys in its (first-shard) header. The video variant is told apart by
    /// folder name — both variants ship byte-identical configs.</summary>
    private static bool IsLanceFolder(T2IModel model, JObject header, bool video)
    {
        string folder = ResolveLanceFolder(model?.RawFilePath);
        if (folder is null)
        {
            return false;
        }
        string llmConfig = $"{folder}/llm_config.json";
        if (!File.Exists(llmConfig) || !File.ReadAllText(llmConfig).Contains("Qwen2_5_VL"))
        {
            return false;
        }
        if (header is null || !header.Properties().Any(p => p.Name.StartsWith("language_model.")))
        {
            return false;
        }
        bool isVideoVariant = folder.Replace('\\', '/').AfterLast('/').ToLowerInvariant().Contains("video");
        return video == isVideoVariant;
    }

    /// <summary>Maps a Swarm model path (the .safetensors file or the checkpoint folder itself) to the Lance
    /// checkpoint folder, or null when neither exists.</summary>
    private static string ResolveLanceFolder(string rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
        {
            return null;
        }
        if (Directory.Exists(rawPath))
        {
            return rawPath;
        }
        if (File.Exists(rawPath))
        {
            return Path.GetDirectoryName(rawPath)?.Replace('\\', '/');
        }
        return null;
    }
}
