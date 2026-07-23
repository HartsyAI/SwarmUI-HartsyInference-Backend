using SwarmUI.Accounts;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Hartsy.Extensions.HartsyInferenceBackend.Backends;
using Hartsy.Extensions.HartsyInferenceBackend.WebAPI;

// NOTE: Namespace must NOT contain "SwarmUI" (reserved for built-ins).
// See docs/00-Overview.md for the broader plan.
namespace Hartsy.Extensions.HartsyInferenceBackend;

// NOTE: This extension used to install an AssemblyLoadContext.Default.Resolving hook
// ([ModuleInitializer]) to locate HartsyInference.*.dll next to the extension DLL. That's
// now handled by Swarm core's SwarmExtensionLoadContext (ExtensionsManager.cs), which
// probes the extension's folder for private deps after host resolution fails. Keeping
// the old hook caused every HartsyInference DLL to load into the DEFAULT context first,
// producing "ships X.dll but host already has it loaded" warnings at startup and a
// version-skew hazard (host copy silently wins over the extension's copy).

/// <summary>Permissions for the HartsyInference backend extension.</summary>
public static class HartsyInferencePermissions
{
    public static readonly PermInfoGroup Group = new(
        "HartsyInference",
        "Permissions for the pure-C# HartsyInference backend.");

    public static readonly PermInfo PermUseHartsyInference = Permissions.Register(new(
        "use_hartsyinference",
        "Use HartsyInference backend",
        "Allows generating images using the in-process HartsyInference backend.",
        PermissionDefault.POWERUSERS, Group));

    public static readonly PermInfo PermAdminHartsyInference = Permissions.Register(new(
        "admin_hartsyinference",
        "Administer HartsyInference",
        "Allows clearing the pipeline cache, probing models, and managing devices.",
        PermissionDefault.ADMINS, Group));
}

/// <summary>
/// Extension entry point. Registers the HartsyInferenceBackend backend type, custom
/// parameters, feature flags, and HTTP routes.
/// See docs/01-Architecture.md for the component diagram.
/// </summary>
public class SwarmUIHartsyInference : Extension
{
    // HartsyInference-specific param group (see docs/07-Parameters-And-Feature-Flags.md).
    public static T2IParamGroup HartsyInferenceParamGroup;

    // HartsyInference-specific params. Registered under feature flag "hartsyinference"
    // so they only show when our backend is the active target.
    public static T2IRegisteredParam<Image> AnimateReferenceImageParam;
    public static T2IRegisteredParam<bool> AnimateAutoPreprocessParam;
    public static T2IRegisteredParam<Image> AnimatePoseVideoParam;
    public static T2IRegisteredParam<Image> AnimateFaceVideoParam;
    public static T2IRegisteredParam<string> DtypeOverrideParam;
    public static T2IRegisteredParam<int> TileVaeThresholdParam;
    public static T2IRegisteredParam<string> SamplerParam;

    // IP-Adapter FaceID-PlusV2 shortcut strength (read by Generation.IpAdapterResolver).
    public static T2IRegisteredParam<double> FaceIdV2WeightParam;

    // Ideogram 4 magic-prompt params (see Generation.Ideogram4MagicPrompt).
    public static T2IParamGroup Ideogram4ParamGroup;
    public static T2IRegisteredParam<bool> Ideogram4MagicPromptParam;
    public static T2IRegisteredParam<string> Ideogram4MagicPromptModelParam;

    public override void OnPreInit()
    {
        Logs.Init("HartsyInference extension pre-init");

        // Engine performance config: since engine 1.0.0-alpha.45, the standard performance profile
        // (cuDNN fused SDPA, fp8-native GEMM on SM 8.9+, F16 DiT activations, resident DiT weights, raised
        // mem-pool release threshold) is DEFAULT-ON inside the engine itself — nothing to set here, every
        // install reproduces the published benchmark times out of the box. Operators can disable any feature
        // with HARTSY_<FEATURE>=0 in the launcher environment; the engine logs the resolved set at backend
        // init ("[Cuda] perf flags: ..."). See the engine's docs/PERFORMANCE.md.

        // Register feature flags here if needed before settings load.
        // Most feature flags are advertised dynamically via HartsyInferenceBackend.SupportedFeatures.

        // Model classes core doesn't know: ACE-Step v1 (core only registers v1.5), Lance image + video
        // (core has no Lance classes at all — folder models with llm_config.json), MusicGen, YuE, F-Lite.
        // Must register before model folders are scanned.
        Generation.ModelClassRegistrations.RegisterAll();
        // Boogu: core now detects it (compat class "boogu") and excludes it from the OmniGen2 probe, so
        // the extension reuses core's classification — no registration needed.
    }

    /// <summary>Lists <c>&lt;ModelRoot&gt;/ipadapter/*.safetensors|*.bin</c> into the Comfy extension's IP-Adapter dropdown values.</summary>
    private static void PopulateIpAdapterModels()
    {
        try
        {
            string folder = System.IO.Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, "ipadapter");
            if (!System.IO.Directory.Exists(folder))
            {
                return;
            }
            List<string> found = [];
            foreach (string file in System.IO.Directory.EnumerateFiles(folder, "*.*", System.IO.SearchOption.AllDirectories))
            {
                if (file.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
                {
                    found.Add(System.IO.Path.GetRelativePath(folder, file).Replace('\\', '/'));
                }
            }
            if (found.Count > 0)
            {
                T2IParamTypes.ConcatDropdownValsClean(ref SwarmUI.Builtin_ComfyUIBackend.ComfyUIBackendExtension.IPAdapterModels, found);
                Logs.Verbose($"[HartsyInference] IP-Adapter dropdown populated with {found.Count} local file(s).");
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] Failed to populate IP-Adapter model list: {ex.Message}");
        }
    }

    public override void OnInit()
    {
        Logs.Init("HartsyInference extension init");

        // The Comfy extension's Use IP-Adapter dropdown normally populates from a live ComfyUI
        // backend's object_info. When only HartsyInference backends run, populate it from the
        // ipadapter model folder directly (the same folder-listing approach the style-model
        // dropdown uses), so IPA works Comfy-free. Refresh keeps it current after downloads.
        PopulateIpAdapterModels();
        Program.ModelRefreshEvent += PopulateIpAdapterModels;

        // 1. Param group + HartsyInference-specific params.
        HartsyInferenceParamGroup = new("HartsyInference", Toggles: false, Open: false, IsAdvanced: true);

        AnimateReferenceImageParam = T2IParamTypes.Register<Image>(new(
            "Animate Reference Image",
            "Wan-Animate: the character/identity image to animate.\nThe Init Image slot carries the driving (pose/motion) video; this image is who performs that motion.\nRequired for Wan-Animate generations on the HartsyInference backend.",
            null,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        // Wan-Animate driving preprocessing. When on (default), the backend auto-derives the pose skeleton +
        // cropped face from the Init-Image driving clip (the way the checkpoint was trained), instead of feeding
        // the raw clip. Toggle off (or supply the overrides below) to hand the backend pre-rendered inputs.
        AnimateAutoPreprocessParam = T2IParamTypes.Register<bool>(new(
            "Animate Auto-Preprocess Driving",
            "Wan-Animate: auto-derive the pose skeleton + cropped face from the Init-Image driving video (the format the model was trained on).\nOn (default) = best motion fidelity; off = feed the raw clip (legacy). The pose/face override params below take precedence when set.",
            "true",
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        AnimatePoseVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Pose Video",
            "Wan-Animate: an already-rendered pose/skeleton driving video (OpenPose/DWPose colored limbs).\nOverrides auto-preprocessing for the pose branch — supply this when you have a pre-rendered skeleton.",
            null,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        AnimateFaceVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Face Video",
            "Wan-Animate: an already-cropped, face-centered driving video (square, ~512px) for the facial-motion branch.\nOverrides auto-preprocessing for the face branch.",
            null,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        DtypeOverrideParam = T2IParamTypes.Register<string>(new(
            "HartsyInference Dtype",
            "Override the loaded model dtype. fp16 saves VRAM, fp32 is most accurate (especially on CPU).",
            "fp16",
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            GetValues: _ => new List<string> { "fp16", "bf16", "fp32" }));

        TileVaeThresholdParam = T2IParamTypes.Register<int>(new(
            "HartsyInference Tile VAE Threshold",
            "Tile VAE decode for outputs above this dimension (px). 0 = always tile.",
            "1024",
            Toggleable: true,
            Min: 0,
            Max: 4096,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference"));

        // Named "HartsyInference Sampler" because Comfy already owns the "sampler" param ID.
        // When this is unset but Comfy's Sampler param is, SamplingParamResolver maps the
        // Comfy value over as a courtesy (euler/ddim/dpmpp_2m/lcm map; others → Euler).
        SamplerParam = T2IParamTypes.Register<string>(new(
            "HartsyInference Sampler",
            "Sampler for SD 1.5 / SDXL generations on the HartsyInference backend.\n'euler' is the safe default; 'dpm++2m' is popular for SDXL; 'lcm' is for LCM/turbo checkpoints.\nFlow-matching models (Flux, SD3, Z-Image, etc.) use their canonical sampler and ignore this.",
            "euler",
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            GetValues: _ => new List<string> { "euler", "ddim", "dpm++2m", "lcm" }));

        // FaceID-PlusV2 shortcut strength: sits with the Comfy extension's IP-Adapter params in the
        // Image Prompting group (same "ipadapter" feature flag + dropdown dependency) so it appears
        // exactly when an IP-Adapter is selected. Only consumed for faceid-plusv2 checkpoints; the
        // 1.0 default matches the official IPAdapterFaceIDPlus pipeline (s_scale).
        FaceIdV2WeightParam = T2IParamTypes.Register<double>(new(
            "FaceID V2 Weight",
            "Strength of the FaceID-PlusV2 CLIP-face shortcut mix (the official pipeline's 's_scale').\nHigher = the CLIP appearance of the face crop contributes more on top of the ArcFace identity tokens.\nOnly used with ip-adapter-faceid-plusv2 models; 1.0 is the official default.",
            "1", Min: 0, Max: 2, Step: 0.05, IgnoreIf: "1",
            FeatureFlag: "ipadapter",
            Group: T2IParamTypes.GroupImagePrompting,
            ViewType: ParamViewType.SLIDER,
            OrderPriority: 19.5,
            IsAdvanced: true,
            Examples: ["0.5", "1", "1.5"],
            DependNonDefault: SwarmUI.Builtin_ComfyUIBackend.ComfyUIBackendExtension.UseIPAdapterForRevision.Type.ID));

        // Ideogram 4 magic prompt: optional LLM rewrite of a plain prompt into Ideogram's structured JSON
        // caption (Generation.Ideogram4MagicPrompt). Ideogram 4 is trained on structured captions; plain
        // text also works, so this is opt-in and only fires for Ideogram 4 generations.
        Ideogram4ParamGroup = new("Ideogram 4", Toggles: false, Open: false, IsAdvanced: false);

        Ideogram4MagicPromptParam = T2IParamTypes.Register<bool>(new(
            "Ideogram 4 Magic Prompt",
            "Rewrite your plain prompt into Ideogram 4's structured JSON caption using a running LLM backend, the way Ideogram's own stack does.\nRequires an LLM backend (Server > Backends — LlamaSharp, Anthropic, or remote). When off (default), your prompt is sent to the model as-is (which also works).",
            "false",
            // Toggleable so the frontend only sends it (and thus only requires the
            // "hartsyinference" flag) when actually enabled. A non-toggleable flagged param
            // is sent on every request, forcing "hartsyinference" onto unrelated generations
            // and refusing the Comfy backend (which lacks the flag) — see [[backend feature flags]].
            Toggleable: true,
            Group: Ideogram4ParamGroup,
            FeatureFlag: "hartsyinference"));

        Ideogram4MagicPromptModelParam = T2IParamTypes.Register<string>(new(
            "Ideogram 4 Magic Prompt LLM",
            "Optional: which LLM model the magic prompt should use (must be available on a running LLM backend).\nLeave unset to use the running LLM backend's default model. Only used when 'Ideogram 4 Magic Prompt' is on.",
            "",
            Toggleable: true,
            Group: Ideogram4ParamGroup,
            FeatureFlag: "hartsyinference"));

        // 2. Register the backend type (single type — no _selfstart vs _api split,
        //    we always run in-process).
        // Fully qualify Backends.HartsyInferenceBackend because the trailing namespace
        // segment of this file (`Hartsy.Extensions.HartsyInferenceBackend`) collides
        // with the unqualified class name.
        Program.Backends.RegisterBackendType<Backends.HartsyInferenceBackend>(
            "hartsyinference",
            "HartsyInference (Pure C# Inference)",
            "In-process pure-C# diffusion backend. CPU / Vulkan / CUDA support, no Python required.",
            isStandard: false);

        // 3. Register HTTP routes.
        HartsyInferenceWebAPI.Register();

        // 4. Pre-register all built-in architecture handlers.
        Generation.ModelSupport.RegisterBuiltins();
    }

    public override void OnPreLaunch()
    {
        // No special wiring needed — backend instances are created on demand by Swarm.
    }

    public override void OnShutdown()
    {
        Logs.Init("HartsyInference extension shutdown");
        // Per-instance shutdown is handled by the BackendHandler;
        // nothing extension-level to clean up here.
    }
}
