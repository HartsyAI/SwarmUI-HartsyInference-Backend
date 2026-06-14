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
    public static T2IRegisteredParam<string> DtypeOverrideParam;
    public static T2IRegisteredParam<int> TileVaeThresholdParam;
    public static T2IRegisteredParam<string> SamplerParam;

    public override void OnPreInit()
    {
        Logs.Init("HartsyInference extension pre-init");
        // Register feature flags here if needed before settings load.
        // Most feature flags are advertised dynamically via HartsyInferenceBackend.SupportedFeatures.

        // ACE-Step v1 model class: Swarm core only knows the v1.5 class, so v1 checkpoints would
        // otherwise be unclassified. Must register before model folders are scanned.
        Generation.AceStepLoader.RegisterModelClass();
        // Lance compat + model classes: core has no Lance classes at all; the checkpoints are
        // folder-models (sharded safetensors + llm_config.json), surfaced by the core
        // folder-model scanning support. Must register before model folders are scanned.
        Generation.LanceLoader.RegisterModelClass();
        // MusicGen + YuE: gen-tab music models (audio params light up via IsAudioModel).
        Generation.MusicGenLoader.RegisterModelClass();
        Generation.YueLoader.RegisterModelClass();
    }

    public override void OnInit()
    {
        Logs.Init("HartsyInference extension init");

        // 1. Param group + HartsyInference-specific params.
        HartsyInferenceParamGroup = new("HartsyInference", Toggles: false, Open: false, IsAdvanced: true);

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
