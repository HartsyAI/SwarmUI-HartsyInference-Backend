using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using SwarmUI.Accounts;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Hartsy.Extensions.SharpInferenceBackend.Backends;
using Hartsy.Extensions.SharpInferenceBackend.WebAPI;

// NOTE: Namespace must NOT contain "SwarmUI" (reserved for built-ins).
// See docs/00-Overview.md for the broader plan.
namespace Hartsy.Extensions.SharpInferenceBackend;

/// <summary>Module initializer: installs the AssemblyLoadContext.Default.Resolving handler
/// that locates SharpInference.*.dll next to our extension DLL. This MUST run before any
/// type metadata in the extension is resolved — Swarm calls <c>asm.GetTypes()</c> on the
/// freshly-loaded extension, and any type with a field/parameter referencing a SharpInference
/// type will trigger an assembly-load attempt that the default ALC can't satisfy (the SharpInference
/// DLLs sit in the extension's subfolder, not in Swarm's main bin). A <c>[ModuleInitializer]</c>
/// fires at assembly-load time, strictly before any type's metadata is loaded — earlier than
/// any static constructor, which only runs on first type use.</summary>
internal static class AssemblyResolverModuleInit
{
    [ModuleInitializer]
    public static void Init()
    {
        try
        {
            string extensionDir = Path.GetDirectoryName(typeof(AssemblyResolverModuleInit).Assembly.Location);
            if (string.IsNullOrEmpty(extensionDir)) return;
            AssemblyLoadContext.Default.Resolving += (ctx, name) =>
            {
                string candidate = Path.Combine(extensionDir, name.Name + ".dll");
                if (File.Exists(candidate))
                {
                    return ctx.LoadFromAssemblyPath(candidate);
                }
                return null;
            };
        }
        catch
        {
            // Swallow — if hooking fails, downstream load errors will produce a clearer message
            // than an unhandled exception in a module initializer would.
        }
    }
}

/// <summary>Permissions for the SharpInference backend extension.</summary>
public static class SharpInferencePermissions
{
    public static readonly PermInfoGroup Group = new(
        "SharpInference",
        "Permissions for the pure-C# SharpInference backend.");

    public static readonly PermInfo PermUseSharpInference = Permissions.Register(new(
        "use_sharpinference",
        "Use SharpInference backend",
        "Allows generating images using the in-process SharpInference backend.",
        PermissionDefault.POWERUSERS, Group));

    public static readonly PermInfo PermAdminSharpInference = Permissions.Register(new(
        "admin_sharpinference",
        "Administer SharpInference",
        "Allows clearing the pipeline cache, probing models, and managing devices.",
        PermissionDefault.ADMINS, Group));
}

/// <summary>
/// Extension entry point. Registers the SharpInferenceBackend backend type, custom
/// parameters, feature flags, and HTTP routes.
/// See docs/01-Architecture.md for the component diagram.
/// </summary>
public class SwarmUISharpInference : Extension
{
    // Assembly resolver lives in AssemblyResolverModuleInit (above) — it must hook before
    // Swarm's GetTypes() call resolves type metadata, which is earlier than any static ctor.

    // SharpInference-specific param group (see docs/07-Parameters-And-Feature-Flags.md).
    public static T2IParamGroup SharpInferenceParamGroup;

    // SharpInference-specific params. Registered under feature flag "sharpinference"
    // so they only show when our backend is the active target.
    public static T2IRegisteredParam<string> DtypeOverrideParam;
    public static T2IRegisteredParam<int> TileVaeThresholdParam;
    public static T2IRegisteredParam<string> SamplerParam;

    public override void OnPreInit()
    {
        Logs.Init("SharpInference extension pre-init");
        // Register feature flags here if needed before settings load.
        // Most feature flags are advertised dynamically via SharpInferenceBackend.SupportedFeatures.

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
        Logs.Init("SharpInference extension init");

        // 1. Param group + SharpInference-specific params.
        SharpInferenceParamGroup = new("SharpInference", Toggles: false, Open: false, IsAdvanced: true);

        DtypeOverrideParam = T2IParamTypes.Register<string>(new(
            "SharpInference Dtype",
            "Override the loaded model dtype. fp16 saves VRAM, fp32 is most accurate (especially on CPU).",
            "fp16",
            Toggleable: true,
            Group: SharpInferenceParamGroup,
            FeatureFlag: "sharpinference",
            GetValues: _ => new List<string> { "fp16", "bf16", "fp32" }));

        TileVaeThresholdParam = T2IParamTypes.Register<int>(new(
            "SharpInference Tile VAE Threshold",
            "Tile VAE decode for outputs above this dimension (px). 0 = always tile.",
            "1024",
            Toggleable: true,
            Min: 0,
            Max: 4096,
            Group: SharpInferenceParamGroup,
            FeatureFlag: "sharpinference"));

        // Named "SharpInference Sampler" because Comfy already owns the "sampler" param ID.
        // When this is unset but Comfy's Sampler param is, SamplingParamResolver maps the
        // Comfy value over as a courtesy (euler/ddim/dpmpp_2m/lcm map; others → Euler).
        SamplerParam = T2IParamTypes.Register<string>(new(
            "SharpInference Sampler",
            "Sampler for SD 1.5 / SDXL generations on the SharpInference backend.\n'euler' is the safe default; 'dpm++2m' is popular for SDXL; 'lcm' is for LCM/turbo checkpoints.\nFlow-matching models (Flux, SD3, Z-Image, etc.) use their canonical sampler and ignore this.",
            "euler",
            Toggleable: true,
            Group: SharpInferenceParamGroup,
            FeatureFlag: "sharpinference",
            GetValues: _ => new List<string> { "euler", "ddim", "dpm++2m", "lcm" }));

        // 2. Register the backend type (single type — no _selfstart vs _api split,
        //    we always run in-process).
        // Fully qualify Backends.SharpInferenceBackend because the trailing namespace
        // segment of this file (`Hartsy.Extensions.SharpInferenceBackend`) collides
        // with the unqualified class name.
        Program.Backends.RegisterBackendType<Backends.SharpInferenceBackend>(
            "sharpinference",
            "SharpInference (Pure C# Inference)",
            "In-process pure-C# diffusion backend. CPU / Vulkan / CUDA support, no Python required.",
            isStandard: false);

        // 3. Register HTTP routes.
        SharpInferenceWebAPI.Register();

        // 4. Pre-register all built-in architecture handlers.
        Generation.ModelSupport.RegisterBuiltins();
    }

    public override void OnPreLaunch()
    {
        // No special wiring needed — backend instances are created on demand by Swarm.
    }

    public override void OnShutdown()
    {
        Logs.Init("SharpInference extension shutdown");
        // Per-instance shutdown is handled by the BackendHandler;
        // nothing extension-level to clean up here.
    }
}
