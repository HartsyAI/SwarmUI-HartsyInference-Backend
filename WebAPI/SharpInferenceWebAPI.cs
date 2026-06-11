using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Utils;
using SwarmUI.WebAPI;

namespace Hartsy.Extensions.SharpInferenceBackend.WebAPI;

/// <summary>
/// Extra HTTP routes added by the SharpInference extension.
/// See docs/08-Web-API-Routes.md for the contract of each route.
/// </summary>
public static class SharpInferenceWebAPI
{
    public static void Register()
    {
        // TODO phase 1: API.RegisterAPICall<JObject>(SharpInferenceProbeModel,            true,  SharpInferencePermissions.PermAdminSharpInference);
        // TODO phase 1: API.RegisterAPICall<JObject>(SharpInferenceListLoadedPipelines,   false, SharpInferencePermissions.PermAdminSharpInference);
        // TODO phase 1: API.RegisterAPICall<JObject>(SharpInferenceClearCache,            true,  SharpInferencePermissions.PermAdminSharpInference);
        // TODO phase 1: API.RegisterAPICall<JObject>(SharpInferenceGetDeviceInfo,         false, SharpInferencePermissions.PermAdminSharpInference);
        // TODO phase 1: API.RegisterAPICall<JObject>(SharpInferenceGetSupportedArchs,     false, SharpInferencePermissions.PermUseSharpInference);
        Logs.Init("SharpInference WebAPI register stub — no routes wired yet (see docs/08).");
    }

    /// <summary>POST /API/SharpInferenceProbeModel — diagnose a checkpoint without loading it.
    /// See docs/08-Web-API-Routes.md §SharpInferenceProbeModel for inputs/outputs.</summary>
    public static async Task<JObject> SharpInferenceProbeModel(Session session, string model_path)
    {
        // TODO phase 1: open .safetensors via SafeTensorsLoader, walk tensor keys, infer arch.
        await Task.CompletedTask;
        return new JObject
        {
            ["success"] = false,
            ["error"] = "Not implemented yet — see docs/08-Web-API-Routes.md."
        };
    }

    /// <summary>POST /API/SharpInferenceListLoadedPipelines — show pipeline cache state across all backends.</summary>
    public static async Task<JObject> SharpInferenceListLoadedPipelines(Session session)
    {
        // TODO phase 1: iterate Program.Backends, find SharpInferenceBackend instances, project their cache.
        await Task.CompletedTask;
        return new JObject { ["backends"] = new JArray() };
    }

    /// <summary>POST /API/SharpInferenceClearCache — drop the pipeline cache for a backend (or one entry).</summary>
    public static async Task<JObject> SharpInferenceClearCache(Session session, int backend_id, string model_name)
    {
        // TODO phase 1.
        await Task.CompletedTask;
        return new JObject { ["success"] = false, ["evicted_count"] = 0 };
    }

    /// <summary>POST /API/SharpInferenceGetDeviceInfo — enumerate available CUDA / Vulkan / CPU devices.</summary>
    public static async Task<JObject> SharpInferenceGetDeviceInfo(Session session)
    {
        // TODO phase 1: probe each backend type for available devices. Most of this delegates
        // to SharpInference's device-enum APIs (which exist per-backend in src/SharpInference.{Cuda,Vulkan}).
        await Task.CompletedTask;
        return new JObject
        {
            ["cuda_available"] = false,
            ["vulkan_available"] = false,
            ["cpu_available"] = true
        };
    }

    /// <summary>POST /API/SharpInferenceGetSupportedArchs — list supported / planned model architectures.</summary>
    public static async Task<JObject> SharpInferenceGetSupportedArchs(Session session)
    {
        await Task.CompletedTask;
        var supported = new JArray();
        foreach (var arch in Generation.ModelSupport.SupportedArchitectures)
        {
            supported.Add(arch);
        }
        return new JObject
        {
            ["supported"] = supported,
            ["planned"] = new JArray { "stable-diffusion-xl-v1-base", "Flux.1-dev", "Flux.1-schnell", "stable-diffusion-v3-medium" }
        };
    }
}
