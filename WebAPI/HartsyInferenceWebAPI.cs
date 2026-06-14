using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SwarmUI.WebAPI;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
// The class name collides with the trailing namespace segment, so alias it explicitly.
using SiBackend = Hartsy.Extensions.HartsyInferenceBackend.Backends.HartsyInferenceBackend;

namespace Hartsy.Extensions.HartsyInferenceBackend.WebAPI;

/// <summary>
/// Admin / diagnostic HTTP routes for the HartsyInference backend. These have no ComfyUI
/// equivalent — they exist because an in-process backend can answer "what will you do with
/// this model?" and "what's resident right now?" directly, without a separate process.
/// All routes are read-only except <see cref="HartsyInferenceClearCache"/>.
/// See docs/08-Web-API-Routes.md.
/// </summary>
public static class HartsyInferenceWebAPI
{
    public static void Register()
    {
        API.RegisterAPICall(HartsyInferenceGetSupportedArchs, false, HartsyInferencePermissions.PermUseHartsyInference);
        API.RegisterAPICall(HartsyInferenceProbeModel, false, HartsyInferencePermissions.PermUseHartsyInference);
        API.RegisterAPICall(HartsyInferenceListLoadedPipelines, false, HartsyInferencePermissions.PermAdminHartsyInference);
        API.RegisterAPICall(HartsyInferenceGetDeviceInfo, false, HartsyInferencePermissions.PermAdminHartsyInference);
        API.RegisterAPICall(HartsyInferenceClearCache, true, HartsyInferencePermissions.PermAdminHartsyInference);
        Logs.Init("HartsyInference WebAPI routes registered (supported-archs, probe-model, list-pipelines, device-info, clear-cache).");
    }

    /// <summary>Enumerate every live HartsyInference backend instance with its handler data.</summary>
    private static IEnumerable<(BackendHandler.T2IBackendData data, SiBackend backend)> EnumerateBackends()
    {
        if (Program.Backends is null) yield break;
        foreach (BackendHandler.T2IBackendData data in Program.Backends.EnumerateT2IBackends)
        {
            if (data?.Backend is SiBackend si)
            {
                yield return (data, si);
            }
        }
    }

    /// <summary>POST /API/HartsyInferenceGetSupportedArchs — list dispatched architectures plus the
    /// engine-blocked "pending" ones with their reasons. Lets a UI explain coverage at a glance.</summary>
    public static async Task<JObject> HartsyInferenceGetSupportedArchs(Session session)
    {
        await Task.CompletedTask;
        JArray supported = [];
        foreach (string arch in ModelSupport.SupportedArchitectures)
        {
            supported.Add(arch);
        }
        JObject pending = [];
        foreach (KeyValuePair<string, string> kv in ModelSupport.PendingArchitectures)
        {
            pending[kv.Key] = kv.Value;
        }
        return new JObject
        {
            ["success"] = true,
            ["supported"] = supported,
            ["pending"] = pending,
        };
    }

    /// <summary>POST /API/HartsyInferenceProbeModel — answer "will HartsyInference run this model, and how?"
    /// for a model by name, WITHOUT loading it. Resolves the model's architecture compat class and
    /// reports supported / pending / unsupported with the human-readable reason.</summary>
    public static async Task<JObject> HartsyInferenceProbeModel(Session session, string model_name)
    {
        await Task.CompletedTask;
        if (string.IsNullOrWhiteSpace(model_name))
        {
            return new JObject { ["success"] = false, ["error"] = "No model_name provided." };
        }
        if (!Program.T2IModelSets.TryGetValue("Stable-Diffusion", out T2IModelHandler handler))
        {
            return new JObject { ["success"] = false, ["error"] = "Stable-Diffusion model set unavailable." };
        }
        T2IModel model = handler.GetModel(model_name);
        if (model is null)
        {
            return new JObject { ["success"] = false, ["error"] = $"Model '{model_name}' not found." };
        }
        string compat = model.ModelClass?.CompatClass?.ID;
        bool supported = ModelSupport.IsArchitectureSupported(compat);
        bool pending = !supported && compat is not null && ModelSupport.PendingArchitectures.ContainsKey(compat);
        string state = supported ? "supported" : pending ? "pending" : "unsupported";
        return new JObject
        {
            ["success"] = true,
            ["model_name"] = model.Name,
            ["arch_id"] = model.ModelClass?.ID,
            ["compat_class"] = compat,
            ["state"] = state,
            ["reason"] = supported ? "HartsyInference can generate with this model." : ModelSupport.WhyNotSupported(compat),
        };
    }

    /// <summary>POST /API/HartsyInferenceListLoadedPipelines — per-backend resident-model + status snapshot.</summary>
    public static async Task<JObject> HartsyInferenceListLoadedPipelines(Session session)
    {
        await Task.CompletedTask;
        JArray backends = [];
        foreach ((BackendHandler.T2IBackendData data, SiBackend backend) in EnumerateBackends())
        {
            backends.Add(new JObject
            {
                ["backend_id"] = data.ID,
                ["status"] = backend.Status.ToString(),
                ["current_model"] = backend.CurrentModelName ?? "",
                ["compute_backend"] = backend.Settings?.ComputeBackend ?? "auto",
                ["device_ordinal"] = backend.Settings?.DeviceOrdinal ?? 0,
                ["max_cached_pipelines"] = backend.Settings?.MaxCachedPipelines ?? 1,
                ["usages"] = data.Usages,
            });
        }
        return new JObject { ["success"] = true, ["backends"] = backends };
    }

    /// <summary>POST /API/HartsyInferenceGetDeviceInfo — report each backend's configured compute
    /// target. Enumerating physical devices ahead of construction isn't exposed by the engine, so
    /// this reflects the live configuration rather than a hardware scan.</summary>
    public static async Task<JObject> HartsyInferenceGetDeviceInfo(Session session)
    {
        await Task.CompletedTask;
        JArray devices = [];
        foreach ((BackendHandler.T2IBackendData data, SiBackend backend) in EnumerateBackends())
        {
            devices.Add(new JObject
            {
                ["backend_id"] = data.ID,
                ["compute_backend"] = backend.Settings?.ComputeBackend ?? "auto",
                ["device_ordinal"] = backend.Settings?.DeviceOrdinal ?? 0,
                ["status"] = backend.Status.ToString(),
            });
        }
        return new JObject { ["success"] = true, ["devices"] = devices };
    }

    /// <summary>POST /API/HartsyInferenceClearCache — evict the pipeline cache. <paramref name="backend_id"/>
    /// of -1 clears every HartsyInference backend; otherwise just the matching one. Frees VRAM without
    /// restarting the server.</summary>
    public static async Task<JObject> HartsyInferenceClearCache(Session session, int backend_id = -1, bool free_system_ram = false)
    {
        int cleared = 0;
        foreach ((BackendHandler.T2IBackendData data, SiBackend backend) in EnumerateBackends())
        {
            if (backend_id != -1 && data.ID != backend_id)
            {
                continue;
            }
            if (await backend.FreeMemory(free_system_ram))
            {
                cleared++;
            }
        }
        return new JObject { ["success"] = true, ["backends_cleared"] = cleared };
    }
}
