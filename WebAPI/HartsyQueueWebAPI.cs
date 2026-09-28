using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Utils;
using SwarmUI.WebAPI;

namespace Hartsy.Extensions.HartsyInferenceBackend.WebAPI;

/// <summary>
/// Whole-server queue visibility.
/// <para><see cref="HartsyGetQueueStatus"/> answers the question core has no call for: how busy is the whole server?
/// Core's <c>GetCurrentStatus</c> is per session, so a caller such as Hartsy's site - which gives every Discord run a
/// session of its own - could only ever see its own jobs, and could not tell a member how many are ahead of them or
/// whether the server is simply saturated.</para>
/// </summary>
public static class HartsyQueueWebAPI
{
    public static void Register()
    {
        API.RegisterAPICall(HartsyGetQueueStatus, false, HartsyInferencePermissions.PermUseHartsyInference);
        Logs.Init("HartsyInference queue route registered (queue-status).");
    }

    /// <summary>POST /API/HartsyGetQueueStatus - the whole server's queue, summed over every session, plus what each
    /// text-to-image backend is doing. Read-only; one pass over the live sessions.</summary>
    public static async Task<JObject> HartsyGetQueueStatus(Session session)
    {
        await Task.CompletedTask;
        int waiting = 0, loading = 0, waitingBackends = 0, live = 0, active = 0;
        foreach (Session each in Program.Sessions.Sessions.Values)
        {
            int w = each.WaitingGenerations, l = each.LoadingModels, b = each.WaitingBackends, g = each.LiveGens;
            waiting += w;
            loading += l;
            waitingBackends += b;
            live += g;
            if (w + l + b + g > 0)
            {
                active++;
            }
        }

        JArray backends = [];
        if (Program.Backends is not null)
        {
            foreach (BackendHandler.T2IBackendData data in Program.Backends.EnumerateT2IBackends)
            {
                if (data?.Backend is not AbstractT2IBackend backend)
                {
                    continue;
                }

                backends.Add(new JObject
                {
                    ["id"] = data.ID,
                    ["type"] = backend.HandlerTypeData?.ID,
                    ["status"] = backend.Status.ToString().ToLowerInvariant(),
                    ["usages"] = data.Usages,
                    ["max_usages"] = backend.MaxUsages,
                });
            }
        }

        return new JObject
        {
            ["waiting_gens"] = waiting,
            ["live_gens"] = live,
            ["loading_models"] = loading,
            ["waiting_backends"] = waitingBackends,
            ["active_sessions"] = active,
            ["backends"] = backends,
        };
    }
}
