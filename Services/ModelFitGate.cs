using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning.Memory;
using SwarmUI.Backends;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
// The class name collides with the trailing namespace segment, so alias it explicitly.
using SiBackend = Hartsy.Extensions.HartsyInferenceBackend.Backends.HartsyInferenceBackend;

namespace Hartsy.Extensions.HartsyInferenceBackend.Services;

/// <summary>Routes each generation to a GPU its model fits on. SwarmUI's scheduler picks backends by load and loaded
/// model only; this gives it the missing memory dimension through the one hook it offers, the backend filter.</summary>
/// <remarks><para><b>Two halves, split by cost.</b> <see cref="Prepare"/> runs once per generation on that request's
/// own task (SwarmUI's <see cref="T2IEngine.PreGenerateEvent"/>), asks each distinct fit profile's engine for a verdict
/// and freezes the routing answer. <see cref="Allows"/> runs inside SwarmUI's single scheduler thread for every pending
/// request against every backend on every pass, so it reads that frozen answer and only touches the registry snapshot
/// when it is about to refuse.</para>
/// <para><b>Works with the VRAM tiers, never over them.</b> The engine judges under the effective tier (backend setting
/// with the request's "VRAM Mode" applied). Performance is the operator sizing workloads themselves, so it is never
/// filtered. Auto and Balanced prefer a card that holds the model resident, then the largest card that can stream it.
/// Aggressive and Maximum chose streaming deliberately, so any card that can stream qualifies.</para>
/// <para><b>Estimates route; only evidence refuses.</b> When no card fits by estimate the largest card is still allowed
/// — the engine's own pre-flight gives the precise refusal. A preference never strands a request: when every card it
/// prefers has stopped taking work, the rest are allowed again. A card is excluded outright only after it actually ran
/// out of memory for this model and request shape (<see cref="RecordFailure"/>); that evidence expires, and is what
/// makes the out-of-memory redirect finite, since every redirect excludes at least one more fit profile.</para></remarks>
public static class ModelFitGate
{
    /// <summary>Capacities within this fraction of the largest count as the largest: two "24 GB" cards rarely report
    /// identical totals, and a few hundred MB must not split them into a preferred and a skipped card.</summary>
    private const double LargestCapacityTolerance = 0.95;

    /// <summary>The backend type id this gate routes for; a request pinned to another type is none of its business.</summary>
    private const string BackendTypeId = "hartsyinference";

    /// <summary>How long an out-of-memory record excludes a card. Long enough to stop a queue of identical requests
    /// hammering a card that cannot take them; short enough that a transient cause (another process, a co-resident
    /// backend's cache) does not bench the card for the rest of the session.</summary>
    private static readonly TimeSpan RecordLifetime = TimeSpan.FromMinutes(30);

    /// <summary>The routing answer for each in-flight request, collected with the request itself.</summary>
    /// <remarks>Keyed by instance: SwarmUI hands the same <see cref="T2IParamInput"/> to <c>PreGenerateEvent</c> and to
    /// the backend filter of one <c>CreateImageTask</c> (it clones per image before both). <see cref="Allows"/> logs if
    /// that ever stops being true, since a miss otherwise looks exactly like no routing at all.</remarks>
    private static readonly ConditionalWeakTable<T2IParamInput, RequestFit> Fits = new();

    /// <summary>What <see cref="Prepare"/> stores when it deliberately has no opinion (not our model, pinned to another
    /// backend type, nothing to assess), so <see cref="Allows"/> can tell that apart from a request it never saw.</summary>
    private static readonly RequestFit NoOpinion = new(null, null, null, 0, false,
        ImmutableDictionary<string, ProfileFit>.Empty);

    /// <summary>Smallest workload that has run out of memory per (checkpoint, request shape, fit profile, effective
    /// tier). Bounded by the models, shapes and profiles in use, not by request volume, and pruned as records expire.</summary>
    private static readonly ConcurrentDictionary<LedgerKey, OutOfMemoryRecord> FailedWorkloads = new();

    /// <summary>Assesses <paramref name="args"/>' request against every dispatchable fit profile and freezes the routing
    /// answer. Subscribed to <see cref="T2IEngine.PreGenerateEvent"/>; never throws — any failure leaves the request
    /// unfiltered, exactly as it was before this gate existed.</summary>
    public static void Prepare(T2IEngine.PreGenerationEventParams args)
    {
        T2IParamInput input = args?.UserInput;
        if (input is null)
        {
            return;
        }
        Fits.Remove(input);
        try
        {
            Fits.AddOrUpdate(input, Assess(input) ?? NoOpinion);
        }
        catch (Exception ex)
        {
            Fits.AddOrUpdate(input, NoOpinion);
            Logs.Warning($"[HartsyInference] VRAM fit check skipped for this generation: {ex.Message}");
        }
    }

    /// <summary>Whether <paramref name="backend"/> should take <paramref name="input"/>; adds the reason to
    /// <see cref="T2IParamInput.RefusalReasons"/> when not. Called from the backend's own validator.</summary>
    public static bool Allows(T2IParamInput input, SiBackend backend)
    {
        if (input is null || backend?.FitProfileKey is null)
        {
            return true;
        }
        if (!Fits.TryGetValue(input, out RequestFit fit))
        {
            // Prepare runs for every generation and always leaves an answer, so a miss means SwarmUI filtered a
            // different instance than it prepared. Said once per request, then remembered, so the scheduler's repeated
            // passes do not repeat it.
            Logs.Verbose($"[HartsyInference] VRAM fit routing has no answer for a request reaching backend "
                + $"{backend.FitProfileKey}; it was not prepared on this instance, so it is routed without memory checks.");
            Fits.AddOrUpdate(input, NoOpinion);
            return true;
        }
        if (!fit.Profiles.TryGetValue(backend.FitProfileKey, out ProfileFit profile) || profile.Allowed)
        {
            return true;
        }
        // A preference is only worth waiting for while a preferred card can still take the work. The answer was
        // frozen when the request was prepared; a preferred backend disabled or reloaded since must not leave the
        // request waiting on nothing.
        if (!profile.OutOfMemory && !AnyPreferredDispatchable(fit))
        {
            return true;
        }
        input.RefusalReasons.Add(profile.Refusal);
        return false;
    }

    /// <summary>Records that <paramref name="input"/> ran out of memory on <paramref name="backend"/>, so requests of this
    /// shape at least this large skip that card, and equally configured cards no larger, until the record expires.</summary>
    /// <returns>True when another fit profile can still take the request and it was not pinned to this backend — the
    /// caller should redirect rather than fail.</returns>
    public static bool RecordFailure(T2IParamInput input, SiBackend backend)
    {
        if (input is null || backend?.FitProfileKey is null || !Fits.TryGetValue(input, out RequestFit fit)
            || !fit.Profiles.TryGetValue(backend.FitProfileKey, out ProfileFit failed)
            || failed.Fit.EffectiveTier == VramTier.Performance)
        {
            return false;
        }
        OutOfMemoryRecord record = new(failed.Fit.CapacityBytes, fit.Workload, failed.FitSettings, DateTime.UtcNow);
        FailedWorkloads.AddOrUpdate(new LedgerKey(fit.Checkpoint, fit.Shape, backend.FitProfileKey, failed.Fit.EffectiveTier),
            record, (_, known) => record with { Workload = Math.Min(known.Workload, fit.Workload) });
        // Exclude the failed card from this request's own frozen answer too, rather than relying on SwarmUI re-firing
        // PreGenerateEvent for the redirected attempt: that is what makes each redirect exclude one more card.
        Fits.AddOrUpdate(input, fit with
        {
            Profiles = fit.Profiles.SetItem(backend.FitProfileKey, failed with
            {
                OutOfMemory = true,
                Allowed = false,
                Refusal = $"HartsyInference: '{fit.ModelName}' just ran out of VRAM on {backend.FitProfileKey} for this "
                    + "request. Lower the resolution or frame count, or use a GPU with more memory.",
            }),
        });
        Logs.Info($"[HartsyInference] Recorded out-of-VRAM for '{fit.ModelName}' on {backend.FitProfileKey} at "
            + $"workload {fit.Workload}; for {RecordLifetime.TotalMinutes:0} minutes, requests like it at least this "
            + "large skip this card and equally configured cards no larger.");
        return !fit.Pinned && fit.Profiles.Any(other => other.Key != backend.FitProfileKey
            && !PresumedOutOfMemory(fit.Checkpoint, fit.Shape, other.Key, other.Value.FitSettings, other.Value.Fit,
                fit.Workload));
    }

    /// <summary>Forgets recorded failures for <paramref name="fitProfileKey"/>: its backend was reconfigured away from
    /// it, so what ran out of memory under those settings is no evidence about the new ones.</summary>
    public static void ForgetProfile(string fitProfileKey)
    {
        foreach (LedgerKey key in FailedWorkloads.Keys)
        {
            if (key.Profile == fitProfileKey)
            {
                FailedWorkloads.TryRemove(key, out _);
            }
        }
    }

    /// <summary>The routing answer for one request, or null when the gate has nothing to say about it.</summary>
    private static RequestFit Assess(T2IParamInput input)
    {
        if (input.TryGet(T2IParamTypes.BackendType, out string backendType)
            && !string.Equals(backendType, "any", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(backendType, BackendTypeId, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }
        List<BackendDeviceEntry> entries = [.. BackendDeviceRegistry.Entries
            .Where(entry => BackendDeviceRegistry.IsDispatchable(entry.Key)).Select(entry => entry.Value)];
        T2IModel model = input.Get(T2IParamTypes.Model);
        string compat = model?.ModelClass?.CompatClass?.ID;
        if (entries.Count == 0 || model is null || !ModelSupport.IsArchitectureSupported(compat))
        {
            return null;
        }
        ModelSupport.Family family = ModelSupport.Resolve(compat);
        if (family.Kind is not (ModelSupport.Kind.Image or ModelSupport.Kind.Video))
        {
            return null;
        }
        ModelSpec spec = ModelSupport.BuildSpec(model, family, input, stageBundles: false);
        if (string.IsNullOrEmpty(spec.LocalPath))
        {
            return null;
        }
        MemoryEstimateRequest request = SiBackend.MapMemoryEstimateRequest(input, family);
        string shape = RequestShape(input, request);

        Dictionary<string, (BackendDeviceEntry Entry, MemoryFit Fit, bool OutOfMemory)> assessed = [];
        foreach (BackendDeviceEntry entry in entries)
        {
            if (assessed.ContainsKey(entry.FitProfileKey))
            {
                continue;
            }
            MemoryFit fit;
            try
            {
                // The event is synchronous, so the wait is unavoidable. Checkpoint headers are cached process-wide,
                // across every engine, so after a model's first request each profile is answered by arithmetic.
                fit = entry.Engine.MemoryEstimation.AssessAsync(spec, request).GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                // One profile's engine failing (disposed mid-shutdown, unreadable side file) costs that profile its
                // say, not the whole request its routing.
                Logs.Verbose($"[HartsyInference] Fit '{model.Name}' on {entry.FitProfileKey} not assessed: {ex.Message}");
                continue;
            }
            bool outOfMemory = fit.EffectiveTier != VramTier.Performance
                && PresumedOutOfMemory(spec.LocalPath, shape, entry.FitProfileKey, entry.FitSettings, fit, request.Workload);
            assessed[entry.FitProfileKey] = (entry, fit, outOfMemory);
            Logs.Verbose($"[HartsyInference] Fit '{model.Name}' on {entry.DeviceName ?? entry.DeviceKey} "
                + $"[{entry.FitProfileKey}]: {fit.Verdict}{(outOfMemory ? " (ran out of VRAM before)" : "")} — {fit.Reason}");
        }
        if (assessed.Count == 0)
        {
            return null;
        }

        bool pinned = input.TryGet(T2IParamTypes.ExactBackendID, out string _);
        return new RequestFit(model.Name, spec.LocalPath, shape, request.Workload, pinned,
            Route(model.Name, assessed, pinned).ToImmutableDictionary());
    }

    /// <summary>Everything besides the base geometry that moves a request's memory, so out-of-memory evidence only
    /// transfers between requests that stress a card the same way: whether the frame count was left to the family
    /// (its default length is not in the workload), the refiner's upscale, the LoRA and ControlNet counts, and the
    /// per-request VRAM levers.</summary>
    private static string RequestShape(T2IParamInput input, MemoryEstimateRequest request)
    {
        double refinerUpscale = input.TryGet(T2IParamTypes.RefinerMethod, out string _)
            && input.TryGet(T2IParamTypes.RefinerUpscale, out double upscale) ? upscale : 1;
        int loras = input.TryGet(T2IParamTypes.Loras, out List<string> loraList) ? loraList.Count : 0;
        int controlNets = T2IParamTypes.Controlnets.Count(holder => holder?.Model is not null
            && input.TryGet(holder.Model, out T2IModel controlNet) && controlNet is not null);
        return $"frames={(request.Frames is null ? "default" : "set")}|refiner={refinerUpscale:0.##}|loras={loras}"
            + $"|controlnets={controlNets}|vram={request.Vram}";
    }

    /// <summary>Whether a profile should be treated as out of memory for this request: it failed on a request of the
    /// same shape at this size or smaller itself, or an equally configured card at least as large did, within the
    /// record lifetime. Only a card that reports its size can be judged by another card's failure.</summary>
    private static bool PresumedOutOfMemory(string checkpoint, string shape, string profile, string fitSettings,
        MemoryFit fit, long workload)
    {
        DateTime now = DateTime.UtcNow;
        foreach ((LedgerKey key, OutOfMemoryRecord record) in FailedWorkloads)
        {
            if (now - record.RecordedUtc > RecordLifetime)
            {
                FailedWorkloads.TryRemove(new KeyValuePair<LedgerKey, OutOfMemoryRecord>(key, record));
                continue;
            }
            if (key.Checkpoint != checkpoint || key.Shape != shape || key.Tier != fit.EffectiveTier
                || record.Workload > workload)
            {
                continue;
            }
            if (key.Profile == profile
                || (fit.CapacityBytes > 0 && record.FitSettings == fitSettings && record.CapacityBytes >= fit.CapacityBytes))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Applies the per-tier preference to every profile's verdict.</summary>
    private static Dictionary<string, ProfileFit> Route(string modelName,
        Dictionary<string, (BackendDeviceEntry Entry, MemoryFit Fit, bool OutOfMemory)> assessed, bool pinned)
    {
        List<(BackendDeviceEntry Entry, MemoryFit Fit, bool OutOfMemory)> judged =
            [.. assessed.Values.Where(a => !a.OutOfMemory && a.Fit.Verdict != MemoryFitVerdict.Unknown)];
        bool anyResident = judged.Any(a => a.Fit.Verdict == MemoryFitVerdict.Resident);
        long largestStreamed = judged.Where(a => a.Fit.Verdict == MemoryFitVerdict.Streamed)
            .Select(a => a.Fit.CapacityBytes).DefaultIfEmpty(0).Max();
        long largest = judged.Select(a => a.Fit.CapacityBytes).DefaultIfEmpty(0).Max();

        Dictionary<string, ProfileFit> routed = [];
        foreach ((string key, (BackendDeviceEntry entry, MemoryFit fit, bool outOfMemory)) in assessed)
        {
            string device = entry.DeviceName ?? entry.DeviceKey;
            bool allowed;
            string refusal = null;
            if (pinned || fit.EffectiveTier == VramTier.Performance || fit.Verdict == MemoryFitVerdict.Unknown)
            {
                // Pinned is the operator choosing the card outright; routing has no say, and a refusal would only
                // turn the engine's own precise error into a vaguer one.
                allowed = true;
            }
            else if (outOfMemory)
            {
                allowed = false;
                refusal = $"HartsyInference: '{modelName}' recently ran out of VRAM on {device} for a request like this "
                    + "one at this size or smaller. Lower the resolution or frame count, or use a GPU with more memory.";
            }
            else
            {
                bool streamingChosen = fit.EffectiveTier is VramTier.Aggressive or VramTier.Maximum;
                allowed = fit.Verdict switch
                {
                    MemoryFitVerdict.Resident => true,
                    MemoryFitVerdict.Streamed => streamingChosen
                        || (!anyResident && fit.CapacityBytes >= largestStreamed * LargestCapacityTolerance),
                    _ => !anyResident && largestStreamed == 0 && fit.CapacityBytes >= largest * LargestCapacityTolerance,
                };
                if (!allowed)
                {
                    refusal = $"HartsyInference: skipped {device} for '{modelName}' ({fit.Reason}) — waiting for a GPU "
                        + "with more memory for this model.";
                }
            }
            routed[key] = new ProfileFit(fit, entry.FitSettings, outOfMemory, allowed, refusal);
        }
        return routed;
    }

    /// <summary>Whether any profile this request prefers still has a backend SwarmUI can hand work to.</summary>
    private static bool AnyPreferredDispatchable(RequestFit fit)
    {
        foreach ((AbstractT2IBackend backend, BackendDeviceEntry entry) in BackendDeviceRegistry.Entries)
        {
            if (fit.Profiles.TryGetValue(entry.FitProfileKey, out ProfileFit profile) && profile.Allowed
                && BackendDeviceRegistry.IsDispatchable(backend))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>What one out-of-memory record is filed under.</summary>
    private readonly record struct LedgerKey(string Checkpoint, string Shape, string Profile, VramTier Tier);

    /// <summary>A card's size and settings, the smallest workload that ran out of memory on it, and when.</summary>
    private sealed record OutOfMemoryRecord(long CapacityBytes, long Workload, string FitSettings, DateTime RecordedUtc);

    /// <summary>The frozen routing answer for one request.</summary>
    private sealed record RequestFit(string ModelName, string Checkpoint, string Shape, long Workload, bool Pinned,
        ImmutableDictionary<string, ProfileFit> Profiles);

    /// <summary>One fit profile's verdict for one request, and whether routing lets it take the request.</summary>
    private sealed record ProfileFit(MemoryFit Fit, string FitSettings, bool OutOfMemory, bool Allowed, string Refusal);
}
