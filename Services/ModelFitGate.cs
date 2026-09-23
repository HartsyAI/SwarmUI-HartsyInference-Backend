using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning.Memory;
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
/// request against every backend on every pass, so it only reads that frozen answer: no I/O, no locks.</para>
/// <para><b>Works with the VRAM tiers, never over them.</b> The engine judges under the effective tier (backend setting
/// with the request's "VRAM Mode" applied). Performance is the operator sizing workloads themselves, so it is never
/// filtered. Auto and Balanced prefer a card that holds the model resident, then the largest card that can stream it.
/// Aggressive and Maximum chose streaming deliberately, so any card that can stream qualifies.</para>
/// <para><b>Estimates route; only evidence refuses.</b> When no card fits by estimate the largest card is still allowed
/// — the engine's own pre-flight gives the precise refusal. A card is excluded outright only after it actually ran out
/// of memory at this size (<see cref="RecordFailure"/>), which is also what makes the out-of-memory redirect finite:
/// every redirect excludes at least one more fit profile.</para></remarks>
public static class ModelFitGate
{
    /// <summary>Capacities within this fraction of the largest count as the largest: two "24 GB" cards rarely report
    /// identical totals, and a few hundred MB must not split them into a preferred and a skipped card.</summary>
    private const double LargestCapacityTolerance = 0.95;

    /// <summary>The routing answer for each in-flight request, collected with the request itself.</summary>
    private static readonly ConditionalWeakTable<T2IParamInput, RequestFit> Fits = new();

    /// <summary>Smallest workload that has run out of memory per (checkpoint, fit profile, effective tier), with that
    /// profile's capacity. Bounded by models x profiles x tiers, not by request volume.</summary>
    private static readonly ConcurrentDictionary<(string Checkpoint, string Profile, VramTier Tier), OutOfMemoryRecord>
        FailedWorkloads = new();

    /// <summary>Assesses <paramref name="args"/>' request against every live fit profile and freezes the routing answer.
    /// Subscribed to <see cref="T2IEngine.PreGenerateEvent"/>; never throws — any failure leaves the request
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
            RequestFit fit = Assess(input);
            if (fit is not null)
            {
                Fits.AddOrUpdate(input, fit);
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference] VRAM fit check skipped for this generation: {ex.Message}");
        }
    }

    /// <summary>Whether <paramref name="backend"/> should take <paramref name="input"/>; adds the reason to
    /// <see cref="T2IParamInput.RefusalReasons"/> when not. Called from the backend's own validator.</summary>
    public static bool Allows(T2IParamInput input, SiBackend backend)
    {
        if (input is null || backend?.FitProfileKey is null || !Fits.TryGetValue(input, out RequestFit fit)
            || !fit.Profiles.TryGetValue(backend.FitProfileKey, out ProfileFit profile) || profile.Allowed)
        {
            return true;
        }
        input.RefusalReasons.Add(profile.Refusal);
        return false;
    }

    /// <summary>Records that <paramref name="input"/> ran out of memory on <paramref name="backend"/>, so this request and
    /// every later one at least this large skips that fit profile.</summary>
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
        OutOfMemoryRecord record = new(failed.Fit.CapacityBytes, fit.Workload);
        FailedWorkloads.AddOrUpdate((fit.Checkpoint, backend.FitProfileKey, failed.Fit.EffectiveTier), record,
            (_, known) => known with { Workload = Math.Min(known.Workload, fit.Workload) });
        Logs.Info($"[HartsyInference] Recorded out-of-VRAM for '{fit.ModelName}' on {backend.FitProfileKey} at "
            + $"workload {fit.Workload}; requests at least this large will skip GPUs this size or smaller.");
        return !fit.Pinned && fit.Profiles.Any(other => other.Key != backend.FitProfileKey
            && !PresumedOutOfMemory(fit.Checkpoint, other.Key, other.Value.Fit, fit.Workload));
    }

    /// <summary>Forgets recorded failures for <paramref name="fitProfileKey"/>: its backend was reconfigured, so what ran
    /// out of memory under the old settings is no evidence about the new ones.</summary>
    public static void ForgetProfile(string fitProfileKey)
    {
        foreach ((string Checkpoint, string Profile, VramTier Tier) key in FailedWorkloads.Keys)
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
        ImmutableDictionary<SwarmUI.Backends.AbstractT2IBackend, BackendDeviceEntry> entries = BackendDeviceRegistry.Entries;
        T2IModel model = input.Get(T2IParamTypes.Model);
        string compat = model?.ModelClass?.CompatClass?.ID;
        if (entries.IsEmpty || model is null || !ModelSupport.IsArchitectureSupported(compat))
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

        Dictionary<string, (BackendDeviceEntry Entry, MemoryFit Fit, bool Failed)> assessed = [];
        foreach (BackendDeviceEntry entry in entries.Values)
        {
            if (assessed.ContainsKey(entry.FitProfileKey))
            {
                continue;
            }
            // The event is synchronous, so the wait is unavoidable; the engine answers from cached headers and
            // arithmetic, so there is nothing to wait on after the first request for a model.
            MemoryFit fit = entry.Engine.MemoryEstimation.AssessAsync(spec, request).GetAwaiter().GetResult();
            bool failed = fit.EffectiveTier != VramTier.Performance
                && PresumedOutOfMemory(spec.LocalPath, entry.FitProfileKey, fit, request.Workload);
            assessed[entry.FitProfileKey] = (entry, fit, failed);
            Logs.Verbose($"[HartsyInference] Fit '{model.Name}' on {entry.DeviceName ?? entry.DeviceKey} "
                + $"[{entry.FitProfileKey}]: {fit.Verdict}{(failed ? " (ran out of VRAM before)" : "")} — {fit.Reason}");
        }

        bool pinned = input.TryGet(T2IParamTypes.ExactBackendID, out string _);
        return new RequestFit(model.Name, spec.LocalPath, request.Workload, pinned,
            Route(model.Name, assessed, pinned).ToImmutableDictionary());
    }

    /// <summary>Whether a profile should be treated as out of memory for this workload: it failed at this size or
    /// smaller itself, or a card at least as large under the same tier did. Only a card that reports its size can be
    /// judged by another card's failure.</summary>
    private static bool PresumedOutOfMemory(string checkpoint, string profile, MemoryFit fit, long workload)
    {
        foreach (KeyValuePair<(string Checkpoint, string Profile, VramTier Tier), OutOfMemoryRecord> known in FailedWorkloads)
        {
            if (known.Key.Checkpoint != checkpoint || known.Key.Tier != fit.EffectiveTier || known.Value.Workload > workload)
            {
                continue;
            }
            if (known.Key.Profile == profile || (fit.CapacityBytes > 0 && known.Value.CapacityBytes >= fit.CapacityBytes))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Applies the per-tier preference to every profile's verdict.</summary>
    private static Dictionary<string, ProfileFit> Route(string modelName,
        Dictionary<string, (BackendDeviceEntry Entry, MemoryFit Fit, bool Failed)> assessed, bool pinned)
    {
        List<(BackendDeviceEntry Entry, MemoryFit Fit, bool Failed)> judged =
            [.. assessed.Values.Where(a => !a.Failed && a.Fit.Verdict != MemoryFitVerdict.Unknown)];
        bool anyResident = judged.Any(a => a.Fit.Verdict == MemoryFitVerdict.Resident);
        long largestStreamed = judged.Where(a => a.Fit.Verdict == MemoryFitVerdict.Streamed)
            .Select(a => a.Fit.CapacityBytes).DefaultIfEmpty(0).Max();
        long largest = judged.Select(a => a.Fit.CapacityBytes).DefaultIfEmpty(0).Max();

        Dictionary<string, ProfileFit> routed = [];
        foreach ((string key, (BackendDeviceEntry entry, MemoryFit fit, bool failed)) in assessed)
        {
            string device = entry.DeviceName ?? entry.DeviceKey;
            bool allowed;
            string refusal;
            if (fit.EffectiveTier == VramTier.Performance || fit.Verdict == MemoryFitVerdict.Unknown)
            {
                (allowed, refusal) = (true, null);
            }
            else if (failed)
            {
                (allowed, refusal) = (false, $"HartsyInference: '{modelName}' already ran out of VRAM on {device} at this "
                    + "size or larger. Lower the resolution or frame count, or use a GPU with more memory.");
            }
            else if (pinned)
            {
                (allowed, refusal) = (true, null);
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
                refusal = allowed ? null : $"HartsyInference: skipped {device} for '{modelName}' ({fit.Reason}) — "
                    + "waiting for a GPU with more memory for this model.";
            }
            routed[key] = new ProfileFit(fit, failed, allowed, refusal);
        }
        return routed;
    }

    /// <summary>A card size and the smallest workload that ran out of memory on it.</summary>
    private sealed record OutOfMemoryRecord(long CapacityBytes, long Workload);

    /// <summary>The frozen routing answer for one request.</summary>
    private sealed record RequestFit(string ModelName, string Checkpoint, long Workload, bool Pinned,
        ImmutableDictionary<string, ProfileFit> Profiles);

    /// <summary>One fit profile's verdict for one request, and whether routing lets it take the request.</summary>
    private sealed record ProfileFit(MemoryFit Fit, bool Failed, bool Allowed, string Refusal);
}
