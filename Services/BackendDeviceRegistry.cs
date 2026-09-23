using System.Collections.Immutable;
using SwarmUI.Backends;

namespace Hartsy.Extensions.HartsyInferenceBackend.Services;

/// <summary>Every live HartsyInference backend and the device it runs on, as one immutable snapshot.</summary>
/// <remarks><para>Read on the hot path — once per queued request — and written only when a backend starts or stops.
/// Writers swap a new snapshot under a lock; readers take the current reference and never lock, so a busy queue never
/// contends with a backend restart.</para>
/// <para>Keyed by the backend instance, not its id: Swarm reuses an instance across settings edits (Init runs again),
/// and a re-registration must replace the old entry rather than add a second one.</para></remarks>
public static class BackendDeviceRegistry
{
    private static readonly object WriteLock = new();

    private static ImmutableDictionary<AbstractT2IBackend, BackendDeviceEntry> _entries =
        ImmutableDictionary.Create<AbstractT2IBackend, BackendDeviceEntry>(ReferenceEqualityComparer.Instance);

    /// <summary>The current snapshot. Safe to enumerate while backends start and stop.</summary>
    public static ImmutableDictionary<AbstractT2IBackend, BackendDeviceEntry> Entries => Volatile.Read(ref _entries);

    /// <summary>Whether SwarmUI's scheduler can currently hand <paramref name="backend"/> a generation — the same
    /// eligibility <c>BackendHandler</c> applies before any filter runs. A card that fails this cannot be the one a
    /// request waits for.</summary>
    public static bool IsDispatchable(AbstractT2IBackend backend) =>
        backend.IsEnabled && !backend.ShutDownReserve && backend.MaxUsages > 0 && backend.Status == BackendStatus.RUNNING;

    /// <summary>Adds or replaces <paramref name="backend"/>'s entry.</summary>
    /// <returns>How many live backends now share <paramref name="entry"/>'s physical device, this one included.</returns>
    public static int Register(AbstractT2IBackend backend, BackendDeviceEntry entry)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(entry);
        lock (WriteLock)
        {
            ImmutableDictionary<AbstractT2IBackend, BackendDeviceEntry> next = _entries.SetItem(backend, entry);
            Volatile.Write(ref _entries, next);
            return next.Values.Count(other => other.DeviceKey == entry.DeviceKey);
        }
    }

    /// <summary>Drops <paramref name="backend"/>'s entry, if it has one.</summary>
    public static void Release(AbstractT2IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        lock (WriteLock)
        {
            Volatile.Write(ref _entries, _entries.Remove(backend));
        }
    }
}
