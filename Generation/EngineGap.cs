namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Helpers for loader paths blocked on missing HartsyInference engine surface (a preset,
/// converter method, or tokenizer that doesn't exist upstream yet). Loaders call
/// <see cref="Throw"/> early in Load() so the user gets a clear message BEFORE any heavy
/// weight loading, and substitute <see cref="Value{T}"/> for the not-yet-existing engine
/// expressions (kept alongside in TODO(engine-blocked) comments) so the file keeps
/// compiling against today's engine. Both always throw at runtime; they're plain method
/// calls (no [DoesNotReturn]) specifically so the compiler doesn't flag the preserved
/// downstream wiring as unreachable code.
/// </summary>
public static class EngineGap
{
    /// <summary>Throws an engine-blocked refusal. Call at the top of Load().</summary>
    public static void Throw(string message) => throw new InvalidOperationException(message);

    /// <summary>Placeholder for an engine expression that doesn't exist yet.</summary>
    public static T Value<T>(string message) => throw new InvalidOperationException(message);
}
