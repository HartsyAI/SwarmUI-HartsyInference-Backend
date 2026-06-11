using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Centralised resolution of Swarm sampling parameters into the concrete numbers we
/// pass to a SharpInference pipeline. Currently handles step count + EndStepsEarly;
/// will grow to cover refiner step splits and other per-stage tweaks.
/// </summary>
public static class SamplingParamResolver
{
    /// <summary>Resolve the effective step count for a single-stage generation,
    /// honoring <see cref="T2IParamTypes.EndStepsEarly"/>. Always returns at least 1.
    ///
    /// EndStepsEarly is a fraction (0..1) of the configured steps to cut off — e.g.
    /// Steps=20, EndStepsEarly=0.25 → run 15 steps. Comfy implements this as
    /// <c>endStep = (int)(steps * (1 - endEarly))</c> (truncating; see
    /// WorkflowGeneratorSteps.cs:1326). We mirror that exactly so users get the same
    /// step count between backends.</summary>
    public static int ResolveSteps(T2IParamInput input, int fallback)
    {
        int steps = input.Get(T2IParamTypes.Steps);
        if (steps <= 0) steps = fallback;

        if (input.TryGet(T2IParamTypes.EndStepsEarly, out double endEarly) && endEarly > 0)
        {
            // Match Comfy's truncating cast (not rounding) so the two backends agree
            // on the step count for the same inputs.
            int reduced = (int)(steps * (1 - endEarly));
            if (reduced < 1)
            {
                Logs.Warning(
                    $"[SharpInference] EndStepsEarly={endEarly} would zero out the step count " +
                    $"(steps={steps}); clamping to 1 step. Consider lowering EndStepsEarly.");
                reduced = 1;
            }
            steps = reduced;
        }

        return steps;
    }

    /// <summary>Resolve the user's sampler choice into a SharpInference
    /// <c>SchedulerFactory</c> name (<c>"ddim"</c> / <c>"dpm++2m"</c> / <c>"lcm"</c>), or null
    /// for the default Euler. Only meaningful for the sigma-domain SD-family pipelines
    /// (SD 1.5 / SDXL / SDXL Refiner); flow-match architectures use their canonical
    /// scheduler and ignore this.
    ///
    /// Priority: our own "SharpInference Sampler" param, then ComfyUI's "sampler" param
    /// as a courtesy fallback (so a user who configured a sampler while on a Comfy
    /// backend gets the same intent honored here when the value maps). Unmappable Comfy
    /// values (euler_ancestral, SDE variants, uni_pc, …) fall back to Euler with a
    /// Verbose log rather than a refusal — sampler choice is a preference, not a
    /// correctness contract.</summary>
    public static string ResolveSchedulerName(T2IParamInput input)
    {
        if (input.TryGet(SwarmUISharpInference.SamplerParam, out string ours) && !string.IsNullOrWhiteSpace(ours))
        {
            return MapSamplerName(ours, logUnmapped: true);
        }
        if (T2IParamTypes.TryGetType("sampler", out T2IParamType comfyType, input)
            && input.TryGetRaw(comfyType, out object comfyRaw)
            && comfyRaw is string comfySampler
            && !string.IsNullOrWhiteSpace(comfySampler))
        {
            return MapSamplerName(comfySampler, logUnmapped: true);
        }
        return null;
    }

    private static string MapSamplerName(string name, bool logUnmapped) => name.ToLowerInvariant() switch
    {
        "euler" => null, // SchedulerFactory default
        "ddim" => "ddim",
        "dpm++2m" or "dpmpp_2m" or "dpmpp2m" => "dpm++2m",
        "lcm" => "lcm",
        _ => LogUnmapped(name, logUnmapped),
    };

    private static string LogUnmapped(string name, bool log)
    {
        if (log)
        {
            Logs.Verbose($"[SharpInference] Sampler '{name}' isn't available in SharpInference (have: euler, ddim, dpm++2m, lcm) — using Euler.");
        }
        return null;
    }

    /// <summary>Resolve Swarm's "CLIP Stop At Layer" (clip skip) param into the
    /// <c>TextToImageRequest.ClipSkip</c> layers-from-end convention: Swarm/Comfy use
    /// negative-from-end (-1 = final layer, -2 = penultimate), SharpInference uses
    /// positive (1 = final, 2 = penultimate). Returns null when unset or default.
    /// Only SD 1.5 honors this (SDXL is penultimate by spec, matching Comfy).</summary>
    public static int? ResolveClipSkip(T2IParamInput input)
    {
        if (!input.TryGet(T2IParamTypes.ClipStopAtLayer, out int stopAt) || stopAt == 0 || stopAt == -1)
        {
            return null;
        }
        if (stopAt > 0)
        {
            // Positive values are "layer N from the start" in some UIs; Swarm uses negatives.
            // Treat positive as already-layers-from-end for robustness.
            return Math.Clamp(stopAt, 1, 12);
        }
        return Math.Clamp(-stopAt, 1, 12);
    }
}
