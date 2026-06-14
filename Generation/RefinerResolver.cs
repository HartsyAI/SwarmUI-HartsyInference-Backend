using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Reads Swarm's refiner-related params and produces a <see cref="RefinerSpec"/> when
/// the user has enabled refining. Returns null when no refiner stage should run.
///
/// Backed Swarm params:
/// <list type="bullet">
///   <item><description><c>RefinerModel</c> — the refiner checkpoint (omitted/"(Use Base)" disables refining).</description></item>
///   <item><description><c>RefinerControl</c> — fraction of steps the refiner runs (Strength in upstream terms).</description></item>
///   <item><description><c>RefinerSteps</c> — toggleable override of the step total used in the refiner sub-range calc.</description></item>
///   <item><description><c>RefinerCFGScale</c> — toggleable override of CFG for the refiner pass.</description></item>
///   <item><description><c>RefinerMethod</c> — only "PostApply" is supported here; StepSwap variants are refused upfront.</description></item>
/// </list>
/// </summary>
public static class RefinerResolver
{
    public sealed class RefinerSpec
    {
        public required T2IModel Model { get; init; }
        public required int Steps { get; init; }
        public required float Strength { get; init; }
        public required float CfgScale { get; init; }
        /// <summary>The user-selected method string. Always "PostApply" by the time
        /// we get here — other methods are refused at validation time.</summary>
        public required string Method { get; init; }
    }

    public static RefinerSpec Resolve(T2IParamInput input)
    {
        if (input is null) return null;

        T2IModel refinerModel = input.Get(T2IParamTypes.RefinerModel);
        if (refinerModel is null) return null; // "(Use Base)" / unset

        double refinerControl = input.Get(T2IParamTypes.RefinerControl);
        if (refinerControl <= 0)
        {
            Logs.Debug("[HartsyInference] RefinerControl=0 → skipping refiner pass.");
            return null;
        }

        // RefinerSteps is toggleable: only honored when the user explicitly enabled it.
        int refinerStepTotal = input.Get(T2IParamTypes.Steps);
        if (input.TryGet(T2IParamTypes.RefinerSteps, out int rs) && rs > 0)
        {
            refinerStepTotal = rs;
        }
        if (refinerStepTotal <= 0) refinerStepTotal = 30;

        // RefinerCFGScale also toggleable — fall back to the base CFG when off.
        double cfg = input.Get(T2IParamTypes.CFGScale);
        if (input.TryGet(T2IParamTypes.RefinerCFGScale, out double rcfg))
        {
            cfg = rcfg;
        }

        string method = input.Get(T2IParamTypes.RefinerMethod) ?? "PostApply";

        return new RefinerSpec
        {
            Model = refinerModel,
            Steps = refinerStepTotal,
            Strength = (float)refinerControl,
            CfgScale = cfg <= 0 ? 7.5f : (float)cfg,
            Method = method,
        };
    }
}
