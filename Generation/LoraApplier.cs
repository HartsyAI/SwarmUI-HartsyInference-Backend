using SwarmUI.Utils;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.Lora;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Builds and applies a SharpInference <see cref="LoraStack"/> from a list of resolved
/// <see cref="LoraResolver.LoraSpec"/> entries against a per-architecture set of weight
/// dictionaries. The dicts passed in are mutated in place: matching keys have their
/// Tensor entries replaced with newly-allocated merged tensors owned by the returned
/// stack. Caller is responsible for disposing the stack <em>after</em> the model
/// components built from these dicts are no longer in use.
/// </summary>
public static class LoraApplier
{
    /// <summary>Open every LoRA file, partition layers by target, and merge into the
    /// supplied per-component weight dicts. Returns the stack so the caller can
    /// dispose it (and free the merged tensors) when the generation is finished.</summary>
    /// <param name="loras">Resolved LoRA specs. Empty list returns null.</param>
    /// <param name="backend">Compute backend used for the matmul/scale/add operations
    /// inside <see cref="LoraStack.ApplyTo"/>.</param>
    /// <param name="unetWeights">Pass when targeting SD 1.5 / SDXL UNet, else null.</param>
    /// <param name="transformerWeights">Pass when targeting Flux Transformer, else null.</param>
    /// <param name="clipLWeights">Pass when the architecture has a CLIP-L encoder.</param>
    /// <param name="clipGWeights">Pass when the architecture has a CLIP-G encoder (SDXL only).</param>
    public static LoraStack BuildAndApply(
        IReadOnlyList<LoraResolver.LoraSpec> loras,
        IBackend backend,
        IDictionary<string, Tensor> unetWeights = null,
        IDictionary<string, Tensor> transformerWeights = null,
        IDictionary<string, Tensor> clipLWeights = null,
        IDictionary<string, Tensor> clipGWeights = null)
    {
        if (loras is null || loras.Count == 0) return null;

        LoraStack stack = new();
        try
        {
            foreach (LoraResolver.LoraSpec spec in loras)
            {
                Logs.Verbose($"[SharpInference][LoRA] Loading '{spec.Model.Name}' (model={spec.ModelStrength}, tenc={spec.TencStrength}).");
                stack.AddFromPath(spec.FilePath, strength: spec.ModelStrength);
            }

            // ApplyToWeights walks the stacked LoRAs and merges per-target. Note that
            // we use a single combined "model" strength here. Per-target strength
            // splitting (model vs tenc) requires a second stack with the tenc
            // strength applied only to ClipL/ClipG — we'd need that if/when users
            // commonly tune the two independently. For now: model strength applies
            // everywhere. TODO: split into two stacks if tenc != model.
            int merged = stack.ApplyToWeights(
                backend,
                unetWeights: unetWeights,
                transformerWeights: transformerWeights,
                clipLWeights: clipLWeights,
                clipGWeights: clipGWeights);

            if (merged == 0)
            {
                Logs.Warning(
                    "[SharpInference] LoRA stack matched 0 weights. The LoRA's target keys may not align with this architecture, " +
                    "or the LoRA format may not be supported by SharpInference. Generation will proceed without LoRA effect.");
            }
            else
            {
                Logs.Verbose($"[SharpInference][LoRA] Stack merged {merged} weights across components.");
            }

            return stack;
        }
        catch
        {
            stack.Dispose();
            throw;
        }
    }

    /// <summary>Shallow-copy a weight dict so that LoRA replacement of entries doesn't
    /// poison the cache's original dict. Tensor instances are referenced (not copied);
    /// only the dict shell is duplicated. Replacement entries created by
    /// <see cref="LoraStack.ApplyTo"/> live in the returned dict and are owned by the
    /// stack returned from <see cref="BuildAndApply"/>.</summary>
    public static Dictionary<string, Tensor> ShallowClone(IReadOnlyDictionary<string, Tensor> source)
    {
        if (source is null) return null;
        Dictionary<string, Tensor> copy = new(source.Count);
        foreach (KeyValuePair<string, Tensor> kv in source)
        {
            copy[kv.Key] = kv.Value;
        }
        return copy;
    }
}
