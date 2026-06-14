using System.Globalization;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's <see cref="T2IParamTypes.Loras"/> + companion params into a list of
/// concrete <see cref="LoraSpec"/> entries (file path + per-component strengths) that
/// the per-architecture LoRA paths can feed into a HartsyInference <c>LoraStack</c>.
///
/// Mirrors the lookup logic in
/// <c>WorkflowGenerator.LoadLorasForConfinement</c> (see
/// <c>src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs</c>) so that a LoRA
/// dropdown selection routes to the same on-disk file Comfy would have used.
///
/// Section-confined LoRAs (segment blocks etc.) are not yet handled by HartsyInference;
/// any LoRA with a non-global confinement is skipped with a warning.
/// </summary>
public static class LoraResolver
{
    /// <summary>One resolved LoRA entry. <see cref="ModelStrength"/> is applied to
    /// UNet/Transformer/CLIP-G targets; <see cref="TencStrength"/> is applied to
    /// CLIP-L (the OpenAI text-encoder side that LoRA tooling traditionally lets users
    /// scale separately from the diffusion side). When the user didn't supply a
    /// distinct tenc weight, <see cref="TencStrength"/> equals <see cref="ModelStrength"/>.</summary>
    public sealed class LoraSpec
    {
        public required T2IModel Model { get; init; }
        public required string FilePath { get; init; }
        public required float ModelStrength { get; init; }
        public required float TencStrength { get; init; }
    }

    /// <summary>Resolve every selected LoRA in the input. Returns an empty list when
    /// the user picked no LoRAs or only picked LoRAs that aren't applicable to this
    /// generation (section-confined to a non-global block).</summary>
    public static List<LoraSpec> Resolve(T2IParamInput input)
    {
        if (input is null) return new List<LoraSpec>();
        if (!input.TryGet(T2IParamTypes.Loras, out List<string> loras) || loras is null || loras.Count == 0)
        {
            return new List<LoraSpec>();
        }

        List<string> weights = input.Get(T2IParamTypes.LoraWeights);
        List<string> tencWeights = input.Get(T2IParamTypes.LoraTencWeights);
        List<string> confinements = input.Get(T2IParamTypes.LoraSectionConfinement);

        if (!Program.T2IModelSets.TryGetValue("LoRA", out T2IModelHandler loraHandler))
        {
            Logs.Warning("[HartsyInference] LoRA model set not registered with Swarm; skipping LoRA application.");
            return new List<LoraSpec>();
        }

        List<LoraSpec> result = new(loras.Count);
        for (int i = 0; i < loras.Count; i++)
        {
            // Skip non-global confinement: HartsyInference doesn't model per-segment
            // LoRA scopes yet, so honoring the global slot only is the safe behavior.
            if (confinements is not null && i < confinements.Count
                && int.TryParse(confinements[i], out int confinementId)
                && confinementId > 0)
            {
                Logs.Debug($"[HartsyInference] LoRA '{loras[i]}' has section confinement={confinementId}; skipping (not yet supported).");
                continue;
            }

            // Match Comfy's resolution order: try with .safetensors suffix first, then raw.
            if (!loraHandler.Models.TryGetValue(loras[i] + ".safetensors", out T2IModel lora)
                && !loraHandler.Models.TryGetValue(loras[i], out lora))
            {
                throw new InvalidOperationException($"LoRA Model '{loras[i]}' not found in the model set.");
            }
            if (string.IsNullOrWhiteSpace(lora?.RawFilePath))
            {
                throw new InvalidOperationException($"LoRA '{lora?.Name ?? loras[i]}' has no resolvable file path.");
            }

            float modelStrength = ParseStrength(weights, i, fallback: 1f);
            float tencStrength = ParseStrength(tencWeights, i, fallback: modelStrength);

            result.Add(new LoraSpec
            {
                Model = lora,
                FilePath = lora.RawFilePath,
                ModelStrength = modelStrength,
                TencStrength = tencStrength,
            });
        }

        return result;
    }

    private static float ParseStrength(List<string> raw, int index, float fallback)
    {
        if (raw is null || index >= raw.Count) return fallback;
        if (float.TryParse(raw[index], NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
        {
            return v;
        }
        return fallback;
    }
}
