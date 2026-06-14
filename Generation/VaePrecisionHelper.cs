using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// SDXL-VAE precision policy. SDXL's AutoencoderKL is famously unstable in F16:
/// resnet activations exceed F16's Â±65504 range, producing +Inf â NaN â all-black
/// output. This is documented industry-wide (madebyollin's sdxl-vae-fp16-fix repo,
/// Diffusers' <c>force_upcast: true</c>, ComfyUI's allow-list <c>[bf16, fp32]</c>).
/// HartsyInference matches ComfyUI's policy here: prefer BF16 on Ampere+ (same byte
/// count as F16 but F32-equivalent dynamic range, so structurally cannot overflow),
/// fall back to F32 on Turing/older where BF16 isn't supported.
/// </summary>
public static class VaePrecisionHelper
{
    /// <summary>Picks the preferred VAE compute dtype for a given backend. BF16 if the
    /// backend reports BF16 support (Ampere+ on CUDA, Vulkan with VK_KHR_shader_bfloat16,
    /// etc.), otherwise F32. Never returns F16 â see class summary.</summary>
    public static DType PreferredSdxlVaeDtype(IBackend backend)
    {
        return backend.Capabilities.SupportsBF16 ? DType.BF16 : DType.F32;
    }

    /// <summary>Casts a VAE weight dictionary to the target dtype in-place (returns a new
    /// dictionary with the same keys). Preserves tensors that are already at the target
    /// dtype to avoid unnecessary copies.</summary>
    public static Dictionary<string, Tensor> CastVaeWeights(
        Dictionary<string, Tensor> weights,
        DType targetDtype)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            result[kvp.Key] = kvp.Value.DType == targetDtype
                ? kvp.Value
                : kvp.Value.CastTo(targetDtype);
        }
        return result;
    }
}
