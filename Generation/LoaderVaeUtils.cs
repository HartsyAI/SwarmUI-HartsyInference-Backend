using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Shared staging for loaders that consume the FLUX.1 16-channel autoencoder (<c>ae.safetensors</c>,
/// original ldm naming): remaps every key to the diffusers naming <c>VaeDecoder</c> expects via
/// <see cref="CheckpointConvertUtils.ConvertVaeKey"/> and upcasts 16-bit weights to F32 (the F32 VAE path).
/// Mirrors the engine tests' <c>ConvertVaeWeights</c> helpers (OmniGen2/Lumina2/Boogu).</summary>
internal static class LoaderVaeUtils
{
    /// <summary>Loads a FLUX.1-family VAE file and returns diffusers-keyed F32 weights. Keys already in
    /// diffusers naming pass through unchanged (ConvertVaeKey is identity-tolerant); unknown keys are dropped.
    /// The returned loader owns the tensor memory — keep it alive with the cache entry.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadFluxVaeF32(string filePath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(filePath);
        try
        {
            Dictionary<string, Tensor> result = new();
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                string diffusersKey = CheckpointConvertUtils.ConvertVaeKey(kvp.Key);
                if (diffusersKey is null)
                    continue;
                DType dt = kvp.Value.DType;
                result[diffusersKey] = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
            }
            return (result, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }
}
