using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Variation seed support (Comfy parity: SwarmKSampler's second-seed blended noise).
/// Builds the initial-noise tensor as <c>slerp(noise(seed), noise(variationSeed), strength)</c>
/// and hands it to the pipeline via <c>TextToImageRequest.InitialNoise</c> (the pipeline
/// takes ownership and disposes it). Strength 0 reproduces the base seed exactly —
/// <c>SeedGenerator.CreateNoise</c> is the same generator the pipeline would have used.
///
/// Currently wired for the spatial-latent architectures (SD 1.5 / SDXL, latent
/// [1, 4, H/8, W/8]); packed-latent archs (Flux, SD3) need their per-arch shapes wired
/// before the validation gate in <c>HartsyInferenceBackend.IsValidForThisBackend</c> lifts.
/// </summary>
public static class VariationSeedResolver
{
    /// <summary>Latent channel count for SD 1.5 / SDXL spatial latents.</summary>
    public const int SdLatentChannels = 4;

    /// <summary>Latent channel count for Flux (and other 16-channel VAE) spatial latents.
    /// Flux's pipeline injects the unpacked <c>[1, 16, H/8, W/8]</c> noise BEFORE the 2×2
    /// patchify, so the variation noise is built in that unpacked space.</summary>
    public const int FluxLatentChannels = 16;

    /// <summary>Returns the blended initial-noise tensor, or null when no variation seed
    /// is requested (param unset, strength 0) — null means "let the pipeline seed normally".
    /// Caller passes the SAME int seed value it puts on the request, so strength→0
    /// continuity holds. <paramref name="latentChannels"/> must match the architecture's
    /// unpacked latent channel count (4 for SD 1.5 / SDXL, 16 for Flux); the pipeline
    /// validates the injected-noise shape and throws on mismatch.</summary>
    public static Tensor Resolve(T2IParamInput input, int width, int height, int? requestSeed, int latentChannels = SdLatentChannels)
    {
        if (!input.TryGet(T2IParamTypes.VariationSeedStrength, out double strength) || strength <= 0)
        {
            return null;
        }
        if (!input.TryGet(T2IParamTypes.VariationSeed, out long varSeedLong))
        {
            return null;
        }
        int baseSeed = requestSeed ?? SeedGenerator.RandomSeed();
        int varSeed = varSeedLong < 0 ? SeedGenerator.RandomSeed() : (int)(varSeedLong & 0x7FFFFFFF);
        strength = Math.Clamp(strength, 0.0, 1.0);

        TensorShape shape = new(1, latentChannels, height / 8, width / 8);
        Tensor baseNoise = SeedGenerator.CreateNoise(shape, baseSeed);
        if (strength >= 1.0)
        {
            // Full replacement: just use the variation seed's noise.
            baseNoise.Dispose();
            Logs.Verbose($"[HartsyInference] Variation seed {varSeed} at strength 1 — replacing base noise entirely.");
            return SeedGenerator.CreateNoise(shape, varSeed);
        }
        Tensor varNoise = SeedGenerator.CreateNoise(shape, varSeed);
        SlerpInPlace(baseNoise, varNoise, (float)strength);
        varNoise.Dispose();
        Logs.Verbose($"[HartsyInference] Variation seed {varSeed} blended at strength {strength} (slerp).");
        return baseNoise;
    }

    /// <summary>Spherical interpolation of <paramref name="a"/> toward <paramref name="b"/>
    /// by <paramref name="t"/>, written into <paramref name="a"/>. Slerp (not lerp) keeps
    /// the result's norm consistent with unit-variance Gaussian noise — straight lerp of
    /// two independent Gaussians shrinks variance and washes the image out. Matches the
    /// slerp in Comfy's SwarmKSampler variation-seed path.</summary>
    private static unsafe void SlerpInPlace(Tensor a, Tensor b, float t)
    {
        int count = (int)a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < count; i++)
        {
            dot += (double)pa[i] * pb[i];
            na += (double)pa[i] * pa[i];
            nb += (double)pb[i] * pb[i];
        }
        na = Math.Sqrt(na);
        nb = Math.Sqrt(nb);
        double cosTheta = Math.Clamp(dot / Math.Max(na * nb, 1e-12), -1.0, 1.0);
        double theta = Math.Acos(cosTheta);
        double sinTheta = Math.Sin(theta);

        float wa, wb;
        if (sinTheta < 1e-6)
        {
            // Nearly colinear — fall back to lerp (slerp is numerically unstable here).
            wa = 1.0f - t;
            wb = t;
        }
        else
        {
            wa = (float)(Math.Sin((1.0 - t) * theta) / sinTheta);
            wb = (float)(Math.Sin(t * theta) / sinTheta);
        }
        for (int i = 0; i < count; i++)
        {
            pa[i] = wa * pa[i] + wb * pb[i];
        }
    }
}
