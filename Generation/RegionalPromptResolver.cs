using System;
using System.Collections.Generic;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Translates SwarmUI <c>&lt;region:x,y,w,h,strength&gt;</c> / <c>&lt;object:…&gt;</c> prompt parts into the engine's
/// <see cref="RegionalPlan"/>. Each part's bbox (fractions of image area) becomes a pixel-space <see cref="RectMask"/>
/// and its prompt is encoded via the supplied delegate; the engine turns the plan into a per-step attention bias.
///
/// <para>Returns <c>null</c> when the prompt carries no region/object parts, so callers keep the plain
/// single-conditioning path byte-identical (zero behavior change, zero cost, cannot break ordinary gens).</para>
///
/// <para><c>&lt;object:&gt;</c> parts contribute their regional CONDITIONING here identically to <c>&lt;region:&gt;</c>;
/// the extra inpaint-back pass that additionally distinguishes objects is a separate feature not yet wired.</para>
/// </summary>
public static class RegionalPromptResolver
{
    /// <summary>Builds a <see cref="RegionalPlan"/> from the raw (untagged-stripped) prompt, or null if it has no
    /// region/object parts. <paramref name="baseCond"/> is the already-encoded global conditioning.
    /// <paramref name="encodeRegion"/> encodes one region's prompt text into a <c>[1, L, featDim]</c> caption
    /// embedding (same encoder/path as the global prompt); the caller owns disposal via <see cref="DisposeRegions"/>.</summary>
    public static RegionalPlan Resolve(
        string rawPrompt, Tensor baseCond, int width, int height, int steps,
        Func<string, Tensor> encodeRegion)
    {
        if (string.IsNullOrEmpty(rawPrompt) || !rawPrompt.Contains('<'))
        {
            return null;
        }
        PromptRegion parsed = new PromptRegion(rawPrompt);
        List<RegionConditioning> regions = new();
        foreach (PromptRegion.Part part in parsed.Parts)
        {
            if (part.Type != PromptRegion.PartType.Region && part.Type != PromptRegion.PartType.Object)
            {
                continue;
            }
            // Fractions → pixel rect, clamped inside the canvas (rounding can overshoot by a pixel).
            int rx = Math.Clamp((int)MathF.Round(part.X * width), 0, Math.Max(0, width - 1));
            int ry = Math.Clamp((int)MathF.Round(part.Y * height), 0, Math.Max(0, height - 1));
            int rw = Math.Clamp((int)MathF.Round(part.Width * width), 1, width - rx);
            int rh = Math.Clamp((int)MathF.Round(part.Height * height), 1, height - ry);
            RegionMask mask = RegionMask.FromRect(new RectMask(rx, ry, rw, rh), width, height);
            Tensor cond = encodeRegion(part.Prompt ?? "");
            regions.Add(new RegionConditioning(cond, mask, (float)part.Strength, 0, steps));
        }
        if (regions.Count == 0)
        {
            return null;
        }
        return new RegionalPlan { BaseCond = baseCond, Regions = regions };
    }

    /// <summary>True when the raw prompt carries region/object parts. Cheap pre-check so callers can decide TE
    /// weight residency (region encodes need the encoder even when the global conditioning is cached).</summary>
    public static bool HasRegionParts(string rawPrompt)
    {
        if (string.IsNullOrEmpty(rawPrompt) || !rawPrompt.Contains('<'))
        {
            return false;
        }
        PromptRegion parsed = new PromptRegion(rawPrompt);
        foreach (PromptRegion.Part part in parsed.Parts)
        {
            if (part.Type == PromptRegion.PartType.Region || part.Type == PromptRegion.PartType.Object)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Disposes the per-region conditioning tensors owned by a plan (call after the gen completes). Null-safe.</summary>
    public static void DisposeRegions(RegionalPlan plan)
    {
        if (plan is null)
        {
            return;
        }
        foreach (RegionConditioning region in plan.Regions)
        {
            region.Cond?.Dispose();
        }
    }
}
