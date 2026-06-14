using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Implements Swarm's <c>&lt;segment:yolo-...&gt;</c> auto-refinement (Comfy parity) for the
/// HartsyInference backend. After the base (and refiner) stage produces an image, each segment
/// part is detected with YOLO (<see cref="SegmentResolver"/>), turned into a mask, and the masked
/// region is re-denoised with the segment's own prompt via the architecture's existing
/// img2img + inpaint-blend path — fed back through the <paramref name="reGenerate"/> delegate.
///
/// <para>This is the in-process equivalent of Comfy's SwarmYoloDetection → crop → KSampler →
/// recomposite chain. We run the re-denoise at full image resolution (the mask localizes the
/// change); the crop-to-bbox optimization (<see cref="T2IParamTypes.SegmentMaskOversize"/>) is a
/// future efficiency follow-up, not a correctness gap.</para>
/// </summary>
public static class SegmentRefiner
{
    /// <summary>True if the prompt contains any <c>&lt;segment:&gt;</c> parts.</summary>
    public static bool HasSegments(T2IParamInput input)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        if (!prompt.Contains("<segment:", StringComparison.OrdinalIgnoreCase)) return false;
        return new PromptRegion(prompt).Parts.Any(p => p.Type == PromptRegion.PartType.Segment);
    }

    /// <summary>True if every segment part targets YOLO (the only detector we support). Used by the
    /// validator to refuse text/CLIP-Seg targets upfront with a clear message.</summary>
    public static bool AllSegmentsAreYolo(T2IParamInput input)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        return new PromptRegion(prompt).Parts
            .Where(p => p.Type == PromptRegion.PartType.Segment)
            .All(SegmentResolver.IsYoloTarget);
    }

    /// <summary>Runs every segment part over each base image, returning the refined images.
    /// <paramref name="reGenerate"/> takes a (cloned) input with InitImage + MaskImage + segment
    /// prompt set and returns the re-denoised image(s) using the active architecture's loader.</summary>
    public static Image[] Apply(
        IBackend backend,
        Image[] baseImages,
        T2IParamInput input,
        Func<T2IParamInput, Image[]> reGenerate,
        Action<string> log,
        CancellationToken cancel)
    {
        if (baseImages is null || baseImages.Length == 0) return baseImages;
        PromptRegion region = new(input.Get(T2IParamTypes.Prompt) ?? "");
        PromptRegion.Part[] parts = region.Parts.Where(p => p.Type == PromptRegion.PartType.Segment).ToArray();
        if (parts.Length == 0) return baseImages;

        PromptRegion negativeRegion = new(input.Get(T2IParamTypes.NegativePrompt) ?? "");

        Image[] result = new Image[baseImages.Length];
        for (int i = 0; i < baseImages.Length; i++)
        {
            Image current = baseImages[i];
            int segIdx = 0;
            foreach (PromptRegion.Part part in parts)
            {
                cancel.ThrowIfCancellationRequested();
                Image mask = SegmentResolver.BuildYoloMask(backend, current, part, input, log);
                if (mask is null)
                {
                    segIdx++;
                    continue; // nothing detected for this segment — leave the image as-is
                }

                // Match this segment's negative to a same-target negative part if present, else the global negative.
                string segNegative = negativeRegion.Parts
                    .FirstOrDefault(p => p.Type == PromptRegion.PartType.Segment && p.DataText == part.DataText)?.Prompt
                    ?? negativeRegion.GlobalPrompt;

                T2IParamInput clone = input.Clone();
                clone.Set(T2IParamTypes.Prompt, string.IsNullOrWhiteSpace(part.Prompt) ? region.GlobalPrompt : part.Prompt);
                clone.Set(T2IParamTypes.NegativePrompt, segNegative ?? "");
                clone.Set(T2IParamTypes.InitImage, current);
                clone.Set(T2IParamTypes.MaskImage, mask);
                // Strength2 (default 0.6) is the segment's denoise amount → img2img creativity.
                clone.Set(T2IParamTypes.InitImageCreativity, part.Strength2);
                // Per-segment step/cfg overrides if the user set them.
                if (input.TryGet(T2IParamTypes.SegmentSteps, out int segSteps) && segSteps > 0)
                {
                    clone.Set(T2IParamTypes.Steps, segSteps);
                }
                if (input.TryGet(T2IParamTypes.SegmentCFGScale, out double segCfg) && segCfg > 0)
                {
                    clone.Set(T2IParamTypes.CFGScale, segCfg);
                }

                long t0 = Environment.TickCount64;
                Image[] refined = reGenerate(clone);
                if (refined is not null && refined.Length > 0 && refined[0] is not null)
                {
                    current = refined[0];
                    log($"[Segment] Applied segment {segIdx + 1}/{parts.Length} (creativity={part.Strength2:F2}) " +
                        $"in {Environment.TickCount64 - t0}ms.");
                }
                segIdx++;
            }
            result[i] = current;
        }
        return result;
    }
}
