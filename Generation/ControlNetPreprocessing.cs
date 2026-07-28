using System.IO;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Engine.Requests;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Turns a raw ControlNet input image into the annotated hint the Engine expects.
/// <para>The Engine's <see cref="ControlNetConditioning"/> contract states the hint is "already preprocessed" —
/// annotation (Canny / Depth / OpenPose / …) is deliberately a host-side concern, because the annotators consume
/// SwarmUI <c>Image</c> objects and download their own weights through Swarm. This is where that happens: pick the
/// annotator from the ControlNet checkpoint's mode, run it, and hand back RGB pixels.</para>
/// <para>Feeding an un-annotated photo to a ControlNet produces near-garbage conditioning, so skipping this step is
/// not a benign no-op.</para></summary>
public static class ControlNetPreprocessing
{
    /// <summary>Infers the control type from the checkpoint's file name. Mirrors the engine's own
    /// <c>ControlNetLoader.DetectMode</c> heuristic (which is private), so the annotator we run always matches the
    /// mode the engine will load the adapter as. Keep the two in sync.</summary>
    public static ControlNetMode DetectMode(string filePath)
    {
        string lowered = Path.GetFileNameWithoutExtension(filePath ?? "").ToLowerInvariant();
        if (lowered.Contains("canny")) return ControlNetMode.Canny;
        if (lowered.Contains("depth") || lowered.Contains("zoedepth") || lowered.Contains("midas")) return ControlNetMode.Depth;
        if (lowered.Contains("openpose") || lowered.Contains("pose")) return ControlNetMode.OpenPose;
        if (lowered.Contains("scribble")) return ControlNetMode.Scribble;
        if (lowered.Contains("tile")) return ControlNetMode.Tile;
        if (lowered.Contains("normal")) return ControlNetMode.Normal;
        if (lowered.Contains("seg")) return ControlNetMode.Segmentation;
        if (lowered.Contains("inpaint")) return ControlNetMode.Inpaint;
        if (lowered.Contains("lineart") || lowered.Contains("line_art")) return ControlNetMode.LineArt;
        if (lowered.Contains("softedge") || lowered.Contains("hed") || lowered.Contains("pidi")) return ControlNetMode.SoftEdge;
        return ControlNetMode.Depth;
    }

    /// <summary>Runs the annotator for <paramref name="mode"/> over <paramref name="image"/> and returns the hint as
    /// Engine-native RGB pixels. <paramref name="backend"/> is invoked only for the model-driven annotators (Depth /
    /// OpenPose / SoftEdge / Scribble / Lineart / Normal / Segmentation) so a pure-Canny job never spins up a device.
    /// Tile and Inpaint carry the source image itself as the control signal and pass through unannotated.</summary>
    public static ImageData Preprocess(ControlNetMode mode, SwarmUI.Utils.Image image, int targetW, int targetH,
        Func<IBackend> backend, Action<string> log)
    {
        if (mode is ControlNetMode.Tile or ControlNetMode.Inpaint)
        {
            // These modes condition on the image itself; there is no annotator to run.
            log($"[ControlNet] mode={mode}: passing the source image through unannotated (by design).");
            (byte[] passthrough, int passW, int passH) = RgbToImage.ToHwcRgb(image);
            return new ImageData { Rgb = passthrough, Width = passW, Height = passH };
        }
        using Tensor hint = mode switch
        {
            ControlNetMode.Canny => CannyPreprocessor.Process(image, targetW, targetH),
            ControlNetMode.Depth => DepthPreprocessor.Process(image, targetW, targetH, backend(), m => log($"[Depth] {m}")),
            ControlNetMode.OpenPose => OpenPoseControlPreprocessor.Process(image, targetW, targetH, backend(), m => log($"[OpenPose] {m}")),
            ControlNetMode.SoftEdge => AnnotatorControlPreprocessors.ProcessSoftEdge(image, targetW, targetH, backend(), m => log($"[SoftEdge] {m}")),
            ControlNetMode.Scribble => AnnotatorControlPreprocessors.ProcessScribble(image, targetW, targetH, backend(), m => log($"[Scribble] {m}")),
            ControlNetMode.LineArt => AnnotatorControlPreprocessors.ProcessLineart(image, targetW, targetH, backend(), m => log($"[Lineart] {m}")),
            ControlNetMode.Normal => AnnotatorControlPreprocessors.ProcessNormal(image, targetW, targetH, backend(), m => log($"[Normal] {m}")),
            ControlNetMode.Segmentation => AnnotatorControlPreprocessors.ProcessSegment(image, targetW, targetH, backend(), m => log($"[Segment] {m}")),
            _ => throw new NotSupportedException(
                $"ControlNet mode '{mode}' has no preprocessor. Supported: Canny, Depth, OpenPose, SoftEdge/HED, "
                + "Scribble, Lineart, Normal, Segmentation (Tile/Inpaint pass through)."),
        };
        return ToImageData(hint);
    }

    /// <summary>Converts an annotator's <c>[1, 3, H, W]</c> F32 tensor in <c>[0, 1]</c> (the shape every preprocessor
    /// here returns) into row-major RGB24 — the Engine's on-the-wire image form.</summary>
    private static ImageData ToImageData(Tensor hint)
    {
        if (hint.Shape.Rank != 4 || hint.Shape[1] != 3)
        {
            throw new InvalidOperationException(
                $"ControlNet preprocessor returned a rank-{hint.Shape.Rank} tensor with {hint.Shape[1]} channels; expected [1, 3, H, W].");
        }
        int height = (int)hint.Shape[2];
        int width = (int)hint.Shape[3];
        int plane = width * height;
        ReadOnlySpan<float> src = hint.AsReadOnlySpan<float>();
        byte[] rgb = new byte[plane * 3];
        for (int c = 0; c < 3; c++)
        {
            int planeOffset = c * plane;
            for (int i = 0; i < plane; i++)
            {
                float v = src[planeOffset + i];
                // Annotators emit [0,1]; clamp defensively so a stray NaN/overshoot can't wrap a byte.
                int b = (int)MathF.Round(float.IsNaN(v) ? 0f : Math.Clamp(v, 0f, 1f) * 255f);
                rgb[(i * 3) + c] = (byte)b;
            }
        }
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }
}
