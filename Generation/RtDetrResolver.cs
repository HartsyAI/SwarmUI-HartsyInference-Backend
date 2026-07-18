using System.IO;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Vision.Detection;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Closed-set (80-class COCO) object detection via the engine's pure-C# RT-DETR (<c>rtdetr_r18vd</c>) —
/// a transformer, NMS-free alternative to YOLO whose decoder uses the GPU-routed multi-scale deformable
/// attention. <see cref="SegmentResolver"/> feeds the chosen box to SAM 2 for a pixel-accurate mask.
///
/// <para>Weights: a <c>.safetensors</c> under a conventional <c>rtdetr</c> folder (sibling of the SD roots).
/// The pipeline is cached per checkpoint (weight load + conversion is expensive to repeat).</para>
/// </summary>
public static class RtDetrResolver
{
    private static readonly object s_lock = new();
    private static readonly Dictionary<string, (IBackend backend, RtDetrPipeline pipe)> s_cache = new();

    /// <summary>Detects COCO objects in the HWC-RGB pixels, returning normalized-xywh results above
    /// <paramref name="threshold"/>. Throws when no <c>rtdetr</c> model is installed.</summary>
    public static IReadOnlyList<DetectionResult> Detect(IBackend backend, byte[] rgb, int w, int h, float threshold, Action<string> log)
    {
        string ckpt = ResolveModelPath();
        if (ckpt is null)
        {
            throw new InvalidOperationException(
                "RT-DETR model not found. Place a converted rtdetr_r18vd .safetensors in a 'rtdetr' folder under your model root.");
        }
        RtDetrPipeline pipe = GetOrLoad(backend, ckpt, log);
        return pipe.Detect(rgb, w, h, threshold);
    }

    private static RtDetrPipeline GetOrLoad(IBackend backend, string ckpt, Action<string> log)
    {
        lock (s_lock)
        {
            if (s_cache.TryGetValue(ckpt, out (IBackend backend, RtDetrPipeline pipe) entry))
            {
                if (ReferenceEquals(entry.backend, backend)) return entry.pipe;
                entry.pipe.Dispose();
                s_cache.Remove(ckpt);
            }
            log($"Loading RT-DETR (r18vd): {ckpt}");
            RtDetrPipeline pipe = new RtDetrPipeline(backend, RtDetrConfig.R18vd, ckpt, inputSize: 640, labels: CocoLabels.Names);
            s_cache[ckpt] = (backend, pipe);
            return pipe;
        }
    }

    /// <summary>Locates an RT-DETR <c>.safetensors</c> under a conventional <c>rtdetr</c> folder (sibling of the
    /// SD roots, plus each model root). Returns null when none is installed.</summary>
    private static string ResolveModelPath()
    {
        List<string> roots = [];
        if (Program.T2IModelSets.TryGetValue("Stable-Diffusion", out T2IModelHandler sd))
        {
            foreach (string fp in sd.FolderPaths)
            {
                roots.Add(Path.Combine(fp, "rtdetr"));
                string parent = Path.GetDirectoryName(fp.TrimEnd('/', '\\'));
                if (!string.IsNullOrEmpty(parent)) roots.Add(Path.Combine(parent, "rtdetr"));
            }
        }
        foreach (string root in Program.ServerSettings.Paths.ActualModelRoots)
        {
            roots.Add(Path.Combine(root, "rtdetr"));
        }
        foreach (string root in roots.Distinct())
        {
            if (!Directory.Exists(root)) continue;
            foreach (string dir in new[] { root }.Concat(Directory.EnumerateDirectories(root)))
            {
                foreach (string f in Directory.EnumerateFiles(dir, "*.safetensors").OrderBy(x => x, StringComparer.Ordinal))
                {
                    return f;
                }
            }
        }
        return null;
    }
}
