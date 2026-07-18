using System.IO;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Vision.Detection.GroundingDino;
using ISImage = SixLabors.ImageSharp.Image;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Open-vocabulary, text-prompted object detection via the engine's pure-C# Grounding DINO
/// (<c>grounding-dino-tiny</c>). Given a base image and a free-text phrase ("a red car"), it runs
/// DETR-style preprocessing → BERT tokenization → the full detector (Swin backbone + cross-modality
/// encoder with the GPU-routed multi-scale deformable attention + two-stage decoder) → post-process,
/// returning boxes in the base-image coordinate frame. <see cref="SegmentResolver"/> feeds the chosen
/// box to SAM 2 for a pixel-accurate mask.
///
/// <para>Weights: a <c>grounding-dino</c> folder containing <c>model.safetensors</c> and the BERT
/// <c>vocab.txt</c> is resolved under the model roots (sibling of the SD roots, mirroring the SAM 2 /
/// YOLO folder convention). The model + tokenizer are cached per checkpoint (construction copies every
/// weight, so this is expensive to repeat).</para>
/// </summary>
public static class GroundingDinoResolver
{
    // GDINO uses ImageNet normalization on a DETR-resized image (shortest edge 800, longest ≤ 1333).
    private static readonly float[] s_mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] s_std = { 0.229f, 0.224f, 0.225f };
    private const int ShortestEdge = 800;
    private const int LongestEdge = 1333;

    private static readonly object s_lock = new();
    private static readonly Dictionary<string, Entry> s_cache = new();

    private sealed class Entry
    {
        public required IBackend Backend { get; init; }
        public required GroundingDinoModel Model { get; init; }
        // The model's weight tensors are owned by the loader — it must stay alive with the cached model.
        public required SafeTensorsLoader Loader { get; init; }
        public required BertWordPieceTokenizer Tokenizer { get; init; }
        public required string[] Vocab { get; init; }
    }

    /// <summary>Detects <paramref name="text"/> in <paramref name="baseImage"/>, returning boxes (base-image
    /// pixel coords) above <paramref name="threshold"/>. Throws when no <c>grounding-dino</c> model is installed.</summary>
    public static List<GroundingDinoDetection> Detect(IBackend backend, Image baseImage, string text, float threshold, Action<string> log)
    {
        (string ckpt, string vocabPath) = ResolveModelPaths();
        if (ckpt is null)
        {
            throw new InvalidOperationException(
                "Grounding DINO model not found. Place 'model.safetensors' and 'vocab.txt' (from " +
                "IDEA-Research/grounding-dino-tiny) in a 'grounding-dino' folder under your model root.");
        }
        Entry e = GetOrLoad(backend, ckpt, vocabPath, log);

        // GDINO convention: lowercase phrase, period-terminated (the period is a query separator token).
        string query = text.Trim();
        if (query.Length == 0) return new List<GroundingDinoDetection>();
        if (!query.EndsWith(".")) query += " .";
        int[] ids = e.Tokenizer.EncodeWithSpecial(query).ToArray();

        Tensor pixels = BuildPixels(baseImage, out int origW, out int origH);
        GroundingDinoDetector.Output outp;
        try
        {
            outp = e.Model.Forward(backend, pixels, ids);
        }
        finally
        {
            pixels.Dispose();
        }

        try
        {
            float textThreshold = Math.Min(0.3f, threshold);
            return GroundingDinoPipeline.PostProcess(outp.Logits, outp.PredBoxes, ids, e.Vocab, origH, origW, threshold, textThreshold);
        }
        finally
        {
            outp.Logits.Dispose();
            outp.PredBoxes.Dispose();
        }
    }

    /// <summary>Builds the DETR-preprocessed pixel tensor <c>[1,3,H,W]</c> (aspect-preserving resize to shortest
    /// edge 800 / longest ≤ 1333, snapped even, then ImageNet-normalized) and reports the original image size for
    /// box rescaling. Exact HF-processor parity isn't needed here — SAM 2 refines the box downstream.</summary>
    private static unsafe Tensor BuildPixels(Image baseImage, out int origW, out int origH)
    {
        using SixLabors.ImageSharp.Image<Rgb24> frame = ISImage.Load<Rgb24>(baseImage.RawData);
        origW = frame.Width;
        origH = frame.Height;
        double scale = (double)ShortestEdge / Math.Min(origW, origH);
        if (Math.Round(Math.Max(origW, origH) * scale) > LongestEdge)
        {
            scale = (double)LongestEdge / Math.Max(origW, origH);
        }
        int newW = Math.Max(2, (int)Math.Round(origW * scale));
        int newH = Math.Max(2, (int)Math.Round(origH * scale));
        newW -= newW % 2;
        newH -= newH % 2;
        frame.Mutate(x => x.Resize(newW, newH));
        byte[] rgb = new byte[newW * newH * 3];
        frame.CopyPixelDataTo(rgb);

        Tensor t = new Tensor(new TensorShape(1, 3, newH, newW), DType.F32);
        float* dp = (float*)t.DataPointer;
        int spatial = newW * newH;
        const float inv255 = 1f / 255f;
        for (int c = 0; c < 3; c++)
        {
            float mean = s_mean[c], invStd = 1f / s_std[c];
            int chOff = c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                dp[chOff + i] = (rgb[i * 3 + c] * inv255 - mean) * invStd;
            }
        }
        return t;
    }

    private static Entry GetOrLoad(IBackend backend, string ckpt, string vocabPath, Action<string> log)
    {
        lock (s_lock)
        {
            if (s_cache.TryGetValue(ckpt, out Entry entry))
            {
                if (ReferenceEquals(entry.Backend, backend)) return entry;
                entry.Model.Dispose();
                entry.Loader.Dispose();
                entry.Tokenizer.Dispose();
                s_cache.Remove(ckpt);
            }
            log($"Loading Grounding DINO (tiny): {ckpt}");
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(ckpt);
            GroundingDinoModel model = new GroundingDinoModel(GroundingDinoConfig.Tiny);
            model.LoadWeights(loader.GetAllTensors());
            BertWordPieceTokenizer tokenizer = new BertWordPieceTokenizer(vocabPath, lowerCase: true);
            string[] vocab = GroundingDinoPipeline.LoadVocab(vocabPath);
            Entry created = new Entry
            {
                Backend = backend,
                Model = model,
                Loader = loader,
                Tokenizer = tokenizer,
                Vocab = vocab,
            };
            s_cache[ckpt] = created;
            return created;
        }
    }

    /// <summary>Locates <c>model.safetensors</c> + <c>vocab.txt</c> under a conventional <c>grounding-dino</c>
    /// folder (sibling of the SD roots, plus each model root). Returns (null, null) when none is installed.</summary>
    private static (string ckpt, string vocab) ResolveModelPaths()
    {
        List<string> roots = [];
        if (Program.T2IModelSets.TryGetValue("Stable-Diffusion", out T2IModelHandler sd))
        {
            foreach (string fp in sd.FolderPaths)
            {
                roots.Add(Path.Combine(fp, "grounding-dino"));
                string parent = Path.GetDirectoryName(fp.TrimEnd('/', '\\'));
                if (!string.IsNullOrEmpty(parent)) roots.Add(Path.Combine(parent, "grounding-dino"));
            }
        }
        foreach (string root in Program.ServerSettings.Paths.ActualModelRoots)
        {
            roots.Add(Path.Combine(root, "grounding-dino"));
        }
        foreach (string root in roots.Distinct())
        {
            if (!Directory.Exists(root)) continue;
            // Accept the safetensors either directly in the folder or nested one level (HF snapshot layout).
            foreach (string dir in new[] { root }.Concat(Directory.EnumerateDirectories(root)))
            {
                string ckpt = Path.Combine(dir, "model.safetensors");
                if (!File.Exists(ckpt)) continue;
                string vocab = Path.Combine(dir, "vocab.txt");
                if (!File.Exists(vocab)) continue;
                return (ckpt, vocab);
            }
        }
        return (null, null);
    }
}
