using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.Vision.DepthAnything;
using ISImage = SixLabors.ImageSharp.Image;
using EngineDepthPreprocessor = HartsyInference.Vision.DepthAnything.DepthAnythingPreprocessor;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Depth-map extraction for Flux Depth / depth ControlNet conditioning, backed by the engine's
/// Depth-Anything-V2 model (ViT-L by default — the checkpoint the Comfy ecosystem validates
/// against). Auto-downloads the annotator on first use, runs preprocess → model → min-max
/// normalize, and returns a <c>[1, 3, H, W]</c> F32 tensor in <c>[0, 1]</c> (relative depth
/// replicated to 3 channels, white = near), matching ComfyUI's DepthAnythingV2Preprocessor output.
/// </summary>
public static class DepthPreprocessor
{
    private const string VitlFileName = "depth_anything_v2_vitl.pth";
    private const string VitlUrl = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth";
    private const string VitlHash = "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345";

    private static readonly object s_lock = new();
    private static DepthAnythingV2Model s_model;
    // The model's weight tensors are owned by the loader — it must stay alive with the cached model.
    private static PytorchPickleLoader s_loader;

    /// <summary>Produces the depth conditioning map for <paramref name="input"/> at the generation resolution. <paramref name="fluxScaling"/> selects BFL's max-only depth scaling (FLUX.1-Depth checkpoints) instead of the min-max stretch SD ControlNets expect.</summary>
    public static unsafe Tensor Process(Image input, int targetWidth, int targetHeight, IBackend backend, Action<string> log, bool fluxScaling = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        DepthAnythingV2Model model = GetOrLoad(log);
        using var src = ISImage.Load<Rgb24>(input.RawData);
        int srcW = src.Width;
        int srcH = src.Height;
        byte[] rgb = new byte[srcW * srcH * 3];
        src.CopyPixelDataTo(rgb);

        EngineDepthPreprocessor pre = new EngineDepthPreprocessor();
        Tensor pixels = pre.Preprocess(rgb, srcW, srcH);
        Tensor depth;
        try
        {
            depth = model.Forward(backend, pixels);
        }
        finally
        {
            pixels.Dispose();
        }

        float[] unitDepth;
        try
        {
            unitDepth = EngineDepthPreprocessor.PostprocessToUnit(depth, targetWidth, targetHeight, minMaxNormalize: !fluxScaling);
        }
        finally
        {
            depth.Dispose();
        }
        backend.FreeWeights(model.EnumerateWeights());

        Tensor output = new Tensor(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        float* dp = (float*)output.DataPointer;
        int spatial = targetWidth * targetHeight;
        for (int c = 0; c < 3; c++)
        {
            fixed (float* up = unitDepth)
            {
                Buffer.MemoryCopy(up, dp + (long)c * spatial, spatial * sizeof(float), spatial * sizeof(float));
            }
        }
        log($"Depth-Anything-V2 map ready: {targetWidth}x{targetHeight} (source {srcW}x{srcH}).");
        return output;
    }

    private static DepthAnythingV2Model GetOrLoad(Action<string> log)
    {
        lock (s_lock)
        {
            if (s_model is not null)
            {
                return s_model;
            }
            string path = AnnotatorDownloader.EnsureAnnotator(VitlFileName, VitlUrl, VitlHash, log);
            log($"Loading Depth-Anything-V2 ViT-L: {path}");
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            DepthAnythingV2Model model = new DepthAnythingV2Model(DepthAnythingPreset.Large);
            model.LoadWeights(loader.GetAllTensors());
            s_loader = loader;
            s_model = model;
            return model;
        }
    }
}
