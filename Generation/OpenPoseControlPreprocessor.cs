using System.IO;
using System.Runtime.CompilerServices;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using ISImage = SixLabors.ImageSharp.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// OpenPose skeleton extraction for ControlNet conditioning, backed by the engine's
/// <see cref="OpenPosePreprocessor"/> (YOLO11-pose detector + OpenPose BODY-18 renderer — the same
/// stack Wan-Animate's driving preprocessing uses). Returns <c>[1, 3, H, W]</c> F32 in <c>[0, 1]</c>
/// (black background, colored limbs), the control_v11p_sd15_openpose / SDXL openpose format.
/// </summary>
public static class OpenPoseControlPreprocessor
{
    private const string PoseWeightsFile = "yolo11n-pose-folded.safetensors";

    /// <summary>Per-backend preprocessor instances: <see cref="OpenPosePreprocessor"/> binds its backend at
    /// construction, so a single static instance would run every caller on whichever GPU touched it first.
    /// Weak keys let a disposed backend's entry die with it.</summary>
    private static readonly ConditionalWeakTable<IBackend, Entry> s_perBackend = new();

    /// <summary>Renders the OpenPose skeleton map for <paramref name="input"/> at the generation resolution.</summary>
    public static Tensor Process(Image input, int targetWidth, int targetHeight, IBackend backend, Action<string> log)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (backend is null) throw new ArgumentNullException(nameof(backend));

        Entry entry = s_perBackend.GetOrCreateValue(backend);
        // One lock spanning load + forward: the pose model is not safe for concurrent forwards, and the load
        // must not race a sibling generation on the same backend.
        lock (entry.RunLock)
        {
            if (entry.Preprocessor is null)
            {
                string path = ResolvePoseWeights();
                log($"Loading YOLO11n-pose for OpenPose ControlNet preprocessing: {path}");
                entry.Preprocessor = new OpenPosePreprocessor(backend, path);
            }
            using var src = ISImage.Load<Rgb24>(input.RawData);
            byte[] rgb = new byte[src.Width * src.Height * 3];
            src.CopyPixelDataTo(rgb);
            Tensor result = entry.Preprocessor.Process(rgb, src.Width, src.Height, targetWidth, targetHeight);
            log($"OpenPose skeleton rendered at {targetWidth}x{targetHeight} (source {src.Width}x{src.Height}).");
            return result;
        }
    }

    /// <summary>Same resolution rule as Wan-Animate: the folded YOLO11n-pose safetensors lives in the shared Clip model folder.</summary>
    private static string ResolvePoseWeights()
    {
        if (!Program.T2IModelSets.TryGetValue("Clip", out T2IModelHandler handler) || handler.FolderPaths.Length == 0)
        {
            throw new SwarmUserErrorException("HartsyInference: cannot locate the model folder for the YOLO11n-pose weights.");
        }
        string path = Path.Combine(handler.FolderPaths[0], PoseWeightsFile);
        if (!File.Exists(path))
        {
            throw new SwarmUserErrorException(
                $"HartsyInference: OpenPose ControlNet preprocessing needs the YOLO11n-pose weights at '{path}'. "
                + "Convert Ultralytics 'yolo11n-pose.pt' with tests/python-reference/convert_yolov8_pt_to_safetensors.py.");
        }
        return path;
    }

    /// <summary>Per-backend cached preprocessor plus its run lock.</summary>
    private sealed class Entry
    {
        public readonly object RunLock = new();
        public OpenPosePreprocessor Preprocessor;
    }
}
