using System.Threading;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Builds Wan-Animate's pose driving clip by running YOLO11-pose per driving frame and rendering an
/// OpenPose-18 skeleton (<see cref="OpenPoseRenderer"/>) — the DWPose/ControlNet convention the checkpoint's
/// <c>pose_patch_embedding</c> was trained on. Replaces feeding the raw driving RGB (out-of-distribution) so the
/// model follows the driving motion faithfully.</summary>
public static class WanAnimatePosePreprocessor
{
    /// <summary>Returns the <c>[1, 3, numFrames, height, width]</c> pose-skeleton clip in [-1, 1] (black background,
    /// colored OpenPose limbs/joints).</summary>
    public static unsafe Tensor BuildPoseClip(
        IBackend backend, YoloPosePipeline pose, Image driving,
        int width, int height, int numFrames, CancellationToken cancel)
    {
        List<byte[]> frames = ControlVideoDecoder.DecodeFramesRgb(driving, width, height, numFrames, cancel);
        Tensor clip = new Tensor(new TensorShape([1L, 3, numFrames, height, width]), DType.F32);
        float* cp = (float*)clip.DataPointer;
        long framePix = (long)height * width;
        long perChannel = (long)numFrames * framePix;

        backend.PreloadWeights(pose.EnumerateWeights());
        int rendered = 0;
        try
        {
            for (int f = 0; f < numFrames; f++)
            {
                cancel.ThrowIfCancellationRequested();
                IReadOnlyList<PoseDetection> people = pose.Detect(frames[f], width, height, confidenceThreshold: 0.25f, iouThreshold: 0.45f);
                backend.FreeActivations();
                byte[] skel = OpenPoseRenderer.RenderBodyPose(people, width, height);   // HWC rgb24, black bg
                if (people.Count > 0) rendered++;
                for (long pix = 0; pix < framePix; pix++)
                    for (int c = 0; c < 3; c++)
                        cp[(long)c * perChannel + (long)f * framePix + pix] = skel[pix * 3 + c] / 127.5f - 1f;
            }
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(pose.EnumerateWeights());
        }
        Logs.Info($"[HartsyInference][Animate] pose preprocess: {rendered}/{numFrames} frames skeleton-rendered → {width}x{height} pose clip.");
        return clip;
    }
}
