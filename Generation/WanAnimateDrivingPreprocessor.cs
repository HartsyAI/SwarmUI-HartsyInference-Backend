using System.Threading;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Prepares the two Wan-Animate driving inputs — the pose skeleton clip and the cropped face clip — from
/// the raw Init-Image driving video. The checkpoint was trained on a rendered OpenPose/DWPose skeleton (pose branch)
/// and a face-cropped square clip (face branch); feeding the raw clip is out-of-distribution and weakens motion
/// following. Pre-rendered overrides always take precedence; otherwise auto-preprocessing derives both in-backend.
///
/// <para><b>Phase 0 (current):</b> the override + raw paths are wired; the auto path falls back to the raw clip with
/// a one-time warning until the pose (YOLOv11-pose → OpenPose render) + face (YOLOv8-face detect+crop) vision models
/// land (Phases 1–2). See <c>docs/Design/WAN_ANIMATE_PREPROCESSING.md</c>.</para></summary>
public static class WanAnimateDrivingPreprocessor
{
    /// <summary>Builds the pose clip (<c>[1,3,T,H,W]</c>) and face clip (<c>[1,3,T−1,S,S]</c>) in [-1, 1].
    /// <paramref name="pose"/> is the shared YOLO11-pose pipeline (null when auto-preprocess is off / not needed).</summary>
    public static (Tensor poseClip, Tensor faceClip) BuildDrivingClips(
        IBackend backend, YoloPosePipeline pose, Image driving, Image poseOverride, Image faceOverride,
        bool autoPreprocess, int width, int height, int numFrames, int motionSize, CancellationToken cancel)
    {
        Tensor poseClip = BuildPoseClip(backend, pose, driving, poseOverride, autoPreprocess, width, height, numFrames, cancel);
        try
        {
            Tensor faceClip = BuildFaceClip(backend, pose, driving, faceOverride, autoPreprocess, width, height, motionSize, numFrames, cancel);
            return (poseClip, faceClip);
        }
        catch
        {
            poseClip.Dispose();
            throw;
        }
    }

    private static Tensor BuildPoseClip(IBackend backend, YoloPosePipeline pose, Image driving, Image poseOverride,
        bool autoPreprocess, int width, int height, int numFrames, CancellationToken cancel)
    {
        if (poseOverride is not null)
        {
            Logs.Info("[HartsyInference][Animate] pose branch: using the supplied pre-rendered pose video.");
            return ControlVideoDecoder.DecodeControlClip(poseOverride, width, height, numFrames, cancel);
        }
        if (autoPreprocess && pose is not null)
        {
            // Phase 2: render the OpenPose skeleton from per-frame YOLO11-pose keypoints (the format the DiT expects).
            return WanAnimatePosePreprocessor.BuildPoseClip(backend, pose, driving, width, height, numFrames, cancel);
        }
        return ControlVideoDecoder.DecodeControlClip(driving, width, height, numFrames, cancel);
    }

    private static Tensor BuildFaceClip(IBackend backend, YoloPosePipeline pose, Image driving, Image faceOverride,
        bool autoPreprocess, int width, int height, int motionSize, int numFrames, CancellationToken cancel)
    {
        if (faceOverride is not null)
        {
            Logs.Info("[HartsyInference][Animate] face branch: using the supplied pre-cropped face video.");
            return ControlVideoDecoder.DecodeControlClip(faceOverride, motionSize, motionSize, numFrames - 1, cancel);
        }
        if (autoPreprocess && pose is not null)
        {
            // Phase 1: detect the face per driving frame (YOLO11-pose keypoints) and crop a face-centered square.
            return WanAnimateFacePreprocessor.BuildFaceClip(backend, pose, driving, width, height, motionSize, numFrames - 1, cancel);
        }
        return ControlVideoDecoder.DecodeControlClip(driving, motionSize, motionSize, numFrames - 1, cancel);
    }
}
