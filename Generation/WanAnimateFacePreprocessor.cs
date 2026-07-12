using System.Threading;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.FaceDetection;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Builds Wan-Animate's face-motion driving clip by localizing the face per driving frame (YOLO11-pose
/// keypoints → <see cref="PoseFaceCrop"/>) and bilinearly resampling a face-centered square to the motion-encoder
/// resolution. Replaces the naïve whole-frame resample so the StyleGAN motion encoder sees a filled, tracked face.</summary>
public static class WanAnimateFacePreprocessor
{
    /// <summary>Returns the <c>[1, 3, faceFrames, motionSize, motionSize]</c> face clip in [-1, 1]. Driving frames are
    /// decoded at <paramref name="workWidth"/>×<paramref name="workHeight"/> (where pose runs — validated accurate at
    /// the output resolution), then each frame's face square is resampled to <paramref name="motionSize"/>².</summary>
    public static unsafe Tensor BuildFaceClip(
        IBackend backend, YoloPosePipeline pose, Image driving,
        int workWidth, int workHeight, int motionSize, int faceFrames, CancellationToken cancel)
    {
        List<byte[]> frames = ControlVideoDecoder.DecodeFramesRgb(driving, workWidth, workHeight, faceFrames, cancel);
        Tensor clip = new Tensor(new TensorShape([1L, 3, faceFrames, motionSize, motionSize]), DType.F32);
        float* cp = (float*)clip.DataPointer;
        long framePix = (long)motionSize * motionSize;
        long perChannel = (long)faceFrames * framePix;

        backend.PreloadWeights(pose.EnumerateWeights());
        int detected = 0, fallbacks = 0;
        try
        {
            for (int f = 0; f < faceFrames; f++)
            {
                cancel.ThrowIfCancellationRequested();
                byte[] rgb = frames[f];
                IReadOnlyList<PoseDetection> people = pose.Detect(rgb, workWidth, workHeight, confidenceThreshold: 0.25f, iouThreshold: 0.45f);
                backend.FreeActivations();
                PoseFaceCrop.Rect crop;
                if (people.Count > 0)
                {
                    PoseDetection best = people[0];
                    for (int i = 1; i < people.Count; i++)
                        if (people[i].Confidence > best.Confidence) best = people[i];
                    crop = PoseFaceCrop.ComputeSquareCrop(best, workWidth, workHeight);
                    detected++;
                }
                else
                {
                    float side = Math.Min(workWidth, workHeight);
                    crop = new PoseFaceCrop.Rect((workWidth - side) * 0.5f, (workHeight - side) * 0.5f, side);
                    fallbacks++;
                }
                SampleSquare(rgb, workWidth, workHeight, crop, motionSize, cp, f, framePix, perChannel);
            }
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(pose.EnumerateWeights());
        }
        Logs.Info($"[HartsyInference][Animate] face preprocess: {detected}/{faceFrames} frames face-detected"
            + $"{(fallbacks > 0 ? $", {fallbacks} center-crop fallback" : "")} → {motionSize}² face clip.");
        return clip;
    }

    /// <summary>Bilinearly samples the source frame's square crop into output frame <paramref name="f"/> of the CHW
    /// clip, in [-1, 1]; samples outside the frame read mid-gray so an off-image crop pads cleanly.</summary>
    private static unsafe void SampleSquare(byte[] rgb, int w, int h, PoseFaceCrop.Rect crop, int outSize,
        float* cp, int f, long framePix, long perChannel)
    {
        fixed (byte* src = rgb)
        {
            float step = crop.Size / outSize;
            for (int oy = 0; oy < outSize; oy++)
            {
                float sy = crop.Y + (oy + 0.5f) * step - 0.5f;
                int y0 = (int)MathF.Floor(sy);
                float fy = sy - y0;
                for (int ox = 0; ox < outSize; ox++)
                {
                    float sx = crop.X + (ox + 0.5f) * step - 0.5f;
                    int x0 = (int)MathF.Floor(sx);
                    float fx = sx - x0;
                    long outPix = (long)oy * outSize + ox;
                    for (int c = 0; c < 3; c++)
                    {
                        float v00 = Sample(src, w, h, x0, y0, c);
                        float v10 = Sample(src, w, h, x0 + 1, y0, c);
                        float v01 = Sample(src, w, h, x0, y0 + 1, c);
                        float v11 = Sample(src, w, h, x0 + 1, y0 + 1, c);
                        float top = v00 + (v10 - v00) * fx;
                        float bot = v01 + (v11 - v01) * fx;
                        float val = top + (bot - top) * fy;   // [0, 255]
                        cp[(long)c * perChannel + (long)f * framePix + outPix] = val / 127.5f - 1f;
                    }
                }
            }
        }
    }

    private static unsafe float Sample(byte* src, int w, int h, int x, int y, int c)
    {
        if (x < 0 || y < 0 || x >= w || y >= h) return 114f;   // mid-gray pad (matches the letterbox fill)
        return src[((long)y * w + x) * 3 + c];
    }
}
