using System.Diagnostics;
using System.IO;
using SwarmUI.Media;
using SwarmUI.Utils;
using HartsyInference.Core.Tensors;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Decodes a control input (the Init Image slot, which for Wan VACE carries a control <em>video</em> —
/// pose/depth/edge/sketch frames) into the <c>[1, 3, T, H, W]</c> RGB tensor in [-1, 1] that
/// <see cref="HartsyInference.Video.Pipelines.WanVacePipeline.GenerateFromControl"/> expects.
///
/// <para>The inverse of <see cref="VideoOutputEncoder"/>: that pipes raw <c>rgb24</c> frames <em>to</em>
/// ffmpeg to mux a container; this pipes a container <em>through</em> ffmpeg (<c>-f rawvideo -pix_fmt
/// rgb24 -s WxH</c>) to recover frames, then packs them into the engine's CHW-per-frame tensor layout
/// (matching <see cref="HartsyInference.Video.VideoRgbFrames"/>'s <c>byte = (v + 1)·127.5</c>
/// convention). A still image (or any non-video media) is decoded once via ImageSharp and tiled across
/// all <paramref name="numFrames"/> frames — a degenerate constant control, so VACE still runs.</para>
/// </summary>
public static class ControlVideoDecoder
{
    /// <summary>Builds the <c>[1, 3, T, H, W]</c> control tensor (caller owns disposal). Frames are
    /// resampled to <paramref name="width"/>×<paramref name="height"/>; the source is truncated to
    /// <paramref name="numFrames"/> (longer) or padded by repeating its last frame (shorter).</summary>
    public static unsafe Tensor DecodeControlClip(Image control, int width, int height, int numFrames, CancellationToken cancel)
    {
        if (control is null) throw new ArgumentNullException(nameof(control));
        if (numFrames < 1) throw new ArgumentOutOfRangeException(nameof(numFrames));

        List<byte[]> frames = control.Type?.MetaType == MediaMetaType.Video
            ? DecodeVideoFrames(control.RawData, width, height, numFrames, cancel)
            : new List<byte[]> { RgbToImage.ToHwcRgbResized(control, width, height) };

        if (frames.Count == 0)
        {
            throw new SwarmUserErrorException(
                "HartsyInference: the VACE control video decoded to zero frames. Check the file is a valid video.");
        }
        // Truncate-or-pad (repeat last) to exactly numFrames.
        while (frames.Count < numFrames) frames.Add(frames[^1]);
        if (frames.Count > numFrames) frames.RemoveRange(numFrames, frames.Count - numFrames);

        long perFrame = (long)height * width;
        Tensor clip = new Tensor(new TensorShape([1L, 3, numFrames, height, width]), DType.F32);
        float* p = (float*)clip.DataPointer;
        for (int f = 0; f < numFrames; f++)
        {
            byte[] src = frames[f];   // interleaved HWC rgb24
            if (src.Length != perFrame * 3)
            {
                clip.Dispose();
                throw new InvalidOperationException(
                    $"Control frame {f} has {src.Length} bytes, expected {perFrame * 3} for {width}x{height}.");
            }
            for (long pix = 0; pix < perFrame; pix++)
            {
                for (int ci = 0; ci < 3; ci++)
                {
                    p[((long)ci * numFrames + f) * perFrame + pix] = src[pix * 3 + ci] / 127.5f - 1f;
                }
            }
        }
        Logs.Verbose($"[HartsyInference][VACE] Control clip decoded: {numFrames}f {width}x{height} ("
            + $"{(control.Type?.MetaType == MediaMetaType.Video ? "video" : "still tiled")}).");
        return clip;
    }

    /// <summary>Runs ffmpeg to decode up to <paramref name="numFrames"/> RGB24 frames from a video
    /// container, each resampled to <paramref name="width"/>×<paramref name="height"/>.</summary>
    private static List<byte[]> DecodeVideoFrames(byte[] videoData, int width, int height, int numFrames, CancellationToken cancel)
    {
        string ffmpeg = Utilities.FfmegLocation.Value;
        if (string.IsNullOrWhiteSpace(ffmpeg))
        {
            throw new SwarmUserErrorException(
                "HartsyInference VACE control-video decoding requires ffmpeg, but none was found. "
                + "Install ffmpeg on your system PATH (or install the ComfyUI self-start backend, whose bundled copy Swarm reuses).");
        }
        string tmpFile = Path.Combine(Path.GetTempPath(), $"hartsyinference_vace_ctrl_{Guid.NewGuid():N}.mp4");
        File.WriteAllBytes(tmpFile, videoData);
        // -i file → scale to WxH, cap to numFrames, drop audio, emit raw rgb24 on stdout.
        string args = $"-v error -i \"{tmpFile}\" -an -vf \"scale={width}:{height}\" -frames:v {numFrames} "
            + "-f rawvideo -pix_fmt rgb24 -";
        ProcessStartInfo psi = new()
        {
            FileName = ffmpeg,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        Logs.Verbose($"[HartsyInference][VACE] Decoding control video via ffmpeg {args}");
        Process proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to launch ffmpeg for control-video decode.");
        try
        {
            // Drain stderr on a side task so a chatty ffmpeg can't deadlock on a full pipe.
            Task<string> stderrTask = proc.StandardError.ReadToEndAsync();
            using MemoryStream raw = new();
            proc.StandardOutput.BaseStream.CopyTo(raw);
            string stderr = stderrTask.GetAwaiter().GetResult();
            proc.WaitForExit();
            cancel.ThrowIfCancellationRequested();
            if (proc.ExitCode != 0)
            {
                throw new SwarmUserErrorException(
                    $"HartsyInference: ffmpeg failed to decode the VACE control video (exit {proc.ExitCode}): {stderr}");
            }
            byte[] all = raw.ToArray();
            long frameBytes = (long)width * height * 3;
            if (frameBytes == 0 || all.Length < frameBytes)
            {
                throw new SwarmUserErrorException(
                    "HartsyInference: the VACE control video produced no decodable frames at the target resolution.");
            }
            int got = (int)(all.Length / frameBytes);
            List<byte[]> frames = new(got);
            for (int i = 0; i < got; i++)
            {
                byte[] frame = new byte[frameBytes];
                Array.Copy(all, i * frameBytes, frame, 0, frameBytes);
                frames.Add(frame);
            }
            return frames;
        }
        finally
        {
            if (!proc.HasExited)
            {
                try { proc.Kill(); } catch (Exception ex) { Logs.Error($"[HartsyInference] Failed to kill ffmpeg: {ex.Message}"); }
            }
            proc.Dispose();
            try { if (File.Exists(tmpFile)) File.Delete(tmpFile); }
            catch (Exception ex) { Logs.Warning($"[HartsyInference] Failed to delete temp control video '{tmpFile}': {ex.Message}"); }
        }
    }
}
