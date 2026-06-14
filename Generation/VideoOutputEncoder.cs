using System.Diagnostics;
using System.IO;
using SwarmUI.Media;
using SwarmUI.Utils;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Muxes raw RGB frames from a HartsyInference video pipeline into an encoded video container,
/// returning a SwarmUI <see cref="Image"/> (videos are Image objects with a video MediaType —
/// Swarm's save path, mime handling, and history previews all key off the MediaType).
///
/// <para>This is the C# twin of ComfyUI's <c>SwarmSaveAnimationWS.py</c> node: same per-format
/// ffmpeg arguments, same <c>rawvideo rgb24</c> stdin piping, same format names (the
/// <c>VideoFormat</c> param values). ffmpeg itself comes from Swarm's own resolver
/// (<see cref="Utilities.FfmegLocation"/> — system ffmpeg or ComfyUI's vendored imageio-ffmpeg);
/// we never bundle an encoder.</para>
/// </summary>
public static class VideoOutputEncoder
{
    /// <summary>Encodes frames to the requested <c>VideoFormat</c> value ("h264-mp4", "h265-mp4",
    /// "webm", "prores", "gif", "gif-hd", "webp"). A single frame short-circuits to a plain PNG image,
    /// matching the Comfy node's behavior.</summary>
    /// <param name="frames">Interleaved RGB24 bytes, one array per frame, each width*height*3 long.</param>
    public static Image Encode(byte[][] frames, int width, int height, int fps, string format, CancellationToken cancel)
    {
        if (frames is null || frames.Length == 0)
        {
            throw new InvalidOperationException("Video pipeline produced no frames.");
        }
        if (frames.Length == 1)
        {
            return RgbToImage.FromHwcRgb(frames[0], width, height);
        }
        string ffmpeg = Utilities.FfmegLocation.Value;
        if (string.IsNullOrWhiteSpace(ffmpeg))
        {
            throw new SwarmUserErrorException(
                "HartsyInference video output requires ffmpeg, but no ffmpeg was found. "
                + "Install ffmpeg on your system PATH (or install the ComfyUI self-start backend, whose bundled copy Swarm reuses).");
        }
        (string videoArgs, string ext, MediaType type) = FormatArgs(format);
        string tmpFile = Path.Combine(Path.GetTempPath(), $"hartsyinference_video_{Guid.NewGuid():N}.{ext}");
        string args = $"-v error -f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i - {videoArgs} -y \"{tmpFile}\"";
        ProcessStartInfo psi = new()
        {
            FileName = ffmpeg,
            Arguments = args,
            RedirectStandardInput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        Logs.Verbose($"[HartsyInference] Encoding {frames.Length} frames {width}x{height}@{fps}fps as '{format}' via ffmpeg {args}");
        Process proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to launch ffmpeg.");
        try
        {
            Stream stdin = proc.StandardInput.BaseStream;
            foreach (byte[] frame in frames)
            {
                cancel.ThrowIfCancellationRequested();
                stdin.Write(frame, 0, frame.Length);
            }
            stdin.Close();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            if (proc.ExitCode != 0)
            {
                throw new InvalidOperationException($"ffmpeg exited with code {proc.ExitCode} while encoding '{format}': {stderr}");
            }
            byte[] bytes = File.ReadAllBytes(tmpFile);
            return new Image(bytes, type);
        }
        finally
        {
            if (!proc.HasExited)
            {
                try { proc.Kill(); } catch (Exception ex) { Logs.Error($"[HartsyInference] Failed to kill ffmpeg: {ex.Message}"); }
            }
            proc.Dispose();
            try { if (File.Exists(tmpFile)) File.Delete(tmpFile); }
            catch (Exception ex) { Logs.Warning($"[HartsyInference] Failed to delete temp video file '{tmpFile}': {ex.Message}"); }
        }
    }

    /// <summary>Applies the shared frame-array post-edits the Comfy path supports:
    /// <c>TrimVideoStartFrames</c>/<c>TrimVideoEndFrames</c> drop corrupted edge frames, and
    /// <c>VideoBoomerang</c> appends the reversed sequence (minus endpoints) for smooth looping.</summary>
    public static byte[][] ApplyFrameEdits(byte[][] frames, int trimStart, int trimEnd, bool boomerang)
    {
        if (trimStart > 0 || trimEnd > 0)
        {
            int keep = frames.Length - Math.Max(0, trimStart) - Math.Max(0, trimEnd);
            if (keep < 1)
            {
                throw new SwarmUserErrorException(
                    $"Trim Video Start/End Frames ({trimStart}/{trimEnd}) would remove all {frames.Length} generated frames.");
            }
            frames = frames[Math.Max(0, trimStart)..(Math.Max(0, trimStart) + keep)];
        }
        if (boomerang && frames.Length > 2)
        {
            byte[][] looped = new byte[frames.Length * 2 - 2][];
            frames.CopyTo(looped, 0);
            for (int i = 1; i < frames.Length - 1; i++)
            {
                looped[frames.Length + i - 1] = frames[frames.Length - 1 - i];
            }
            frames = looped;
        }
        return frames;
    }

    /// <summary>Per-format ffmpeg output arguments, file extension, and Swarm MediaType. Arguments
    /// mirror <c>SwarmSaveAnimationWS.py</c>; webp/gif (PIL-encoded in the python node) use ffmpeg's
    /// libwebp / palette encoders here. Unknown values fall back to h264-mp4.</summary>
    private static (string videoArgs, string ext, MediaType type) FormatArgs(string format) => format switch
    {
        "h265-mp4" => ("-c:v libx265 -pix_fmt yuv420p", "mp4", MediaType.VideoMp4),
        "webm" => ("-pix_fmt yuv420p -crf 23", "webm", MediaType.VideoWebm),
        "prores" => ("-c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le", "mov", MediaType.VideoMov),
        "gif" => ("", "gif", MediaType.ImageGif),
        "gif-hd" => ("-filter_complex \"split=2 [a][b]; [a] palettegen [pal]; [b] [pal] paletteuse\"", "gif", MediaType.ImageGif),
        "webp" => ("-c:v libwebp -lossless 0 -q:v 95 -loop 0", "webp", MediaType.ImageWebp),
        _ => ("-c:v libx264 -pix_fmt yuv420p -crf 19", "mp4", MediaType.VideoMp4),
    };
}
