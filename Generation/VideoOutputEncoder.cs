using System.Diagnostics;
using System.IO;
using SwarmUI.Media;
using SwarmUI.Utils;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Muxes raw RGB frames from a HartsyInference video pipeline into an encoded container, returned as a SwarmUI <see cref="Image"/> with a video MediaType.
///
/// <para>The C# twin of ComfyUI's <c>SwarmSaveAnimationWS.py</c>: same per-format ffmpeg arguments, <c>rawvideo rgb24</c> stdin piping and <c>VideoFormat</c> names. ffmpeg comes from Swarm's resolver (<see cref="Utilities.FfmegLocation"/>: system ffmpeg or ComfyUI's vendored imageio-ffmpeg). No encoder is bundled.</para></summary>
public static class VideoOutputEncoder
{
    /// <summary>A stereo audio track to mux into the video container, e.g. LTX-2's soundtrack or a Wan S2V speech clip. <see cref="Right"/> may be null for mono (Left is duplicated).</summary>
    public sealed class AudioTrack
    {
        public required float[] Left { get; init; }
        public float[] Right { get; init; }
        public required int SampleRate { get; init; }
        public bool HasSamples => Left is not null && Left.Length > 0 && SampleRate > 0;
    }

    /// <summary>Encodes frames to the requested <c>VideoFormat</c> ("h264-mp4", "h265-mp4", "webm", "prores", "gif", "gif-hd", "webp"). A single frame short-circuits to a plain PNG, matching the Comfy node. Audio is muxed in as a second ffmpeg input when the container can carry it (mp4/mov/webm); other formats ignore it.</summary>
    /// <param name="frames">Interleaved RGB24 bytes, one array per frame, each width*height*3 long.</param>
    public static Image Encode(byte[][] frames, int width, int height, int fps, string format, CancellationToken cancel, AudioTrack audio = null)
    {
        if (frames is null || frames.Length == 0)
        {
            throw new InvalidOperationException("Video pipeline produced no frames.");
        }
        if (frames.Length == 1)
        {
            return RgbToImage.FromHwcRgb(frames[0], width, height);
        }
        (Process proc, string tmpFile, string audioTmp, MediaType type) = StartFfmpegEncode(width, height, fps, format, audio, cancel);
        Logs.Debug($"[HartsyInference] Encoding {frames.Length} frames {width}x{height}@{fps}fps as '{format}'"
            + (audioTmp is not null ? $" with a {audio.SampleRate} Hz audio track" : ""));
        try
        {
            Stream stdin = proc.StandardInput.BaseStream;
            foreach (byte[] frame in frames)
            {
                cancel.ThrowIfCancellationRequested();
                stdin.Write(frame, 0, frame.Length);
            }
            stdin.Close();
            return FinishEncode(proc, tmpFile, audioTmp, type, format);
        }
        finally
        {
            CleanupEncode(proc, tmpFile, audioTmp);
        }
    }

    /// <summary>Streaming variant of <see cref="Encode"/>: writes frames to ffmpeg's stdin as they arrive, for pipelines that produce frames incrementally (<c>IVideoService.GenerateFramesAsync</c>).
    /// <para>Shares the ffmpeg setup and cleanup helpers with <see cref="Encode"/>. No frame count is passed to ffmpeg, so duration comes from EOF on stdin. A single frame is not special-cased to PNG, since the count is unknown until the source ends; the only caller streams multi-frame requests (see <c>HartsyInferenceBackend.GenerateVideo</c>).</para></summary>
    public static async Task<Image> EncodeStreamingAsync(IAsyncEnumerable<byte[]> frames, int width, int height, int fps, string format, CancellationToken cancel, AudioTrack audio = null)
    {
        (Process proc, string tmpFile, string audioTmp, MediaType type) = StartFfmpegEncode(width, height, fps, format, audio, cancel);
        Logs.Debug($"[HartsyInference] Streaming-encoding {width}x{height}@{fps}fps as '{format}'"
            + (audioTmp is not null ? $" with a {audio.SampleRate} Hz audio track" : "") + " (frame count unknown until EOF)");
        try
        {
            Stream stdin = proc.StandardInput.BaseStream;
            int count = 0;
            await foreach (byte[] frame in frames.WithCancellation(cancel))
            {
                cancel.ThrowIfCancellationRequested();
                await stdin.WriteAsync(frame, cancel);
                count++;
            }
            if (count == 0)
            {
                throw new InvalidOperationException("Video pipeline produced no frames.");
            }
            stdin.Close();
            return FinishEncode(proc, tmpFile, audioTmp, type, format);
        }
        finally
        {
            CleanupEncode(proc, tmpFile, audioTmp);
        }
    }

    /// <summary>Resolves ffmpeg, builds its argv (shared by the buffered and streaming paths), and starts the subprocess with stdin and stderr redirected. The caller writes the frames.</summary>
    private static (Process Proc, string TmpFile, string AudioTmp, MediaType Type) StartFfmpegEncode(
        int width, int height, int fps, string format, AudioTrack audio, CancellationToken cancel)
    {
        string ffmpeg = Utilities.FfmegLocation.Value;
        if (string.IsNullOrWhiteSpace(ffmpeg))
        {
            throw new SwarmUserErrorException(
                "HartsyInference video output requires ffmpeg, but no ffmpeg was found. "
                + "Install ffmpeg on your system PATH (or install the ComfyUI self-start backend, whose bundled copy Swarm reuses).");
        }
        (string videoArgs, string ext, MediaType type) = FormatArgs(format);
        string audioCodec = audio is not null && audio.HasSamples ? AudioArgsForContainer(ext) : null;
        // Audio goes in as a second ffmpeg input from a temp f32le file (video streams over stdin).
        string audioTmp = audioCodec is not null ? WriteInterleavedF32(audio, cancel) : null;
        string tmpFile = Path.Combine(Path.GetTempPath(), $"hartsyinference_video_{Guid.NewGuid():N}.{ext}");
        string inputArgs = audioTmp is not null
            ? $"-f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i - -f f32le -ar {audio.SampleRate} -ac 2 -i \"{audioTmp}\" -map 0:v:0 -map 1:a:0"
            : $"-f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i -";
        // -shortest trims to the shorter of video/audio so a slight frame-vs-sample mismatch doesn't pad silence.
        string muxArgs = audioTmp is not null ? $"{audioCodec} -shortest" : "";
        string args = $"-v error {inputArgs} {videoArgs} {muxArgs} -y \"{tmpFile}\"";
        ProcessStartInfo psi = new()
        {
            FileName = ffmpeg,
            Arguments = args,
            RedirectStandardInput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        Process proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to launch ffmpeg.");
        return (proc, tmpFile, audioTmp, type);
    }

    /// <summary>Closes out a successful encode: waits for ffmpeg to exit, checks its exit code, and reads back the muxed file. The caller must already have closed stdin.</summary>
    private static Image FinishEncode(Process proc, string tmpFile, string audioTmp, MediaType type, string format)
    {
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        if (proc.ExitCode != 0)
        {
            throw new InvalidOperationException($"ffmpeg exited with code {proc.ExitCode} while encoding '{format}': {stderr}");
        }
        return new Image(File.ReadAllBytes(tmpFile), type);
    }

    /// <summary>Kills ffmpeg if it is still running (e.g. the caller threw before closing stdin) and deletes the temp files. Runs from a <c>finally</c> in both <see cref="Encode"/> and <see cref="EncodeStreamingAsync"/>.</summary>
    private static void CleanupEncode(Process proc, string tmpFile, string audioTmp)
    {
        if (!proc.HasExited)
        {
            try { proc.Kill(); } catch (Exception ex) { Logs.Warning($"[HartsyInference] Could not stop the ffmpeg process: {ex.Message}"); }
        }
        proc.Dispose();
        try { if (File.Exists(tmpFile)) File.Delete(tmpFile); }
        catch (Exception ex) { Logs.Warning($"[HartsyInference] Could not delete temp video file '{tmpFile}' (you can remove it by hand): {ex.Message}"); }
        if (audioTmp is not null)
        {
            try { if (File.Exists(audioTmp)) File.Delete(audioTmp); }
            catch (Exception ex) { Logs.Warning($"[HartsyInference] Could not delete temp audio file '{audioTmp}' (you can remove it by hand): {ex.Message}"); }
        }
    }

    /// <summary>True if the container for this <c>VideoFormat</c> can carry an audio stream (mp4/mov/webm). gif and webp cannot, so a generated soundtrack is dropped for those formats, with no separate audio-file fallback.</summary>
    public static bool FormatSupportsAudio(string format) => AudioArgsForContainer(FormatArgs(format).ext) is not null;

    /// <summary>Per-container audio codec args, or null if the container cannot carry audio (gif/webp). mp4/mov take AAC; webm needs Opus, since AAC is not valid in Matroska.</summary>
    private static string AudioArgsForContainer(string ext) => ext switch
    {
        "mp4" or "mov" => "-c:a aac -b:a 192k",
        "webm" => "-c:a libopus -b:a 192k",
        _ => null, // gif / webp: no audio stream
    };

    /// <summary>Writes a stereo audio track to a temp f32le file (interleaved L/R) for use as a second ffmpeg input. Mono tracks (null Right) duplicate Left into both channels.</summary>
    private static string WriteInterleavedF32(AudioTrack audio, CancellationToken cancel)
    {
        float[] left = audio.Left;
        float[] right = audio.Right;
        int n = right is null ? left.Length : Math.Min(left.Length, right.Length);
        byte[] interleaved = new byte[n * 2 * sizeof(float)];
        for (int i = 0; i < n; i++)
        {
            cancel.ThrowIfCancellationRequested();
            BitConverter.TryWriteBytes(interleaved.AsSpan(i * 8, 4), left[i]);
            BitConverter.TryWriteBytes(interleaved.AsSpan(i * 8 + 4, 4), right is null ? left[i] : right[i]);
        }
        string path = Path.Combine(Path.GetTempPath(), $"hartsyinference_vaudio_{Guid.NewGuid():N}.f32le");
        File.WriteAllBytes(path, interleaved);
        return path;
    }

    /// <summary>Applies the frame-array post-edits the Comfy path supports: <c>TrimVideoStartFrames</c> and <c>TrimVideoEndFrames</c> drop corrupted edge frames, and <c>VideoBoomerang</c> appends the reversed sequence (minus endpoints) for smooth looping.</summary>
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

    /// <summary>Per-format ffmpeg output arguments, file extension, and Swarm MediaType. Arguments mirror <c>SwarmSaveAnimationWS.py</c>, with webp and gif using ffmpeg's libwebp and palette encoders. Unknown values fall back to h264-mp4.</summary>
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
