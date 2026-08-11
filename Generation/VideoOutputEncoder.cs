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
    /// <summary>A stereo audio track to mux into the video container (e.g. LTX-2's generated soundtrack,
    /// or a Wan S2V driving-speech clip). <see cref="Right"/> may be null for mono (Left is duplicated).</summary>
    public sealed class AudioTrack
    {
        public required float[] Left { get; init; }
        public float[] Right { get; init; }
        public required int SampleRate { get; init; }
        public bool HasSamples => Left is not null && Left.Length > 0 && SampleRate > 0;
    }

    /// <summary>Encodes frames to the requested <c>VideoFormat</c> value ("h264-mp4", "h265-mp4",
    /// "webm", "prores", "gif", "gif-hd", "webp"). A single frame short-circuits to a plain PNG image,
    /// matching the Comfy node's behavior. When <paramref name="audio"/> is supplied and the container
    /// can carry sound (mp4/mov/webm), the audio is muxed in as a second ffmpeg input so the output is a
    /// single playable video-with-sound file; formats that can't carry audio (gif/webp/png) ignore it.</summary>
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
        Logs.Verbose($"[HartsyInference] Encoding {frames.Length} frames {width}x{height}@{fps}fps as '{format}'"
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

    /// <summary>Streaming sibling of <see cref="Encode"/> (Tier 3.5 backlog): writes frames to ffmpeg's stdin as
    /// they arrive instead of requiring the whole clip materialized first — lower peak memory and a muxed file
    /// available sooner after the last frame decodes, for the video pipelines that can genuinely produce frames
    /// incrementally (<c>IVideoService.GenerateFramesAsync</c>). The ffmpeg command line itself never needed a
    /// frame count up front (no <c>-frames:v</c> flag; duration is inferred from EOF on stdin), so this differs
    /// from <see cref="Encode"/> only in the write loop — same args, same process lifecycle, same cleanup
    /// (<see cref="StartFfmpegEncode"/>/<see cref="FinishEncode"/>/<see cref="CleanupEncode"/> are shared).
    /// Unlike <see cref="Encode"/>, this does NOT special-case a single frame as a bare PNG — the count isn't
    /// known until the source is exhausted. Acceptable because its only caller gates on multi-frame requests
    /// before choosing this path (see the eligibility check in <c>HartsyInferenceBackend.GenerateVideo</c>).</summary>
    public static async Task<Image> EncodeStreamingAsync(IAsyncEnumerable<byte[]> frames, int width, int height, int fps, string format, CancellationToken cancel, AudioTrack audio = null)
    {
        (Process proc, string tmpFile, string audioTmp, MediaType type) = StartFfmpegEncode(width, height, fps, format, audio, cancel);
        Logs.Verbose($"[HartsyInference] Streaming-encoding {width}x{height}@{fps}fps as '{format}'"
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

    /// <summary>Resolves ffmpeg, builds its argv (same for the buffered and streaming encode paths), and starts
    /// the subprocess with stdin/stderr redirected — writing frames to it is the caller's job.</summary>
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

    /// <summary>Closes out a successful encode: waits for ffmpeg to exit, checks its exit code, and reads back
    /// the muxed file. Caller must have already closed stdin.</summary>
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

    /// <summary>Kills ffmpeg if it's still running (e.g. the caller threw before closing stdin) and deletes the
    /// temp files. Always runs from a <c>finally</c> in both <see cref="Encode"/> and
    /// <see cref="EncodeStreamingAsync"/>.</summary>
    private static void CleanupEncode(Process proc, string tmpFile, string audioTmp)
    {
        if (!proc.HasExited)
        {
            try { proc.Kill(); } catch (Exception ex) { Logs.Error($"[HartsyInference] Failed to kill ffmpeg: {ex.Message}"); }
        }
        proc.Dispose();
        try { if (File.Exists(tmpFile)) File.Delete(tmpFile); }
        catch (Exception ex) { Logs.Warning($"[HartsyInference] Failed to delete temp video file '{tmpFile}': {ex.Message}"); }
        if (audioTmp is not null)
        {
            try { if (File.Exists(audioTmp)) File.Delete(audioTmp); }
            catch (Exception ex) { Logs.Warning($"[HartsyInference] Failed to delete temp audio file '{audioTmp}': {ex.Message}"); }
        }
    }

    /// <summary>True if the container for this <c>VideoFormat</c> value can carry an audio stream
    /// (mp4/mov/webm). gif/webp can't, so LTX-2 falls back to a separate mp3 for those.</summary>
    public static bool FormatSupportsAudio(string format) => AudioArgsForContainer(FormatArgs(format).ext) is not null;

    /// <summary>Per-container audio codec args, or null if the container can't carry audio (gif/webp).
    /// mp4/mov take AAC; webm needs Opus (AAC isn't valid in a WebM/Matroska container).</summary>
    private static string AudioArgsForContainer(string ext) => ext switch
    {
        "mp4" or "mov" => "-c:a aac -b:a 192k",
        "webm" => "-c:a libopus -b:a 192k",
        _ => null, // gif / webp — no audio stream
    };

    /// <summary>Writes a stereo audio track to a temp f32le raw file (interleaved L/R), for use as a
    /// second ffmpeg input. Mono tracks (null Right) duplicate Left into both channels.</summary>
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
