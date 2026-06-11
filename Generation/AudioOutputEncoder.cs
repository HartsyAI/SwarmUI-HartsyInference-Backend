using System.Diagnostics;
using System.IO;
using SwarmUI.Media;
using SwarmUI.Utils;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Muxes raw stereo PCM from a SharpInference audio pipeline into an MP3, returned as a SwarmUI
/// <see cref="Image"/> with <see cref="MediaType.AudioMp3"/> (audio outputs are Image objects keyed by
/// MediaType, same as video). Mirrors ComfyUI's audio save path (<c>SaveAudioMP3</c>, LAME V0 quality);
/// ffmpeg comes from Swarm's own resolver (<see cref="Utilities.FfmegLocation"/>) — never bundled.
/// </summary>
public static class AudioOutputEncoder
{
    /// <summary>Encodes planar stereo float PCM to MP3 (LAME V0 — Comfy's quality setting).</summary>
    public static Image EncodeMp3(float[] left, float[] right, int sampleRate, CancellationToken cancel)
    {
        if (left is null || left.Length == 0)
        {
            throw new InvalidOperationException("Audio pipeline produced no samples.");
        }
        string ffmpeg = Utilities.FfmegLocation.Value;
        if (string.IsNullOrWhiteSpace(ffmpeg))
        {
            throw new SwarmUserErrorException(
                "SharpInference audio output requires ffmpeg, but no ffmpeg was found. "
                + "Install ffmpeg on your system PATH (or install the ComfyUI self-start backend, whose bundled copy Swarm reuses).");
        }
        string tmpFile = Path.Combine(Path.GetTempPath(), $"sharpinference_audio_{Guid.NewGuid():N}.mp3");
        string args = $"-v error -f f32le -ar {sampleRate} -ac 2 -i - -c:a libmp3lame -q:a 0 -y \"{tmpFile}\"";
        ProcessStartInfo psi = new()
        {
            FileName = ffmpeg,
            Arguments = args,
            RedirectStandardInput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        Logs.Verbose($"[SharpInference] Encoding {left.Length} samples/channel @ {sampleRate} Hz to mp3 via ffmpeg {args}");
        Process proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to launch ffmpeg.");
        try
        {
            // ffmpeg's f32le expects interleaved L/R frames.
            int n = Math.Min(left.Length, right?.Length ?? left.Length);
            byte[] interleaved = new byte[n * 2 * sizeof(float)];
            for (int i = 0; i < n; i++)
            {
                cancel.ThrowIfCancellationRequested();
                BitConverter.TryWriteBytes(interleaved.AsSpan(i * 8, 4), left[i]);
                BitConverter.TryWriteBytes(interleaved.AsSpan(i * 8 + 4, 4), right is null ? left[i] : right[i]);
            }
            Stream stdin = proc.StandardInput.BaseStream;
            stdin.Write(interleaved, 0, interleaved.Length);
            stdin.Close();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            if (proc.ExitCode != 0)
            {
                throw new InvalidOperationException($"ffmpeg exited with code {proc.ExitCode} while encoding mp3: {stderr}");
            }
            byte[] bytes = File.ReadAllBytes(tmpFile);
            return new Image(bytes, MediaType.AudioMp3);
        }
        finally
        {
            if (!proc.HasExited)
            {
                try { proc.Kill(); } catch (Exception ex) { Logs.Error($"[SharpInference] Failed to kill ffmpeg: {ex.Message}"); }
            }
            proc.Dispose();
            try { if (File.Exists(tmpFile)) File.Delete(tmpFile); }
            catch (Exception ex) { Logs.Warning($"[SharpInference] Failed to delete temp audio file '{tmpFile}': {ex.Message}"); }
        }
    }
}
