using System.Diagnostics;
using System.IO;
using SwarmUI.Media;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Decodes a SwarmUI <see cref="AudioFile"/> (the <c>Video Audio Input</c> param) into the raw mono 16 kHz
/// PCM the Wan-S2V Wav2Vec2 front-end consumes. ffmpeg comes from Swarm's resolver
/// (<see cref="Utilities.FfmegLocation"/>); the audio is piped through it to <c>f32le</c> samples in [-1, 1].
/// </summary>
public static class AudioDecoder
{
    /// <summary>Decodes the audio to a mono 16 kHz <c>float[]</c> waveform.</summary>
    public static float[] DecodeMono16k(AudioFile audio, CancellationToken cancel)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        string ffmpeg = Utilities.FfmegLocation.Value;
        if (string.IsNullOrWhiteSpace(ffmpeg))
        {
            throw new SwarmUserErrorException(
                "HartsyInference Wan-S2V needs ffmpeg to decode the audio input, but none was found. "
                + "Install ffmpeg on your system PATH (or install the ComfyUI self-start backend, whose bundled copy Swarm reuses).");
        }
        string ext = audio.Type?.Extension ?? "wav";
        string tmpFile = Path.Combine(Path.GetTempPath(), $"hartsyinference_s2v_audio_{Guid.NewGuid():N}.{ext}");
        File.WriteAllBytes(tmpFile, audio.RawData);
        string args = $"-v error -i \"{tmpFile}\" -ac 1 -ar 16000 -f f32le -";
        ProcessStartInfo psi = new()
        {
            FileName = ffmpeg,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        Logs.Verbose($"[HartsyInference][S2V] Decoding audio via ffmpeg {args}");
        Process proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to launch ffmpeg for audio decode.");
        try
        {
            Task<string> stderrTask = proc.StandardError.ReadToEndAsync();
            using MemoryStream raw = new();
            proc.StandardOutput.BaseStream.CopyTo(raw);
            string stderr = stderrTask.GetAwaiter().GetResult();
            proc.WaitForExit();
            cancel.ThrowIfCancellationRequested();
            if (proc.ExitCode != 0)
            {
                throw new SwarmUserErrorException($"HartsyInference: ffmpeg failed to decode the S2V audio input (exit {proc.ExitCode}): {stderr}");
            }
            byte[] bytes = raw.ToArray();
            if (bytes.Length < 4)
            {
                throw new SwarmUserErrorException("HartsyInference: the S2V audio input decoded to no samples.");
            }
            float[] samples = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);
            Logs.Verbose($"[HartsyInference][S2V] Decoded {samples.Length} mono samples @16kHz ({samples.Length / 16000.0:0.0}s).");
            return samples;
        }
        finally
        {
            if (!proc.HasExited)
            {
                try { proc.Kill(); } catch (Exception ex) { Logs.Error($"[HartsyInference] Failed to kill ffmpeg: {ex.Message}"); }
            }
            proc.Dispose();
            try { if (File.Exists(tmpFile)) File.Delete(tmpFile); }
            catch (Exception ex) { Logs.Warning($"[HartsyInference] Failed to delete temp audio file '{tmpFile}': {ex.Message}"); }
        }
    }
}
