using System.IO;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using SwarmUI.Core;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Download-on-demand for preprocessor/annotator checkpoints (depth estimators, pose detectors,
/// edge nets). These are NOT Swarm T2I models — they never appear in the model browser and are
/// often <c>.pth</c> rather than <c>.safetensors</c> — so <see cref="ModelAutoDownloader"/>'s
/// model-set registration doesn't apply. Files land under <c>&lt;ModelRoot&gt;/preprocessors/</c>
/// (the comfyui_controlnet_aux convention, adapted), with the same tmp-file + SHA-256 + atomic
/// rename discipline as ModelAutoDownloader.
/// </summary>
public static class AnnotatorDownloader
{
    private static readonly ConcurrentDictionary<string, object> _locks = new();

    /// <summary>Returns the local path of the annotator checkpoint, downloading it first when missing. Hash is SHA-256 hex; empty skips verification (discouraged).</summary>
    public static string EnsureAnnotator(string fileName, string url, string hash, Action<string> log)
    {
        string folder = Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, "preprocessors");
        string path = Path.Combine(folder, fileName);
        if (File.Exists(path))
        {
            return path;
        }
        object dlLock = _locks.GetOrAdd(fileName, _ => new object());
        lock (dlLock)
        {
            if (File.Exists(path))
            {
                return path;
            }
            Directory.CreateDirectory(folder);
            string tmp = path + ".tmp";
            log($"Downloading annotator '{fileName}' from {url} ...");
            try
            {
                Utilities.DownloadFile(url, tmp, null).Wait(Program.GlobalProgramCancel);
                if (!string.IsNullOrEmpty(hash))
                {
                    using FileStream fs = File.OpenRead(tmp);
                    string actual = Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
                    if (actual != hash.ToLowerInvariant())
                    {
                        throw new InvalidDataException(
                            $"Annotator '{fileName}' failed hash verification (expected {hash}, got {actual}) — corrupt download?");
                    }
                }
                File.Move(tmp, path);
                log($"Annotator '{fileName}' ready.");
                return path;
            }
            catch (Exception ex)
            {
                Logs.Error($"[HartsyInference] Annotator download failed for '{fileName}': {ex.Message}");
                try { if (File.Exists(tmp)) File.Delete(tmp); } catch (IOException) { }
                throw;
            }
        }
    }
}
