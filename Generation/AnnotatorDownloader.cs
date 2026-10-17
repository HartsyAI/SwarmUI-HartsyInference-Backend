using System.IO;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using SwarmUI.Core;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Download-on-demand for preprocessor and annotator checkpoints (depth, pose, edge nets). These are not T2I models, so they skip <see cref="ModelAutoDownloader"/>'s model-set registration. Annotator files land in <c>&lt;ModelRoot&gt;/preprocessors/</c>.</summary>
public static class AnnotatorDownloader
{
    private static readonly ConcurrentDictionary<string, object> _locks = new();

    /// <summary>Returns the local path of the annotator checkpoint, downloading it first when missing. Hash is SHA-256 hex; empty skips verification (discouraged).</summary>
    public static string EnsureAnnotator(string fileName, string url, string hash, Action<string> log)
        => EnsureFileInFolder("preprocessors", fileName, url, hash, log);

    /// <summary>Like <see cref="EnsureAnnotator"/> but for any <c>&lt;ModelRoot&gt;/&lt;subFolder&gt;</c>, e.g. IP-Adapter FaceID checkpoints and their LoRAs under <c>ipadapter/</c>.</summary>
    public static string EnsureFileInFolder(string subFolder, string fileName, string url, string hash, Action<string> log)
    {
        string folder = Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, subFolder);
        string path = Path.Combine(folder, fileName);
        if (File.Exists(path))
        {
            return path;
        }
        object dlLock = _locks.GetOrAdd($"{subFolder}/{fileName}", _ => new object());
        lock (dlLock)
        {
            if (File.Exists(path))
            {
                return path;
            }
            Directory.CreateDirectory(folder);
            string tmp = path + ".tmp";
            log($"Downloading '{fileName}' from {url} ...");
            Logs.Info($"[HartsyInference] Downloading '{fileName}' to '{folder}' because it is not installed yet.");
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
                            $"'{fileName}' failed hash verification (expected {hash}, got {actual}). The download may be corrupt.");
                    }
                }
                File.Move(tmp, path);
                log($"'{fileName}' ready.");
                Logs.Info($"[HartsyInference] Finished downloading '{fileName}'.");
                return path;
            }
            catch (Exception ex)
            {
                Logs.Error($"[HartsyInference] Download failed for '{fileName}': {ex.Message}");
                try { if (File.Exists(tmp)) File.Delete(tmp); } catch (IOException) { }
                throw;
            }
        }
    }
}
