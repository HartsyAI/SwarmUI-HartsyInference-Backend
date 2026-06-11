using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading.Tasks;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Ensures a required side-component model (Qwen text encoder, Flux VAE, etc.) is on disk
/// before a pipeline that needs it tries to load. Mirrors the auto-download behaviour
/// ComfyUI's <c>WorkflowGeneratorModelSupport.RequireClipModel</c> provides — when the
/// canonical file isn't already in the expected folder, fetch it from the registered
/// URL, verify hash, refresh Swarm's model set, and return the resulting <see cref="T2IModel"/>.
///
/// <para>Hardening vs the original (matched 1:1 to Comfy's <c>WorkflowGenerator.DownloadModel</c>):
/// <list type="bullet">
/// <item>Per-canonical-name locking — two pipelines that load concurrently and both want the
/// same Qwen3-4B file don't race; one downloads, the other waits and re-uses the result.</item>
/// <item>Atomic <c>.tmp</c> stage + final move — if the download is interrupted, the partial
/// file does NOT live at the canonical name. The next attempt starts fresh.</item>
/// <item>Hash verification — corrupt mid-flight downloads are caught and the partial file
/// is deleted before the .tmp move so the next run re-downloads cleanly.</item>
/// </list></para>
///
/// <para>This is what makes "user picks Z-Image-Turbo and just hits Generate" work without
/// the user having to track down which Qwen3-4B encoder to download manually. The download
/// happens during model-load (not during request-validation) so the user sees the same
/// LOADING status the model itself uses, and the Verbose logs show download progress
/// just like Comfy's <c>"qwen_3_4b.safetensors download at 35.0%..."</c> lines.</para>
/// </summary>
public static class ModelAutoDownloader
{
    /// <summary>One <see cref="object"/> per canonical model name — the simple equivalent of Comfy's
    /// <c>MultiLockSet&lt;string&gt;</c>. Concurrent jobs that both want the same Qwen3-4B file
    /// serialize on the same lock; the second one finds the file already present after the
    /// first finishes and skips its own download.</summary>
    private static readonly ConcurrentDictionary<string, object> _downloadLocks = new();

    /// <summary>Resolves a side-model from the central <see cref="SideModels"/> registry. Preferred
    /// entry point — keeps URLs/hashes out of caller code so updating a download endpoint is a
    /// one-line change in <see cref="SideModels"/>.</summary>
    public static T2IModel EnsureSideModel(
        T2IModel userPick,
        SideModels.Entry entry,
        Action<string> log)
    {
        return EnsureSideModel(userPick, entry.FolderType, entry.CanonicalName, entry.Url, entry.Hash, log);
    }

    /// <summary>Lower-level overload accepting raw fields. Use the <see cref="SideModels.Entry"/>
    /// overload for new code; this exists for callers that need to inject custom URLs (e.g.
    /// from a per-architecture override).</summary>
    /// <param name="userPick">User's explicit model selection from the relevant T2IParam (may be null).</param>
    /// <param name="folderType">Swarm's <c>T2IModelSets</c> key — typically "Clip" for text encoders, "VAE" for VAEs.</param>
    /// <param name="canonicalName">File name (with optional subfolder) to look up / save as. Must end in <c>.safetensors</c>.</param>
    /// <param name="url">HTTPS download URL. Use the same URL Comfy registers for parity.</param>
    /// <param name="hash">SHA-256 hex hash for verification. Empty allowed but discouraged — corruption mid-download
    /// has happened in practice, especially over slow links, and a silent corrupt model is
    /// worse than a noisy failed download.</param>
    /// <param name="log">Logging callback (typically Logs.Verbose / Logs.Info wrapper).</param>
    public static T2IModel EnsureSideModel(
        T2IModel userPick,
        string folderType,
        string canonicalName,
        string url,
        string hash,
        Action<string> log)
    {
        if (userPick is not null)
        {
            return userPick;
        }
        if (!Program.T2IModelSets.TryGetValue(folderType, out T2IModelHandler handler))
        {
            throw new InvalidOperationException(
                $"Unknown model folder type '{folderType}'. Available: {string.Join(", ", Program.T2IModelSets.Keys)}");
        }
        // First lookup outside the lock — fast path for already-installed models.
        T2IModel existing = handler.GetModel(canonicalName);
        if (existing is not null)
        {
            log($"Auto-resolved {folderType} model '{canonicalName}' (already present, no download).");
            return existing;
        }

        if (handler.FolderPaths.Length == 0)
        {
            throw new InvalidOperationException($"No folder paths configured for '{folderType}'.");
        }
        string folder = handler.FolderPaths[0];
        string targetPath = Path.Combine(folder, canonicalName);
        string tmpPath = targetPath + ".tmp";

        // Per-name lock: two concurrent generations that both want the same file should
        // not both download it. This matches Comfy's MultiLockSet<string> behavior.
        object lockObj = _downloadLocks.GetOrAdd(canonicalName, _ => new object());
        lock (lockObj)
        {
            // Re-check inside the lock — by the time we hold it, another thread may have
            // finished the download. Without this re-check we'd re-download for no reason.
            existing = handler.GetModel(canonicalName);
            if (existing is not null)
            {
                log($"Auto-resolved {folderType} model '{canonicalName}' (downloaded by concurrent job, no re-download).");
                return existing;
            }

            log($"Downloading {folderType} model '{canonicalName}' from {url}...");
            // Make sure the target subdirectory exists (canonicalName may have nested
            // path segments like "Flux/ae.safetensors").
            string targetDir = Path.GetDirectoryName(targetPath);
            if (!string.IsNullOrEmpty(targetDir) && !Directory.Exists(targetDir))
            {
                Directory.CreateDirectory(targetDir);
            }

            // Atomic stage: download to .tmp, hash-verify, move into place. If we crash or
            // get interrupted, the canonical name still doesn't exist and the next run
            // re-attempts cleanly (vs leaving a partial file that fakes being "complete").
            if (File.Exists(tmpPath))
            {
                File.Delete(tmpPath);
            }
            try
            {
                double nextLoggedPct = 0.05;
                Task downloadTask = Utilities.DownloadFile(url, tmpPath, (bytes, total, perSec) =>
                {
                    if (total <= 0) return;
                    double pct = bytes / (double)total;
                    if (pct >= nextLoggedPct)
                    {
                        log($"  {canonicalName} download at {pct * 100:0.0}% ({bytes / (1024.0 * 1024.0):F0}/{total / (1024.0 * 1024.0):F0} MB, {perSec / (1024.0 * 1024.0):F1} MB/s)");
                        nextLoggedPct = Math.Round(pct / 0.05) * 0.05 + 0.05;
                    }
                }, verifyHash: string.IsNullOrEmpty(hash) ? null : hash);
                downloadTask.Wait();
                File.Move(tmpPath, targetPath);
            }
            catch (Exception ex)
            {
                log($"  download FAILED: {ex.GetType().Name}: {ex.Message}");
                if (File.Exists(tmpPath))
                {
                    try { File.Delete(tmpPath); } catch { /* best-effort */ }
                }
                throw;
            }

            // Make Swarm rescan so the new file appears in T2IModelSets — without this,
            // GetModel below would still miss and we'd fail the round-trip. Comfy does the
            // exact same dance (DownloadNow().Wait() + RefreshAllModelSets()).
            Program.RefreshAllModelSets();
            T2IModel downloaded = handler.GetModel(canonicalName)
                ?? throw new InvalidOperationException(
                    $"Downloaded '{canonicalName}' to '{targetPath}' but it didn't appear in the model registry after refresh. " +
                    "Check folder paths and file extension.");
            log($"  {canonicalName} download complete; registered in '{folderType}'.");
            return downloaded;
        }
    }
}
