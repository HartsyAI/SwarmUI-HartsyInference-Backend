#nullable enable
using System;
using System.Collections.Concurrent;
using System.IO;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Diffusion.Utilities;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>Lazy-loads and process-wide-caches the per-architecture TAESD preview decoders.
/// First call for a given architecture triggers a one-time download (via
/// <see cref="ModelAutoDownloader"/>) + safetensors mmap; subsequent calls return the cached
/// instance instantly. The mmap-backed weight tensors are read-only, so a single decoder
/// instance is safe to share across concurrent generations.
///
/// <para>Returns <c>null</c> for architectures that don't have a published TAESD checkpoint
/// (Flux.2's 32-channel latent, Chroma — although Chroma uses Flux.1 weights so it works,
/// AuraFlow, F-Lite, Z-Image). Callers should fall back to <see cref="LatentPreview"/> in
/// that case — <see cref="PreviewEncoder"/> already does this transparently.</para></summary>
public static class TaesdResolver
{
    /// <summary>One slot per arch. Lazy because we don't want to download all four files just
    /// because a user toggled the setting on — only the one the current model uses.</summary>
    private static readonly ConcurrentDictionary<LatentArchitecture, Lazy<TaesdDecoder?>> _cache = new();

    /// <summary>Returns a TAESD decoder for the given arch, downloading + loading on first
    /// call. Subsequent calls are O(1) dictionary lookups. Returns <c>null</c> if no
    /// published TAESD checkpoint exists for <paramref name="arch"/>, in which case the
    /// caller should fall back to <see cref="LatentPreview"/>.</summary>
    /// <param name="arch">The model family asking for a preview decoder.</param>
    /// <param name="log">Progress / status callback (download lines + load status).</param>
    public static TaesdDecoder? Resolve(LatentArchitecture arch, Action<string> log)
    {
        SideModels.Entry? entry = EntryFor(arch);
        if (entry is null)
        {
            // No published TAESD for this architecture — caller falls back to latent2rgb.
            return null;
        }

        Lazy<TaesdDecoder?> lazy = _cache.GetOrAdd(arch, _ => new Lazy<TaesdDecoder?>(() =>
        {
            try
            {
                // EnsureSideModel handles the locking, hash-check, atomic .tmp staging, and
                // Swarm model-set refresh. Returns a T2IModel whose RawFilePath we pass to
                // the HartsyInference loader.
                T2IModel model = ModelAutoDownloader.EnsureSideModel(userPick: null, entry, log);
                if (string.IsNullOrWhiteSpace(model.RawFilePath) || !File.Exists(model.RawFilePath))
                {
                    log($"[TAESD] {entry.DisplayName}: resolved model has no usable file path. Falling back to latent2rgb.");
                    return null;
                }
                log($"[TAESD] Loading {entry.DisplayName} from {model.RawFilePath}");
                TaesdDecoder dec = TaesdDecoder.LoadFromSafetensors(arch, model.RawFilePath);
                log($"[TAESD] {entry.DisplayName} ready ({dec.LatentChannels}-channel latent).");
                return dec;
            }
            catch (Exception ex)
            {
                // A failed download / corrupt file shouldn't break generation — fall back.
                Logs.Warning($"[HartsyInference][TAESD] Failed to load decoder for {arch}: {ex.GetType().Name}: {ex.Message}. Falling back to latent2rgb.");
                return null;
            }
        }, isThreadSafe: true));

        return lazy.Value;
    }

    /// <summary>Maps a latent architecture to its TAESD weight registry entry. Architectures
    /// without a published TAESD checkpoint return <c>null</c>; Chroma / Z-Image reuse the
    /// Flux.1 weights since they share Flux's VAE.</summary>
    private static SideModels.Entry? EntryFor(LatentArchitecture arch) => arch switch
    {
        LatentArchitecture.Sd15 => SideModels.TaesdSd15,
        LatentArchitecture.Sdxl => SideModels.TaesdSdxl,
        LatentArchitecture.Sd3 => SideModels.TaesdSd3,
        // Flux.1 weights work for any 16-channel Flux-family VAE.
        LatentArchitecture.Flux or LatentArchitecture.Chroma or LatentArchitecture.ZImage
            => SideModels.TaesdFlux,
        // No published TAESD for these yet — Flux.2 (32-ch) needs a different weight set,
        // AuraFlow / F-Lite haven't been distilled by upstream.
        _ => null,
    };
}
