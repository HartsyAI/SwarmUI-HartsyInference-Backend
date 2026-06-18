using HartsyInference.Core.Backends;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Per-backend cache of loaded pipelines, with global LRU eviction across all
/// architecture maps. One entry per (architecture, model name).
///
/// See docs/01-Architecture.md §Model + weight cache.
/// </summary>
public sealed class PipelineCache
{
    private readonly IBackend _backend;
    private readonly int _maxEntries;
    private readonly Dictionary<string, FluxCacheEntry> _flux = new();
    private readonly Dictionary<string, Flux2CacheEntry> _flux2 = new();
    private readonly Dictionary<string, ChromaCacheEntry> _chroma = new();
    private readonly Dictionary<string, AuraFlowCacheEntry> _auraFlow = new();
    private readonly Dictionary<string, FLiteCacheEntry> _fLite = new();
    private readonly Dictionary<string, Ideogram4CacheEntry> _ideogram4 = new();
    private readonly Dictionary<string, ErnieImageCacheEntry> _ernieImage = new();
    private readonly Dictionary<string, ChromaRadianceCacheEntry> _chromaRadiance = new();
    private readonly Dictionary<string, ZetaChromaCacheEntry> _zetaChroma = new();
    private readonly Dictionary<string, ZImageCacheEntry> _zImage = new();
    private readonly Dictionary<string, AnimaCacheEntry> _anima = new();
    private readonly Dictionary<string, HiDreamCacheEntry> _hiDream = new();
    private readonly Dictionary<string, QwenImageCacheEntry> _qwenImage = new();
    private readonly Dictionary<string, Sd15CacheEntry> _sd15 = new();
    private readonly Dictionary<string, SdxlCacheEntry> _sdxl = new();
    private readonly Dictionary<string, Sd3CacheEntry> _sd3 = new();
    private readonly Dictionary<string, WanVideoCacheEntry> _wanVideo = new();
    private readonly Dictionary<string, LtxVideoCacheEntry> _ltxVideo = new();
    private readonly Dictionary<string, AceStepCacheEntry> _aceStep = new();
    private readonly Dictionary<string, AceStep15CacheEntry> _aceStep15 = new();
    private readonly Dictionary<string, MusicGenCacheEntry> _musicGen = new();
    private readonly Dictionary<string, YueCacheEntry> _yue = new();
    private readonly Dictionary<string, LanceCacheEntry> _lance = new();
    private readonly Dictionary<string, LensCacheEntry> _lens = new();
    private readonly Dictionary<string, RefinerCacheEntry> _refiner = new();
    private readonly Dictionary<string, IpAdapterCacheEntry> _ipAdapter = new();
    private readonly object _lock = new();

    public PipelineCache(IBackend backend, int maxEntries)
    {
        _backend = backend;
        _maxEntries = Math.Max(1, maxEntries);
    }

    public FluxCacheEntry TryGetFlux(string modelName) => Touch(_flux, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public Flux2CacheEntry TryGetFlux2(string modelName) => Touch(_flux2, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public ChromaCacheEntry TryGetChroma(string modelName) => Touch(_chroma, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public AuraFlowCacheEntry TryGetAuraFlow(string modelName) => Touch(_auraFlow, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public FLiteCacheEntry TryGetFLite(string modelName) => Touch(_fLite, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public Ideogram4CacheEntry TryGetIdeogram4(string modelName) => Touch(_ideogram4, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public ErnieImageCacheEntry TryGetErnieImage(string modelName) => Touch(_ernieImage, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public ChromaRadianceCacheEntry TryGetChromaRadiance(string modelName) => Touch(_chromaRadiance, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public ZetaChromaCacheEntry TryGetZetaChroma(string modelName) => Touch(_zetaChroma, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public ZImageCacheEntry TryGetZImage(string modelName) => Touch(_zImage, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public AnimaCacheEntry TryGetAnima(string modelName) => Touch(_anima, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public HiDreamCacheEntry TryGetHiDream(string modelName) => Touch(_hiDream, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public QwenImageCacheEntry TryGetQwenImage(string modelName) => Touch(_qwenImage, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public Sd15CacheEntry TryGetSd15(string modelName) => Touch(_sd15, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public SdxlCacheEntry TryGetSdxl(string modelName) => Touch(_sdxl, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public Sd3CacheEntry TryGetSd3(string modelName) => Touch(_sd3, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public WanVideoCacheEntry TryGetWanVideo(string modelName) => Touch(_wanVideo, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public LtxVideoCacheEntry TryGetLtxVideo(string modelName) => Touch(_ltxVideo, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public AceStepCacheEntry TryGetAceStep(string modelName) => Touch(_aceStep, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public AceStep15CacheEntry TryGetAceStep15(string modelName) => Touch(_aceStep15, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public MusicGenCacheEntry TryGetMusicGen(string modelName) => Touch(_musicGen, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public YueCacheEntry TryGetYue(string modelName) => Touch(_yue, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public LanceCacheEntry TryGetLance(string modelName) => Touch(_lance, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public LensCacheEntry TryGetLens(string modelName) => Touch(_lens, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public RefinerCacheEntry TryGetRefiner(string modelName) => Touch(_refiner, modelName, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);
    public IpAdapterCacheEntry TryGetIpAdapter(string filePath) => Touch(_ipAdapter, filePath, e => e.LastUsedUtc, (e, t) => e.LastUsedUtc = t);

    public void PutFlux(FluxCacheEntry entry) => Put(_flux, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutFlux2(Flux2CacheEntry entry) => Put(_flux2, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutChroma(ChromaCacheEntry entry) => Put(_chroma, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutAuraFlow(AuraFlowCacheEntry entry) => Put(_auraFlow, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutFLite(FLiteCacheEntry entry) => Put(_fLite, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutIdeogram4(Ideogram4CacheEntry entry) => Put(_ideogram4, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutErnieImage(ErnieImageCacheEntry entry) => Put(_ernieImage, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutChromaRadiance(ChromaRadianceCacheEntry entry) => Put(_chromaRadiance, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutZetaChroma(ZetaChromaCacheEntry entry) => Put(_zetaChroma, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutZImage(ZImageCacheEntry entry) => Put(_zImage, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutAnima(AnimaCacheEntry entry) => Put(_anima, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutHiDream(HiDreamCacheEntry entry) => Put(_hiDream, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutQwenImage(QwenImageCacheEntry entry) => Put(_qwenImage, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutSd15(Sd15CacheEntry entry) => Put(_sd15, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutSdxl(SdxlCacheEntry entry) => Put(_sdxl, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutSd3(Sd3CacheEntry entry) => Put(_sd3, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutWanVideo(WanVideoCacheEntry entry) => Put(_wanVideo, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutLtxVideo(LtxVideoCacheEntry entry) => Put(_ltxVideo, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutAceStep(AceStepCacheEntry entry) => Put(_aceStep, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutAceStep15(AceStep15CacheEntry entry) => Put(_aceStep15, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutMusicGen(MusicGenCacheEntry entry) => Put(_musicGen, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutYue(YueCacheEntry entry) => Put(_yue, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutLance(LanceCacheEntry entry) => Put(_lance, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutLens(LensCacheEntry entry) => Put(_lens, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutRefiner(RefinerCacheEntry entry) => Put(_refiner, entry.ModelName, entry, e => e.LastUsedUtc = DateTime.UtcNow);
    public void PutIpAdapter(IpAdapterCacheEntry entry) => Put(_ipAdapter, entry.FilePath, entry, e => e.LastUsedUtc = DateTime.UtcNow);

    private TEntry Touch<TEntry>(Dictionary<string, TEntry> map, string key, Func<TEntry, DateTime> getTime, Action<TEntry, DateTime> setTime)
        where TEntry : class
    {
        lock (_lock)
        {
            if (map.TryGetValue(key, out TEntry entry))
            {
                setTime(entry, DateTime.UtcNow);
                return entry;
            }
            return null;
        }
    }

    private void Put<TEntry>(Dictionary<string, TEntry> map, string key, TEntry entry, Action<TEntry> stamp)
        where TEntry : class
    {
        lock (_lock)
        {
            // If an old entry with the same name lives in this map, dispose it first
            // (replacement). Cross-map collisions (same name in two arch maps) are
            // unlikely; we leave them.
            if (map.TryGetValue(key, out TEntry old) && old is IDisposable oldDisp)
            {
                oldDisp.Dispose();
            }
            map[key] = entry;
            stamp(entry);
            EvictIfOverCapacity();
        }
    }

    private int TotalCount => _flux.Count + _flux2.Count + _chroma.Count + _chromaRadiance.Count + _zetaChroma.Count + _auraFlow.Count + _fLite.Count + _ideogram4.Count + _ernieImage.Count + _zImage.Count + _anima.Count + _hiDream.Count + _qwenImage.Count + _sd15.Count + _sdxl.Count + _sd3.Count + _wanVideo.Count + _ltxVideo.Count + _aceStep.Count + _aceStep15.Count + _musicGen.Count + _yue.Count + _lance.Count + _lens.Count + _refiner.Count + _ipAdapter.Count;

    /// <summary>Evict the globally-oldest entry across all architecture maps until we're
    /// at or under <see cref="_maxEntries"/>.</summary>
    private void EvictIfOverCapacity()
    {
        while (TotalCount > _maxEntries)
        {
            DateTime oldestTime = DateTime.MaxValue;
            Action evictAction = null;

            foreach (KeyValuePair<string, FluxCacheEntry> kv in _flux)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _flux[key].Dispose(); _flux.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, Flux2CacheEntry> kv in _flux2)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _flux2[key].Dispose(); _flux2.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, ChromaCacheEntry> kv in _chroma)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _chroma[key].Dispose(); _chroma.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, AuraFlowCacheEntry> kv in _auraFlow)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _auraFlow[key].Dispose(); _auraFlow.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, FLiteCacheEntry> kv in _fLite)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _fLite[key].Dispose(); _fLite.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, Ideogram4CacheEntry> kv in _ideogram4)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _ideogram4[key].Dispose(); _ideogram4.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, ErnieImageCacheEntry> kv in _ernieImage)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _ernieImage[key].Dispose(); _ernieImage.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, ChromaRadianceCacheEntry> kv in _chromaRadiance)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _chromaRadiance[key].Dispose(); _chromaRadiance.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, ZetaChromaCacheEntry> kv in _zetaChroma)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _zetaChroma[key].Dispose(); _zetaChroma.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, ZImageCacheEntry> kv in _zImage)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _zImage[key].Dispose(); _zImage.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, AnimaCacheEntry> kv in _anima)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _anima[key].Dispose(); _anima.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, HiDreamCacheEntry> kv in _hiDream)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _hiDream[key].Dispose(); _hiDream.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, QwenImageCacheEntry> kv in _qwenImage)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _qwenImage[key].Dispose(); _qwenImage.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, Sd15CacheEntry> kv in _sd15)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _sd15[key].Dispose(); _sd15.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, SdxlCacheEntry> kv in _sdxl)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _sdxl[key].Dispose(); _sdxl.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, Sd3CacheEntry> kv in _sd3)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _sd3[key].Dispose(); _sd3.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, WanVideoCacheEntry> kv in _wanVideo)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _wanVideo[key].Dispose(); _wanVideo.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, LtxVideoCacheEntry> kv in _ltxVideo)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _ltxVideo[key].Dispose(); _ltxVideo.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, AceStepCacheEntry> kv in _aceStep)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _aceStep[key].Dispose(); _aceStep.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, AceStep15CacheEntry> kv in _aceStep15)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _aceStep15[key].Dispose(); _aceStep15.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, MusicGenCacheEntry> kv in _musicGen)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _musicGen[key].Dispose(); _musicGen.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, YueCacheEntry> kv in _yue)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _yue[key].Dispose(); _yue.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, LanceCacheEntry> kv in _lance)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _lance[key].Dispose(); _lance.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, LensCacheEntry> kv in _lens)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _lens[key].Dispose(); _lens.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, RefinerCacheEntry> kv in _refiner)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _refiner[key].Dispose(); _refiner.Remove(key); };
                }
            }
            foreach (KeyValuePair<string, IpAdapterCacheEntry> kv in _ipAdapter)
            {
                if (kv.Value.LastUsedUtc < oldestTime)
                {
                    oldestTime = kv.Value.LastUsedUtc;
                    string key = kv.Key;
                    evictAction = () => { _ipAdapter[key].Dispose(); _ipAdapter.Remove(key); };
                }
            }

            if (evictAction is null) break;
            evictAction();
        }
    }

    public bool EvictAll()
    {
        lock (_lock)
        {
            if (TotalCount == 0) return false;
            foreach (FluxCacheEntry entry in _flux.Values) entry.Dispose();
            foreach (Flux2CacheEntry entry in _flux2.Values) entry.Dispose();
            foreach (ChromaCacheEntry entry in _chroma.Values) entry.Dispose();
            foreach (AuraFlowCacheEntry entry in _auraFlow.Values) entry.Dispose();
            foreach (FLiteCacheEntry entry in _fLite.Values) entry.Dispose();
            foreach (Ideogram4CacheEntry entry in _ideogram4.Values) entry.Dispose();
            foreach (ErnieImageCacheEntry entry in _ernieImage.Values) entry.Dispose();
            foreach (ChromaRadianceCacheEntry entry in _chromaRadiance.Values) entry.Dispose();
            foreach (ZetaChromaCacheEntry entry in _zetaChroma.Values) entry.Dispose();
            foreach (ZImageCacheEntry entry in _zImage.Values) entry.Dispose();
            foreach (AnimaCacheEntry entry in _anima.Values) entry.Dispose();
            foreach (HiDreamCacheEntry entry in _hiDream.Values) entry.Dispose();
            foreach (QwenImageCacheEntry entry in _qwenImage.Values) entry.Dispose();
            foreach (Sd15CacheEntry entry in _sd15.Values) entry.Dispose();
            foreach (SdxlCacheEntry entry in _sdxl.Values) entry.Dispose();
            foreach (Sd3CacheEntry entry in _sd3.Values) entry.Dispose();
            foreach (WanVideoCacheEntry entry in _wanVideo.Values) entry.Dispose();
            foreach (LtxVideoCacheEntry entry in _ltxVideo.Values) entry.Dispose();
            foreach (AceStepCacheEntry entry in _aceStep.Values) entry.Dispose();
            foreach (AceStep15CacheEntry entry in _aceStep15.Values) entry.Dispose();
            foreach (MusicGenCacheEntry entry in _musicGen.Values) entry.Dispose();
            foreach (YueCacheEntry entry in _yue.Values) entry.Dispose();
            foreach (LanceCacheEntry entry in _lance.Values) entry.Dispose();
            foreach (LensCacheEntry entry in _lens.Values) entry.Dispose();
            foreach (RefinerCacheEntry entry in _refiner.Values) entry.Dispose();
            foreach (IpAdapterCacheEntry entry in _ipAdapter.Values) entry.Dispose();
            _flux.Clear();
            _flux2.Clear();
            _chroma.Clear();
            _auraFlow.Clear();
            _fLite.Clear();
            _ideogram4.Clear();
            _ernieImage.Clear();
            _chromaRadiance.Clear();
            _zetaChroma.Clear();
            _zImage.Clear();
            _anima.Clear();
            _hiDream.Clear();
            _qwenImage.Clear();
            _sd15.Clear();
            _sdxl.Clear();
            _sd3.Clear();
            _wanVideo.Clear();
            _ltxVideo.Clear();
            _aceStep.Clear();
            _aceStep15.Clear();
            _musicGen.Clear();
            _yue.Clear();
            _lance.Clear();
            _lens.Clear();
            _refiner.Clear();
            _ipAdapter.Clear();
            return true;
        }
    }

    public void DisposeAll() => EvictAll();
}
