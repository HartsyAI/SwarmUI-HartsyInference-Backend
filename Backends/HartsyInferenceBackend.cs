using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Reflection;
using System.Runtime.CompilerServices;
using SwarmUI.Builtin_ComfyUIBackend;
using FreneticUtilities.FreneticDataSyntax;
using Newtonsoft.Json.Linq;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.MemoryManagement;
using SiLogs = HartsyInference.Core.Logging.Logs;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using EngineImage = HartsyInference.Engine.Requests.ImageData;

namespace Hartsy.Extensions.HartsyInferenceBackend.Backends;

/// <summary>
/// SwarmUI backend that runs inference in-process through <c>HartsyInference.Engine</c>.
///
/// <para>This class is a <b>thin mapper</b> and nothing more. The Engine is the single source of truth for
/// "load a model + generate": it owns architecture detection, per-family recipes, pipeline caching, side-model
/// download, composition (LoRA / ControlNet / IP-Adapter / refiner / img2img / inpaint / regional) and sampling
/// defaults. The four things that remain SwarmUI's problem, and therefore ours, are:</para>
/// <list type="number">
/// <item><b>Load</b> — resolve a <see cref="T2IModel"/> to a checkpoint path and an Engine <see cref="ModelSpec"/>.</item>
/// <item><b>Map request</b> — the only place <c>input.Get(T2IParamTypes.*)</c> is read; produces an
/// <see cref="ImageRequest"/> / <see cref="VideoRequest"/> / <see cref="MusicRequest"/>.</item>
/// <item><b>Stream progress</b> — bridge <see cref="StepPreview"/> onto Swarm's <c>takeOutput</c> contract.</item>
/// <item><b>Map result</b> — Engine pixels/frames/samples back into a SwarmUI <see cref="Image"/>.</item>
/// </list>
/// </summary>
public class HartsyInferenceBackend : AbstractT2IBackend
{
    /// <summary>Settings persisted to Data/Backends.fds. Must be nested inside the
    /// backend class so BackendHandler.RegisterBackendType discovers it via reflection.</summary>
    public class HartsyInferenceBackendSettings : AutoConfiguration
    {
        [ConfigComment("Compute backend to use. 'auto' tries CUDA, then CPU.")]
        public string ComputeBackend = "auto";

        [ConfigComment("Which GPU to use, if multiple are available.\nShould be a single number, like '0' (first GPU), '1' (second GPU), etc.\nIgnored for the CPU compute backend.\nThis is a CUDA device ordinal, which is NOT necessarily the same order nvidia-smi shows: CUDA enumerates fastest-first by default, so on a mixed-GPU machine '0' is the fastest card. To confirm which physical GPU you got, watch nvidia-smi memory while a generation runs.\nRun one backend per GPU to use several cards at once.")]
        public string GPU_ID = "0";

        [ConfigComment("How to handle models that do not fit in VRAM.\n'auto' (default): measure free VRAM and stream weights from system RAM only when the model would not otherwise fit. Cards with headroom keep the full-speed resident path, so this costs nothing when it isn't needed.\n'on': always stream, even when the model would fit. Useful when sharing the GPU with another program (e.g. a second backend), or to test the streamed path.\n'off': never stream and never auto-evict — load everything and let an oversized model fail with an out-of-VRAM error. For operators who size their own workloads and want a hard failure rather than a slow generation.\nStreaming is typically 5-8x slower than a fully-resident model, but it is what lets large models run on a 12GB card at all.")]
        public string LowVram = "auto";

        [ConfigComment("GPU for the text encoders (CLIP / T5 / umT5), separate from the main GPU_ID.\nEmpty (default) = same GPU as everything else.\nSet to another CUDA ordinal (e.g. '1') to keep the multi-GB text encoders off the main card — the biggest VRAM win on video models (Wan's umT5 is T5-XXL-class) and Flux.\nMiniMax-H3 honours this too (its Qwen3-VL encoder is ~15 GB), and because the placement is deliberate the encoder's weights STAY RESIDENT between generations instead of being freed for the DiT — the second card has nothing else competing for the space.\nHeads up on mixed cards: the encoder produces slightly different embeddings on a different GPU architecture, so moving it changes the output for a given seed. It is deterministic run-to-run, just not identical to the same seed on the main card.\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string TextEncoderGpuId = "";

        [ConfigComment("GPU for the VAE encode/decode, separate from the main GPU_ID.\nEmpty (default) = same GPU as everything else.\nSet to another CUDA ordinal (e.g. '1') to run the decode's large activation footprint off the main card — useful when a big DiT stays resident and the full-res decode would otherwise force an evict/re-upload cycle every generation.\nMiniMax-H3 honours this for BOTH halves (keyframe/reference encode and the video+audio decode), and the weights stay resident between generations for the same reason as TextEncoderGpuId.\nThe latents cross to the other card as host tensors, so the denoise is bit-identical; only the decode kernels differ on a different GPU architecture (measured: a handful of 1/255 steps on H3, no structural change).\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string VaeGpuId = "";

        [ConfigComment("Second GPU to run CFG's negative-prompt branch on, concurrent with the positive branch on the main GPU_ID — a latency win (not a VRAM win) when the denoiser fits on BOTH cards, since the weights are REPLICATED, not split.\nEmpty (default) = off; CFG runs sequentially on GPU_ID as usual.\nSet to another CUDA ordinal (e.g. '1') to enable it. Wired for Wan video (T2V/TI2V single-expert; A14B MoE falls back to sequential) and Flux true-CFG (negative prompt + CFG > 1; guidance-embedded runs without a real negative branch record 'inapplicable').\nWhen the second card cannot hold a full replica of the denoiser, the generation SUCCEEDS but silently falls back to sequential — check the engine log for the '[CfgParallel]' line ('active' vs 'fell-back(...)') to confirm which path actually ran.\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string CfgParallelGpuId = "";

        [ConfigComment("Second GPU to pool VRAM with for large DiTs that don't fit on GPU_ID alone — the denoiser's block loop is SPLIT across both cards (not replicated), so this is a VRAM win, not a latency win (sequential pipeline split, same per-step speed as one card).\nVerified end-to-end on real weights for: Krea 2, Qwen-Image (20B — the flagship 'does not fit 24GB' case), MiniMax-H3 (fp8 build only; the bf16 build exceeds any two-consumer-card pool and keeps streaming), and Flux.1 (plain generations only — ControlNet/Kontext/inpaint/regional requests automatically fall back to unsharded with a log line). Chroma and HunyuanImage use the same machinery pending their checkpoints' verification runs.\nSharding disables the per-step CUDA graph and step-cache for the sharded model (a captured graph cannot span devices) — expect eager-path step times.\nEmpty (default) = off.\nSet to another CUDA ordinal (e.g. '1') to enable it. Cannot be combined with CfgParallelGpuId — they are two different ways to use a second GPU for the same model (VRAM pooling vs weight replication for latency) and were not designed to compose; the backend will fail to start if both are set.\nNote this also feeds the same placement list LLM text generation uses for layer-split placement, so enabling it may also change where text models place their layers.\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string DitShardGpuId = "";

        [ConfigComment("Second GPU to pool VRAM with for large LANGUAGE models (text LLMs and big audio LMs), WITHOUT enabling DiT sharding — the LM's layer stack is SPLIT across both cards (weights pooled, not replicated).\nText models (LLMAssistant etc.) layer-split exactly as they do when DitShardGpuId is set.\nBig audio LMs (YuE's 7B Stage-1 in AudioLab) additionally switch from the single-card Q4_K quantization default to UN-QUANTIZED checkpoint precision (bf16) pooled across both cards — the quality win the pooling exists for. Override the precision with the HARTSY_AUDIO_LM_QUANT env var (q4k|q8|off) on the service if needed.\nEmpty (default) = off. Redundant when DitShardGpuId is already set (that feeds the same shard list); if both are set they must agree.\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string LmShardGpuId = "";

        [ConfigComment("Path to the compiled kernel directory (the folder CONTAINING 'Ptx' and 'Spirv').\nEmpty = resolve next to the engine assemblies (the extension's own output folder), which is correct for a normal install.")]
        public string KernelDirectory = "";

        [ConfigComment("How many extra requests may queue up on this backend while one is generating.\n0 means a single live generation with nothing waiting (the scheduler routes further requests to other backends/GPUs immediately).\n1 (default) means a live generation plus one extra waiting in line before further requests route elsewhere.\n-1 makes this a UI-only instance that cannot do actual generations.\nGenerations always run one at a time on this backend (the queue just lets requests wait here instead of being sent elsewhere).")]
        public int OverQueue = 1;

        [ConfigComment("Per-step progress previews.\nThe Engine decides whether a given pipeline can produce preview pixels; when it doesn't, only the progress bar moves.\nTurn this off to skip JPEG-encoding previews entirely.")]
        public bool Previews = true;

        [ConfigComment("Whether to auto-update the HartsyInference engine (the in-process NuGet library) when this backend starts.\n'false' (default): never check.\n'true': on start, check NuGet for a newer engine build and, if found, download + rebuild the extension against it.\n'aggressive': same as 'true' but also clears the NuGet caches first (fixes a stuck floating-version restore) and automatically restarts SwarmUI to load the new build.\nThe engine is loaded in-process, so a staged update applies on the NEXT SwarmUI restart (a loaded DLL can't hot-swap). With 'true' you'll get a log line telling you to restart; 'aggressive' restarts for you.")]
        public string AutoUpdate = "false";
    }

    public HartsyInferenceBackendSettings Settings => SettingsRaw as HartsyInferenceBackendSettings;

    /// <summary>The Engine facade. Owns the compute backend, every loaded pipeline, and all generation.</summary>
    private InferenceEngine _engine;

    /// <summary>Device used by the model-driven ControlNet annotators (Depth / OpenPose / SoftEdge / …). Normally
    /// borrowed from the Engine (<c>InferenceEngine.ComputeBackend</c>) — a second backend on the same device would
    /// collide with the Engine's per-context CUDA state and its disposal would evict the Engine's resident weights.
    /// Only when no Engine exists is a standalone device created (and then owned + disposed by us).</summary>
    private IBackend _preprocessBackend;
    private bool _ownsPreprocessBackend;
    private readonly object _preprocessBackendLock = new();

    /// <summary>Cancellation source for the in-flight generation.</summary>
    private CancellationTokenSource _cancelCts;

    /// <summary>Serializes generations so a backend with <c>OverQueue &gt; 0</c> (MaxUsages &gt; 1) accepts
    /// extra requests into a queue but still runs them ONE AT A TIME. The Engine's caches and device are shared
    /// across a generation, so concurrent execution would collide — this lock makes the over-queue safe by holding
    /// extra dispatched jobs here until the current one finishes.</summary>
    private readonly SemaphoreSlim _genLock = new(1, 1);

    /// <summary>Live backend count per device key ("cuda:0"), to warn when two backends share one GPU.</summary>
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _liveDeviceCounts = new();

    /// <summary>The device key this instance registered in <see cref="_liveDeviceCounts"/>, for Shutdown to release.</summary>
    private string _registeredDeviceKey;

    /// <summary>Every feature flag this backend advertises. Static so the startup self-check in
    /// <c>SwarmUIHartsyInference</c> can compare it against the flags our registered params actually carry —
    /// <see cref="T2IEngine"/> refuses any job whose required flags this set does not cover (unless the flag is in
    /// <c>T2IEngine.DisregardedFeatureFlags</c>), and that refusal names no param, so a flag added to a param and
    /// forgotten here is invisible until someone's generation stops working.</summary>
    public static readonly IReadOnlyList<string> DeclaredFeatures =
    [
        // We advertise "comfyui" so that, when a Comfy backend is ALSO configured, requests
        // carrying Comfy-tagged params reach our validator instead of being pre-filtered out
        // by T2IEngine's flag check. SwarmUI can't split one request across two backends, so
        // a request that picked up both "comfyui" (from Comfy's shared params, eg Sampler) and
        // "hartsyinference" would otherwise match NEITHER backend and refuse the generation
        // entirely (Comfy lacks "hartsyinference", we lack "comfyui").
        //
        // Honesty is enforced one layer down: IsValidForThisBackend refuses any comfyui-only
        // param we can't actually service, and every composition feature is checked against the
        // Engine recipe's own declared ImageFeatures — so advertising a flag here never means
        // "we silently serve everything tagged with it".
        "hartsyinference",
        "comfyui",
        "text2image",
        "flux-dev",       // in DisregardedFeatureFlags — informational
        "lora",           // per-family; gated by the recipe's ImageFeatures.Lora
        "endstepsearly",  // ImageRequest.EndStepsEarly
        "refiners",       // per-family; gated by ImageFeatures.Refiner
        "img2img",        // per-family; gated by ImageFeatures.Img2Img
        "inpaint",        // per-family; gated by ImageFeatures.Inpaint
        "controlnet",     // per-family; gated by ImageFeatures.ControlNet
        "ipadapter",      // per-family; gated by ImageFeatures.IpAdapter
        "variation_seed", // per-family; gated by ImageFeatures.VariationSeed
        "seamless",       // SeamlessTileable (shared core param, own flag); per-family, gated by ImageFeatures.SeamlessTiling
        "video",          // Wan (T2V/I2V + VACE / Animate / S2V) + LTX-Video + LTX-2 + Lance Video + MiniMax-H3
    ];

    public override IEnumerable<string> SupportedFeatures => DeclaredFeatures;

    /// <summary>Bridges HartsyInference's internal logger into Swarm's logging system so
    /// diagnostics like the OOM probe in CudaMemory.Allocate appear in the main log file
    /// instead of falling into Console.Error. Idempotent — safe to call multiple times.</summary>
    private static int _loggerWired = 0;

    private static void EnsureLoggerWired()
    {
        if (Interlocked.Exchange(ref _loggerWired, 1) != 0)
        {
            return;
        }
        // Don't double-filter: set HartsyInference's level to Verbose so every message reaches the
        // bridge, and let Swarm's own MinimumLevel decide what actually appears.
        SiLogs.MinLevel = HartsyInference.Core.Logging.LogLevel.Verbose;
        SiLogs.SetLogger((level, msg) =>
        {
            switch (level)
            {
                case HartsyInference.Core.Logging.LogLevel.Verbose: Logs.Verbose(msg); break;
                case HartsyInference.Core.Logging.LogLevel.Debug: Logs.Debug(msg); break;
                case HartsyInference.Core.Logging.LogLevel.Info: Logs.Info(msg); break;
                case HartsyInference.Core.Logging.LogLevel.Warning: Logs.Warning(msg); break;
                case HartsyInference.Core.Logging.LogLevel.Error: Logs.Error(msg); break;
                default: Logs.Info(msg); break;
            }
        });
    }

    // ─────────────────────────────── 1. Lifecycle ───────────────────────────────

    public override async Task Init()
    {
        EnsureLoggerWired();
        if (await MaybeAutoUpdateEngine())
        {
            // A newer engine was staged but can't hot-swap into this process. Fail loud instead of
            // silently serving the stale in-process version — the user must restart to load it.
            Status = BackendStatus.ERRORED;
            AddLoadStatus("Engine update staged — RESTART SwarmUI to load the new engine. This backend is disabled until you restart.");
            Logs.Warning("[HartsyInference] Backend disabled: a newer engine is staged. Restart SwarmUI to load it (set AutoUpdate=aggressive to auto-restart).");
            return;
        }
        try
        {
            string requested = Settings?.ComputeBackend?.ToLowerInvariant() ?? "auto";
            int? deviceOrdinal = ParseGpuId(Settings?.GPU_ID);
            LowVramMode? lowVram = ParseLowVramSetting(Settings?.LowVram);

            // Kernels ship in our extension's own output dir, NOT Swarm's main runtime dir. The Engine's
            // BackendFactory already resolves relative to the engine assemblies (which live beside us), so
            // an override is only needed when the operator moved them.
            if (!string.IsNullOrWhiteSpace(Settings?.KernelDirectory))
            {
                BackendFactory.KernelDirOverride = Settings.KernelDirectory;
            }
            string ptxDir = BackendFactory.KernelDir(BackendFactory.PtxDirName);
            string spvDir = BackendFactory.KernelDir(BackendFactory.SpirvDirName);
            AddLoadStatus($"Kernel paths: PTX={ptxDir} (exists={Directory.Exists(ptxDir)}), SPIR-V={spvDir} (exists={Directory.Exists(spvDir)})");

            // Fail HERE (backend shows ERRORED with the reason) instead of on the first generation: the engine
            // constructs its device lazily, so a bad GPU_ID would otherwise report RUNNING and die mid-request.
            BackendFactory.Validate(BackendFactory.WithOrdinal(requested, deviceOrdinal ?? 0));

            // Optional split placement: text encoders on their own GPU. Validated eagerly like the main device.
            int? textEncoderOrdinal = ParseGpuId(Settings?.TextEncoderGpuId);
            PlacementConfig? placement = null;
            string teSelector = null;
            if (textEncoderOrdinal.HasValue && textEncoderOrdinal != (deviceOrdinal ?? 0))
            {
                teSelector = BackendFactory.WithOrdinal("cuda", textEncoderOrdinal.Value);
                BackendFactory.Validate(teSelector);
                AddLoadStatus($"Text encoders placed on {teSelector} (denoiser stays on GPU {deviceOrdinal ?? 0}).");
            }

            // Optional split placement: VAE encode/decode on its own GPU. Same eager validation.
            int? vaeOrdinal = ParseGpuId(Settings?.VaeGpuId);
            string vaeSelector = null;
            if (vaeOrdinal.HasValue && vaeOrdinal != (deviceOrdinal ?? 0))
            {
                vaeSelector = BackendFactory.WithOrdinal("cuda", vaeOrdinal.Value);
                BackendFactory.Validate(vaeSelector);
                AddLoadStatus($"VAE placed on {vaeSelector} (denoiser stays on GPU {deviceOrdinal ?? 0}).");
            }

            // Optional CFG-branch parallelism: uncond on a second GPU, concurrent with cond on GPU_ID. Validated
            // eagerly like the main device; a second backend is constructed lazily by the Engine on first use.
            int? cfgParallelOrdinal = ParseGpuId(Settings?.CfgParallelGpuId);
            string cfgParallelSelector = null;
            if (cfgParallelOrdinal.HasValue && cfgParallelOrdinal != (deviceOrdinal ?? 0))
            {
                cfgParallelSelector = BackendFactory.WithOrdinal("cuda", cfgParallelOrdinal.Value);
                BackendFactory.Validate(cfgParallelSelector);
                AddLoadStatus($"CFG uncond branch placed on {cfgParallelSelector}, concurrent with cond on GPU {deviceOrdinal ?? 0}.");
            }

            // Optional Phase 8 DiT sharding: a large DiT's block loop split across two GPUs to pool VRAM (not a
            // latency win — sequential pipeline split). Mutually exclusive with CfgParallelGpuId; the Engine
            // rejects a PlacementConfig with both set (InferenceEngine ctor → PlacementPlanner.ValidatePlacement),
            // which surfaces here as this method's outer catch setting Status = ERRORED with the reason.
            int? ditShardOrdinal = ParseGpuId(Settings?.DitShardGpuId);
            string ditShardSelector = null;
            bool enableDitSharding = false;
            if (ditShardOrdinal.HasValue && ditShardOrdinal != (deviceOrdinal ?? 0))
            {
                ditShardSelector = BackendFactory.WithOrdinal("cuda", ditShardOrdinal.Value);
                BackendFactory.Validate(ditShardSelector);
                enableDitSharding = true;
                AddLoadStatus($"DiT sharding enabled: denoiser block loop split across GPU {deviceOrdinal ?? 0} and " +
                    $"{ditShardSelector} (VRAM pooling, not a latency win).");
            }

            // LM-only shard route: same shard device list, no DiT flag. Text LLMs layer-split; big audio LMs
            // (YuE Stage-1) additionally default to un-quantized weights pooled across the pair.
            int? lmShardOrdinal = ParseGpuId(Settings?.LmShardGpuId);
            string lmShardSelector = null;
            if (lmShardOrdinal.HasValue && lmShardOrdinal != (deviceOrdinal ?? 0))
            {
                if (ditShardOrdinal.HasValue && ditShardOrdinal != lmShardOrdinal)
                {
                    Status = BackendStatus.ERRORED;
                    AddLoadStatus($"LmShardGpuId ({lmShardOrdinal}) and DitShardGpuId ({ditShardOrdinal}) point at " +
                        "different ordinals — they share one shard device list. Set just DitShardGpuId (it implies " +
                        "the LM split) or make them match.");
                    return;
                }
                lmShardSelector = BackendFactory.WithOrdinal("cuda", lmShardOrdinal.Value);
                BackendFactory.Validate(lmShardSelector);
                if (!enableDitSharding)
                {
                    AddLoadStatus($"LM sharding enabled: large LMs layer-split across GPU {deviceOrdinal ?? 0} and " +
                        $"{lmShardSelector} (VRAM pooled; YuE Stage-1 runs un-quantized by default).");
                }
            }
            string shardSelector = ditShardSelector ?? lmShardSelector;

            if (teSelector is not null || vaeSelector is not null || cfgParallelSelector is not null || shardSelector is not null)
            {
                placement = new PlacementConfig
                {
                    TextEncoderDevice = teSelector,
                    VaeDevice = vaeSelector,
                    CfgParallelDevice = cfgParallelSelector,
                    ShardDevices = shardSelector is not null
                        ? new[] { BackendFactory.WithOrdinal(requested, deviceOrdinal ?? 0), shardSelector }
                        : Array.Empty<string>(),
                    EnableDitSharding = enableDitSharding,
                };
            }

            // 'auto' silently degrades to CPU when CUDA can't load, and the CPU kernels are F32-only: fp8/bf16
            // checkpoint weights read past the end of their allocation, corrupting the heap and aborting the
            // process. Refuse to come up rather than serve a backend that cannot generate safely.
            if (BackendFactory.Resolve(requested) == "cpu" && BackendFactory.Kind(requested) != "cpu")
            {
                string why = CudaUnavailableReason() ?? "no reason reported";
                Status = BackendStatus.ERRORED;
                string msg = $"CUDA is unavailable, so compute='{requested}' resolved to CPU — refusing to start. "
                    + $"Reason: {why} "
                    + "The CPU backend cannot run fp8/bf16 image models (it is F32-only and will corrupt memory). "
                    + "Fix the GPU/driver, or set this backend's ComputeBackend to 'cpu' explicitly to override.";
                AddLoadStatus(msg);
                Logs.Error($"[HartsyInference] Backend #{BackendData?.ID} refusing CPU fallback: {msg}");
                return;
            }

            AddLoadStatus($"Constructing HartsyInference.Engine (compute='{requested}', device={deviceOrdinal?.ToString() ?? "auto"})...");
            EngineOptions engineOptions = new EngineOptions { LowVram = lowVram, Placement = placement };
            _engine = deviceOrdinal.HasValue
                ? new InferenceEngine(requested, deviceOrdinal.Value, engineOptions)
                : new InferenceEngine(requested, engineOptions);
            AddLoadStatus($"Engine ready: {_engine.BackendDescription}");
            RegisterDeviceUsage(requested, deviceOrdinal ?? 0);

            // MaxUsages is what the scheduler checks to decide when to route a request to a different
            // backend (BackendHandler: in-use once Usages >= MaxUsages). Mirror ComfyUI's model:
            // MaxUsages = 1 (the live gen) + OverQueue (extra waiting slots). OverQueue = -1 → MaxUsages 0
            // → UI-only (no gens). The _genLock keeps the extra slots queued rather than concurrent.
            int overQueue = Math.Max(-1, Settings?.OverQueue ?? 0);
            MaxUsages = 1 + overQueue;

            // RUNNING = alive and ready to accept generations. IDLE in Swarm means "alive but currently
            // unavailable", which would make BackendHandler skip us for routing.
            Status = BackendStatus.RUNNING;
            AddLoadStatus($"Ready. Drivable image families: {string.Join(", ", RecipeRegistry.RegisteredNames)}.");
            Logs.Init($"[HartsyInference] Backend #{BackendData?.ID} live ({_engine.BackendDescription})");
        }
        catch (Exception ex)
        {
            Status = BackendStatus.ERRORED;
            AddLoadStatus($"Init failed: {ex.Message}");
            Logs.Error($"[HartsyInference] Backend #{BackendData?.ID} init failed: {ex}");
            throw;
        }
    }

    /// <summary>Tracks how many live backends target one device and notes the sharing tradeoffs: co-residency
    /// means both models must fit in that GPU's VRAM together, and the engine serializes same-device generations
    /// (per-backend state is fully isolated; concurrent same-GPU execution arrives with the DeviceGate flip).</summary>
    private void RegisterDeviceUsage(string requested, int ordinal)
    {
        try
        {
            string kind = BackendFactory.Resolve(BackendFactory.WithOrdinal(requested, ordinal));
            if (kind != "cuda" && kind != "vulkan")
            {
                return;
            }
            string key = $"{kind}:{ordinal}";
            _registeredDeviceKey = key;
            int live = _liveDeviceCounts.AddOrUpdate(key, 1, (_, n) => n + 1);
            if (live > 1)
            {
                string warning = $"{live} HartsyInference backends now share device {key}. Their models must co-fit " +
                    "in that GPU's VRAM, and generations on this device run one at a time (the engine serializes " +
                    "same-GPU work; concurrent same-GPU execution is planned). Use distinct GPU_IDs for parallel throughput.";
                AddLoadStatus($"WARNING: {warning}");
                Logs.Warning($"[HartsyInference] {warning}");
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] Device-usage registration failed (non-fatal): {ex.Message}");
        }
    }

    public override async Task Shutdown()
    {
        Status = BackendStatus.DISABLED;
        _cancelCts?.Cancel();
        _cancelCts = null;
        _engine?.Dispose();
        _engine = null;
        DisposePreprocessBackend();
        if (_registeredDeviceKey is not null)
        {
            _liveDeviceCounts.AddOrUpdate(_registeredDeviceKey, 0, (_, n) => Math.Max(0, n - 1));
            _registeredDeviceKey = null;
        }
        CurrentModelName = null;
        await Task.CompletedTask;
    }

    public override async Task<bool> FreeMemory(bool systemRam)
    {
        if (_engine is null)
        {
            return false;
        }
        _engine.FreeMemory();
        // Annotator weights live on the Engine's backend (borrowed), so the engine free above covers them; dropping
        // the reference here re-borrows a fresh backend next use in case the engine rebuilt its device. A standalone
        // annotator device (no engine) is disposed outright.
        DisposePreprocessBackend();
        if (systemRam)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        CurrentModelName = null;
        await Task.CompletedTask;
        return true;
    }

    /// <summary>Triggered by Swarm when the user clicks Cancel.</summary>
    public void RequestCancel()
    {
        _cancelCts?.Cancel();
    }

    /// <summary>Why the Engine could not use CUDA, or null when that Engine build does not report it.</summary>
    // Reflection, not a direct reference: Swarm's own extension rebuild never passes UseLocalHartsy, so a
    // compile-time dependency on this property silently drops both image backends when the git hash changes.
    private static string CudaUnavailableReason()
        => typeof(BackendFactory).GetProperty("CudaUnavailableReason", BindingFlags.Public | BindingFlags.Static)
            ?.GetValue(null) as string;

    /// <summary>Parses the configured GPU_ID into the device ordinal handed to the Engine. Swarm allows a
    /// comma-separated list (one backend instance per device); a single Engine drives one device, so the first
    /// entry wins. Null (blank/unparseable) lets the Engine pick its own default device.</summary>
    /// <remarks><b>The ordinal is CUDA's, not <c>nvidia-smi</c>'s.</b> CUDA enumerates fastest-first by default, so
    /// on a mixed-GPU host <c>GPU_ID=0</c> is the fastest card, which need not be <c>nvidia-smi</c>'s index 0.
    /// Verified on the dev box: ordinal 0 is the RTX 4090 while <c>nvidia-smi</c> index 0 is the RTX 3060.</remarks>
    private static int? ParseGpuId(string gpuId)
    {
        if (string.IsNullOrWhiteSpace(gpuId))
        {
            return null;
        }
        string first = gpuId.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? "";
        if (!int.TryParse(first, out int ordinal) || ordinal < 0)
        {
            Logs.Warning($"[HartsyInference] GPU_ID='{gpuId}' is not a valid device ordinal; letting the Engine choose.");
            return null;
        }
        return ordinal;
    }

    /// <summary>Maps the backend's low-VRAM setting to the engine's per-engine policy override. Per-engine (not the
    /// old <c>HARTSY_LOWVRAM</c> env var, which was process-wide last-writer-wins) so two backends on different-size
    /// cards keep their own policies. Null = auto (engine measures free VRAM per phase).</summary>
    private static LowVramMode? ParseLowVramSetting(string mode)
    {
        string normalized = string.IsNullOrWhiteSpace(mode) ? "auto" : mode.Trim().ToLowerInvariant();
        LowVramMode? parsed = normalized switch
        {
            "auto" => null,
            "on" or "1" or "true" => LowVramMode.ForceOn,
            "off" or "0" or "false" => LowVramMode.ForceOff,
            _ => null,
        };
        if (parsed is null && normalized != "auto")
        {
            Logs.Warning($"[HartsyInference] LowVram='{mode}' is not recognized; using 'auto'. Valid: auto, on, off.");
            normalized = "auto";
        }
        Logs.Info($"[HartsyInference] Low-VRAM handling: {normalized}"
            + (parsed == LowVramMode.ForceOff ? " (models larger than VRAM will fail rather than stream)." : "."));
        return parsed;
    }

    // ─────────────────────────────── 2. Load ───────────────────────────────

    public override async Task<bool> LoadModel(T2IModel model, T2IParamInput input)
    {
        await Task.CompletedTask;
        if (model is null || _engine is null)
        {
            return false;
        }
        string compat = model.ModelClass?.CompatClass?.ID;
        if (!ModelSupport.IsArchitectureSupported(compat))
        {
            Logs.Warning($"[HartsyInference] LoadModel: architecture '{compat}' isn't drivable. Returning false so another backend can handle it.");
            return false;
        }
        // The Engine loads (and caches) the pipeline lazily inside its generate call, keyed on the checkpoint
        // path plus the request's LoRA/component identity — there is no separate load step to drive here, and
        // duplicating one would only fight its cache.
        CurrentModelName = model.Name;
        return true;
    }

    // ─────────────────────────────── 3. Generate ───────────────────────────────

    public override async Task<Image[]> Generate(T2IParamInput input)
    {
        List<Image> images = [];
        await GenerateLive(input, "single", obj =>
        {
            if (obj is Image img)
            {
                images.Add(img);
            }
            else if (obj is T2IEngine.ImageOutput o && o.File is Image of)
            {
                images.Add(of);
            }
        });
        return [.. images];
    }

    public override async Task GenerateLive(T2IParamInput input, string batchId, Action<object> takeOutput)
    {
        // Status stays RUNNING throughout — it's the resting healthy state, not a "currently busy" flag.
        // BackendData.Usages is what tracks active utilization.
        //
        // Serialize: when OverQueue > 0 the scheduler may dispatch multiple jobs to us; run them one at a
        // time. Acquire BEFORE touching _cancelCts so the per-gen token belongs to the job holding the GPU.
        await _genLock.WaitAsync();
        // Link to the input's InterruptToken (session.SessInterrupt) so the gen-page "stop" button
        // actually cancels us — that's the token Swarm trips on stop.
        _cancelCts = CancellationTokenSource.CreateLinkedTokenSource(input.InterruptToken);
        long startMs = Environment.TickCount64;
        try
        {
            T2IModel model = input.Get(T2IParamTypes.Model)
                ?? throw new InvalidOperationException("No model selected.");
            string compat = model.ModelClass?.CompatClass?.ID;
            ModelSupport.Family family = ModelSupport.Resolve(compat);
            if (family is null || !ModelSupport.IsArchitectureSupported(compat))
            {
                throw new InvalidOperationException($"HartsyInference: {ModelSupport.WhyNotSupported(compat)}");
            }

            string promptPreview = input.Get(T2IParamTypes.Prompt) ?? "";
            if (promptPreview.Length > 80)
            {
                promptPreview = promptPreview[..80] + "…";
            }
            Logs.Verbose($"[HartsyInference] Backend #{BackendData?.ID} accepted job batch='{batchId}' model='{model.Name}' "
                + $"compat='{compat}' family='{family.Id}' ({family.Kind}) prompt=\"{promptPreview}\"");

            CurrentModelName = model.Name;
            CancellationToken cancel = _cancelCts.Token;
            ModelSpec spec = ModelSupport.BuildSpec(model, family);
            IProgress<StepPreview> progress = BuildProgressBridge(batchId, takeOutput);

            async Task<Image[]> Dispatch() => family.Kind switch
            {
                ModelSupport.Kind.Video => [await GenerateVideo(spec, input, progress, cancel)],
                ModelSupport.Kind.Music => [await GenerateMusic(spec, input, progress, cancel)],
                _ => [await GenerateImage(spec, input, family, progress, cancel)],
            };

            Image[] outputs;
            try
            {
                outputs = await Dispatch();
            }
            catch (OutOfVramException first) when (!cancel.IsCancellationRequested)
            {
                // Worth exactly one retry: the usual cause is a pipeline cached from an earlier, larger model still
                // holding VRAM this request needs, which FreeMemory releases. A second failure is the real thing —
                // this request does not fit — so it becomes a readable refusal rather than another retry.
                Logs.Warning($"[HartsyInference] {first.Message} Freeing cached models and retrying once.");
                await FreeMemory(false);
                try
                {
                    outputs = await Dispatch();
                }
                catch (OutOfVramException second)
                {
                    throw new SwarmReadableErrorException(DescribeVramFailure(second, family));
                }
            }
            catch (OutOfVramException only)
            {
                throw new SwarmReadableErrorException(DescribeVramFailure(only, family));
            }

            long totalMs = Environment.TickCount64 - startMs;
            int idx = 0;
            foreach (Image img in outputs)
            {
                Logs.Verbose($"[HartsyInference] Yielding output {idx + 1}/{outputs.Length} batch='{batchId}' "
                    + $"(genTime={totalMs}ms, bytes={img.RawData?.Length ?? 0})");
                takeOutput(new T2IEngine.ImageOutput
                {
                    File = img,
                    IsReal = true,
                    GenTimeMS = totalMs,
                });
                idx++;
            }
            Logs.Verbose($"[HartsyInference] Job batch='{batchId}' complete: {outputs.Length} output(s) in {totalMs}ms.");
        }
        catch (OperationCanceledException)
        {
            Logs.Info("[HartsyInference] Generation cancelled by user.");
        }
        finally
        {
            _cancelCts?.Dispose();
            _cancelCts = null;
            _genLock.Release(); // hand the GPU to the next queued generation, if any
            // No Status change here: we stay RUNNING (alive+ready) regardless of whether the generation
            // succeeded. ERRORED would be wrong — the backend is healthy, only this one request failed.
        }
    }

    /// <summary>Runs a still-image generation and marshals the Engine's RGB result into a SwarmUI image.</summary>
    private async Task<Image> GenerateImage(ModelSpec spec, T2IParamInput input, ModelSupport.Family family,
        IProgress<StepPreview> progress, CancellationToken cancel)
    {
        ImageRequest request = BuildImageRequest(input, family);
        ImageResult result = await _engine.Images.GenerateAsync(spec, request, progress, cancel);
        byte[] rgb = result.Rgb;
        int width = result.Width, height = result.Height;
        // Optional SeedVR2 restore/upscale pass over the still (same param group as the video path). No
        // FreeMemory here: RestoreService frees on pipeline load, and evicting a cached image model every
        // generation would defeat the pipeline cache.
        if (input.TryGet(SwarmUIHartsyInference.VideoRestoreModelParam, out string restoreModel)
            && !string.IsNullOrWhiteSpace(restoreModel))
        {
            ModelSpec restoreSpec = ModelResolver.Resolve(restoreModel, null, Modality.Restore);
            RestoreRequest restoreRequest = BuildRestoreKnobs(input) with
            {
                Image = new ImageData { Rgb = rgb, Width = width, Height = height },
            };
            await foreach (VideoFrame restored in _engine.Restore.RestoreAsync(restoreSpec, restoreRequest, progress, cancel))
            {
                rgb = restored.Rgb;
                width = restored.Width;
                height = restored.Height;
            }
        }
        // Engine-side metadata (resolved arch/seed/steps) folds into Swarm's saved-image metadata; "arch" in
        // particular is checkpoint-sniffed by the engine and not derivable Swarm-side.
        foreach (KeyValuePair<string, string> meta in result.Meta)
        {
            input.ExtraMeta[$"hartsy_{meta.Key}"] = meta.Value;
        }
        input.ExtraMeta["hartsy_engine_seed"] = result.Seed;
        return RgbToImage.FromHwcRgb(rgb, width, height);
    }

    /// <summary>The restore knobs shared by the still and video paths (target size, strength, seed).</summary>
    private static RestoreRequest BuildRestoreKnobs(T2IParamInput input) => new()
    {
        TargetWidth = input.TryGet(SwarmUIHartsyInference.VideoRestoreWidthParam, out int rw) ? rw : null,
        TargetHeight = input.TryGet(SwarmUIHartsyInference.VideoRestoreHeightParam, out int rh) ? rh : null,
        Strength = input.TryGet(SwarmUIHartsyInference.VideoRestoreStrengthParam, out double rst) ? (float)rst : null,
        Seed = input.Get(T2IParamTypes.Seed, -1L) is long seed and >= 0 ? (int)(seed & 0x7FFFFFFF) : null,
    };

    /// <summary>Runs a video generation, collecting the streamed frames and muxing them into a container.</summary>
    private async Task<Image> GenerateVideo(ModelSpec spec, T2IParamInput input, IProgress<StepPreview> progress, CancellationToken cancel)
    {
        VideoRequest request = BuildVideoRequest(input);
        string format = input.Get(T2IParamTypes.VideoFormat, "h264-mp4");

        // Tier 3.5: stream frames straight to ffmpeg's stdin instead of buffering the whole clip, for the subset
        // of requests that can't need the full clip in memory first — no boomerang/trim (need the final frame
        // count / random access to reorder) and no audio (VideoAudioResolver.Resolve needs the final frame count
        // too, to compute clip duration for trimming/looping the track). IVideoService.GenerateFramesAsync itself
        // throws NotSupportedException for any family/variant that can't stream at all (as of 2026-08-11, only
        // Wan's plain T2V/I2V-TI2V path) — caught below and falls back to the buffered path rather than refusing
        // the generation outright, since these are core Swarm params a user may legitimately have set without
        // knowing which families can stream.
        bool wantsBoomerang = input.Get(T2IParamTypes.VideoBoomerang, false);
        bool wantsTrim = input.Get(T2IParamTypes.TrimVideoStartFrames, 0) > 0 || input.Get(T2IParamTypes.TrimVideoEndFrames, 0) > 0;
        bool wantsRestore = input.TryGet(SwarmUIHartsyInference.VideoRestoreModelParam, out string restoreModelCheck) && !string.IsNullOrWhiteSpace(restoreModelCheck);
        bool wantsAudio = input.Get(T2IParamTypes.VideoAudioInput) is not null;
        // ...and no GENERATED soundtrack either. The streaming path pipes frames straight to ffmpeg and has no
        // AudioTrack to give it, so a family that samples audio jointly with video would have its soundtrack
        // silently dropped. Core's own HasJointAVLatents names exactly that set (LTX-2.x and MiniMax-H3), so this
        // stays right as families are added rather than needing a list here.
        bool generatesAudio = input.Get(T2IParamTypes.Model)?.ModelClass?.CompatClass?.HasJointAVLatents == true;
        if (!wantsBoomerang && !wantsTrim && !wantsRestore && !wantsAudio && !generatesAudio)
        {
            try
            {
                return await GenerateVideoStreaming(spec, request, format, progress, cancel);
            }
            catch (NotSupportedException ex)
            {
                Logs.Verbose($"[HartsyInference] Streaming generation not available ({ex.Message}); falling back to buffered.");
            }
        }

        VideoGenerationResult generated = await _engine.Video.GenerateAsync(spec, request, progress, cancel);
        List<byte[]> frames = [];
        int width = 0, height = 0;
        foreach (VideoFrame frame in generated.Frames)
        {
            frames.Add(frame.Rgb);
            width = frame.Width;
            height = frame.Height;
        }
        VideoOutputEncoder.AudioTrack audio = ToAudioTrack(generated.Audio);
        if (frames.Count == 0)
        {
            throw new InvalidOperationException("HartsyInference: the video pipeline produced no frames.");
        }
        // Boomerang is NOT applied here: BuildVideoRequest already sets VideoRequest.VideoBoomerang, and every
        // *RecipePipeline.Generate routes through VideoRecipeUtils.ToVideoFrames, which applies it before the
        // frames ever leave _engine.Video.GenerateAsync above. Re-applying it here used to ping-pong the
        // already-ping-ponged sequence (found scoping Tier 3.5 — see VideoRecipeUtilsFrameEditsTests.cs in the
        // engine repo for the frame-count/order proof) instead of the single loop the user asked for.
        // Optional SeedVR2 restore pass (Video Restore param group): frames go straight into the engine's
        // restore service (no container round-trip), replacing the frame list before muxing.
        if (input.TryGet(SwarmUIHartsyInference.VideoRestoreModelParam, out string restoreModel)
            && !string.IsNullOrWhiteSpace(restoreModel))
        {
            // The resident T2V DiT and SeedVR2's VAE peak cannot share 24 GB.
            _engine.FreeMemory();
            ModelSpec restoreSpec = ModelResolver.Resolve(restoreModel, null, Modality.Restore);
            RestoreRequest restoreRequest = BuildRestoreKnobs(input) with
            {
                Frames = [.. frames.Select(f => new ImageData { Rgb = f, Width = width, Height = height })],
                ClipFrames = input.TryGet(SwarmUIHartsyInference.VideoRestoreClipFramesParam, out int rcf) ? rcf : 5,
                Overlap = input.TryGet(SwarmUIHartsyInference.VideoRestoreOverlapParam, out int rov) ? rov : 1,
            };
            List<byte[]> restoredFrames = [];
            await foreach (VideoFrame restoredFrame in _engine.Restore.RestoreAsync(restoreSpec, restoreRequest, progress, cancel))
            {
                restoredFrames.Add(restoredFrame.Rgb);
                width = restoredFrame.Width;
                height = restoredFrame.Height;
            }
            frames = restoredFrames;
        }
        if (audio is not null && !VideoOutputEncoder.FormatSupportsAudio(format))
        {
            Logs.Warning($"[HartsyInference] Video format '{format}' cannot carry audio; the {audio.SampleRate} Hz track is not muxed.");
            audio = null;
        }
        return VideoOutputEncoder.Encode([.. frames], width, height, request.Fps ?? 25, format, cancel, audio);
    }

    /// <summary>The streaming half of <see cref="GenerateVideo"/> (Tier 3.5): pulls frames from
    /// <c>IVideoService.GenerateFramesAsync</c> and pipes them straight to <see cref="VideoOutputEncoder.EncodeStreamingAsync"/>
    /// as they decode. Peeks exactly one frame to learn width/height (the encoder needs them before the first
    /// write; the request's own Width/Height aren't reliable here — resolutions get snapped/defaulted per family)
    /// and re-prepends it via a small local iterator so nothing is lost. No audio track — callers only reach this
    /// when <c>VideoAudioInput</c> is unset, matching <see cref="GenerateVideo"/>'s eligibility gate.</summary>
    private async Task<Image> GenerateVideoStreaming(ModelSpec spec, VideoRequest request, string format, IProgress<StepPreview> progress, CancellationToken cancel)
    {
        IAsyncEnumerator<VideoFrame> frames = _engine.Video.GenerateFramesAsync(spec, request, progress, cancel).GetAsyncEnumerator(cancel);
        if (!await frames.MoveNextAsync())
        {
            await frames.DisposeAsync();
            throw new InvalidOperationException("HartsyInference: the video pipeline produced no frames.");
        }
        VideoFrame first = frames.Current;
        // RgbFramesFrom takes ownership of `frames` from here: its own finally disposes it once
        // EncodeStreamingAsync's `await foreach` drains it or unwinds — the compiler-generated disposal on an
        // `await foreach` target runs in both cases, so there's no double-dispose or leak on either path.
        return await VideoOutputEncoder.EncodeStreamingAsync(
            RgbFramesFrom(first, frames), first.Width, first.Height, request.Fps ?? 25, format, cancel);
    }

    /// <summary>Yields <paramref name="first"/>'s RGB bytes, then the rest of <paramref name="remaining"/>'s —
    /// the peek-and-reprepend glue <see cref="GenerateVideoStreaming"/> needs to learn width/height before
    /// handing the stream to the encoder.</summary>
    private static async IAsyncEnumerable<byte[]> RgbFramesFrom(VideoFrame first, IAsyncEnumerator<VideoFrame> remaining)
    {
        try
        {
            yield return first.Rgb;
            while (await remaining.MoveNextAsync())
            {
                yield return remaining.Current.Rgb;
            }
        }
        finally
        {
            await remaining.DisposeAsync();
        }
    }

    /// <summary>Adapts the Engine's planar <see cref="AudioBuffer"/> onto the encoder's stereo mux track.</summary>
    private static VideoOutputEncoder.AudioTrack ToAudioTrack(AudioBuffer audio)
    {
        if (audio is null || audio.IsEmpty)
        {
            return null;
        }
        (float[] left, float[] right) = audio.ToStereo();
        return new VideoOutputEncoder.AudioTrack { Left = left, Right = right, SampleRate = audio.SampleRate };
    }

    /// <summary>Runs a music generation; the Engine returns an encoded WAV container, which Swarm carries as an
    /// <see cref="Image"/> keyed by <see cref="MediaType.AudioWav"/> exactly like every other audio output.</summary>
    private async Task<Image> GenerateMusic(ModelSpec spec, T2IParamInput input, IProgress<StepPreview> progress, CancellationToken cancel)
    {
        MusicRequest request = BuildMusicRequest(input);
        AudioResult result = await _engine.Music.GenerateAsync(spec, request, progress, cancel);
        Logs.Verbose($"[HartsyInference] Music: {result.DurationSeconds:0.0}s @ {result.SampleRate} Hz ({result.Format}).");
        foreach (KeyValuePair<string, string> meta in result.Meta)
        {
            input.ExtraMeta[$"hartsy_{meta.Key}"] = meta.Value;
        }
        return new Image(result.Data, MediaType.AudioWav);
    }

    /// <summary>Bridges the Engine's <see cref="StepPreview"/> ticks onto Swarm's <c>takeOutput</c> progress
    /// contract: every tick forwards a <c>{batch_index, overall_percent, current_percent}</c> JObject (or the
    /// richer preview JObject when the Engine supplied preview pixels), and every 5% boundary also logs an
    /// ASCII progress bar.</summary>
    private IProgress<StepPreview> BuildProgressBridge(string batchId, Action<object> takeOutput)
    {
        PreviewEncoder previewEncoder = new(Settings?.Previews ?? true);
        int lastLoggedThreshold = -1;
        return new InlineProgress<StepPreview>(p =>
        {
            double overall = p.TotalSteps > 0 ? Math.Clamp((double)p.Step / p.TotalSteps, 0.0, 1.0) : 0.0;
            int threshold = (int)(overall * 20); // 0..20 buckets of 5%
            if (threshold != lastLoggedThreshold)
            {
                lastLoggedThreshold = threshold;
                Logs.Verbose($"[HartsyInference] Progress batch='{batchId}' {RenderProgressBar(overall)} step {p.Step}/{p.TotalSteps}");
            }
            JObject previewObj = previewEncoder.TryEncode(p, batchId, overall);
            // When we have a preview encoded, send the richer JObject (preview + percent in one message —
            // matches Comfy's contract). Otherwise just the percent.
            takeOutput(previewObj ?? new JObject
            {
                ["batch_index"] = batchId,
                ["overall_percent"] = overall,
                ["current_percent"] = overall,
            });
        });
    }

    /// <summary>An <see cref="IProgress{T}"/> that invokes on the reporting thread. <see cref="Progress{T}"/>
    /// would post to a captured SynchronizationContext and reorder (or drop) ticks fired from the sampler thread.</summary>
    private sealed class InlineProgress<T>(Action<T> handler) : IProgress<T>
    {
        private readonly Action<T> _handler = handler;

        public void Report(T value) => _handler(value);
    }

    /// <summary>Renders an ASCII progress bar for log lines: <c>[████████░░░░░░░░░░░░] 40.0%</c>.
    /// 20 cells wide so each cell is exactly 5% — matching the 5%-threshold log throttling.</summary>
    private static string RenderProgressBar(double fraction, int width = 20)
    {
        fraction = Math.Clamp(fraction, 0.0, 1.0);
        int filled = (int)Math.Round(fraction * width);
        if (filled > width)
        {
            filled = width;
        }
        return $"[{new string('█', filled)}{new string('░', width - filled)}] {fraction * 100:F1}%";
    }

    // ─────────────────────────────── 4. Request mapping ───────────────────────────────
    //
    // Everything below is the ONLY place T2IParamTypes values are read. Each mapper produces one Engine-native
    // request record; anything the Engine's flat contract doesn't name rides in the Extra bag under a documented
    // key. Params we cannot express at all are refused in IsValidForThisBackend rather than silently dropped.

    /// <summary>Maps the Swarm request onto the Engine's <see cref="ImageRequest"/>.</summary>
    private ImageRequest BuildImageRequest(T2IParamInput input, ModelSupport.Family family)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        if (family.Id == "ideogram4" && input.Get(SwarmUIHartsyInference.Ideogram4MagicPromptParam, false))
        {
            // Ideogram 4 is trained on structured JSON captions; the expander rewrites a plain prompt through a
            // running LLM backend. Opt-in, and a no-op that returns the prompt unchanged when unavailable.
            prompt = Ideogram4MagicPrompt.Expand(
                prompt,
                input.Get(T2IParamTypes.Width, 1024),
                input.Get(T2IParamTypes.Height, 1024),
                input.Get(SwarmUIHartsyInference.Ideogram4MagicPromptModelParam, ""),
                input.SourceSession,
                msg => Logs.Verbose($"[HartsyInference][Ideogram4] {msg}"));
        }
        Image initImage = input.Get(T2IParamTypes.InitImage);
        Image maskImage = input.Get(T2IParamTypes.MaskImage);
        IReadOnlyList<ControlNetConditioning> controlNets = BuildControlNets(input, out List<(int Index, string UnionType)> unionTypes);
        return new ImageRequest
        {
            Prompt = prompt,
            NegativePrompt = input.Get(T2IParamTypes.NegativePrompt),
            Width = NullableInt(input, T2IParamTypes.Width),
            Height = NullableInt(input, T2IParamTypes.Height),
            Steps = NullableInt(input, T2IParamTypes.Steps),
            CfgScale = input.TryGet(T2IParamTypes.CFGScale, out double cfg) ? (float)cfg : null,
            CfgRescale = input.TryGet(SwarmUIHartsyInference.CfgRescaleParam, out double cfgRescale) ? (float)cfgRescale : null,
            Tcfg = input.TryGet(SwarmUIHartsyInference.TcfgParam, out bool tcfg) ? tcfg : null,
            // Reads SwarmUI core's shared SeamlessTileable param directly, unlike the Tier 2 CFG cluster above —
            // it already carries its own "seamless" FeatureFlag (not "comfyui"), so there's no AND-semantics
            // problem to work around by duplicating it (see SwarmUIHartsyInference.TcfgParam's doc comment for
            // when duplication IS required).
            SeamlessTiling = input.TryGet(T2IParamTypes.SeamlessTileable, out string seamless) ? seamless : null,
            Seed = input.Get(T2IParamTypes.Seed, -1L),
            ClipSkip = NullableInt(input, T2IParamTypes.ClipStopAtLayer),
            Sampler = input.Get(SwarmUIHartsyInference.SamplerParam, null),
            Scheduler = null, // the Engine resolves the family's canonical schedule; Comfy's Scheduler param has no analogue
            SigmaShift = input.TryGet(T2IParamTypes.SigmaShift, out double shift) ? shift : null,
            EndStepsEarly = input.TryGet(T2IParamTypes.EndStepsEarly, out double endEarly) ? endEarly : null,
            InstructPix2PixCfg = input.TryGet(T2IParamTypes.IP2PCFG2, out double ip2p) ? ip2p : null,
            Batch = 1, // Swarm drives batching itself: one Generate call per image
            Components = BuildComponents(input),
            Loras = BuildLoras(input),
            ControlNets = controlNets,
            IpAdapter = BuildIpAdapter(input),
            Refiner = BuildRefiner(input),
            Img2Img = initImage is null ? null : new Img2Img
            {
                InitImage = ToEngineImage(initImage),
                Creativity = input.Get(T2IParamTypes.InitImageCreativity, 0.6),
                Mode = input.Get(SwarmUIHartsyInference.InitImageModeParam, "auto") switch
                {
                    "denoise" => Img2ImgMode.Denoise,
                    "reference" => Img2ImgMode.Reference,
                    _ => Img2ImgMode.Auto,
                },
            },
            Inpaint = maskImage is null ? null : new Inpaint
            {
                Mask = ToEngineImage(maskImage),
                Grow = input.Get(T2IParamTypes.MaskGrow, 0),
                Blur = input.Get(T2IParamTypes.MaskBlur, 0),
                ShrinkGrow = input.Get(T2IParamTypes.MaskShrinkGrow, 0),
            },
            Regional = BuildRegional(input, prompt),
            VariationSeed = BuildVariationSeed(input),
            Extra = BuildImageExtra(input, family, initImage, unionTypes),
        };
    }

    /// <summary>Per-request swappable-component overrides (VAE + the text encoders the user picked).</summary>
    private static ComponentOverrides BuildComponents(T2IParamInput input)
    {
        ComponentOverrides overrides = new()
        {
            Vae = ModelPath(input.Get(T2IParamTypes.VAE)),
            T5Xxl = ModelPath(input.Get(T2IParamTypes.T5XXLModel)),
            ClipL = ModelPath(input.Get(T2IParamTypes.ClipLModel)),
            ClipG = ModelPath(input.Get(T2IParamTypes.ClipGModel)),
            ClipVision = ModelPath(input.Get(T2IParamTypes.ClipVisionModel)),
            Qwen = ModelPath(input.Get(T2IParamTypes.QwenModel)),
            Llama = ModelPath(input.Get(T2IParamTypes.LLaMAModel)),
            Gemma = ModelPath(input.Get(T2IParamTypes.GemmaModel)),
        };
        bool any = overrides.Vae is not null || overrides.T5Xxl is not null || overrides.ClipL is not null
            || overrides.ClipG is not null || overrides.ClipVision is not null || overrides.Qwen is not null
            || overrides.Llama is not null || overrides.Gemma is not null;
        return any ? overrides : null;
    }

    /// <summary>Maps the Loras / LoraWeights / LoraTencWeights / LoraSectionConfinement parallel lists onto the
    /// Engine's <see cref="LoraStack"/>. Names are resolved to on-disk paths through Swarm's LoRA model set.</summary>
    private static LoraStack BuildLoras(T2IParamInput input)
    {
        if (!input.TryGet(T2IParamTypes.Loras, out List<string> names) || names is null || names.Count == 0)
        {
            return null;
        }
        input.TryGet(T2IParamTypes.LoraWeights, out List<string> weights);
        input.TryGet(T2IParamTypes.LoraTencWeights, out List<string> tencWeights);
        input.TryGet(T2IParamTypes.LoraSectionConfinement, out List<string> confinements);
        Program.T2IModelSets.TryGetValue("LoRA", out T2IModelHandler loraHandler);
        List<LoraEntry> entries = new(names.Count);
        for (int i = 0; i < names.Count; i++)
        {
            string name = names[i];
            if (string.IsNullOrWhiteSpace(name))
            {
                continue;
            }
            T2IModel loraModel = loraHandler?.GetModel(name);
            string path = loraModel?.RawFilePath;
            if (path is null)
            {
                throw new SwarmUserErrorException($"HartsyInference: LoRA '{name}' was not found in the LoRA model folder.");
            }
            entries.Add(new LoraEntry
            {
                Model = path,
                Weight = ParseAt(weights, i, 1.0),
                TextEncoderWeight = tencWeights is not null && i < tencWeights.Count ? ParseAt(tencWeights, i, 1.0) : null,
                SectionConfinement = confinements is not null && i < confinements.Count && !string.IsNullOrWhiteSpace(confinements[i])
                    ? confinements[i]
                    : null,
            });
        }
        return entries.Count == 0 ? null : new LoraStack { Entries = entries };
    }

    /// <summary>Maps Swarm's three ControlNet slots onto the Engine's conditioning list, running the matching
    /// annotator (Canny/Depth/OpenPose/…) over each hint image first. The Engine's contract expects an
    /// already-preprocessed hint — annotation is deliberately host-side, since the annotators consume SwarmUI
    /// images and fetch their own weights through Swarm.</summary>
    private IReadOnlyList<ControlNetConditioning> BuildControlNets(T2IParamInput input, out List<(int Index, string UnionType)> unionTypes)
    {
        unionTypes = [];
        T2IParamTypes.ControlNetParamHolder[] holders = T2IParamTypes.Controlnets;
        if (holders is null)
        {
            return null;
        }
        List<ControlNetConditioning> layers = [];
        foreach (T2IParamTypes.ControlNetParamHolder holder in holders)
        {
            if (holder?.Model is null)
            {
                continue;
            }
            T2IModel cnModel = input.Get(holder.Model);
            if (cnModel is null)
            {
                continue;
            }
            Image hint = input.Get(holder.Image) ?? input.Get(T2IParamTypes.InitImage);
            if (hint is null)
            {
                throw new SwarmUserErrorException(
                    $"HartsyInference: ControlNet '{cnModel.Name}' has no control image. Set the ControlNet Image (or an Init Image).");
            }
            HartsyInference.Diffusion.Adapters.ControlNetMode mode = ControlNetPreprocessing.DetectMode(cnModel.RawFilePath);
            AddLoadStatus($"ControlNet '{cnModel.Name}' → {mode} preprocessing.");
            EngineImage annotated = ControlNetPreprocessing.Preprocess(
                mode, hint,
                input.Get(T2IParamTypes.Width, 1024), input.Get(T2IParamTypes.Height, 1024),
                PreprocessBackend, msg => AddLoadStatus(msg));
            // Keyed by the EMITTED slot index (the engine indexes request.ControlNets positionally); a skipped
            // holder must not shift the pairing. Only union checkpoints consult it engine-side.
            unionTypes.Add((layers.Count, UnionTypeString(mode)));
            layers.Add(new ControlNetConditioning
            {
                Model = cnModel.RawFilePath,
                Image = annotated,
                Strength = input.Get(holder.Strength, 1.0),
                Start = input.Get(holder.Start, 0.0),
                End = input.Get(holder.End, 1.0),
            });
        }
        return layers.Count == 0 ? null : layers;
    }

    /// <summary>The engine's union-control-type token for a detected preprocessing mode. Every output must be a
    /// string <c>ControlNetResolver.ResolveUnionType</c> recognizes — an unknown token throws engine-side.</summary>
    private static string UnionTypeString(HartsyInference.Diffusion.Adapters.ControlNetMode mode) => mode switch
    {
        HartsyInference.Diffusion.Adapters.ControlNetMode.Depth => "depth",
        HartsyInference.Diffusion.Adapters.ControlNetMode.OpenPose => "openpose",
        HartsyInference.Diffusion.Adapters.ControlNetMode.Scribble => "softedge",
        HartsyInference.Diffusion.Adapters.ControlNetMode.SoftEdge => "softedge",
        HartsyInference.Diffusion.Adapters.ControlNetMode.Normal => "normal",
        HartsyInference.Diffusion.Adapters.ControlNetMode.Segmentation => "segment",
        HartsyInference.Diffusion.Adapters.ControlNetMode.Tile => "tile",
        HartsyInference.Diffusion.Adapters.ControlNetMode.Inpaint => "repaint",
        // Canny + LineArt both drive the thin-line head.
        _ => "canny",
    };

    /// <summary>Maps the image-prompt inputs onto the Engine's IP-Adapter conditioning. The adapter model itself
    /// rides the Extra bag under the Engine's documented <c>ipadapter.model</c> key.</summary>
    private static IpAdapter BuildIpAdapter(T2IParamInput input)
    {
        if (!input.TryGet(T2IParamTypes.PromptImages, out List<Image> promptImages) || promptImages is null || promptImages.Count == 0)
        {
            return null;
        }
        return new IpAdapter
        {
            PromptImages = [.. promptImages.Select(ToEngineImage)],
            Grouping = input.Get(T2IParamTypes.SmartImagePromptResizing, false),
            FaceIdV2Weight = input.TryGet(SwarmUIHartsyInference.FaceIdV2WeightParam, out double faceId) ? faceId : null,
        };
    }

    /// <summary>Maps the refiner group onto the Engine's second-pass config.</summary>
    private static Refiner BuildRefiner(T2IParamInput input)
    {
        T2IModel refinerModel = input.Get(T2IParamTypes.RefinerModel);
        if (refinerModel is null)
        {
            return null;
        }
        return new Refiner
        {
            Model = refinerModel.RawFilePath,
            Vae = ModelPath(input.Get(T2IParamTypes.RefinerVAE)),
            Method = input.Get(T2IParamTypes.RefinerMethod, null),
            Control = input.TryGet(T2IParamTypes.RefinerControl, out double control) ? control : null,
            Steps = NullableInt(input, T2IParamTypes.RefinerSteps),
            CfgScale = input.TryGet(T2IParamTypes.RefinerCFGScale, out double refinerCfg) ? refinerCfg : null,
            Upscale = input.TryGet(T2IParamTypes.RefinerUpscale, out double upscale) && upscale > 1e-6 ? upscale : null,
        };
    }

    /// <summary>Regional / segment prompting: the plan is the raw prompt (the Engine parses the
    /// <c>&lt;region&gt;</c>/<c>&lt;segment&gt;</c> syntax), plus the mask-shaping and per-segment overrides.</summary>
    private static Regional BuildRegional(T2IParamInput input, string prompt)
    {
        if (!HasRegionalSyntax(prompt) && !HasRegionalSyntax(input.Get(T2IParamTypes.NegativePrompt)))
        {
            return null;
        }
        return new Regional
        {
            Plan = prompt,
            SortOrder = input.Get(T2IParamTypes.SegmentSortOrder, null),
            MaskGrow = input.Get(T2IParamTypes.SegmentMaskGrow, 0),
            MaskBlur = input.Get(T2IParamTypes.SegmentMaskBlur, 0),
            MaskOversize = input.Get(T2IParamTypes.SegmentMaskOversize, 0),
            Steps = NullableInt(input, T2IParamTypes.SegmentSteps),
            CfgScale = input.TryGet(T2IParamTypes.SegmentCFGScale, out double segCfg) ? segCfg : null,
        };
    }

    /// <summary>True when the text carries Swarm's regional / segment prompt syntax.</summary>
    private static bool HasRegionalSyntax(string text) =>
        !string.IsNullOrEmpty(text)
        && (text.Contains("<region:", StringComparison.OrdinalIgnoreCase)
            || text.Contains("<segment:", StringComparison.OrdinalIgnoreCase));

    /// <summary>Variation-seed blending; null unless a non-zero strength AND a seed were both supplied.</summary>
    private static VariationSeed BuildVariationSeed(T2IParamInput input)
    {
        if (!input.TryGet(T2IParamTypes.VariationSeedStrength, out double strength) || strength <= 0)
        {
            return null;
        }
        if (!input.TryGet(T2IParamTypes.VariationSeed, out long varSeed))
        {
            return null;
        }
        return new VariationSeed { Seed = varSeed, Strength = strength };
    }

    /// <summary>Everything the Engine's flat image contract doesn't name: its own documented Extra keys plus this
    /// extension's registered custom params.</summary>
    private IReadOnlyDictionary<string, object> BuildImageExtra(T2IParamInput input, ModelSupport.Family family, Image initImage,
        List<(int Index, string UnionType)> controlNetUnionTypes)
    {
        Dictionary<string, object> extra = new(StringComparer.Ordinal);
        // Engine-documented keys (Features/HartsyInference.Engine.Features.RequestExtras.cs).
        if (T2IParamTypes.TryGetType("useipadapter", out T2IParamType ipaType, input)
            && input.TryGetRaw(ipaType, out object ipaRaw)
            && ipaRaw is string ipaModel && !string.IsNullOrWhiteSpace(ipaModel) && ipaModel != "None")
        {
            extra[HartsyInference.Engine.Features.RequestExtras.IpAdapterModel] = ipaModel;
        }
        if (input.Get(T2IParamTypes.ClipVisionModel) is T2IModel clipVision)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.IpAdapterClipVision] = clipVision.RawFilePath;
        }
        // IP-Adapter scheduling/weighting: forward Comfy's own params (we don't re-register them).
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterWeight, out double ipaWeight))
        {
            extra[HartsyInference.Engine.Features.RequestExtras.IpAdapterWeight] = ipaWeight;
        }
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterStart, out double ipaStart))
        {
            extra[HartsyInference.Engine.Features.RequestExtras.IpAdapterStart] = ipaStart;
        }
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterEnd, out double ipaEnd))
        {
            extra[HartsyInference.Engine.Features.RequestExtras.IpAdapterEnd] = ipaEnd;
        }
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterWeightType, out string ipaWeightType)
            && !string.IsNullOrWhiteSpace(ipaWeightType))
        {
            // Comfy live-appends "name///name (New)" variants; the engine wants the bare name and safely
            // falls back to "standard" for anything it doesn't recognize.
            int marker = ipaWeightType.IndexOf("///", StringComparison.Ordinal);
            extra[HartsyInference.Engine.Features.RequestExtras.IpAdapterWeightType] = marker < 0 ? ipaWeightType : ipaWeightType[..marker];
        }
        foreach ((int index, string unionType) in controlNetUnionTypes)
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ControlNetUnionTypePrefix + index.ToString(CultureInfo.InvariantCulture)] = unionType;
        }
        // FLUX.1 Redux: Comfy's style-model params map onto the engine's redux.* keys. The model rides by name —
        // the engine's ModelFileLocator resolves it against the style_models folders, same as ipadapter.model.
        if (input.TryGet(ComfyUIBackendExtension.UseStyleModel, out string styleModel)
            && !string.IsNullOrWhiteSpace(styleModel) && styleModel != "None")
        {
            extra[HartsyInference.Engine.Features.RequestExtras.ReduxStyleModel] = styleModel;
            if (input.TryGet(ComfyUIBackendExtension.StyleModelMultiplyStrength, out double reduxMultiply))
            {
                extra[HartsyInference.Engine.Features.RequestExtras.ReduxMultiply] = reduxMultiply;
            }
            if (input.TryGet(ComfyUIBackendExtension.StyleModelMergeStrength, out double reduxMerge))
            {
                extra[HartsyInference.Engine.Features.RequestExtras.ReduxMerge] = reduxMerge;
            }
            if (input.TryGet(ComfyUIBackendExtension.StyleModelApplyStart, out double reduxApplyStart))
            {
                extra[HartsyInference.Engine.Features.RequestExtras.ReduxApplyStart] = reduxApplyStart;
            }
        }
        // Extension-registered custom params.
        if (input.TryGet(SwarmUIHartsyInference.Ideogram4MagicPromptParam, out bool magic))
        {
            extra["hartsy.ideogram4_magic_prompt"] = magic;
        }
        if (input.TryGet(SwarmUIHartsyInference.Ideogram4MagicPromptModelParam, out string magicModel) && !string.IsNullOrWhiteSpace(magicModel))
        {
            extra["hartsy.ideogram4_magic_prompt_model"] = magicModel;
        }
        // FLUX.1 Canny/Depth: these are ordinary Flux.1 checkpoints with a wider x_embedder baked in — no separate
        // ControlNet adapter to select, so there is no ControlNet-slot model to key preprocessing off. The engine
        // only learns the checkpoint is Canny/Depth/Fill after loading its weights (Flux1RecipePipeline has no
        // earlier hook), so detection here is a filename heuristic on the MAIN model, mirroring
        // ControlNetPreprocessing.DetectMode's convention for real ControlNet checkpoints. A wrong guess degrades
        // to FluxPipeline's own clear "requires/does not accept a control image" error, not silent corruption —
        // Fill deliberately isn't matched here since it conditions via Img2Img + Inpaint, not a control image.
        if (family.Id == "flux1" && initImage is not null)
        {
            string modelPath = input.Get(T2IParamTypes.Model)?.RawFilePath ?? "";
            string modelName = System.IO.Path.GetFileNameWithoutExtension(modelPath).ToLowerInvariant();
            HartsyInference.Diffusion.Adapters.ControlNetMode? toolsMode = modelName.Contains("canny") ? HartsyInference.Diffusion.Adapters.ControlNetMode.Canny
                : modelName.Contains("depth") ? HartsyInference.Diffusion.Adapters.ControlNetMode.Depth
                : null;
            if (toolsMode is not null)
            {
                AddLoadStatus($"FLUX.1 Tools checkpoint detected ('{modelName}') → {toolsMode} preprocessing.");
                EngineImage annotated = ControlNetPreprocessing.Preprocess(
                    toolsMode.Value, initImage,
                    input.Get(T2IParamTypes.Width, 1024), input.Get(T2IParamTypes.Height, 1024),
                    PreprocessBackend, msg => AddLoadStatus(msg));
                extra[HartsyInference.Engine.Features.RequestExtras.FluxToolsControlImage] = annotated;
            }
        }
        return extra;
    }

    /// <summary>Maps the Swarm request onto the Engine's <see cref="VideoRequest"/>.</summary>
    private static VideoRequest BuildVideoRequest(T2IParamInput input)
    {
        Image initImage = input.Get(T2IParamTypes.InitImage);
        Image endFrame = input.Get(T2IParamTypes.VideoEndImage);
        int? frames = ResolveFrames(input);
        RefuseIncompatibleH3Conditioning(input, initImage, endFrame);
        Dictionary<string, object> extra = new(StringComparer.Ordinal);
        Image reference = input.Get(SwarmUIHartsyInference.AnimateReferenceImageParam);
        if (reference is not null)
        {
            extra[HartsyInference.Engine.Recipes.Video.WanAnimateRecipePipeline.ReferenceImageKey] = ToEngineImage(reference);
        }
        Image poseVideo = input.Get(SwarmUIHartsyInference.AnimatePoseVideoParam);
        Image faceVideo = input.Get(SwarmUIHartsyInference.AnimateFaceVideoParam);
        // Animate intent + a video-typed Init media ⇒ the whole clip drives motion (VideoClip, all frames);
        // a still keeps the engine's tile-across-frames fallback. Without animate intent the init media stays a
        // plain start frame — DrivingVideo on a non-animate family would (correctly) trip the feature gate.
        bool animateIntent = reference is not null || poseVideo is not null || faceVideo is not null;
        bool initIsVideo = initImage?.Type?.MetaType == MediaMetaType.Video;
        bool initDrives = animateIntent && initIsVideo;
        return new VideoRequest
        {
            Prompt = input.Get(T2IParamTypes.Prompt) ?? "",
            NegativePrompt = input.Get(T2IParamTypes.NegativePrompt),
            Width = input.Get(T2IParamTypes.Width, 704),
            Height = input.Get(T2IParamTypes.Height, 480),
            Steps = NullableInt(input, T2IParamTypes.VideoSteps) ?? NullableInt(input, T2IParamTypes.Steps),
            CfgScale = input.TryGet(T2IParamTypes.VideoCFG, out double videoCfg) ? (float)videoCfg
                : input.TryGet(T2IParamTypes.CFGScale, out double baseCfg) ? (float)baseCfg : null,
            Seed = input.Get(T2IParamTypes.Seed, -1L),
            InitImage = initImage is null || initDrives ? null : ToEngineImage(initImage),
            DrivingVideo = initDrives ? ToVideoClip(initImage) : null,
            DrivingPoseVideo = poseVideo is null ? null : ToVideoClip(poseVideo),
            DrivingFaceVideo = faceVideo is null ? null : ToVideoClip(faceVideo),
            DrivingAutoPreprocess = input.Get(SwarmUIHartsyInference.AnimateAutoPreprocessParam, true),
            VideoModel = ModelPath(input.Get(T2IParamTypes.VideoModel)),
            VideoSwapModel = ModelPath(input.Get(T2IParamTypes.VideoSwapModel)),
            // Untouched slider = auto: Swarm sends group params whenever the group is open, so the 0.5 default must
            // not override Wan 2.2's official boundary. The engine warps an explicit fraction via the flow shift.
            VideoSwapPercent = input.TryGet(T2IParamTypes.VideoSwapPercent, out double swapPercent) && swapPercent != 0.5
                ? swapPercent : null,
            VideoResolution = input.Get(T2IParamTypes.VideoResolution, null),
            // 25 is Swarm's cross-family fallback and is wrong for LTX-2: the recipe's own default, core's param
            // default and Swarm's docs ("LTXV prefers 24") all say 24. Sent as a value rather than null because the
            // LTX-2 pipeline doesn't report the fps it used on VideoGenerationResult — a null would generate at the
            // recipe's 24 and then mux at the encoder's 25 fallback, i.e. play back 4% fast.
            Fps = input.Get(T2IParamTypes.VideoFPS, IsLtxVideo2(input) ? 24 : 25),
            VideoFormat = input.Get(T2IParamTypes.VideoFormat, null),
            // Boomerang is applied to the decoded frames on the way out (see GenerateVideo) so the Engine
            // doesn't waste a second pass; the flag is still carried for pipelines that want it.
            VideoBoomerang = input.Get(T2IParamTypes.VideoBoomerang, false),
            VideoEndFrame = endFrame is null ? null : ToEngineImage(endFrame),
            VideoAudioInput = ToAudioClip(input.Get(T2IParamTypes.VideoAudioInput)),
            VideoAudioReference = ToAudioClip(input.Get(SwarmUIHartsyInference.VideoAudioReferenceParam)),
            ReferenceImages = BuildReferenceImages(input),
            ReferenceVideos = BuildReferenceVideos(input),
            ReferenceAudios = BuildReferenceAudios(input),
            Frames = frames,
            TrimVideoStartFrames = input.Get(T2IParamTypes.TrimVideoStartFrames, 0),
            TrimVideoEndFrames = input.Get(T2IParamTypes.TrimVideoEndFrames, 0),
            Components = BuildComponents(input),
            Loras = BuildLoras(input),
            Extra = extra,
        };
    }

    /// <summary>Refuses start/end-frame conditioning combined with references on MiniMax-H3, whichever checkpoint is
    /// loaded. fl2va and ref2va are separate tasks: the packed layout restarts its position cursor for reference
    /// blocks, so keyframe and reference rows would occupy overlapping coordinates. The engine throws on this too —
    /// this catches it at request construction, where the message can name the params the user actually set.</summary>
    private static void RefuseIncompatibleH3Conditioning(T2IParamInput input, Image initImage, Image endFrame)
    {
        if (!IsMiniMaxH3(input))
        {
            return;
        }
        bool hasFrames = initImage is not null || endFrame is not null;
        bool hasRefs = (input.TryGet(T2IParamTypes.PromptImages, out List<Image> pi) && pi is { Count: > 0 })
            || (input.TryGet(T2IParamTypes.PromptAudios, out List<AudioFile> pa) && pa is { Count: > 0 })
            || (input.TryGet(T2IParamTypes.PromptVideos, out List<VideoFile> pv) && pv is { Count: > 0 });
        if (hasFrames && hasRefs)
        {
            throw new SwarmUserErrorException(
                "MiniMax-H3 cannot combine start/end-frame conditioning with reference media — fl2va and ref2va are "
                + "separate tasks shipped as separate checkpoints, and their packed-layout coordinates overlap. "
                + $"Remove either the {(initImage is not null ? "Init Image" : "Video End Frame")} or the media "
                + "attached to the prompt.");
        }
    }

    /// <summary>The requested frame count, or null to let the family's own default stand.</summary>
    /// <remarks>SwarmUI's core frame params default to 25 for every video family. For MiniMax-H3 that is not a
    /// neutral value — 25 snaps up onto the 17k+5 grid to 39 frames (1.6 s), so a user who never touched the slider
    /// silently got a sixth of the model's native 124-frame clip and no indication why. Returning null when neither
    /// param was actually set hands the decision to the recipe, which uses 124 — the same "unset falls back to 124"
    /// the reference workflow does. Scoped to H3 deliberately: the other families' defaults are wrong in their own
    /// ways, but fixing those is not this change.</remarks>
    private static int? ResolveFrames(T2IParamInput input)
    {
        if (input.TryGet(T2IParamTypes.Text2VideoFrames, out int t2vFrames) && t2vFrames > 0)
        {
            return t2vFrames;
        }
        if (input.TryGet(T2IParamTypes.VideoFrames, out int videoFrames) && videoFrames > 0)
        {
            return videoFrames;
        }
        if (IsMiniMaxH3(input))
        {
            Logs.Verbose("[HartsyInference] No frame count set — leaving it to MiniMax-H3's own 124-frame default "
                + "rather than Swarm's cross-family 25 (which would snap to 39 on H3's 17k+5 grid).");
            return null;
        }
        if (IsLtxVideo2(input))
        {
            Logs.Verbose("[HartsyInference] No frame count set — leaving it to the LTX-2 recipe's own default rather "
                + "than Swarm's cross-family 25, which is not on LTX-2's 8n+1 grid.");
            return null;
        }
        return 25;
    }

    /// <summary>Whether the selected model is MiniMax-H3, by core's own compat class rather than a name guess.</summary>
    private static bool IsMiniMaxH3(T2IParamInput input) =>
        input.Get(T2IParamTypes.Model)?.ModelClass?.CompatClass?.ID == T2IModelClassSorter.CompatMiniMaxH3.ID;

    /// <summary>Whether the selected model is any LTX-2.x (2, 2.3 or 2.5 — they share core's compat class).</summary>
    private static bool IsLtxVideo2(T2IParamInput input) =>
        input.Get(T2IParamTypes.Model)?.ModelClass?.CompatClass?.ID == T2IModelClassSorter.CompatLtxv2.ID;

    /// <summary>Reference media caps the model was trained under, mirroring what the reference node accepts.</summary>
    private const int MaxRefImages = 9, MaxRefAudios = 3, MaxRefVideos = 3;

    /// <summary>Reference images come from the prompt box, not a param control: core's
    /// <see cref="T2IParamTypes.PromptImages"/> is the internal carrier the textbox fills when media is dragged or
    /// pasted onto it, and the model's own text encoder resolves the <c>&lt;Picture N&gt;</c> tags the user types
    /// inline. Swarm never parses those tags — it just supplies the ordered list, which is exactly what the
    /// MiniMax-H3 reference node consumes, index for index.</summary>
    private static IReadOnlyList<EngineImage> BuildReferenceImages(T2IParamInput input)
    {
        if (!input.TryGet(T2IParamTypes.PromptImages, out List<Image> images) || images is not { Count: > 0 })
        {
            return null;
        }
        List<EngineImage> kept = [.. images.Where(i => i is not null).Take(MaxRefImages).Select(ToEngineImage)];
        WarnIfTruncated(images.Count, kept.Count, "images");
        return kept.Count == 0 ? null : kept;
    }

    /// <summary>Standalone reference audio, from the prompt box — <c>&lt;Audio N&gt;</c> in the prompt.</summary>
    private static IReadOnlyList<AudioClip> BuildReferenceAudios(T2IParamInput input)
    {
        if (!input.TryGet(T2IParamTypes.PromptAudios, out List<AudioFile> audios) || audios is not { Count: > 0 })
        {
            return null;
        }
        List<AudioClip> kept = [.. audios.Where(a => a is not null).Take(MaxRefAudios)
            .Select(ToAudioClip).Where(c => c is not null).Cast<AudioClip>()];
        WarnIfTruncated(audios.Count, kept.Count, "audio clips");
        return kept.Count == 0 ? null : kept;
    }

    /// <summary>Reference videos from the prompt box. Deliberately UNPAIRED with the reference audio: the reference
    /// node sends two independent positional lists and lets the model correlate them through the inline
    /// <c>&lt;Video N&gt;</c>/<c>&lt;Audio N&gt;</c> tags. The engine's <see cref="ReferenceVideo.Audio"/> can carry
    /// an explicit per-clip soundtrack and still does — it is simply not something this UI expresses.</summary>
    private static IReadOnlyList<ReferenceVideo> BuildReferenceVideos(T2IParamInput input)
    {
        if (!input.TryGet(T2IParamTypes.PromptVideos, out List<VideoFile> videos) || videos is not { Count: > 0 })
        {
            return null;
        }
        List<ReferenceVideo> kept = [.. videos.Where(v => v is not null).Take(MaxRefVideos)
            .Select(v => new ReferenceVideo { Video = ToVideoClip(v), Audio = null })];
        WarnIfTruncated(videos.Count, kept.Count, "videos");
        return kept.Count == 0 ? null : kept;
    }

    /// <summary>Says so when media past the model's cap is dropped — silently ignoring the 10th image would look
    /// like the model ignored it.</summary>
    private static void WarnIfTruncated(int supplied, int used, string what)
    {
        if (supplied > used)
        {
            Logs.Warning($"[HartsyInference] {supplied} reference {what} attached but this model takes at most "
                + $"{used} — the extras were dropped.");
        }
    }

    /// <summary>Maps Swarm's Text2Audio group plus this extension's music params onto the Engine's
    /// <see cref="MusicRequest"/>. One source-audio param + one edit-mode dropdown populate exactly one of
    /// Continuation/Repaint/Cover, so the engine's "more than one edit mode" throw is structurally unreachable.</summary>
    private static MusicRequest BuildMusicRequest(T2IParamInput input)
    {
        long seed = input.Get(T2IParamTypes.Seed, -1L);
        AudioClip sourceAudio = ToAudioClip(input.Get(SwarmUIHartsyInference.MusicSourceAudioParam));
        string editMode = input.Get(SwarmUIHartsyInference.MusicEditModeParam, "continuation");
        string lmModel = input.Get(SwarmUIHartsyInference.MusicLmModelParam, "none");
        MusicRequest request = new MusicRequest
        {
            // ACE-Step convention (which the Engine follows): the prompt carries the lyrics, the style/genre tags
            // come from Text2AudioStyle. An empty prompt means instrumental.
            Prompt = input.Get(T2IParamTypes.Prompt) ?? "",
            Genre = input.Get(T2IParamTypes.Text2AudioStyle, "") ?? "",
            Duration = input.Get(T2IParamTypes.Text2AudioDuration, 10d),
            Seed = seed < 0 ? Random.Shared.Next() : (int)(seed & 0x7FFFFFFF),
            InferSteps = NullableInt(input, T2IParamTypes.Steps),
            CfgScale = input.TryGet(T2IParamTypes.CFGScale, out double cfg) ? cfg : null,
            Shift = input.TryGet(T2IParamTypes.SigmaShift, out double shift) ? shift : null,
            Bpm = input.TryGet(T2IParamTypes.Text2AudioBPM, out long bpm) && bpm > 0 ? (int)bpm : null,
            KeyScale = input.Get(T2IParamTypes.Text2AudioKeyScale, "") ?? "",
            TimeSignature = input.Get(T2IParamTypes.Text2AudioTimeSignature, "") ?? "",
            VocalLanguage = input.Get(T2IParamTypes.Text2AudioLanguage, "") ?? "",
            Continuation = editMode == "continuation" ? sourceAudio : null,
            Repaint = editMode == "repaint" ? sourceAudio : null,
            Cover = editMode == "cover" ? sourceAudio : null,
        };
        if (input.TryGet(SwarmUIHartsyInference.MusicRepaintStartParam, out double repaintStart))
        {
            request = request with { RepaintStart = repaintStart };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicRepaintEndParam, out double repaintEnd))
        {
            request = request with { RepaintEnd = repaintEnd };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicCoverStrengthParam, out double coverStrength))
        {
            request = request with { CoverStrength = coverStrength };
        }
        if (lmModel != "none")
        {
            request = request with
            {
                LmModel = lmModel,
                Thinking = input.Get(SwarmUIHartsyInference.MusicLmThinkingParam, true),
                LmTemperature = input.Get(SwarmUIHartsyInference.MusicLmTemperatureParam, 0.85),
                LmCfgScale = input.Get(SwarmUIHartsyInference.MusicLmCfgParam, 2.0),
                LmTopK = input.Get(SwarmUIHartsyInference.MusicLmTopKParam, 0),
                LmTopP = input.Get(SwarmUIHartsyInference.MusicLmTopPParam, 0.9),
                LmNegativePrompt = input.Get(SwarmUIHartsyInference.MusicLmNegativePromptParam, "") ?? "",
            };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicInferMethodParam, out string inferMethod))
        {
            request = request with { InferMethod = inferMethod };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicUseAdgParam, out bool useAdg))
        {
            request = request with { UseAdg = useAdg };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicCfgIntervalStartParam, out double cfgStart))
        {
            request = request with { CfgIntervalStart = cfgStart };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicCfgIntervalEndParam, out double cfgEnd))
        {
            request = request with { CfgIntervalEnd = cfgEnd };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicTemperatureParam, out double temperature))
        {
            request = request with { Temperature = temperature };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicTopKParam, out int topK))
        {
            request = request with { TopK = topK };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicTopPParam, out double topP))
        {
            request = request with { TopP = topP };
        }
        if (input.TryGet(SwarmUIHartsyInference.MusicRepetitionPenaltyParam, out double repetitionPenalty))
        {
            request = request with { RepetitionPenalty = repetitionPenalty };
        }
        return request;
    }

    // ─── mapping helpers ───

    /// <summary>An int param's value, or null when the request didn't set it (so the Engine's family default wins).</summary>
    private static int? NullableInt(T2IParamInput input, T2IRegisteredParam<int> param) =>
        input.TryGet(param, out int value) && value > 0 ? value : null;

    /// <summary>Parses the i-th entry of a Swarm parallel string list as a double.</summary>
    private static double ParseAt(List<string> list, int index, double fallback)
    {
        if (list is null || index >= list.Count || !double.TryParse(list[index], System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out double value))
        {
            return fallback;
        }
        return value;
    }

    /// <summary>The annotator device: the Engine's own backend when available (same device, shared caches, no second
    /// CUDA context), else a lazily created standalone device that we own.</summary>
    private IBackend PreprocessBackend()
    {
        IBackend existing = _preprocessBackend;
        if (existing is not null)
        {
            return existing;
        }
        lock (_preprocessBackendLock)
        {
            if (_preprocessBackend is null)
            {
                InferenceEngine engine = _engine;
                if (engine is not null)
                {
                    _preprocessBackend = engine.ComputeBackend;
                    _ownsPreprocessBackend = false;
                }
                else
                {
                    string selector = Settings?.ComputeBackend?.ToLowerInvariant() ?? "auto";
                    int? ordinal = ParseGpuId(Settings?.GPU_ID);
                    AddLoadStatus($"Creating ControlNet annotator device (compute='{selector}', device={ordinal?.ToString() ?? "auto"})...");
                    // Create takes only a selector; the ordinal rides as a 'cuda:1' suffix (0 composes back to the bare kind).
                    _preprocessBackend = BackendFactory.Create(BackendFactory.WithOrdinal(selector, ordinal ?? 0));
                    _ownsPreprocessBackend = true;
                }
            }
            return _preprocessBackend;
        }
    }

    /// <summary>Drops the annotator device reference; disposes it only when it was standalone (a borrowed Engine
    /// backend must never be disposed here — that would evict the Engine's resident weights mid-lifecycle).</summary>
    private void DisposePreprocessBackend()
    {
        lock (_preprocessBackendLock)
        {
            if (_ownsPreprocessBackend)
            {
                _preprocessBackend?.Dispose();
            }
            _preprocessBackend = null;
            _ownsPreprocessBackend = false;
        }
    }

    /// <summary>A picked model's on-disk path, or null when nothing was picked.</summary>
    private static string ModelPath(T2IModel model) => model?.RawFilePath;

    /// <summary>Decodes a SwarmUI image into the Engine's raw RGB24 payload.</summary>
    private static EngineImage ToEngineImage(Image image)
    {
        (byte[] rgb, int width, int height) = RgbToImage.ToHwcRgb(image);
        return new EngineImage { Rgb = rgb, Width = width, Height = height };
    }

    /// <summary>Wraps a SwarmUI video upload as the Engine's encoded-video payload; the Engine decodes it via ffmpeg.</summary>
    private static VideoClip ToVideoClip(Image video) =>
        video is null ? null : new VideoClip { Data = video.RawData, Format = video.Type?.Extension };

    /// <summary>Same, for the core prompt-box video carrier (<see cref="T2IParamTypes.PromptVideos"/>).</summary>
    private static VideoClip ToVideoClip(VideoFile video) =>
        video is null ? null : new VideoClip { Data = video.RawData, Format = video.Type?.Extension };

    /// <summary>Turns an out-of-VRAM failure into something a user can act on. The Engine's own message is already
    /// specific — the pre-flight names the geometry, the byte requirement and the longest clip that would fit; a
    /// mid-forward failure names the allocation that could not be served — so it is quoted verbatim rather than
    /// re-derived, and only the advice is added here. Without this the whole thing surfaced as T2IEngine's generic
    /// "Something went wrong".</summary>
    private static string DescribeVramFailure(OutOfVramException ex, ModelSupport.Family family)
    {
        string sizes = ex.RequestedBytes > 0
            ? $" (needed {ex.RequestedBytes / (1024 * 1024)} MB, {ex.AvailableBytes / (1024 * 1024)} MB free)"
            : "";
        string lever = family?.Kind == ModelSupport.Kind.Video
            ? "Lower the frame count or the resolution"
            : "Lower the resolution or batch size";
        return $"Out of VRAM on the HartsyInference backend{sizes}. {ex.Message}\n\n"
            + $"{lever}, or set this backend's DitShardGpuId to a second CUDA ordinal to pool both cards' VRAM, "
            + "or set LowVram to stream weights instead of keeping them resident. Freeing cached models and "
            + "retrying once already failed, so this is not a stale-cache problem.";
    }

    /// <summary>Wraps a SwarmUI audio param as the Engine's encoded-audio payload.</summary>
    private static AudioClip ToAudioClip(AudioFile audio) =>
        audio is null ? null : new AudioClip { Data = audio.RawData, Format = audio.Type?.Extension };

    // ─────────────────────────────── 5. Validation (the honesty guard) ───────────────────────────────

    /// <summary>Cleaned IDs of "comfyui"-tagged params we genuinely service, so the comfy-only guard in
    /// <see cref="IsValidForThisBackend"/> doesn't falsely refuse them. <c>refinersampler</c>/<c>refinerscheduler</c>/
    /// <c>refinerupscalemethod</c> were removed 2026-08-09: they allow-listed but <see cref="BuildRefiner"/> never
    /// read any of them — StepSwap shares the base loop's scheduler by construction (no independent refiner
    /// sampler/scheduler exists to honor), and upscale-method has no consumer until hires-fix ships. Re-add
    /// <c>refinerupscalemethod</c> only once a real consumer exists.</summary>
    private static readonly HashSet<string> HonoredComfyParams =
        ["sampler", "scheduler",
         // Style-model (FLUX.1 Redux) strengths — mapped onto the engine's redux.* Extra keys.
         "stylemodelmergestrength", "stylemodelmultiplystrength", "stylemodelapplystart",
         // IP-Adapter scheduling knobs — mapped onto the engine's ipadapter.* Extra keys.
         "ipadapterweight", "ipadapterstart", "ipadapterend", "ipadapterweighttype"];

    /// <summary>Swarm prompt-syntax tags that need conditioning machinery the Engine's recipes do not expose on the
    /// request contract at all. Feeding these raw into a text encoder would silently corrupt the generation.
    /// <c>region</c>/<c>segment</c> are NOT here — they map to <see cref="Regional"/> and are gated per-family.</summary>
    private static readonly System.Text.RegularExpressions.Regex UnsupportedPromptSyntax =
        new(@"<(object|clear|embed)\s*:|<break\s*>",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.Compiled);

    public override bool IsValidForThisBackend(T2IParamInput input)
    {
        T2IModel model = input.Get(T2IParamTypes.Model);
        if (model is null)
        {
            return true; // let other validators speak
        }
        string compat = model.ModelClass?.CompatClass?.ID;
        if (!ModelSupport.IsArchitectureSupported(compat))
        {
            input.RefusalReasons.Add($"HartsyInference: {ModelSupport.WhyNotSupported(compat)}");
            return false;
        }
        ModelSupport.Family family = ModelSupport.Resolve(compat);

        // Prompt-syntax features with no Engine contract: refuse with the tag named rather than silently feeding
        // the tag text into the tokenizer (silent bad output is worse than a clean refusal that routes to Comfy).
        foreach (string promptText in new[] { input.Get(T2IParamTypes.Prompt), input.Get(T2IParamTypes.NegativePrompt) })
        {
            if (string.IsNullOrEmpty(promptText))
            {
                continue;
            }
            System.Text.RegularExpressions.Match match = UnsupportedPromptSyntax.Match(promptText);
            if (match.Success)
            {
                string tag = match.Value.TrimStart('<').TrimEnd(':', '>', ' ');
                input.RefusalReasons.Add(
                    $"HartsyInference: the '<{tag}:...>' prompt syntax isn't supported (no equivalent on the engine's request contract). "
                    + "Remove the tag, or use a ComfyUI backend for this generation.");
                return false;
            }
        }

        // The two-stage "generate an image, then animate it with a separate video model" flow (Comfy's
        // ImageToVideoGenInfo, driven by the Video Model param) has no Engine equivalent — refuse upfront rather
        // than silently returning a still image.
        if (family.Kind != ModelSupport.Kind.Video && input.Get(T2IParamTypes.VideoModel) is not null)
        {
            input.RefusalReasons.Add(
                "HartsyInference: the image-then-animate flow (Video Model param) isn't supported. "
                + "For image-to-video, select a video model as the main model and set an Init Image instead.");
            return false;
        }

        if (family.Kind == ModelSupport.Kind.Music && !ValidateMusic(input, family))
        {
            return false;
        }
        if (family.Kind == ModelSupport.Kind.Video && !ValidateVideo(input, compat, family))
        {
            return false;
        }
        if (family.Kind == ModelSupport.Kind.Image && !ValidateImageFeatures(input, compat, family))
        {
            return false;
        }
        return ValidateComfyOnlyParams(input);
    }

    /// <summary>Music models: refuse the image-only knobs that make no sense for an audio output, plus any music
    /// param the selected family would silently ignore. Music has no per-family feature model (no MusicFeatures
    /// enum), so these are hard-coded per family id — reachable families today: acestep, musicgen, yue.</summary>
    private static bool ValidateMusic(T2IParamInput input, ModelSupport.Family family)
    {
        if (input.Get(T2IParamTypes.InitImage) is not null)
        {
            input.RefusalReasons.Add("HartsyInference: this is a music model — remove the Init Image.");
            return false;
        }
        if (input.Get(T2IParamTypes.RefinerModel) is not null)
        {
            input.RefusalReasons.Add("HartsyInference: refiners can't run over audio outputs. Remove the Refiner Model selection.");
            return false;
        }
        if (input.TryGet(T2IParamTypes.Loras, out List<string> loras) && loras is not null && loras.Count > 0)
        {
            input.RefusalReasons.Add("HartsyInference: LoRAs aren't supported for music models. Remove the LoRA selection.");
            return false;
        }
        bool hasSourceAudio = input.Get(SwarmUIHartsyInference.MusicSourceAudioParam) is not null;
        bool isAceStep = family.Id == "acestep";
        if (hasSourceAudio && !isAceStep)
        {
            input.RefusalReasons.Add(
                $"HartsyInference: music editing (continuation/repaint/cover) is ACE-Step only; '{family.Id}' has no "
                + "audio-conditioned edit path. Remove Hartsy Music Source Audio or pick an ACE-Step model.");
            return false;
        }
        // Pre-empt the engine's src_latents-conflict throw with a routable refusal.
        if (hasSourceAudio && input.TryGet(SwarmUIHartsyInference.MusicLmModelParam, out string lmModel) && lmModel != "none")
        {
            input.RefusalReasons.Add(
                "HartsyInference: the ACE-Step edit modes and the 5 Hz LM planner both occupy the src_latents slot — "
                + "set Hartsy Music LM Planner to none, or remove the Source Audio.");
            return false;
        }
        if (hasSourceAudio && input.Get(SwarmUIHartsyInference.MusicEditModeParam, "continuation") == "repaint"
            && input.Get(SwarmUIHartsyInference.MusicRepaintEndParam, 0.0) <= input.Get(SwarmUIHartsyInference.MusicRepaintStartParam, 0.0))
        {
            input.RefusalReasons.Add(
                $"HartsyInference: repaint needs Repaint End > Repaint Start (got {input.Get(SwarmUIHartsyInference.MusicRepaintStartParam, 0.0)}"
                + $"..{input.Get(SwarmUIHartsyInference.MusicRepaintEndParam, 0.0)} s).");
            return false;
        }
        if (!isAceStep)
        {
            (bool Set, string Name)[] aceOnly =
            [
                (input.TryGet(SwarmUIHartsyInference.MusicLmModelParam, out string lm) && lm != "none", "Hartsy Music LM Planner"),
                (input.TryGet(SwarmUIHartsyInference.MusicInferMethodParam, out string im) && im != "ode", "Hartsy Music Infer Method"),
                (input.TryGet(SwarmUIHartsyInference.MusicUseAdgParam, out bool adg) && adg, "Hartsy Music Use ADG"),
                (input.TryGet(SwarmUIHartsyInference.MusicCfgIntervalStartParam, out double cis) && cis > 0, "Hartsy Music CFG Interval Start"),
                (input.TryGet(SwarmUIHartsyInference.MusicCfgIntervalEndParam, out double cie) && cie < 1, "Hartsy Music CFG Interval End"),
            ];
            foreach ((bool set, string name) in aceOnly)
            {
                if (set)
                {
                    input.RefusalReasons.Add(
                        $"HartsyInference: {name} is ACE-Step only; '{family.Id}' would silently ignore it. Unset it or pick an ACE-Step model.");
                    return false;
                }
            }
        }
        if (family.Id != "yue")
        {
            (bool Set, string Name)[] yueOnly =
            [
                (input.TryGet(SwarmUIHartsyInference.MusicTemperatureParam, out double _), "Hartsy Music Temperature"),
                (input.TryGet(SwarmUIHartsyInference.MusicTopKParam, out int _), "Hartsy Music Top K"),
                (input.TryGet(SwarmUIHartsyInference.MusicTopPParam, out double _), "Hartsy Music Top P"),
                (input.TryGet(SwarmUIHartsyInference.MusicRepetitionPenaltyParam, out double _), "Hartsy Music Repetition Penalty"),
            ];
            foreach ((bool set, string name) in yueOnly)
            {
                if (set)
                {
                    input.RefusalReasons.Add(
                        $"HartsyInference: {name} is a YuE sampling knob; '{family.Id}' would silently ignore it. Unset it or pick a YuE model.");
                    return false;
                }
            }
        }
        return true;
    }

    /// <summary>Video models: the Engine's <see cref="VideoRequest"/> has no composition phase yet (its
    /// VideoService rejects a LoRA stack outright), and a refiner over an encoded clip is meaningless.</summary>
    private static bool ValidateVideo(T2IParamInput input, string compat, ModelSupport.Family family)
    {
        // Init/end-frame conditioning is per-family. Without this check the Engine used to accept the image and
        // silently generate text-to-video, which looks like a working generation and is not. Checkpoint-aware:
        // Wan's Animate/VACE/S2V variants share the family compat classes (header-sniffed engine-side).
        VideoFeatures videoSupported = ModelSupport.SupportedVideoFeatures(compat, input.Get(T2IParamTypes.Model)?.RawFilePath);
        // LTX-2.5 ships split across four files and must be handed to the Engine as a folder. Caught here rather than
        // at load: an incomplete bundle makes the recipe fall back to LTX-2.3's Gemma 3 and 2.3 VAEs and generate a
        // video with them, so "wrong model, plausible output" is the failure this prevents.
        if (input.Get(T2IParamTypes.Model) is T2IModel videoModel
            && videoModel.ModelClass?.ID == ModelSupport.Ltx25ModelClassId
            && !ModelSupport.TryResolveLtx25Bundle(videoModel, out _, out string bundleProblem))
        {
            input.RefusalReasons.Add($"HartsyInference: LTX-2.5 bundle is incomplete — {bundleProblem}");
            return false;
        }
        // Reference media rides the prompt box (core's internal PromptImages/Audios/Videos carriers), so these read
        // what the user attached there rather than a param control of ours.
        bool hasRefImages = input.TryGet(T2IParamTypes.PromptImages, out List<Image> refImgs) && refImgs is { Count: > 0 };
        bool hasRefAudios = input.TryGet(T2IParamTypes.PromptAudios, out List<AudioFile> refAuds) && refAuds is { Count: > 0 };
        bool hasRefVideos = input.TryGet(T2IParamTypes.PromptVideos, out List<VideoFile> refVids) && refVids is { Count: > 0 };
        (VideoFeatures Feature, string Name, bool Requested)[] videoChecks =
        [
            (VideoFeatures.InitImage, "image-to-video (Init Image)", input.Get(T2IParamTypes.InitImage) is not null),
            (VideoFeatures.EndFrame, "end-frame conditioning (Video End Frame)", input.Get(T2IParamTypes.VideoEndImage) is not null),
            (VideoFeatures.ReferenceImages, "reference images (attached to the prompt)", hasRefImages),
            (VideoFeatures.ReferenceVideos, "reference videos (attached to the prompt)", hasRefVideos),
            (VideoFeatures.ReferenceAudios, "reference audio (attached to the prompt, or Video Audio Reference)",
                hasRefAudios || input.Get(SwarmUIHartsyInference.VideoAudioReferenceParam) is not null),
            (VideoFeatures.DrivingVideo, "Wan-Animate driving (Animate Reference Image / Pose Video / Face Video)",
                input.Get(SwarmUIHartsyInference.AnimateReferenceImageParam) is not null
                || input.Get(SwarmUIHartsyInference.AnimatePoseVideoParam) is not null
                || input.Get(SwarmUIHartsyInference.AnimateFaceVideoParam) is not null),
        ];
        foreach ((VideoFeatures feature, string name, bool requested) in videoChecks)
        {
            if (requested && (videoSupported & feature) == 0)
            {
                // For H3 the limit is the CHECKPOINT, not the architecture — saying "architecture doesn't support
                // references" about a model whose sibling checkpoint is built for references sends people looking
                // in the wrong place.
                string h3Hint = family.Id == "minimax-h3"
                    ? " MiniMax-H3 ships this as two checkpoints: fl2va does start/end frames, ref2va does reference"
                        + " media. Load the other one for this."
                    : " Remove it or pick a video model from a supported architecture.";
                input.RefusalReasons.Add(
                    $"HartsyInference: {name} isn't supported by this checkpoint (architecture '{compat}', engine "
                    + $"family '{family.Id}'). It applies: "
                    + $"{(videoSupported == VideoFeatures.None ? "text-to-video only" : videoSupported.ToString())}."
                    + h3Hint);
                return false;
            }
        }
        // Feature-driven, not blanket: MiniMax-H3 merges LoRAs, the other video families still do not read
        // context.Loras at all, and silently dropping a selected LoRA is worse than refusing it.
        if (input.TryGet(T2IParamTypes.Loras, out List<string> loras) && loras is not null && loras.Count > 0
            && (videoSupported & VideoFeatures.Lora) == 0)
        {
            input.RefusalReasons.Add(
                $"HartsyInference: LoRAs aren't supported on architecture '{compat}' (engine family '{family.Id}') — "
                + "that family's recipe never merges them. Remove the LoRA selection.");
            return false;
        }
        if (input.Get(T2IParamTypes.RefinerModel) is not null)
        {
            input.RefusalReasons.Add("HartsyInference: refiners can't run over video outputs. Remove the Refiner Model selection.");
            return false;
        }
        return true;
    }

    /// <summary>Checks every composition object the request would produce against the features the Engine's recipe
    /// for this family actually declares, and refuses by name. This is the honesty guard: it is driven by
    /// <see cref="IArchitectureRecipe.Supports"/>, so it can never claim more than the Engine will really apply.</summary>
    private static bool ValidateImageFeatures(T2IParamInput input, string compat, ModelSupport.Family family)
    {
        ImageFeatures supported = ModelSupport.SupportedFeatures(compat);
        List<(ImageFeatures Feature, string Name, bool Requested)> checks =
        [
            (ImageFeatures.Lora, "LoRAs", input.TryGet(T2IParamTypes.Loras, out List<string> loras) && loras is not null && loras.Count > 0),
            (ImageFeatures.ControlNet, "ControlNet", AnyControlNetSelected(input)),
            (ImageFeatures.IpAdapter, "IP-Adapter / image prompting", input.TryGet(T2IParamTypes.PromptImages, out List<Image> imgs) && imgs is not null && imgs.Count > 0),
            (ImageFeatures.Refiner, "Refiners", input.Get(T2IParamTypes.RefinerModel) is not null),
            // An Init Image is satisfied by EITHER mode: strength-based img2img, or reference-image editing on an edit
            // model (Mage-Flow, OmniGen2, Boogu, Qwen-Image-Edit) where Creativity has nothing to select. Checking only
            // Img2Img here would refuse the edit families outright.
            (ImageFeatures.Img2Img | ImageFeatures.RefEdit, "img2img (Init Image)", input.Get(T2IParamTypes.InitImage) is not null),
            (ImageFeatures.Inpaint, "inpainting (Mask Image)", input.Get(T2IParamTypes.MaskImage) is not null),
            (ImageFeatures.Regional, "regional / segment prompting",
                HasRegionalSyntax(input.Get(T2IParamTypes.Prompt)) || HasRegionalSyntax(input.Get(T2IParamTypes.NegativePrompt))),
            (ImageFeatures.VariationSeed, "Variation Seed",
                input.TryGet(T2IParamTypes.VariationSeedStrength, out double varStrength) && varStrength > 0
                && input.TryGet(T2IParamTypes.VariationSeed, out long _)),
        ];
        foreach ((ImageFeatures feature, string name, bool requested) in checks)
        {
            if (requested && (supported & feature) == 0)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: {name} isn't supported on architecture '{compat}' (engine family '{family.Id}'). "
                    + $"That family's recipe applies: {(supported == ImageFeatures.None ? "text-to-image only" : supported.ToString())}. "
                    + "Remove it or pick a model from a supported architecture.");
                return false;
            }
        }
        // An EXPLICIT init-image mode must match the family: reference needs RefEdit, denoise needs Img2Img.
        // RefEdit-only families never branch on Mode, so an explicit "denoise" there would silently run as an edit.
        if (input.Get(T2IParamTypes.InitImage) is not null
            && input.TryGet(SwarmUIHartsyInference.InitImageModeParam, out string initMode))
        {
            if (initMode == "reference" && (supported & ImageFeatures.RefEdit) == 0)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: reference-edit Init Image Mode isn't supported on '{compat}' (engine family '{family.Id}') — "
                    + "that family only does strength-based img2img. Set Init Image Mode to denoise or auto.");
                return false;
            }
            if (initMode == "denoise" && (supported & ImageFeatures.Img2Img) == 0)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: denoise Init Image Mode isn't supported on '{compat}' (engine family '{family.Id}') — "
                    + "that family is a reference-edit model with no denoise-strength path. Set Init Image Mode to reference or auto.");
                return false;
            }
        }
        return true;
    }

    /// <summary>True when any of Swarm's ControlNet slots has a model picked.</summary>
    private static bool AnyControlNetSelected(T2IParamInput input)
    {
        T2IParamTypes.ControlNetParamHolder[] holders = T2IParamTypes.Controlnets;
        if (holders is null)
        {
            return false;
        }
        foreach (T2IParamTypes.ControlNetParamHolder holder in holders)
        {
            if (holder?.Model is not null && input.Get(holder.Model) is not null)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>ComfyUI-only param guard. We advertise "comfyui" (see <see cref="SupportedFeatures"/>) so
    /// comfyui-tagged requests reach this validator rather than being pre-filtered out; here we honour that by
    /// refusing — and cleanly routing to a Comfy backend — any comfyui-tagged param actually set that we can't
    /// service. The check is driven by the params' own FeatureFlag so it auto-covers future Comfy params.</summary>
    private static bool ValidateComfyOnlyParams(T2IParamInput input)
    {
        // input.InternalSet.ValuesInput is the map of params actually set on the request, keyed by cleaned
        // param ID — the authoritative "what did the user send" list.
        Dictionary<string, object> setParams = input.InternalSet.ValuesInput;

        // Explicit safety net first: a custom ComfyUI workflow (the raw node-graph IR, or the stored-workflow
        // selector that expands into it) is the one we must never accept.
        foreach (string wfKey in new[] { "comfyworkflowraw", "comfyuicustomworkflow" })
        {
            if (setParams.TryGetValue(wfKey, out object wfVal) && !string.IsNullOrWhiteSpace($"{wfVal}"))
            {
                input.RefusalReasons.Add(
                    "HartsyInference: custom ComfyUI workflows aren't supported by this backend. "
                    + "Use a ComfyUI backend for this generation.");
                return false;
            }
        }
        foreach (string paramId in setParams.Keys.ToArray())
        {
            if (HonoredComfyParams.Contains(paramId))
            {
                continue;
            }
            if (!T2IParamTypes.TryGetType(paramId, out T2IParamType pType, input) || pType.FeatureFlag is null)
            {
                continue;
            }
            if (!pType.FeatureFlag.Split(',').Contains("comfyui"))
            {
                continue;
            }
            string valStr = $"{setParams[paramId]}";
            // Skip params left at their default / ignore value — they have no effect.
            if (valStr == pType.Default || (pType.IgnoreIf is not null && valStr == pType.IgnoreIf))
            {
                continue;
            }
            input.RefusalReasons.Add(
                $"HartsyInference: the '{pType.Name}' parameter is a ComfyUI-only feature this backend "
                + "can't service. Remove it, or use a ComfyUI backend for this generation.");
            return false;
        }
        return true;
    }

    // ─────────────────────────────── 6. Engine auto-update ───────────────────────────────

    /// <summary>Checks NuGet for a newer HartsyInference engine and, when the <c>AutoUpdate</c> setting opts in,
    /// rebuilds this extension against it. Because the engine is an <b>in-process</b> library (not an external
    /// process Swarm can relaunch), a fetched update is <i>staged</i> into the extension's build output and only
    /// takes effect on the next SwarmUI start. 'aggressive' clears NuGet caches and calls
    /// <see cref="Program.RequestRestart"/> so the new build loads automatically.
    /// <para>Returns true if a newer engine was staged and the backend should now refuse to run on the stale
    /// in-process version (forcing the user to restart). False when up to date, when staging failed, when
    /// 'aggressive' is restarting anyway, or when the loop-guard tripped (run degraded, don't brick).</para></summary>
    private async Task<bool> MaybeAutoUpdateEngine()
    {
        string mode = (Settings?.AutoUpdate ?? "false").Trim().ToLowerInvariant();
        if (mode is "false" or "" or "0" or "no" or "off")
        {
            return false;
        }
        bool aggressive = mode is "aggressive" or "force";
        // Boot-loop guard: when 'aggressive' stages an update and restarts, we record the target version. If we
        // come back up STILL behind that exact version, the restore isn't advancing — refuse to auto-restart
        // again and surface an error instead of cycling forever.
        string updateMarker = Path.Combine(
            Path.GetDirectoryName(typeof(HartsyInferenceBackend).Assembly.Location) ?? ".",
            ".hartsy-engine-update-pending");
        try
        {
            string loaded = LoadedEngineVersion();
            string latest = await LatestEnginePackageVersion();
            AddLoadStatus($"Auto-update: loaded engine={loaded ?? "unknown"}, latest published={latest ?? "unknown"}.");
            if (latest is null)
            {
                AddLoadStatus("Auto-update: could not query NuGet; skipping.");
                return false;
            }
            if (loaded is not null && !IsNewerAlpha(latest, loaded))
            {
                // Up to date — a prior staged update (if any) applied successfully, so clear the marker.
                if (File.Exists(updateMarker))
                {
                    try { File.Delete(updateMarker); }
                    catch (Exception mEx) { Logs.Debug($"[HartsyInference] Auto-update: couldn't clear update marker: {mEx.Message}"); }
                }
                AddLoadStatus("Auto-update: engine is already up to date.");
                return false;
            }
            string pending = File.Exists(updateMarker) ? File.ReadAllText(updateMarker).Trim() : null;
            if (aggressive && string.Equals(pending, latest, StringComparison.Ordinal))
            {
                Logs.Error($"[HartsyInference] Auto-update: engine {latest} was already staged on a previous restart but the running engine is still {loaded}. "
                    + $"Not auto-restarting again (avoiding a boot loop). The rebuild's NuGet restore isn't resolving {latest}.");
                AddLoadStatus($"Auto-update: {latest} did not apply after a restart — auto-restart paused to avoid a loop (see logs).");
                return false; // run degraded on the old engine rather than bricking the backend
            }

            string csproj = ExtensionProjectPath();
            if (csproj is null)
            {
                Logs.Warning("[HartsyInference] Auto-update: extension .csproj not found next to the assembly; cannot rebuild.");
                return false;
            }
            AddLoadStatus($"Auto-update: verifying engine {latest} compiles against this extension (this can take a minute)...");
            string dir = Path.GetDirectoryName(csproj);
            // Verification build BEFORE we touch anything: a freshly-published engine can have breaking API
            // changes. This build goes to the csproj's DEFAULT output — NOT Swarm's load folder — purely as a
            // compile check. RestoreForceEvaluate re-resolves the floating alpha.*, pre-warming the packages folder.
            (int code, string output) = await RunDotnet($"build \"{csproj}\" -c Release /p:RestoreForceEvaluate=true", dir);
            if (code != 0)
            {
                Logs.Error($"[HartsyInference] Auto-update: engine {latest} does NOT compile against this extension — staying on the current engine. Build output:\n{output}");
                AddLoadStatus($"Auto-update: engine {latest} failed to compile — keeping the current engine (see logs).");
                return false;
            }

            // The engine is an IN-PROCESS library: it can't hot-swap, and SwarmUI's ExtensionsManager.BuildExtension
            // keys its "already built, skip rebuild" decision on the extension's GIT COMMIT hash — NOT on the engine
            // NuGet version. So to actually apply `latest` we must INVALIDATE that cached build.
            string selfDll = typeof(HartsyInferenceBackend).Assembly.Location;
            try
            {
                // On Linux, unlinking the loaded/mmap'd assembly is allowed — the running process keeps its
                // mapping so THIS session is unaffected; the file is simply gone from disk for next start.
                if (File.Exists(selfDll))
                {
                    File.Delete(selfDll);
                }
            }
            catch (Exception delEx)
            {
                Logs.Warning($"[HartsyInference] Auto-update: couldn't invalidate cached build '{selfDll}' ({delEx.Message}). "
                    + $"Delete that file (or its folder under src/bin/extensions/) and restart to load engine {latest}.");
                AddLoadStatus($"Auto-update: couldn't invalidate cached build — delete '{Path.GetFileName(selfDll)}' manually and restart.");
                return false;
            }
            Logs.Warning($"[HartsyInference] Engine {latest} verified + cached extension build invalidated. SwarmUI will REBUILD against the new engine on next start (an in-process library can't hot-swap).");
            if (aggressive)
            {
                Logs.Warning("[HartsyInference] AutoUpdate=aggressive — requesting a SwarmUI restart to rebuild and load the new engine.");
                AddLoadStatus($"Auto-update: engine {latest} verified — restarting SwarmUI to rebuild and apply.");
                try { File.WriteAllText(updateMarker, latest); }
                catch (Exception mEx) { Logs.Debug($"[HartsyInference] Auto-update: couldn't write update marker: {mEx.Message}"); }
                Program.RequestRestart();
                return false; // process is restarting; no need to error the backend
            }
            AddLoadStatus($"Auto-update: engine {latest} staged — RESTART SwarmUI to load it.");
            return true; // staged but not loaded: caller errors the backend so it won't run stale
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] Auto-update failed: {ex.ReadableString()}");
            AddLoadStatus("Auto-update: failed (continuing with the current engine).");
            return false;
        }
    }

    /// <summary>The NuGet version baked into the loaded engine assembly (e.g. "1.0.0-alpha.11"), or null.</summary>
    private static string LoadedEngineVersion()
    {
        Assembly asm = typeof(IBackend).Assembly;
        string info = asm.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        // Informational version may carry a build-metadata suffix ("1.0.0-alpha.11+abc123"); trim it.
        if (info is not null)
        {
            int plus = info.IndexOf('+');
            if (plus >= 0)
            {
                info = info[..plus];
            }
        }
        return info;
    }

    /// <summary>Queries NuGet's flat-container index for the highest published HartsyInference version.</summary>
    private static async Task<string> LatestEnginePackageVersion()
    {
        try
        {
            using HttpClient http = new() { Timeout = TimeSpan.FromSeconds(20) };
            string json = await http.GetStringAsync("https://api.nuget.org/v3-flatcontainer/hartsyinference/index.json");
            JObject parsed = JObject.Parse(json);
            if (parsed["versions"] is not JArray versions)
            {
                return null;
            }
            string best = null;
            foreach (JToken v in versions)
            {
                string s = v.ToString();
                if (best is null || IsNewerAlpha(s, best))
                {
                    best = s;
                }
            }
            return best;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference] Auto-update: NuGet version query failed: {ex.ReadableString()}");
            return null;
        }
    }

    /// <summary>True if <paramref name="candidate"/> is a newer "1.0.0-alpha.N" than <paramref name="current"/>
    /// (compares the trailing alpha number; unparseable forms fall back to ordinal string compare).</summary>
    private static bool IsNewerAlpha(string candidate, string current)
    {
        static int Num(string v)
        {
            int dash = v.LastIndexOf("alpha.", StringComparison.OrdinalIgnoreCase);
            if (dash < 0)
            {
                return -1;
            }
            string tail = v[(dash + "alpha.".Length)..];
            int dot = tail.IndexOfAny(['.', '-', '+']);
            if (dot >= 0)
            {
                tail = tail[..dot];
            }
            return int.TryParse(tail, out int n) ? n : -1;
        }
        int a = Num(candidate), b = Num(current);
        if (a >= 0 && b >= 0)
        {
            return a > b;
        }
        return string.CompareOrdinal(candidate, current) > 0;
    }

    /// <summary>Locates this extension's <c>.csproj</c> by walking up from the loaded assembly's directory.</summary>
    private static string ExtensionProjectPath()
    {
        string dir = Path.GetDirectoryName(typeof(HartsyInferenceBackend).Assembly.Location);
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            string[] found = Directory.GetFiles(dir, "*.csproj", SearchOption.TopDirectoryOnly);
            if (found.Length > 0)
            {
                return found[0];
            }
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    /// <summary>Runs <c>dotnet &lt;args&gt;</c> in <paramref name="workDir"/> and returns (exitCode, combined output).</summary>
    private static async Task<(int Code, string Output)> RunDotnet(string args, string workDir)
    {
        System.Diagnostics.ProcessStartInfo psi = new("dotnet", args)
        {
            WorkingDirectory = workDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using System.Diagnostics.Process proc = System.Diagnostics.Process.Start(psi);
        string stdout = await proc.StandardOutput.ReadToEndAsync();
        string stderr = await proc.StandardError.ReadToEndAsync();
        await proc.WaitForExitAsync();
        return (proc.ExitCode, (stdout + "\n" + stderr).Trim());
    }
}
