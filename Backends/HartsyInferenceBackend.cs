using System.IO;
using System.Net.Http;
using System.Reflection;
using FreneticUtilities.FreneticDataSyntax;
using Newtonsoft.Json.Linq;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using HartsyInference.Core.Backends;
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

        [ConfigComment("GPU for the text encoders (CLIP / T5 / umT5), separate from the main GPU_ID.\nEmpty (default) = same GPU as everything else.\nSet to another CUDA ordinal (e.g. '1') to keep the multi-GB text encoders off the main card — the biggest VRAM win on video models (Wan's umT5 is T5-XXL-class) and Flux.\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string TextEncoderGpuId = "";

        [ConfigComment("Second GPU to run CFG's negative-prompt branch on, concurrent with the positive branch on the main GPU_ID — a latency win (not a VRAM win) when the denoiser fits on BOTH cards, since the weights are REPLICATED, not split.\nEmpty (default) = off; CFG runs sequentially on GPU_ID as usual.\nSet to another CUDA ordinal (e.g. '1') to enable it. Currently wired for Wan video (T2V/TI2V, single-expert checkpoints only — Wan2.2 A14B MoE checkpoints fall back to sequential automatically).\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string CfgParallelGpuId = "";

        [ConfigComment("Second GPU to pool VRAM with for large DiTs that don't fit on GPU_ID alone (experimental; currently wired for Krea 2 image generation only) — the denoiser's block loop is SPLIT across both cards (not replicated), so this is a VRAM win, not a latency win (sequential pipeline split, same per-step speed as one card).\nEmpty (default) = off.\nSet to another CUDA ordinal (e.g. '1') to enable it. Cannot be combined with CfgParallelGpuId — they are two different ways to use a second GPU for the same model (VRAM pooling vs weight replication for latency) and were not designed to compose; the backend will fail to start if both are set.\nNote this also feeds the same placement list LLM text generation uses for layer-split placement, so enabling it may also change where text models place their layers.\nThe number is a CUDA ordinal like GPU_ID (fastest-first, not nvidia-smi order).")]
        public string DitShardGpuId = "";

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

    public override IEnumerable<string> SupportedFeatures
    {
        get
        {
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
            yield return "hartsyinference";
            yield return "comfyui";
            yield return "text2image";
            yield return "flux-dev";       // in DisregardedFeatureFlags — informational
            yield return "lora";           // per-family; gated by the recipe's ImageFeatures.Lora
            yield return "endstepsearly";  // ImageRequest.EndStepsEarly
            yield return "refiners";       // per-family; gated by ImageFeatures.Refiner
            yield return "img2img";        // per-family; gated by ImageFeatures.Img2Img
            yield return "inpaint";        // per-family; gated by ImageFeatures.Inpaint
            yield return "controlnet";     // per-family; gated by ImageFeatures.ControlNet
            yield return "ipadapter";      // per-family; gated by ImageFeatures.IpAdapter
            yield return "variation_seed"; // per-family; gated by ImageFeatures.VariationSeed
            yield return "video";          // Wan (T2V/I2V + VACE / Animate / S2V) + LTX-Video + LTX-2 + Lance Video
        }
    }

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
                AddLoadStatus($"Text encoders placed on {teSelector} (denoiser/VAE stay on GPU {deviceOrdinal ?? 0}).");
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

            if (teSelector is not null || cfgParallelSelector is not null || enableDitSharding)
            {
                placement = new PlacementConfig
                {
                    TextEncoderDevice = teSelector,
                    CfgParallelDevice = cfgParallelSelector,
                    ShardDevices = enableDitSharding
                        ? new[] { BackendFactory.WithOrdinal(requested, deviceOrdinal ?? 0), ditShardSelector }
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

            Image[] outputs = family.Kind switch
            {
                ModelSupport.Kind.Video => [await GenerateVideo(spec, input, progress, cancel)],
                ModelSupport.Kind.Music => [await GenerateMusic(spec, input, progress, cancel)],
                _ => [await GenerateImage(spec, input, family, progress, cancel)],
            };

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
        return RgbToImage.FromHwcRgb(result.Rgb, result.Width, result.Height);
    }

    /// <summary>Runs a video generation, collecting the streamed frames and muxing them into a container.</summary>
    private async Task<Image> GenerateVideo(ModelSpec spec, T2IParamInput input, IProgress<StepPreview> progress, CancellationToken cancel)
    {
        VideoRequest request = BuildVideoRequest(input);
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
        if (input.Get(T2IParamTypes.VideoBoomerang, false))
        {
            for (int i = frames.Count - 2; i > 0; i--)
            {
                frames.Add(frames[i]);
            }
        }
        // Optional SeedVR2 restore pass (Video Restore param group): frames go straight into the engine's
        // restore service (no container round-trip), replacing the frame list before muxing.
        if (input.TryGet(SwarmUIHartsyInference.VideoRestoreModelParam, out string restoreModel)
            && !string.IsNullOrWhiteSpace(restoreModel))
        {
            // The resident T2V DiT and SeedVR2's VAE peak cannot share 24 GB.
            _engine.FreeMemory();
            ModelSpec restoreSpec = ModelResolver.Resolve(restoreModel, null, Modality.Restore);
            RestoreRequest restoreRequest = new()
            {
                Frames = [.. frames.Select(f => new ImageData { Rgb = f, Width = width, Height = height })],
                TargetWidth = input.TryGet(SwarmUIHartsyInference.VideoRestoreWidthParam, out int rw) ? rw : null,
                TargetHeight = input.TryGet(SwarmUIHartsyInference.VideoRestoreHeightParam, out int rh) ? rh : null,
                ClipFrames = input.TryGet(SwarmUIHartsyInference.VideoRestoreClipFramesParam, out int rcf) ? rcf : 5,
                Overlap = input.TryGet(SwarmUIHartsyInference.VideoRestoreOverlapParam, out int rov) ? rov : 1,
                Strength = input.TryGet(SwarmUIHartsyInference.VideoRestoreStrengthParam, out double rst) ? (float)rst : null,
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
        string format = input.Get(T2IParamTypes.VideoFormat, "h264-mp4");
        if (audio is not null && !VideoOutputEncoder.FormatSupportsAudio(format))
        {
            Logs.Warning($"[HartsyInference] Video format '{format}' cannot carry audio; the {audio.SampleRate} Hz track is not muxed.");
            audio = null;
        }
        return VideoOutputEncoder.Encode([.. frames], width, height, request.Fps ?? 25, format, cancel, audio);
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
        return new ImageRequest
        {
            Prompt = prompt,
            NegativePrompt = input.Get(T2IParamTypes.NegativePrompt),
            Width = NullableInt(input, T2IParamTypes.Width),
            Height = NullableInt(input, T2IParamTypes.Height),
            Steps = NullableInt(input, T2IParamTypes.Steps),
            CfgScale = input.TryGet(T2IParamTypes.CFGScale, out double cfg) ? (float)cfg : null,
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
            ControlNets = BuildControlNets(input),
            IpAdapter = BuildIpAdapter(input),
            Refiner = BuildRefiner(input),
            Img2Img = initImage is null ? null : new Img2Img
            {
                InitImage = ToEngineImage(initImage),
                Creativity = input.Get(T2IParamTypes.InitImageCreativity, 0.6),
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
            Extra = BuildImageExtra(input),
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
    private IReadOnlyList<ControlNetConditioning> BuildControlNets(T2IParamInput input)
    {
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
    private static IReadOnlyDictionary<string, object> BuildImageExtra(T2IParamInput input)
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
        // Extension-registered custom params.
        if (input.TryGet(SwarmUIHartsyInference.DtypeOverrideParam, out string dtype) && !string.IsNullOrWhiteSpace(dtype))
        {
            extra["hartsy.dtype_override"] = dtype;
        }
        if (input.TryGet(SwarmUIHartsyInference.TileVaeThresholdParam, out int tileThreshold))
        {
            extra["hartsy.tile_vae_threshold"] = tileThreshold;
        }
        if (input.TryGet(SwarmUIHartsyInference.Ideogram4MagicPromptParam, out bool magic))
        {
            extra["hartsy.ideogram4_magic_prompt"] = magic;
        }
        if (input.TryGet(SwarmUIHartsyInference.Ideogram4MagicPromptModelParam, out string magicModel) && !string.IsNullOrWhiteSpace(magicModel))
        {
            extra["hartsy.ideogram4_magic_prompt_model"] = magicModel;
        }
        return extra;
    }

    /// <summary>Maps the Swarm request onto the Engine's <see cref="VideoRequest"/>.</summary>
    private static VideoRequest BuildVideoRequest(T2IParamInput input)
    {
        Image initImage = input.Get(T2IParamTypes.InitImage);
        Image endFrame = input.Get(T2IParamTypes.VideoEndFrame);
        int frames = input.TryGet(T2IParamTypes.Text2VideoFrames, out int t2vFrames) && t2vFrames > 0
            ? t2vFrames
            : input.Get(T2IParamTypes.VideoFrames, 25);
        Dictionary<string, object> extra = new(StringComparer.Ordinal);
        if (input.Get(SwarmUIHartsyInference.AnimateReferenceImageParam) is Image reference)
        {
            extra[HartsyInference.Engine.Recipes.Video.WanAnimateRecipePipeline.ReferenceImageKey] = ToEngineImage(reference);
        }
        if (input.TryGet(SwarmUIHartsyInference.AnimateAutoPreprocessParam, out bool autoPreprocess))
        {
            extra["hartsy.animate_auto_preprocess"] = autoPreprocess;
        }
        if (input.Get(SwarmUIHartsyInference.AnimatePoseVideoParam) is Image poseVideo)
        {
            extra["hartsy.animate_pose_video"] = ToEngineImage(poseVideo);
        }
        if (input.Get(SwarmUIHartsyInference.AnimateFaceVideoParam) is Image faceVideo)
        {
            extra["hartsy.animate_face_video"] = ToEngineImage(faceVideo);
        }
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
            InitImage = initImage is null ? null : ToEngineImage(initImage),
            VideoModel = ModelPath(input.Get(T2IParamTypes.VideoModel)),
            VideoSwapModel = ModelPath(input.Get(T2IParamTypes.VideoSwapModel)),
            VideoSwapPercent = input.TryGet(T2IParamTypes.VideoSwapPercent, out double swapPercent) ? swapPercent : null,
            VideoExtendModel = ModelPath(input.Get(T2IParamTypes.VideoExtendModel)),
            VideoResolution = input.Get(T2IParamTypes.VideoResolution, null),
            Fps = input.Get(T2IParamTypes.VideoFPS, 25),
            VideoFormat = input.Get(T2IParamTypes.VideoFormat, null),
            // Boomerang is applied to the decoded frames on the way out (see GenerateVideo) so the Engine
            // doesn't waste a second pass; the flag is still carried for pipelines that want it.
            VideoBoomerang = input.Get(T2IParamTypes.VideoBoomerang, false),
            VideoEndFrame = endFrame is null ? null : ToEngineImage(endFrame),
            VideoAudioInput = ToAudioClip(input.Get(T2IParamTypes.VideoAudioInput)),
            VideoAudioReference = ToAudioClip(input.Get(T2IParamTypes.VideoAudioReference)),
            Frames = frames,
            TrimVideoStartFrames = input.Get(T2IParamTypes.TrimVideoStartFrames, 0),
            TrimVideoEndFrames = input.Get(T2IParamTypes.TrimVideoEndFrames, 0),
            Components = BuildComponents(input),
            Loras = BuildLoras(input),
            Extra = extra,
        };
    }

    /// <summary>Maps Swarm's Text2Audio group onto the Engine's <see cref="MusicRequest"/>.</summary>
    private static MusicRequest BuildMusicRequest(T2IParamInput input)
    {
        long seed = input.Get(T2IParamTypes.Seed, -1L);
        return new MusicRequest
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
        };
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

    /// <summary>Wraps a SwarmUI audio param as the Engine's encoded-audio payload.</summary>
    private static AudioClip ToAudioClip(AudioFile audio) =>
        audio is null ? null : new AudioClip { Data = audio.RawData, Format = audio.Type?.Extension };

    // ─────────────────────────────── 5. Validation (the honesty guard) ───────────────────────────────

    /// <summary>Cleaned IDs of "comfyui"-tagged params we genuinely service, so the comfy-only guard in
    /// <see cref="IsValidForThisBackend"/> doesn't falsely refuse them.</summary>
    private static readonly HashSet<string> HonoredComfyParams =
        ["sampler", "scheduler", "refinersampler", "refinerscheduler", "refinerupscalemethod"];

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

        if (family.Kind == ModelSupport.Kind.Music && !ValidateMusic(input))
        {
            return false;
        }
        if (family.Kind == ModelSupport.Kind.Video && !ValidateVideo(input))
        {
            return false;
        }
        if (family.Kind == ModelSupport.Kind.Image && !ValidateImageFeatures(input, compat, family))
        {
            return false;
        }
        return ValidateComfyOnlyParams(input);
    }

    /// <summary>Music models: refuse the image-only knobs that make no sense for an audio output.</summary>
    private static bool ValidateMusic(T2IParamInput input)
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
        return true;
    }

    /// <summary>Video models: the Engine's <see cref="VideoRequest"/> has no composition phase yet (its
    /// VideoService rejects a LoRA stack outright), and a refiner over an encoded clip is meaningless.</summary>
    private static bool ValidateVideo(T2IParamInput input)
    {
        if (input.TryGet(T2IParamTypes.Loras, out List<string> loras) && loras is not null && loras.Count > 0)
        {
            input.RefusalReasons.Add(
                "HartsyInference: LoRAs aren't supported for video models yet — the engine's video composition phase "
                + "isn't wired. Remove the LoRA selection.");
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
            (ImageFeatures.Img2Img, "img2img (Init Image)", input.Get(T2IParamTypes.InitImage) is not null),
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
