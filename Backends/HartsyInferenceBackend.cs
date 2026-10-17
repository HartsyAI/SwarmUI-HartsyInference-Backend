using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using SwarmUI.Builtin_ComfyUIBackend;
using FreneticUtilities.FreneticDataSyntax;
using Newtonsoft.Json.Linq;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using Hartsy.Extensions.HartsyInferenceBackend.Services;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.MemoryManagement;
using SiLogs = HartsyInference.Core.Logging.Logs;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using EngineImage = HartsyInference.Engine.Requests.ImageData;

namespace Hartsy.Extensions.HartsyInferenceBackend.Backends;

/// <summary>SwarmUI backend that runs inference in-process through <c>HartsyInference.Engine</c>.
///
/// <para>This class is a <b>thin mapper</b> and nothing more. The Engine is the single source of truth for
/// "load a model + generate": it owns architecture detection, per-family recipes, pipeline caching, side-model
/// download, composition (LoRA / ControlNet / IP-Adapter / refiner / img2img / inpaint / regional) and sampling
/// defaults. The four things that remain SwarmUI's problem, and therefore ours, are:</para>
/// <list type="number">
/// <item><b>Load</b>: resolve a <see cref="T2IModel"/> to a checkpoint path and an Engine <see cref="ModelSpec"/>.</item>
/// <item><b>Map request</b>: the only place <c>input.Get(T2IParamTypes.*)</c> is read; produces an
/// <see cref="ImageRequest"/> / <see cref="VideoRequest"/> / <see cref="MusicRequest"/>.</item>
/// <item><b>Stream progress</b>: bridge <see cref="StepPreview"/> onto Swarm's <c>takeOutput</c> contract.</item>
/// <item><b>Map result</b>: Engine pixels/frames/samples back into a SwarmUI <see cref="Image"/>.</item>
/// </list>
/// </summary>
public class HartsyInferenceBackend : AbstractT2IBackend
{
    /// <summary>Settings saved to Data/Backends.fds. Nested in the backend class so BackendHandler.RegisterBackendType can find it by reflection.</summary>
    public class HartsyInferenceBackendSettings : AutoConfiguration
    {
        [ConfigComment("Compute backend to use.\n'Auto' (default) tries CUDA, then Vulkan, then CPU. Vulkan is how a non-NVIDIA GPU (AMD, Intel) gets used, so 'Auto' does not mean 'CUDA or nothing'.\nOnly options this machine can actually use are offered here: 'Cuda' only appears when an NVIDIA GPU is present AND a real test computation on it succeeds, not just that a driver reports a device (this catches a kernel built for the wrong GPU architecture, a missing PTX directory, or a driver/toolkit mismatch, before it surfaces mid-generation instead of here).\n'Vulkan' only appears when a real Vulkan GPU is present. Software rasterizers such as Mesa's lavapipe enumerate as Vulkan devices but are CPU implementations of it, slower than the CPU backend, so they do not count as a GPU here.\nThis list is built once per SwarmUI process start. If you plug in a GPU or fix a driver, restart SwarmUI to see it here.")]
        [SettingsOptions(Impl = typeof(ComputeBackendOptions))]
        public string ComputeBackend = "auto";

        [ConfigComment("Which GPU to use, if multiple are available.\nShould be a single number, like '0' (first GPU), '1' (second GPU), etc.\nIgnored for the CPU compute backend.\nUnder CUDA this is a CUDA device ordinal, which is NOT necessarily the same order nvidia-smi shows: CUDA enumerates fastest-first by default, so on a mixed-GPU machine '0' is the fastest card. To confirm which physical GPU you got, watch nvidia-smi memory while a generation runs.\nUnder Vulkan the numbering works differently. '0' (the default) means 'let the engine pick the best GPU'; any other number is the Vulkan loader's RAW device index. That raw list can include software rasterizers, which take up an index without being usable GPUs, so '1' is not necessarily your second card. The backend's startup log names the device it actually bound to, which is the reliable way to check.\nRun one backend per GPU to use several cards at once.")]
        public string GPU_ID = "0";

        [ConfigComment("How hard the engine should work to fit a model in VRAM.\n\n'Auto' (default) reads your card's size to pick a starting posture, then measures free VRAM before each phase and streams only when the model would not otherwise fit. Cards with headroom keep the full-speed resident path, so this costs nothing when it isn't needed. Leave it here unless you have a reason not to.\n\n'Performance' never streams and never auto-evicts: everything loads and an oversized model fails with an out-of-VRAM error instead of running slowly. For operators who size their own workloads and want a hard failure.\n\n'Balanced' releases each phase's weights at its boundary so the next phase gets the space, and streams only when the measurement says it must.\n\n'Aggressive' always streams, halves the precision of cross-step caches, and shrinks chunk/tile sizes. Useful when sharing the GPU with another program, or to test the streamed path.\n\n'Maximum' adds activation offload, quantized compute and freeing after every generation, and will spill onto a second GPU when one is configured and idle. Last resort — every one of those costs speed, and some change numerics.\n\nStreaming is typically 5-8x slower than a fully-resident model, but it is what lets large models run on a 12GB card at all. Individual levers can be overridden per-generation under the 'VRAM & Memory' parameter group.\nThe older 'auto'/'on'/'off' values still load: 'on' becomes Aggressive and 'off' becomes Performance.")]
        [ManualSettingsOptions(Impl = null, Vals = ["Auto", "Performance", "Balanced", "Aggressive", "Maximum"],
            ManualNames = ["Auto (recommended)", "Performance (never stream, fail instead)", "Balanced (unload between phases)",
                "Aggressive (always stream, half-precision caches)", "Maximum (every lever, may spill to a 2nd GPU)"])]
        public string LowVram = "Auto";

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

    /// <summary>Builds the ComputeBackend dropdown from the kinds this machine can use, using <see cref="BackendFactory.ValidSelectors"/>.</summary>
    /// <remarks>Offers kinds only: GPU_ID already picks the device. Cached for the process, since a probe creates a real GPU context.</remarks>
    public class ComputeBackendOptions : SettingsOptionsAttribute.AbstractImpl
    {
        private static readonly Lazy<(string[] Vals, string[] Names)> _options = new(Build);

        /// <summary>Friendly labels for known kinds. Any other kind is shown under the engine's own token.</summary>
        private static readonly Dictionary<string, string> KindLabels = new(StringComparer.OrdinalIgnoreCase)
        {
            ["auto"] = "Auto (recommended)",
            ["cpu"] = "CPU (slow; F32-only, cannot run fp8/bf16 models)",
            ["cuda"] = "CUDA (NVIDIA GPU)",
            ["vulkan"] = "Vulkan (any GPU)",
        };

        private static (string[] Vals, string[] Names) Build()
        {
            List<string> vals = [];
            List<string> names = [];
            foreach (string kind in Guard(() => BackendFactory.ValidSelectors.ToList(), "backend list") ?? [])
            {
                if (!IsUsable(kind))
                {
                    continue;
                }
                vals.Add(kind);
                names.Add(Label(kind));
            }
            if (vals.Count == 0)
            {
                // Nothing validated (e.g. engine assemblies not resolvable). 'auto' always resolves
                // (falls back to CPU), so never ship an empty dropdown.
                vals.Add("auto");
                names.Add(KindLabels["auto"]);
            }
            return ([.. vals], [.. names]);
        }

        /// <summary>'cuda' needs a real GPU computation, not just driver presence. Other kinds use <see cref="BackendFactory.Validate"/>.</summary>
        private static bool IsUsable(string kind)
        {
            if (!kind.Equals("cuda", StringComparison.OrdinalIgnoreCase))
            {
                return Guard(() => { BackendFactory.Validate(kind); return true; }, $"validate '{kind}'");
            }
            bool ok = Guard(() => BackendFactory.ProbeCuda(), "CUDA probe");
            if (!ok)
            {
                // Warning, not Debug: hiding an option the user would otherwise expect deserves a visible reason.
                string why = Guard(() => BackendFactory.CudaProbeFailureReason, "CUDA probe reason") ?? "no reason reported";
                Logs.Warning($"[HartsyInference] ComputeBackend dropdown: excluding 'cuda': {why}");
            }
            return ok;
        }

        /// <summary>Names the CUDA card from the engine's <see cref="CudaTopology"/>, which is in CUDA ordinal order.</summary>
        /// <remarks>Not <see cref="NvidiaUtil"/>: nvidia-smi index order can differ from CUDA ordinals, so it could name the wrong card.</remarks>
        private static string Label(string kind)
        {
            string baseLabel = KindLabels.TryGetValue(kind, out string label) ? label : kind;
            if (!kind.Equals("cuda", StringComparison.OrdinalIgnoreCase))
            {
                return baseLabel;
            }
            IReadOnlyList<GpuTopologyInfo> gpus = Guard(() => CudaTopology.Probe(), "CUDA topology probe") ?? [];
            GpuTopologyInfo gpu0 = gpus.FirstOrDefault(d => d.Ordinal == 0 && d.Name is not null);
            if (gpu0.Name is null)
            {
                return baseLabel;
            }
            string detected = gpus.Count == 1 ? gpu0.Name : $"{gpu0.Name} + {gpus.Count - 1} more";
            return $"CUDA ({detected})";
        }

        /// <summary>Runs a probe. A throw (no engine, no driver, timeout) means "not usable", so this returns default instead.</summary>
        private static T Guard<T>(Func<T> probe, string what)
        {
            try
            {
                return probe();
            }
            catch (Exception ex)
            {
                Logs.Debug($"[HartsyInference] ComputeBackend dropdown: {what} failed ({ex.Message}).");
                return default;
            }
        }

        public override string[] GetOptions => _options.Value.Vals;

        public override string[] Names => _options.Value.Names;
    }

    public HartsyInferenceBackendSettings Settings => SettingsRaw as HartsyInferenceBackendSettings;

    /// <summary>The Engine facade. Owns the compute backend, every loaded pipeline, and all generation.</summary>
    private InferenceEngine _engine;

    /// <summary>Device for the ControlNet annotators: borrowed from the Engine when one exists, else created and owned here.</summary>
    /// <remarks>Borrowing avoids a second CUDA context on the same device, which would collide with the Engine's state.</remarks>
    private IBackend _preprocessBackend;
    private bool _ownsPreprocessBackend;
    private readonly object _preprocessBackendLock = new();

    /// <summary>Cancellation source for the in-flight generation.</summary>
    private CancellationTokenSource _cancelCts;

    /// <summary>Runs generations one at a time, even when OverQueue lets jobs wait here, since the Engine's caches and device are shared.</summary>
    private readonly SemaphoreSlim _genLock = new(1, 1);

    /// <summary>The device plus every setting that changes a memory verdict, as registered in <see cref="BackendDeviceRegistry"/>. Null before the first successful Init.</summary>
    /// <remarks>Backends sharing a key share one fit assessment. Kept across Shutdown, so a settings change can forget the old key's out-of-memory evidence.</remarks>
    public string FitProfileKey { get; private set; }

    /// <summary>The compute kind <see cref="Init"/>'s probe resolved, so device builders do not re-derive it via the cheaper <see cref="BackendFactory.Resolve"/>. Null before Init.</summary>
    /// <remarks>Volatile: <see cref="PreprocessBackend"/> reads it under its own lock while Init writes it unlocked. Init completes before any generation.</remarks>
    private volatile string _resolvedComputeKind;

    /// <summary>Every feature flag this backend advertises. Static so the startup self-check in <c>HartsyInference</c> can compare it with our registered params' flags.</summary>
    /// <remarks><see cref="T2IEngine"/> refuses any job whose required flags this set does not cover (unless listed in <c>T2IEngine.DisregardedFeatureFlags</c>).
    /// That refusal names no param, so a flag added to a param and missed here shows up only as a refused generation.</remarks>
    public static readonly IReadOnlyList<string> DeclaredFeatures =
    [
        // "comfyui" lets Comfy-tagged params (e.g. Sampler) reach our validator. SwarmUI cannot split one request across
        // backends, so a request carrying both flags would otherwise match neither. Advertising is not a promise:
        // IsValidForThisBackend refuses comfyui-only params we cannot service, and composition features are checked per recipe.
        "hartsyinference",
        "comfyui",
        "text2image",
        "flux-dev",       // listed in DisregardedFeatureFlags (informational)
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
        // Model-scoped UI flags, granted per selected model by Assets/hartsy-params.js. T2IEngine needs them declared here.
        "hartsy_ideogram4",
        "hartsy_acestep",
        "hartsy_minimaxmusic",
        "hartsy_wan_animate",
        "hartsy_audio_ref",
        "hartsy_h3_chain",
        "hartsy_refedit_choice",
    ];

    public override IEnumerable<string> SupportedFeatures => DeclaredFeatures;

    /// <summary>Bridges HartsyInference's logger into Swarm's log, so engine diagnostics (such as the OOM probe) reach the main log file. Idempotent.</summary>
    private static int _loggerWired = 0;

    private static void EnsureLoggerWired()
    {
        if (Interlocked.Exchange(ref _loggerWired, 1) != 0)
        {
            return;
        }
        // Pass everything through the bridge; Swarm's MinimumLevel does the filtering.
        SiLogs.MinLevel = global::HartsyInference.Core.Logging.LogLevel.Verbose;
        SiLogs.SetLogger((level, msg) =>
        {
            msg = "[HartsyInference][Engine] " + msg;
            switch (level)
            {
                case global::HartsyInference.Core.Logging.LogLevel.Verbose: Logs.Verbose(msg); break;
                case global::HartsyInference.Core.Logging.LogLevel.Debug: Logs.Debug(msg); break;
                case global::HartsyInference.Core.Logging.LogLevel.Info: Logs.Info(msg); break;
                case global::HartsyInference.Core.Logging.LogLevel.Warning: Logs.Warning(msg); break;
                case global::HartsyInference.Core.Logging.LogLevel.Error: Logs.Error(msg); break;
                default: Logs.Info(msg); break;
            }
        });
    }

    // ─────────────────────────────── 1. Lifecycle ───────────────────────────────

    public override async Task Init()
    {
        EnsureLoggerWired();
        // Swarm owns the model root. Pin the engine to that resolved directory so side-model files are shared with Comfy.
        KnobStore.Set(EngineKnobs.ModelsRoot, Program.ServerSettings.Paths.ActualModelRoot);
        if (await MaybeAutoUpdateEngine())
        {
            // A staged engine update cannot hot-swap into this process, so fail loudly rather than serve the old version.
            Status = BackendStatus.ERRORED;
            AddLoadStatus("Engine update staged — RESTART SwarmUI to load the new engine. This backend is disabled until you restart.");
            Logs.Warning("[HartsyInference] Backend disabled: a newer engine is staged. Restart SwarmUI to load it (set AutoUpdate=aggressive to auto-restart).");
            return;
        }
        try
        {
            string requested = Settings?.ComputeBackend?.ToLowerInvariant() ?? "auto";
            int? deviceOrdinal = ParseGpuId(Settings?.GPU_ID);
            VramPolicy? vramPolicy = ParseVramSetting(Settings?.LowVram);

            // Kernels live in the extension's output dir. BackendFactory resolves them beside the engine assemblies,
            // so the override is only for operators who moved them.
            if (!string.IsNullOrWhiteSpace(Settings?.KernelDirectory))
            {
                BackendFactory.KernelDirOverride = Settings.KernelDirectory;
            }
            string ptxDir = BackendFactory.KernelDir(BackendFactory.PtxDirName);
            string spvDir = BackendFactory.KernelDir(BackendFactory.SpirvDirName);
            AddLoadStatus($"Kernel paths: PTX={ptxDir} (exists={Directory.Exists(ptxDir)}), SPIR-V={spvDir} (exists={Directory.Exists(spvDir)})");

            // Validate here so a bad device shows ERRORED now, not on the first generation (the engine builds its device lazily).
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

            // Optional DiT sharding: the block loop is split across two GPUs to pool VRAM. Exclusive with CfgParallelGpuId;
            // the Engine rejects a placement with both set, which the outer catch reports as ERRORED.
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

            // One ordinal for the probe, the engine and the device-sharing key, so the three cannot silently
            // disagree about which device this backend is talking about.
            int ordinal = deviceOrdinal ?? 0;

            // ResolveProbed, not Resolve: Resolve only checks that a driver sees a device, which a broken CUDA install still passes.
            // Probing also tries Vulkan, so AMD/Intel boxes land on their GPU. A throw counts as no usable GPU, so the refusal below
            // is reported instead of a raw exception.
            string resolved;
            try
            {
                resolved = BackendFactory.ResolveProbed(BackendFactory.WithOrdinal(requested, ordinal));
            }
            catch (Exception ex)
            {
                Logs.Warning($"[HartsyInference] Backend probe threw, treating as no usable GPU: {ex.Message}");
                resolved = "cpu";
            }

            // The CPU kernels are F32-only: fp8/bf16 checkpoint weights read past the end of their allocation,
            // corrupting the heap and aborting the process. Refuse to come up rather than serve a backend that
            // cannot generate safely. Only an explicit 'cpu' opts into it.
            if (resolved == "cpu" && BackendFactory.Kind(requested) != "cpu")
            {
                Status = BackendStatus.ERRORED;
                // Both reasons are shown: on a non-NVIDIA box the Vulkan reason is the actionable one. Brackets, since neither reason ends in punctuation.
                string msg = $"No usable GPU, so compute='{requested}' resolved to CPU, and this backend refuses to "
                    + "start on CPU. The CPU backend cannot run fp8/bf16 image models (it is F32-only and will "
                    + $"corrupt memory). CUDA: [{CudaReason()}] Vulkan: [{VulkanReason()}] "
                    + "Fix whichever of those is closest to working, or set this backend's ComputeBackend to 'cpu' "
                    + "explicitly to override.";
                AddLoadStatus(msg);
                Logs.Error($"[HartsyInference] Backend #{BackendData?.ID} refusing CPU fallback: {msg}");
                return;
            }
            _resolvedComputeKind = resolved;

            AddLoadStatus($"Constructing HartsyInference.Engine (compute='{requested}' resolved to '{resolved}', device={deviceOrdinal?.ToString() ?? "auto"})...");
            EngineOptions engineOptions = new EngineOptions { VramPolicy = vramPolicy, Placement = placement };
            // The RESOLVED kind, not the raw setting: BackendFactory.Create would re-resolve 'auto' cheaply and could build a device the probe rejected.
            _engine = deviceOrdinal.HasValue
                ? new InferenceEngine(resolved, deviceOrdinal.Value, engineOptions)
                : new InferenceEngine(resolved, engineOptions);
            // Asked, not assumed: a selector with no ordinal lets the engine rank devices, so only this call knows the card.
            // It also builds the backend, so an unusable explicit selector fails here.
            (string Key, string Name) device = EngineDeviceIdentity();
            AddLoadStatus($"Engine ready: {_engine.BackendDescription}{(device.Key is null ? "" : $" on {device.Name ?? device.Key} [{device.Key}]")}");
            Logs.Info($"[HartsyInference] Using '{resolved}' compute on {device.Name ?? device.Key ?? "the engine's default device"}.");
            // Everything besides the device that changes what fits: the tier and where the components run.
            string fitSettings = $"vram={Settings?.LowVram?.Trim().ToLowerInvariant()}|te={teSelector}|vae={vaeSelector}"
                + $"|cfg={cfgParallelSelector}|shard={(enableDitSharding ? ditShardSelector : "")}";
            RegisterDeviceUsage(resolved, ordinal, device, fitSettings);

            // MaxUsages = one live generation + OverQueue waiting slots, as in ComfyUI. OverQueue -1 gives MaxUsages 0 (UI-only).
            // The scheduler routes elsewhere once Usages reaches MaxUsages; _genLock keeps the extra slots queued.
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
            Logs.Debug($"[HartsyInference] Backend #{BackendData?.ID} init failed: {ex}");
            throw;
        }
    }

    /// <summary>Registers this backend's device with <see cref="BackendDeviceRegistry"/>, which memory-aware routing (<see cref="ModelFitGate"/>) reads.</summary>
    /// <remarks>Backends sharing a device must co-fit in its VRAM, and the engine runs their generations on it one at a time.</remarks>
    /// <param name="kind">An ALREADY-RESOLVED backend kind, never <c>auto</c>. Re-resolving here is what this
    /// signature exists to prevent: the cheap <see cref="BackendFactory.Resolve"/> can disagree with the probe
    /// <see cref="Init"/> ran, and this keys the registry, so it would name a device the engine is not on.</param>
    /// <param name="device">What the engine reports it bound to. Key is null when this engine build cannot say, in
    /// which case the selector is all there is to key on; Name is for the warning, since a Vulkan key is a UUID and
    /// nobody can tell which of their cards that is.</param>
    /// <param name="fitSettings">Every setting besides the device that changes a memory verdict, joined into the
    /// backend's <see cref="FitProfileKey"/>.</param>
    private void RegisterDeviceUsage(string kind, int ordinal, (string Key, string Name) device, string fitSettings)
    {
        try
        {
            // Idempotent: Init reruns on this instance when settings change, and the registry is keyed by instance.
            string previousProfile = FitProfileKey;
            ReleaseDeviceUsage();
            if (!BackendFactory.IsDeviceKind(kind))
            {
                return;
            }
            // The engine's key first. The composed fallback uses the factory's grammar, so a device has one spelling.
            string key = device.Key ?? BackendFactory.CanonicalDeviceKey(BackendFactory.WithOrdinal(kind, ordinal));
            FitProfileKey = $"{key}|{fitSettings}";
            // A changed profile forgets its out-of-memory evidence. An unchanged one keeps it, including evidence other backends gathered.
            if (previousProfile is not null && previousProfile != FitProfileKey)
            {
                ModelFitGate.ForgetProfile(previousProfile);
            }
            int live = BackendDeviceRegistry.Register(this,
                new BackendDeviceEntry(key, device.Name, FitProfileKey, fitSettings, _engine));
            if (live > 1)
            {
                // Name the device too: a Vulkan key is a UUID, which does not say which card is shared.
                string named = string.IsNullOrWhiteSpace(device.Name) ? key : $"{device.Name} [{key}]";
                string warning = $"{live} HartsyInference backends now share one physical device: {named}. Their models " +
                    "must co-fit in that GPU's VRAM, and generations on this device run one at a time (the engine " +
                    "serializes same-GPU work; concurrent same-GPU execution is planned). If you gave these backends " +
                    "different GPU_IDs expecting different cards, note that under Vulkan the ordinal is the loader's raw " +
                    "device index and software rasterizers occupy indices, so two different numbers can land on one card.";
                AddLoadStatus($"WARNING: {warning}");
                Logs.Warning($"[HartsyInference] {warning}");
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference] Device-usage registration failed; memory-aware routing may not account for this backend's device: {ex.Message}");
        }
    }

    /// <summary>Drops this backend from <see cref="BackendDeviceRegistry"/>, so routing stops considering its device.</summary>
    private void ReleaseDeviceUsage()
    {
        BackendDeviceRegistry.Release(this);
    }

    /// <summary>The device the engine bound to, as a key and a display name. Both are null when this engine build cannot say.</summary>
    /// <remarks>Reflective, because the property is newer than the compile-time floor (see <see cref="VulkanReason"/>).
    /// The engine ranks devices, so a composed <c>vulkan:0</c> can name a card the backend is not on.
    /// Reading the key builds the backend, so an unusable explicit <c>cuda</c>/<c>vulkan</c> selector fails here with a reason.
    /// The throw is unwrapped for <see cref="Init"/>'s handler.</remarks>
    private (string Key, string Name) EngineDeviceIdentity()
    {
        PropertyInfo prop = _engine?.GetType().GetProperty("DeviceKey", BindingFlags.Public | BindingFlags.Instance);
        if (prop is null)
        {
            // Engine build without DeviceKey: report nothing and keep the backend lazy.
            return (null, null);
        }
        try
        {
            string key = prop.GetValue(_engine) as string;
            // Reading the key above already built the backend, so this does not force construction.
            return (key, _engine.ComputeBackend?.Capabilities?.Name);
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw; // unreachable, and the compiler needs it said
        }
    }

    public override async Task Shutdown()
    {
        Status = BackendStatus.DISABLED;
        // Out of routing before the engine goes: a request being prepared must not ask a disposed engine for a fit.
        ReleaseDeviceUsage();
        _cancelCts?.Cancel();
        _cancelCts = null;
        _engine?.Dispose();
        _engine = null;
        DisposePreprocessBackend();
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
        // Annotator weights live on the borrowed Engine backend, so the engine free covers them. Dropping the reference
        // re-borrows a fresh backend next use, in case the engine rebuilt its device. A standalone device is disposed.
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

    /// <summary>Why the Engine did not use CUDA, as a display string.</summary>
    /// <remarks>Direct references, not reflection: <c>ComputeBackendOptions</c> already names these members at compile time, so reflection here would imply a tolerance this file lacks.</remarks>
    private static string CudaReason()
        => (BackendFactory.CudaUnavailableReason ?? BackendFactory.CudaProbeFailureReason)?.Trim()
            ?? "no reason reported";

    /// <summary>Why the Engine did not use Vulkan, as a display string.</summary>
    /// <remarks>Reflective, because these properties are newer than the compile-time floor (see <see cref="CudaReason"/>), and a direct reference would stop an older engine building.
    /// A missing property is reported as unsupported, not as "no reason reported", which would claim the build looked and found nothing.</remarks>
    private static string VulkanReason()
    {
        PropertyInfo unavailable = typeof(BackendFactory)
            .GetProperty("VulkanUnavailableReason", BindingFlags.Public | BindingFlags.Static);
        if (unavailable is null)
        {
            return "not supported by this engine build; update HartsyInference to 2.0.0-alpha.158 or newer";
        }
        string probeFailure = typeof(BackendFactory)
            .GetProperty("VulkanProbeFailureReason", BindingFlags.Public | BindingFlags.Static)
            ?.GetValue(null) as string;
        return (unavailable.GetValue(null) as string ?? probeFailure)?.Trim() ?? "no reason reported";
    }

    /// <summary>Parses GPU_ID into the device ordinal handed to the Engine. Only the first entry of a comma-separated list is used; null lets the Engine choose.</summary>
    /// <remarks><b>The ordinal is CUDA's, not <c>nvidia-smi</c>'s.</b> CUDA enumerates fastest-first, so on a mixed-GPU host <c>GPU_ID=0</c> may be a different card than <c>nvidia-smi</c> index 0.</remarks>
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

    /// <summary>Maps the backend's VRAM setting to the engine's per-backend policy, so backends on different cards keep their own.</summary>
    /// <remarks>The older <c>auto</c>/<c>on</c>/<c>off</c> spellings still load from a saved <c>Backends.fds</c>: <c>on</c> is Aggressive and <c>off</c> is Performance.
    /// Auto returns null, so the backend inherits the environment's policy.</remarks>
    private static VramPolicy? ParseVramSetting(string mode)
    {
        string normalized = string.IsNullOrWhiteSpace(mode) ? "auto" : mode.Trim().ToLowerInvariant();
        VramTier? tier = normalized switch
        {
            "auto" => null,
            "performance" or "off" or "0" or "false" => VramTier.Performance,
            "balanced" => VramTier.Balanced,
            "aggressive" or "on" or "1" or "true" => VramTier.Aggressive,
            "maximum" or "max" => VramTier.Maximum,
            _ => null,
        };
        if (tier is null && normalized != "auto")
        {
            Logs.Warning($"[HartsyInference] LowVram='{mode}' is not recognized; using Auto. "
                + "Valid: Auto, Performance, Balanced, Aggressive, Maximum.");
            normalized = "auto";
        }
        Logs.Info($"[HartsyInference] VRAM mode: {(tier?.ToString() ?? "Auto")}"
            + (tier == VramTier.Performance ? " (models larger than VRAM will fail rather than stream)." : "."));
        return tier is VramTier resolved ? VramPolicy.For(resolved) : null;
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
            Logs.Debug($"[HartsyInference] LoadModel: architecture '{compat}' is not supported here. Returning false so another backend can handle it.");
            return false;
        }
        // The Engine loads and caches pipelines lazily inside generate, keyed on checkpoint path and LoRA/component identity.
        // A separate load step would only fight its cache.
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
        // Status stays RUNNING throughout: it is the resting healthy state. BackendData.Usages tracks active use.
        //
        // Serialize: when OverQueue > 0 the scheduler may dispatch multiple jobs to us; run them one at a
        // time. Acquire BEFORE touching _cancelCts so the per-gen token belongs to the job holding the GPU.
        await _genLock.WaitAsync();
        // Linked to the input's InterruptToken, which Swarm trips when the Stop button is pressed.
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
            ModelSpec spec = ModelSupport.BuildSpec(model, family, input);
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
                // One retry: a pipeline cached from an earlier, larger model often holds VRAM this request needs, and FreeMemory releases it.
                // A second failure means the request does not fit.
                Logs.Warning($"[HartsyInference] {first.Message} Freeing cached models and retrying once.");
                await FreeMemory(false);
                try
                {
                    outputs = await Dispatch();
                }
                catch (OutOfVramException second)
                {
                    // Too big for this card: record it so routing skips this card and any with no more memory, then let Swarm re-route.
                    if (ModelFitGate.RecordFailure(input, this))
                    {
                        Logs.Warning($"[HartsyInference] {second.Message} Redirecting '{model.Name}' to a GPU with more memory.");
                        throw new PleaseRedirectException();
                    }
                    throw new SwarmReadableErrorException(DescribeVramFailure(second, family));
                }
            }
            catch (OutOfVramException only)
            {
                throw new SwarmReadableErrorException(DescribeVramFailure(only, family));
            }
            catch (NotSupportedException unsupported)
            {
                // Backstop for sampler/schedule refusals ValidateSamplingChoice missed. Some families accept a selection on
                // text-to-image but not on a secondary path (Boogu, OmniGen 2, Flux.1), which the capability table cannot express.
                throw new SwarmUserErrorException($"HartsyInference: {unsupported.Message}");
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
            Logs.Debug($"[HartsyInference] Job batch='{batchId}' complete: {outputs.Length} output(s) in {totalMs}ms.");
        }
        catch (OperationCanceledException)
        {
            Logs.Debug("[HartsyInference] Generation cancelled by user.");
        }
        finally
        {
            _cancelCts?.Dispose();
            _cancelCts = null;
            _genLock.Release(); // hand the GPU to the next queued generation, if any
            // No Status change: the backend stays RUNNING even when this generation failed, since only this request is at fault.
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
        // Passes below run on the engine's output, even when Init Image Creativity 0 returns the init image untouched.
        // Optional SeedVR2 restore pass. No FreeMemory here: RestoreService frees on pipeline load, and evicting cached models every generation would defeat the cache.
        if (input.TryGet(HartsyInference.RestoreModelParam, out string restoreModel)
            && !string.IsNullOrWhiteSpace(restoreModel))
        {
            (rgb, width, height) = await RestoreStill(restoreModel, rgb, width, height, BuildRestoreKnobs(input), progress, cancel);
        }
        // Core's Refiner Upscale with no refiner model is a plain enlargement of the finished image (Comfy reads
        // that combination the same way), so it runs here as a Real-ESRGAN or SeedVR2 pixel-space pass. With a
        // refiner model the engine's PostApply hand-off owns the resize instead (BuildRefiner passes Upscale through).
        if (input.Get(T2IParamTypes.RefinerModel) is null
            && input.TryGet(T2IParamTypes.RefinerUpscale, out double upscaleFactor) && Math.Abs(upscaleFactor - 1.0) > 1e-6)
        {
            (rgb, width, height) = await UpscaleStill(input, upscaleFactor, rgb, width, height, progress, cancel);
        }
        // Engine-side metadata (resolved arch/seed/steps) folds into Swarm's saved-image metadata; "arch" in
        // particular is checkpoint-sniffed by the engine and not derivable Swarm-side.
        foreach (KeyValuePair<string, string> meta in result.Meta)
        {
            input.ExtraMeta[$"hartsy_{meta.Key}"] = meta.Value;
        }
        input.ExtraMeta["hartsy_engine_seed"] = result.Seed;
        // Core's Remove Background runs last, at the final resolution, and yields a real RGBA cutout: the engine's
        // RMBG-1.4 matte over the ORIGINAL colours (its gray composite is for the image→3D preprocessors, not here).
        if (input.Get(T2IParamTypes.RemoveBackground, false))
        {
            byte[] alpha = await RemoveBackground(rgb, width, height, cancel);
            return RgbToImage.FromHwcRgba(rgb, alpha, width, height);
        }
        return RgbToImage.FromHwcRgb(rgb, width, height);
    }

    // ─────────────────────────── Post-generation passes (restore / upscale / cutout) ───────────────────────────

    /// <summary>The Refiner Upscale Method value that routes the pixel-space pass to SeedVR2 instead of Real-ESRGAN.</summary>
    private const string SeedVr2UpscaleMethod = "seedvr2";

    /// <summary>Resolves a catalog model for an auxiliary pass (SeedVR2, Real-ESRGAN, RMBG), fetching any missing assets on first use.</summary>
    /// <remarks>The engine never downloads at generate time, so a missing id, a failed download or a gated repo (which needs <c>HF_TOKEN</c>) becomes one readable error.</remarks>
    private static async Task<ModelSpec> ResolveAuxModel(string id, Modality modality, CancellationToken cancel)
    {
        ModelSpec spec = ModelResolver.Resolve(id, null, modality);
        if (spec.Catalog is null)
        {
            if (spec.LocalPath is not null)
            {
                return spec;
            }
            throw new SwarmReadableErrorException(
                $"HartsyInference: '{id}' is neither a catalog model id nor a checkpoint path on this machine.");
        }
        IReadOnlyList<ModelAsset> missing = ModelDownloader.MissingAssets(spec.Catalog);
        if (missing.Count == 0)
        {
            return spec;
        }
        string files = string.Join(", ", missing.Select(a => $"{a.Repo}/{a.RepoPath}"));
        Logs.Info($"[HartsyInference] {spec.Catalog.DisplayName}: fetching {missing.Count} file(s) on first use ({files}).");
        try
        {
            Dictionary<string, int> lastBucket = new();
            await ModelDownloader.DownloadAsync(missing, (asset, fraction) =>
            {
                int bucket = (int)(fraction * 4);
                if (bucket > lastBucket.GetValueOrDefault(asset.FileName, -1))
                {
                    lastBucket[asset.FileName] = bucket;
                    Logs.Debug($"[HartsyInference] Downloading '{asset.FileName}': {fraction * 100:0}%");
                }
            }, cancel);
            Logs.Info($"[HartsyInference] {spec.Catalog.DisplayName}: finished fetching {missing.Count} file(s).");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            throw new SwarmReadableErrorException(
                $"HartsyInference: could not fetch {spec.Catalog.DisplayName} ({files}): {ex.Message}\n\n"
                + $"Place the file(s) under '{Path.Combine(RepoPaths.ModelsRoot(), missing[0].TargetSubdir)}' by hand, "
                + "or set HF_TOKEN in SwarmUI's environment if the repo is gated or private.");
        }
        return ModelResolver.Resolve(id, null, modality);
    }

    /// <summary>SeedVR2 over one still, with the engine's failure text kept readable (the HTTP API otherwise shows "Something went wrong").</summary>
    private async Task<(byte[] Rgb, int Width, int Height)> RestoreStill(string restoreModel, byte[] rgb, int width, int height,
        RestoreRequest knobs, IProgress<StepPreview> progress, CancellationToken cancel)
    {
        ModelSpec restoreSpec = await ResolveAuxModel(restoreModel, Modality.Restore, cancel);
        RestoreRequest restoreRequest = knobs with
        {
            Image = new EngineImage { Rgb = rgb, Width = width, Height = height },
        };
        try
        {
            await foreach (VideoFrame restored in _engine.Restore.RestoreAsync(restoreSpec, restoreRequest, progress, cancel))
            {
                rgb = restored.Rgb;
                width = restored.Width;
                height = restored.Height;
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not OutOfVramException and not SwarmReadableErrorException)
        {
            throw new SwarmReadableErrorException($"HartsyInference: the SeedVR2 restore pass ({restoreModel}) failed: {ex.Message}");
        }
        return (rgb, width, height);
    }

    /// <summary>Refiner Upscale with no refiner model: enlarges the finished still by <paramref name="factor"/> in pixel space.</summary>
    /// <remarks>The Refiner Upscale Method picks the engine (<see cref="ResolveUpscaleModel"/>). Real-ESRGAN runs model passes then resizes down, while SeedVR2 treats the size as an area target.</remarks>
    private async Task<(byte[] Rgb, int Width, int Height)> UpscaleStill(T2IParamInput input, double factor, byte[] rgb, int width, int height,
        IProgress<StepPreview> progress, CancellationToken cancel)
    {
        int targetWidth = Math.Max(1, (int)Math.Round(width * factor));
        int targetHeight = Math.Max(1, (int)Math.Round(height * factor));
        if (factor < 1.0)
        {
            return (RgbToImage.ResizeHwcRgb(rgb, width, height, targetWidth, targetHeight), targetWidth, targetHeight);
        }
        input.TryGet(ComfyUIBackendExtension.RefinerUpscaleMethod, out string method);
        string model = ResolveUpscaleModel(method, factor);
        Logs.Debug($"[HartsyInference] Refiner upscale x{factor}: {width}x{height} to {targetWidth}x{targetHeight} via '{model}' (method '{method ?? "default"}').");
        if (model == SeedVr2UpscaleMethod)
        {
            string restoreModel = input.TryGet(HartsyInference.RestoreModelParam, out string picked) && !string.IsNullOrWhiteSpace(picked)
                ? picked
                : "seedvr2-3b";
            RestoreRequest knobs = BuildRestoreKnobs(input) with { TargetWidth = targetWidth, TargetHeight = targetHeight };
            return await RestoreStill(restoreModel, rgb, width, height, knobs, progress, cancel);
        }
        ModelSpec spec = await ResolveAuxModel(model, Modality.Vision, cancel);
        VisionResult result = await RunVision(spec, new VisionRequest
        {
            Image = new EngineImage { Rgb = rgb, Width = width, Height = height },
            Mode = VisionMode.Upscale,
            TargetWidth = targetWidth,
            TargetHeight = targetHeight,
        }, "Real-ESRGAN upscale", cancel);
        EngineImage up = result.Image
            ?? throw new SwarmReadableErrorException("HartsyInference: the Real-ESRGAN upscale pass returned no image.");
        return (up.Rgb, up.Width, up.Height);
    }

    /// <summary>Maps Comfy's Refiner Upscale Method onto the pass this backend runs.</summary>
    /// <remarks>Unset and <c>pixel-*</c> become Real-ESRGAN (x2plus up to 2x, x4plus above, since a model pass beats a resampler).
    /// <c>real-esrgan-*</c> and <c>seedvr2</c> are used as given. A <c>model-&lt;file&gt;</c> picks the nearest Real-ESRGAN variant by name,
    /// as Comfy's ESRGAN architectures are not loadable here. <c>latent-*</c> never reaches this: <see cref="ValidateImageFeatures"/> refuses it without a refiner model.</remarks>
    private static string ResolveUpscaleModel(string method, double factor)
    {
        string m = (method ?? "").Trim().ToLowerInvariant();
        if (m.Contains("seedvr"))
        {
            return SeedVr2UpscaleMethod;
        }
        if (m.StartsWith("real-esrgan-", StringComparison.Ordinal))
        {
            return m;
        }
        if (m.StartsWith("model-", StringComparison.Ordinal))
        {
            if (m.Contains("anime"))
            {
                return "real-esrgan-anime6b";
            }
            return m.Contains("x2") || m.Contains("2x") ? "real-esrgan-x2plus" : "real-esrgan-x4plus";
        }
        return factor <= 2.0 + 1e-6 ? "real-esrgan-x2plus" : "real-esrgan-x4plus";
    }

    /// <summary>RMBG-1.4 matte for the finished still. Only the alpha plane of the engine's result is used.</summary>
    /// <remarks>Needs engine 2.0.0-alpha.56 or newer.</remarks>
    private async Task<byte[]> RemoveBackground(byte[] rgb, int width, int height, CancellationToken cancel)
    {
        ModelSpec spec = await ResolveAuxModel("rmbg", Modality.Vision, cancel);
        VisionResult result = await RunVision(spec, new VisionRequest
        {
            Image = new EngineImage { Rgb = rgb, Width = width, Height = height },
            Mode = VisionMode.BackgroundRemoval,
        }, "background removal", cancel);
        if (result.Image?.Alpha is not { } alpha || alpha.Length != width * height)
        {
            throw new SwarmReadableErrorException(
                "HartsyInference: background removal returned no alpha plane — the engine build predates 2.0.0-alpha.56.");
        }
        return alpha;
    }

    /// <summary>One vision call with readable failure text. VRAM and cancellation errors pass through so the retry and cancel paths still see them.</summary>
    private async Task<VisionResult> RunVision(ModelSpec spec, VisionRequest request, string label, CancellationToken cancel)
    {
        try
        {
            return await _engine.Vision.RunAsync(spec, request, cancel);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not OutOfVramException and not SwarmReadableErrorException)
        {
            throw new SwarmReadableErrorException($"HartsyInference: the {label} pass failed: {ex.Message}");
        }
    }

    /// <summary>The restore knobs shared by the still and video paths (target size, strength, seed).</summary>
    private static RestoreRequest BuildRestoreKnobs(T2IParamInput input) => new()
    {
        TargetWidth = input.TryGet(HartsyInference.RestoreWidthParam, out int rw) ? rw : null,
        TargetHeight = input.TryGet(HartsyInference.RestoreHeightParam, out int rh) ? rh : null,
        Strength = input.TryGet(HartsyInference.RestoreStrengthParam, out double rst) ? (float)rst : null,
        Seed = input.Get(T2IParamTypes.Seed, -1L) is long seed and >= 0 ? (int)(seed & 0x7FFFFFFF) : null,
    };

    /// <summary>Runs a video generation, collecting the streamed frames and muxing them into a container.</summary>
    private async Task<Image> GenerateVideo(ModelSpec spec, T2IParamInput input, IProgress<StepPreview> progress, CancellationToken cancel)
    {
        VideoRequest request = BuildVideoRequest(input);
        string format = input.Get(T2IParamTypes.VideoFormat, "h264-mp4");

        // Stream frames straight to ffmpeg when nothing needs the whole clip in memory (no boomerang, trim, restore or audio).
        // Families that cannot stream throw NotSupportedException, and those requests fall back to buffering.
        bool wantsBoomerang = input.Get(T2IParamTypes.VideoBoomerang, false);
        bool wantsTrim = input.Get(T2IParamTypes.TrimVideoStartFrames, 0) > 0 || input.Get(T2IParamTypes.TrimVideoEndFrames, 0) > 0;
        bool wantsRestore = input.TryGet(HartsyInference.RestoreModelParam, out string restoreModelCheck) && !string.IsNullOrWhiteSpace(restoreModelCheck);
        bool wantsAudio = input.Get(T2IParamTypes.VideoAudioInput) is not null;
        // No generated soundtrack either: streaming has no AudioTrack, so one would be dropped. Core's HasJointAVLatents names those families.
        bool generatesAudio = input.Get(T2IParamTypes.Model)?.ModelClass?.CompatClass?.HasJointAVLatents == true;
        if (!wantsBoomerang && !wantsTrim && !wantsRestore && !wantsAudio && !generatesAudio)
        {
            try
            {
                return await GenerateVideoStreaming(spec, request, format, progress, cancel);
            }
            catch (NotSupportedException ex)
            {
                Logs.Debug($"[HartsyInference] Streaming generation not available ({ex.Message}); falling back to buffered.");
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
        // Boomerang is applied by the engine (VideoRecipeUtils.ToVideoFrames, via VideoRequest.VideoBoomerang), so it is not applied again here.
        // Optional SeedVR2 restore pass: frames go straight to the restore service, with no container round-trip.
        if (input.TryGet(HartsyInference.RestoreModelParam, out string restoreModel)
            && !string.IsNullOrWhiteSpace(restoreModel))
        {
            // The resident T2V DiT and SeedVR2's VAE peak cannot share 24 GB.
            _engine.FreeMemory();
            ModelSpec restoreSpec = await ResolveAuxModel(restoreModel, Modality.Restore, cancel);
            RestoreRequest restoreRequest = BuildRestoreKnobs(input) with
            {
                Frames = [.. frames.Select(f => new ImageData { Rgb = f, Width = width, Height = height })],
                ClipFrames = input.TryGet(HartsyInference.RestoreClipFramesParam, out int rcf) ? rcf : 5,
                Overlap = input.TryGet(HartsyInference.RestoreOverlapParam, out int rov) ? rov : 1,
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
            Logs.Warning($"[HartsyInference] Video format '{format}' cannot carry audio, so the {audio.SampleRate} Hz track is dropped. Choose 'h264-mp4', 'h265-mp4', 'prores' or 'webm' to keep it.");
            audio = null;
        }
        return VideoOutputEncoder.Encode([.. frames], width, height, request.Fps ?? 25, format, cancel, audio);
    }

    /// <summary>The streaming path of <see cref="GenerateVideo"/>: pipes frames from <c>IVideoService.GenerateFramesAsync</c> into <see cref="VideoOutputEncoder.EncodeStreamingAsync"/> as they decode.</summary>
    /// <remarks>Peeks one frame for width/height, since the request's own size is snapped per family. Callers only reach this with no audio track, matching the GenerateVideo gate.</remarks>
    private async Task<Image> GenerateVideoStreaming(ModelSpec spec, VideoRequest request, string format, IProgress<StepPreview> progress, CancellationToken cancel)
    {
        IAsyncEnumerator<VideoFrame> frames = _engine.Video.GenerateFramesAsync(spec, request, progress, cancel).GetAsyncEnumerator(cancel);
        if (!await frames.MoveNextAsync())
        {
            await frames.DisposeAsync();
            throw new InvalidOperationException("HartsyInference: the video pipeline produced no frames.");
        }
        VideoFrame first = frames.Current;
        // RgbFramesFrom takes ownership of `frames`: its finally disposes it once the encoder drains or unwinds the stream.
        return await VideoOutputEncoder.EncodeStreamingAsync(
            RgbFramesFrom(first, frames), first.Width, first.Height, request.Fps ?? 25, format, cancel);
    }

    /// <summary>Yields <paramref name="first"/>'s RGB bytes, then the rest of <paramref name="remaining"/>: the peek-and-reprepend glue for <see cref="GenerateVideoStreaming"/>.</summary>
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

    /// <summary>Runs a music generation. The Engine returns an encoded WAV, carried as an <see cref="Image"/> keyed by <see cref="MediaType.AudioWav"/>.</summary>
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

    /// <summary>Bridges the Engine's <see cref="StepPreview"/> ticks onto Swarm's <c>takeOutput</c> progress contract, and logs a progress bar at each 5% step.</summary>
    /// <remarks>Each tick sends a <c>{batch_index, overall_percent, current_percent}</c> JObject, or the richer preview JObject when the Engine supplied pixels.</remarks>
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
            // Send the preview JObject when one was encoded (preview and percent in one message, as Comfy does), else just the percent.
            takeOutput(previewObj ?? new JObject
            {
                ["batch_index"] = batchId,
                ["overall_percent"] = overall,
                ["current_percent"] = overall,
            });
        });
    }

    /// <summary>An <see cref="IProgress{T}"/> that runs on the reporting thread. <see cref="Progress{T}"/> would reorder or drop sampler ticks.</summary>
    private sealed class InlineProgress<T>(Action<T> handler) : IProgress<T>
    {
        private readonly Action<T> _handler = handler;

        public void Report(T value) => _handler(value);
    }

    /// <summary>Renders a text progress bar for log lines, such as <c>[████████░░░░░░░░░░░░] 40.0%</c>, 20 cells wide so each cell is 5%.</summary>
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
    // T2IParamTypes values are read only in this section. Each mapper produces one Engine request record; anything the flat
    // contract does not name rides in Extra under a documented key. Params we cannot express are refused in IsValidForThisBackend.

    /// <summary>Maps the Swarm request onto the Engine's <see cref="ImageRequest"/>.</summary>
    private ImageRequest BuildImageRequest(T2IParamInput input, ModelSupport.Family family)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        if (family.Id == "ideogram4" && input.Get(HartsyInference.Ideogram4MagicPromptParam, false))
        {
            // Ideogram 4 is trained on structured JSON captions; the expander rewrites a plain prompt through a
            // running LLM backend. Opt-in, and a no-op that returns the prompt unchanged when unavailable.
            prompt = Ideogram4MagicPrompt.Expand(
                prompt,
                input.Get(T2IParamTypes.Width, DefaultImageSize),
                input.Get(T2IParamTypes.Height, DefaultImageSize),
                input.Get(HartsyInference.Ideogram4MagicPromptModelParam, ""),
                input.SourceSession,
                msg => Logs.Verbose($"[HartsyInference][Ideogram4] {msg}"));
        }
        Image initImage = input.Get(T2IParamTypes.InitImage);
        Image maskImage = input.Get(T2IParamTypes.MaskImage);
        // For reference-edit recipes, prompt images are the references: the Init Image comes first, then the prompt images in prompt order.
        bool refEditFamily = (ModelSupport.SupportedFeatures(input.Get(T2IParamTypes.Model)) & ImageFeatures.RefEdit) != 0;
        IReadOnlyList<ControlNetConditioning> controlNets = BuildControlNets(input, out List<(int Index, string UnionType)> unionTypes);
        return new ImageRequest
        {
            Prompt = prompt,
            NegativePrompt = input.Get(T2IParamTypes.NegativePrompt),
            Width = NullableInt(input, T2IParamTypes.Width),
            Height = NullableInt(input, T2IParamTypes.Height),
            Steps = NullableInt(input, T2IParamTypes.Steps),
            // Core's toggle; the Engine also reads it, to gate Krea 2's joint-attention prompt-weight patch.
            ModelSpecificEnhancements = input.Get(T2IParamTypes.ModelSpecificEnhancements, true),
            CfgScale = input.TryGet(T2IParamTypes.CFGScale, out double cfg) ? (float)cfg : null,
            CfgRescale = input.TryGet(HartsyInference.CfgRescaleParam, out double cfgRescale) ? (float)cfgRescale : null,
            // Comfy's TCFG param (same paper, arXiv 2503.18137, same boolean). ValidateComfyOnlyParams passes it because "usetcfg" is in HonoredComfyParams.
            Tcfg = input.TryGet(ComfyUIBackendExtension.UseTCFG, out bool tcfg) ? tcfg : null,
            SeamlessTiling = input.TryGet(T2IParamTypes.SeamlessTileable, out string seamless) ? seamless : null,
            Seed = input.Get(T2IParamTypes.Seed, -1L),
            ClipSkip = NullableInt(input, T2IParamTypes.ClipStopAtLayer),
            // Comfy's Sampler and Scheduler are two orthogonal selections; the Engine combines them (dpmpp_2m + karras gives dpmpp_2m_karras).
            // ValidateSamplingChoice refuses unsupported values before generation, so the request can route to Comfy.
            Sampler = input.Get(ComfyUIBackendExtension.SamplerParam, null),
            Scheduler = input.Get(ComfyUIBackendExtension.SchedulerParam, null),
            SigmaShift = input.TryGet(T2IParamTypes.SigmaShift, out double shift) ? shift : null,
            EndStepsEarly = input.TryGet(T2IParamTypes.EndStepsEarly, out double endEarly) ? endEarly : null,
            InstructPix2PixCfg = input.TryGet(T2IParamTypes.IP2PCFG2, out double ip2p) ? ip2p : null,
            Batch = 1, // Swarm drives batching itself: one Generate call per image
            Components = BuildComponents(input),
            Loras = BuildLoras(input),
            ControlNets = controlNets,
            IpAdapter = refEditFamily ? null : BuildIpAdapter(input),
            ReferenceImages = refEditFamily ? BuildReferenceImages(input) : null,
            Refiner = BuildRefiner(input),
            Img2Img = initImage is null ? null : new Img2Img
            {
                InitImage = ToEngineImage(initImage),
                Creativity = input.Get(T2IParamTypes.InitImageCreativity, 0.6),
                Mode = input.Get(HartsyInference.InitImageModeParam, "auto") switch
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
                // Shrink Grow is a toggle: on at 0 crops tight to the mask, which ShrinkGrow alone cannot say.
                CropToMask = input.TryGet(T2IParamTypes.MaskShrinkGrow, out int _),
                RecompositeMask = input.Get(T2IParamTypes.InitImageRecompositeMask, true),
            },
            MaskCompositeUnthresholded = input.Get(T2IParamTypes.MaskCompositeUnthresholded, false),
            Regional = BuildRegional(input, prompt),
            VariationSeed = BuildVariationSeed(input),
            Extra = BuildImageExtra(input, family, initImage, unionTypes),
            Vram = MapVramOverrides(input),
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

    /// <summary>Maps the Loras, LoraWeights, LoraTencWeights and LoraSectionConfinement lists onto the Engine's <see cref="LoraStack"/>.</summary>
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

    /// <summary>Maps Swarm's ControlNet slots onto the Engine's conditioning list, running the matching annotator over each hint image first.</summary>
    /// <remarks>Annotation stays host-side: the Engine expects preprocessed hints, and the annotators consume SwarmUI images and fetch weights through Swarm.</remarks>
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
            global::HartsyInference.Diffusion.Adapters.ControlNetMode mode = ControlNetPreprocessing.DetectMode(cnModel.RawFilePath);
            AddLoadStatus($"ControlNet '{cnModel.Name}' → {mode} preprocessing.");
            EngineImage annotated = ControlNetPreprocessing.Preprocess(
                mode, hint,
                input.Get(T2IParamTypes.Width, DefaultImageSize), input.Get(T2IParamTypes.Height, DefaultImageSize),
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

    /// <summary>The engine's union-control token for a detected preprocessing mode. Each must be one that <c>ControlNetResolver.ResolveUnionType</c> recognizes.</summary>
    private static string UnionTypeString(global::HartsyInference.Diffusion.Adapters.ControlNetMode mode) => mode switch
    {
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.Depth => "depth",
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.OpenPose => "openpose",
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.Scribble => "softedge",
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.SoftEdge => "softedge",
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.Normal => "normal",
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.Segmentation => "segment",
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.Tile => "tile",
        global::HartsyInference.Diffusion.Adapters.ControlNetMode.Inpaint => "repaint",
        // Canny + LineArt both drive the thin-line head.
        _ => "canny",
    };

    /// <summary>Maps the image-prompt inputs onto the Engine's IP-Adapter conditioning. The adapter model rides in Extra under <c>ipadapter.model</c>.</summary>
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
            FaceIdV2Weight = input.TryGet(HartsyInference.FaceIdV2WeightParam, out double faceId) ? faceId : null,
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
            // Swarm knows every model's family from its compat class, so any model can act as a refiner. The engine's header sniff covers only classic architectures.
            FamilyId = ModelSupport.Resolve(refinerModel.ModelClass?.CompatClass?.ID)?.Id,
            Vae = ModelPath(input.Get(T2IParamTypes.RefinerVAE)),
            Method = input.Get(T2IParamTypes.RefinerMethod, null),
            Control = input.TryGet(T2IParamTypes.RefinerControl, out double control) ? control : null,
            Steps = NullableInt(input, T2IParamTypes.RefinerSteps),
            CfgScale = input.TryGet(T2IParamTypes.RefinerCFGScale, out double refinerCfg) ? refinerCfg : null,
            Upscale = input.TryGet(T2IParamTypes.RefinerUpscale, out double upscale) && upscale > 1e-6 ? upscale : null,
        };
    }

    /// <summary>Regional prompting: the Engine parses the raw prompt as the plan, plus mask and per-segment overrides.</summary>
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
            // Same fallbacks as core's Comfy path: these params are absent (not 0) when their toggle is off.
            MaskGrow = input.Get(T2IParamTypes.SegmentMaskGrow, 16),
            MaskBlur = input.Get(T2IParamTypes.SegmentMaskBlur, 10),
            // Exact so a toggled-on 0 crops tight, as in core; absent means core's default of 16.
            MaskOversize = input.Get(T2IParamTypes.SegmentMaskOversize, 16),
            ExactMaskOversize = true,
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

    /// <summary>Everything the Engine's flat image contract does not name: its documented Extra keys plus this extension's custom params.</summary>
    private IReadOnlyDictionary<string, object> BuildImageExtra(T2IParamInput input, ModelSupport.Family family, Image initImage,
        List<(int Index, string UnionType)> controlNetUnionTypes)
    {
        Dictionary<string, object> extra = new(StringComparer.Ordinal);
        // Engine-documented keys (Features/HartsyInference.Engine.Features.RequestExtras.cs).
        if (T2IParamTypes.TryGetType("useipadapter", out T2IParamType ipaType, input)
            && input.TryGetRaw(ipaType, out object ipaRaw)
            && ipaRaw is string ipaModel && !string.IsNullOrWhiteSpace(ipaModel) && ipaModel != "None")
        {
            extra[global::HartsyInference.Engine.Features.RequestExtras.IpAdapterModel] = ipaModel;
        }
        if (input.Get(T2IParamTypes.ClipVisionModel) is T2IModel clipVision)
        {
            extra[global::HartsyInference.Engine.Features.RequestExtras.IpAdapterClipVision] = clipVision.RawFilePath;
        }
        // IP-Adapter scheduling/weighting: forward Comfy's own params (we don't re-register them).
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterWeight, out double ipaWeight))
        {
            extra[global::HartsyInference.Engine.Features.RequestExtras.IpAdapterWeight] = ipaWeight;
        }
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterStart, out double ipaStart))
        {
            extra[global::HartsyInference.Engine.Features.RequestExtras.IpAdapterStart] = ipaStart;
        }
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterEnd, out double ipaEnd))
        {
            extra[global::HartsyInference.Engine.Features.RequestExtras.IpAdapterEnd] = ipaEnd;
        }
        if (input.TryGet(ComfyUIBackendExtension.IPAdapterWeightType, out string ipaWeightType)
            && !string.IsNullOrWhiteSpace(ipaWeightType))
        {
            // Comfy live-appends "name///name (New)" variants; the engine wants the bare name and safely
            // falls back to "standard" for anything it doesn't recognize.
            int marker = ipaWeightType.IndexOf("///", StringComparison.Ordinal);
            extra[global::HartsyInference.Engine.Features.RequestExtras.IpAdapterWeightType] = marker < 0 ? ipaWeightType : ipaWeightType[..marker];
        }
        foreach ((int index, string unionType) in controlNetUnionTypes)
        {
            extra[global::HartsyInference.Engine.Features.RequestExtras.ControlNetUnionTypePrefix + index.ToString(CultureInfo.InvariantCulture)] = unionType;
        }
        // FLUX.1 Redux: Comfy's style-model params map onto the engine's redux.* keys. The model rides by name, resolved from the style_models folders.
        if (input.TryGet(ComfyUIBackendExtension.UseStyleModel, out string styleModel)
            && !string.IsNullOrWhiteSpace(styleModel) && styleModel != "None")
        {
            extra[global::HartsyInference.Engine.Features.RequestExtras.ReduxStyleModel] = styleModel;
            if (input.TryGet(ComfyUIBackendExtension.StyleModelMultiplyStrength, out double reduxMultiply))
            {
                extra[global::HartsyInference.Engine.Features.RequestExtras.ReduxMultiply] = reduxMultiply;
            }
            if (input.TryGet(ComfyUIBackendExtension.StyleModelMergeStrength, out double reduxMerge))
            {
                extra[global::HartsyInference.Engine.Features.RequestExtras.ReduxMerge] = reduxMerge;
            }
            if (input.TryGet(ComfyUIBackendExtension.StyleModelApplyStart, out double reduxApplyStart))
            {
                extra[global::HartsyInference.Engine.Features.RequestExtras.ReduxApplyStart] = reduxApplyStart;
            }
        }
        // Extension-registered custom params.
        if (input.TryGet(HartsyInference.Ideogram4MagicPromptParam, out bool magic))
        {
            extra["hartsy.ideogram4_magic_prompt"] = magic;
        }
        if (input.TryGet(HartsyInference.Ideogram4MagicPromptModelParam, out string magicModel) && !string.IsNullOrWhiteSpace(magicModel))
        {
            extra["hartsy.ideogram4_magic_prompt_model"] = magicModel;
        }
        // FLUX.1 Canny/Depth are plain Flux.1 checkpoints with a wider x_embedder, so there is no ControlNet slot to key preprocessing off.
        // The engine only learns the variant after loading weights, so a filename heuristic on the main model stands in, as in
        // ControlNetPreprocessing.DetectMode. A wrong guess fails with FluxPipeline's own error, not silent corruption. Fill is not matched: it uses Img2Img + Inpaint.
        if (family.Id == "flux1" && initImage is not null)
        {
            string modelPath = input.Get(T2IParamTypes.Model)?.RawFilePath ?? "";
            string modelName = System.IO.Path.GetFileNameWithoutExtension(modelPath).ToLowerInvariant();
            global::HartsyInference.Diffusion.Adapters.ControlNetMode? toolsMode = modelName.Contains("canny") ? global::HartsyInference.Diffusion.Adapters.ControlNetMode.Canny
                : modelName.Contains("depth") ? global::HartsyInference.Diffusion.Adapters.ControlNetMode.Depth
                : null;
            if (toolsMode is not null)
            {
                AddLoadStatus($"FLUX.1 Tools checkpoint detected ('{modelName}') → {toolsMode} preprocessing.");
                EngineImage annotated = ControlNetPreprocessing.Preprocess(
                    toolsMode.Value, initImage,
                    input.Get(T2IParamTypes.Width, DefaultImageSize), input.Get(T2IParamTypes.Height, DefaultImageSize),
                    PreprocessBackend, msg => AddLoadStatus(msg));
                extra[global::HartsyInference.Engine.Features.RequestExtras.FluxToolsControlImage] = annotated;
            }
        }
        return extra;
    }

    /// <summary>Reads the video end image, by name as the Swarm field name varies across versions.</summary>
    private static Image GetVideoEndImage(T2IParamInput input)
    {
        if (T2IParamTypes.TryGetType("videoendimage", out T2IParamType type, input) && input.TryGetRaw(type, out object raw) && raw is Image endImage)
        {
            return endImage;
        }
        return null;
    }

    /// <summary>Maps the Swarm request onto the Engine's <see cref="VideoRequest"/>.</summary>
    private static VideoRequest BuildVideoRequest(T2IParamInput input)
    {
        Image initImage = input.Get(T2IParamTypes.InitImage);
        Image endFrame = GetVideoEndImage(input);
        int? frames = ResolveFrames(input);
        RefuseIncompatibleH3Conditioning(input, initImage, endFrame);
        (int? chainTotal, int chainContext) = ResolveH3Chain(input, frames);
        Dictionary<string, object> extra = new(StringComparer.Ordinal);
        Image reference = input.Get(HartsyInference.AnimateReferenceImageParam);
        // Comfy parity: a prompt-attached image is the Animate character reference, as core's WanPhantomSubjectToVideo workflow
        // treats PromptImages. The explicit param wins when both are given.
        bool isAnimate = IsWanAnimate(input);
        if (isAnimate && input.TryGet(T2IParamTypes.PromptImages, out List<Image> animateRefs) && animateRefs is { Count: > 0 })
        {
            if (animateRefs.Count > 1)
            {
                throw new SwarmUserErrorException(
                    $"Wan-Animate takes a single character reference image, but {animateRefs.Count} images are "
                    + "attached to the prompt. Remove the extras (or use the Animate Reference Image param).");
            }
            if (reference is null)
            {
                reference = animateRefs[0];
            }
            else
            {
                Logs.Warning("[HartsyInference] Both the Animate Reference Image param and a prompt-attached image "
                    + "are set; the param wins and the prompt-attached image is ignored. Clear one of them to avoid this warning.");
            }
        }
        if (reference is not null)
        {
            extra[global::HartsyInference.Engine.Recipes.Video.WanAnimateRecipePipeline.ReferenceImageKey] = ToEngineImage(reference);
        }
        Image poseVideo = input.Get(HartsyInference.AnimatePoseVideoParam);
        Image faceVideo = input.Get(HartsyInference.AnimateFaceVideoParam);
        // With Animate intent, a video Init Image drives motion across all frames; a still uses the engine's tile-across-frames fallback.
        // Without Animate intent, Init stays a plain start frame.
        bool animateIntent = reference is not null || poseVideo is not null || faceVideo is not null;
        bool initIsVideo = initImage?.Type?.MetaType == MediaMetaType.Video;
        bool initDrives = animateIntent && initIsVideo;
        return new VideoRequest
        {
            Prompt = input.Get(T2IParamTypes.Prompt) ?? "",
            NegativePrompt = input.Get(T2IParamTypes.NegativePrompt),
            Width = input.Get(T2IParamTypes.Width, DefaultVideoWidth),
            Height = input.Get(T2IParamTypes.Height, DefaultVideoHeight),
            Steps = NullableInt(input, T2IParamTypes.VideoSteps) ?? NullableInt(input, T2IParamTypes.Steps),
            CfgScale = input.TryGet(T2IParamTypes.VideoCFG, out double videoCfg) ? (float)videoCfg
                : input.TryGet(T2IParamTypes.CFGScale, out double baseCfg) ? (float)baseCfg : null,
            Seed = input.Get(T2IParamTypes.Seed, -1L),
            // Same two orthogonal selections as the image path. The video families that cannot honor them refuse by
            // name in the Engine; ValidateSamplingChoice catches that here first so the request can route instead.
            Sampler = input.Get(ComfyUIBackendExtension.SamplerParam, null),
            Scheduler = input.Get(ComfyUIBackendExtension.SchedulerParam, null),
            InitImage = initImage is null || initDrives ? null : ToEngineImage(initImage),
            DrivingVideo = initDrives ? ToVideoClip(initImage) : null,
            DrivingPoseVideo = poseVideo is null ? null : ToVideoClip(poseVideo),
            DrivingBackgroundVideo = input.Get(HartsyInference.AnimateBackgroundVideoParam) is Image bg ? ToVideoClip(bg) : null,
            DrivingMaskVideo = input.Get(HartsyInference.AnimateMaskVideoParam) is Image maskVid ? ToVideoClip(maskVid) : null,
            DrivingFaceVideo = faceVideo is null ? null : ToVideoClip(faceVideo),
            DrivingAutoPreprocess = input.Get(HartsyInference.AnimateAutoPreprocessParam, true),
            // 0, unset, or at-or-below the chunk length all mean one chunk. The engine floors the total at the chunk length, so this only adds chunks.
            AnimateTotalFrames = input.TryGet(HartsyInference.AnimateTotalFramesParam, out int animateTotal)
                && animateTotal > 0 ? animateTotal : null,
            AnimateContinueMotionFrames = input.Get(HartsyInference.AnimateMotionContextFramesParam, 5),
            AnimateColorCorrection = input.TryGet(HartsyInference.AnimateColorCorrectionParam, out double animateColorFix)
                ? animateColorFix : null,
            AnimatePoseStrength = input.TryGet(HartsyInference.AnimatePoseStrengthParam, out double animatePose)
                ? animatePose : null,
            AnimateReferenceImageStrength = input.TryGet(HartsyInference.AnimateReferenceStrengthParam, out double animateRefStrength)
                ? animateRefStrength : null,
            VideoModel = ModelPath(input.Get(T2IParamTypes.VideoModel)),
            VideoSwapModel = ModelPath(input.Get(T2IParamTypes.VideoSwapModel)),
            // Untouched slider = auto: Swarm sends group params whenever the group is open, so the 0.5 default must
            // not override Wan 2.2's official boundary. The engine warps an explicit fraction via the flow shift.
            VideoSwapPercent = input.TryGet(T2IParamTypes.VideoSwapPercent, out double swapPercent) && swapPercent != 0.5
                ? swapPercent : null,
            VideoResolution = input.Get(T2IParamTypes.VideoResolution, null),
            // Swarm's cross-family fallback of 25 is wrong for LTX-2, whose default is 24. Send a value, not null: LTX-2 does not report
            // the fps it used, so a null would mux at the encoder's 25 and play 4% fast.
            Fps = input.Get(T2IParamTypes.VideoFPS, IsLtxVideo2(input) ? 24 : 25),
            VideoFormat = input.Get(T2IParamTypes.VideoFormat, null),
            // Boomerang is applied to the decoded frames on the way out (see GenerateVideo) so the Engine
            // doesn't waste a second pass; the flag is still carried for pipelines that want it.
            VideoBoomerang = input.Get(T2IParamTypes.VideoBoomerang, false),
            VideoEndFrame = endFrame is null ? null : ToEngineImage(endFrame),
            VideoAudioInput = ToAudioClip(input.Get(T2IParamTypes.VideoAudioInput)),
            VideoAudioReference = ToAudioClip(input.Get(HartsyInference.VideoAudioReferenceParam)),
            // On Animate checkpoints the prompt-attached image is consumed above as the character reference; passing it here too would trip the feature gate.
            ReferenceImages = isAnimate ? null : BuildReferenceImages(input),
            ReferenceVideos = BuildReferenceVideos(input),
            ReferenceAudios = BuildReferenceAudios(input),
            Frames = frames,
            TrimVideoStartFrames = input.Get(T2IParamTypes.TrimVideoStartFrames, 0),
            TrimVideoEndFrames = input.Get(T2IParamTypes.TrimVideoEndFrames, 0),
            Components = BuildComponents(input),
            Loras = BuildLoras(input),
            ChainTotalFrames = chainTotal,
            ChainContextFrames = chainContext,
            Extra = extra,
            Vram = MapVramOverrides(input),
        };
    }

    /// <summary>Refuses start/end-frame conditioning combined with reference media on MiniMax-H3, whichever checkpoint is loaded.</summary>
    /// <remarks>fl2va and ref2va are separate tasks: reference blocks restart the packed layout's position cursor, so their rows overlap keyframe rows.
    /// Checked here, at request construction, so the message can name the params the user set.</remarks>
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

    /// <summary>Validates MiniMax-H3's chain params by dry-running the real planner (pure arithmetic, no GPU) rather than re-deriving its rules.</summary>
    /// <remarks>The chain grid is 39+51k (17k+5 intersected with whole 40 Hz audio rows), not the plain 17k+5 grid that <see cref="MiniMaxH3Geometry.AlignFrameCount"/> uses.</remarks>
    private static (int? Total, int Context) ResolveH3Chain(T2IParamInput input, int? frames)
    {
        int? total = input.TryGet(HartsyInference.H3ChainTotalFramesParam, out int totalRaw) && totalRaw > 0
            ? totalRaw : null;
        bool contextSet = input.TryGet(HartsyInference.H3ChainContextFramesParam, out int contextRaw);
        int context = contextSet ? SnapChainContextFrames(contextRaw)
            : global::HartsyInference.Engine.Recipes.Video.MiniMaxH3ChainPlanner.DefaultContextFrames;
        if (contextSet && total is null)
        {
            throw new SwarmUserErrorException(
                "HartsyInference: H3 Chain Context Frames is set without H3 Chain Total Frames, so it has nothing "
                + "to apply to. Set Total Frames past one segment's length to actually chain, or remove Context Frames.");
        }
        if (total is int t)
        {
            int maxSegment = global::HartsyInference.Diffusion.Models.Denoisers.MiniMaxH3Geometry.AlignFrameCount(frames ?? 124);
            try
            {
                global::HartsyInference.Engine.Recipes.Video.MiniMaxH3ChainPlanner.Plan(t, context, maxSegment);
            }
            catch (ArgumentOutOfRangeException ex)
            {
                throw new SwarmUserErrorException(
                    $"HartsyInference: H3 chain settings are invalid: {ex.Message.Replace("\r\n", " ").Replace('\n', ' ')}");
            }
        }
        return (total, context);
    }

    /// <summary>Snaps to the nearest valid context length, not up, since rounding up can push a value past the segment cap.</summary>
    /// <remarks>For example, 100 at the default 124-frame segment: rounding up gives 141 (fails), nearest gives 90 (passes).</remarks>
    private static int SnapChainContextFrames(int frames)
    {
        int k = (int)Math.Round((frames - 39) / 51.0, MidpointRounding.AwayFromZero);
        return 39 + Math.Max(0, k) * 51;
    }

    /// <summary>The requested frame count, or null to let the family's own default stand.</summary>
    /// <remarks>Core's frame params default to 25 for every family. On MiniMax-H3 that snaps to 39 frames (1.6 s), and 25 is off LTX-2's 8n+1 grid.
    /// Null when neither param is set lets each recipe pick its own default (124 frames for H3).</remarks>
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
            Logs.Debug("[HartsyInference] No frame count set, so leaving it to MiniMax-H3's own 124-frame default "
                + "rather than Swarm's cross-family 25 (which would snap to 39 on H3's 17k+5 grid).");
            return null;
        }
        if (IsLtxVideo2(input))
        {
            Logs.Debug("[HartsyInference] No frame count set, so leaving it to the LTX-2 recipe's own default rather "
                + "than Swarm's cross-family 25, which is not on LTX-2's 8n+1 grid.");
            return null;
        }
        return 25;
    }

    /// <summary>Whether the selected model is MiniMax-H3, by core's own compat class rather than a name guess.</summary>
    private static bool IsMiniMaxH3(T2IParamInput input) =>
        input.Get(T2IParamTypes.Model)?.ModelClass?.CompatClass?.ID == T2IModelClassSorter.CompatMiniMaxH3.ID;

    /// <summary>Whether the selected model is any LTX-2.x (2, 2.3 or 2.5, which share core's compat class).</summary>
    private static bool IsLtxVideo2(T2IParamInput input) =>
        input.Get(T2IParamTypes.Model)?.ModelClass?.CompatClass?.ID == T2IModelClassSorter.CompatLtxv2.ID;

    /// <summary>Whether the selected checkpoint is Wan-Animate or Wan-Animate-2.</summary>
    /// <remarks>Animate shares Wan's compat classes, so the engine's header-sniffed variant detection decides: only Animate claims <see cref="VideoFeatures.DrivingVideo"/>.</remarks>
    private static bool IsWanAnimate(T2IParamInput input)
    {
        T2IModel model = input.Get(T2IParamTypes.Model);
        string compat = model?.ModelClass?.CompatClass?.ID;
        return compat is not null && compat.StartsWith("wan-2", StringComparison.Ordinal)
            && (ModelSupport.SupportedVideoFeatures(model) & VideoFeatures.DrivingVideo) != 0;
    }

    /// <summary>Reference media caps the model was trained under, mirroring what the reference node accepts.</summary>
    private const int MaxRefImages = ImageInputs.MaxReferenceImages, MaxRefAudios = 3, MaxRefVideos = 3;

    /// <summary>Reference images from the prompt box, as an ordered list. Swarm does not parse the <c>&lt;Picture N&gt;</c> tags; the model's encoder does.</summary>
    /// <remarks>Core's <see cref="T2IParamTypes.PromptImages"/> carrier is what the prompt box fills on drag or paste.
    /// The consuming recipe applies its own slot cap, which can be tighter.</remarks>
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

    /// <summary>Standalone reference audio from the prompt box, referenced as <c>&lt;Audio N&gt;</c> in the prompt.</summary>
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

    /// <summary>Reference videos from the prompt box, unpaired with the reference audio. The model correlates them through the inline <c>&lt;Video N&gt;</c>/<c>&lt;Audio N&gt;</c> tags.</summary>
    /// <remarks><see cref="ReferenceVideo.Audio"/> can carry a per-clip soundtrack, which this UI does not expose.</remarks>
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

    /// <summary>Logs a warning when media past the model's cap is dropped, so it does not look as if the model ignored it.</summary>
    private static void WarnIfTruncated(int supplied, int used, string what)
    {
        if (supplied > used)
        {
            Logs.Warning($"[HartsyInference] {supplied} reference {what} attached but this model takes at most "
                + $"{used}; the extra {what} were dropped. Remove the extras to avoid this warning.");
        }
    }

    /// <summary>Maps the Text2Audio group and this extension's music params onto the Engine's <see cref="MusicRequest"/>.</summary>
    /// <remarks>One source-audio param and one edit-mode dropdown fill exactly one of Continuation, Repaint or Cover, so the engine's multiple-mode error cannot occur.</remarks>
    private static MusicRequest BuildMusicRequest(T2IParamInput input)
    {
        long seed = input.Get(T2IParamTypes.Seed, -1L);
        AudioClip sourceAudio = ToAudioClip(input.Get(HartsyInference.AceStepSourceAudioParam));
        string editMode = input.Get(HartsyInference.AceStepEditModeParam, "continuation");
        string lmModel = input.Get(HartsyInference.AceStepLmPlannerParam, "none");
        MusicRequest request = new MusicRequest
        {
            // ACE-Step convention (which the Engine follows): the prompt carries the lyrics, the style/genre tags
            // come from Text2AudioStyle. An empty prompt means instrumental.
            Prompt = input.Get(T2IParamTypes.Prompt) ?? "",
            Genre = input.Get(T2IParamTypes.Text2AudioStyle, "") ?? "",
            Duration = input.Get(T2IParamTypes.Text2AudioDuration, 10d),
            Vram = MapVramOverrides(input),
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
        if (input.TryGet(HartsyInference.AceStepRepaintStartParam, out double repaintStart))
        {
            request = request with { RepaintStart = repaintStart };
        }
        if (input.TryGet(HartsyInference.AceStepRepaintEndParam, out double repaintEnd))
        {
            request = request with { RepaintEnd = repaintEnd };
        }
        if (input.TryGet(HartsyInference.AceStepCoverStrengthParam, out double coverStrength))
        {
            request = request with { CoverStrength = coverStrength };
        }
        if (lmModel != "none")
        {
            request = request with
            {
                LmModel = lmModel,
                Thinking = input.Get(HartsyInference.AceStepLmThinkingParam, true),
                LmTemperature = input.Get(HartsyInference.AceStepLmTemperatureParam, 0.85),
                LmCfgScale = input.Get(HartsyInference.AceStepLmCfgParam, 2.0),
                LmTopK = input.Get(HartsyInference.AceStepLmTopKParam, 0),
                LmTopP = input.Get(HartsyInference.AceStepLmTopPParam, 0.9),
                LmNegativePrompt = input.Get(HartsyInference.AceStepLmNegativePromptParam, "") ?? "",
            };
        }
        if (input.TryGet(HartsyInference.AceStepSolverParam, out string inferMethod))
        {
            request = request with { InferMethod = inferMethod };
        }
        if (input.TryGet(HartsyInference.AceStepGuidanceTypeParam, out string guidanceType))
        {
            request = request with { GuidanceType = guidanceType };
        }
        if (input.TryGet(HartsyInference.AceStepErgTagParam, out bool ergTag))
        {
            request = request with { UseErgTag = ergTag };
        }
        if (input.TryGet(HartsyInference.AceStepErgLyricParam, out bool ergLyric))
        {
            request = request with { UseErgLyric = ergLyric };
        }
        if (input.TryGet(HartsyInference.AceStepErgDiffusionParam, out bool ergDiffusion))
        {
            request = request with { UseErgDiffusion = ergDiffusion };
        }
        if (input.TryGet(HartsyInference.AceStepCfgIntervalStartParam, out double cfgStart))
        {
            request = request with { CfgIntervalStart = cfgStart };
        }
        if (input.TryGet(HartsyInference.AceStepCfgIntervalEndParam, out double cfgEnd))
        {
            request = request with { CfgIntervalEnd = cfgEnd };
        }
        return request;
    }

    // ─── mapping helpers ───

    /// <summary>Image width/height when the request leaves them unset.</summary>
    public const int DefaultImageSize = 1024;

    /// <summary>Video width when the request leaves it unset.</summary>
    public const int DefaultVideoWidth = 704;

    /// <summary>Video height when the request leaves it unset.</summary>
    public const int DefaultVideoHeight = 480;

    /// <summary>The geometry and VRAM overrides a generation will run with, for the engine's memory estimate, so routing judges what will actually run.</summary>
    /// <remarks>Frames stay null when the request leaves them to the family; the engine then substitutes the recipe's own.</remarks>
    public static MemoryEstimateRequest MapMemoryEstimateRequest(T2IParamInput input, ModelSupport.Family family)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(family);
        bool video = family.Kind == ModelSupport.Kind.Video;
        return new MemoryEstimateRequest(
            input.Get(T2IParamTypes.Width, video ? DefaultVideoWidth : DefaultImageSize),
            input.Get(T2IParamTypes.Height, video ? DefaultVideoHeight : DefaultImageSize),
            video ? ResolveFrames(input) : 1,
            Vram: MapVramOverrides(input));
    }

    /// <summary>An int param's value, or null when the request didn't set it (so the Engine's family default wins).</summary>
    private static int? NullableInt(T2IParamInput input, T2IRegisteredParam<int> param) =>
        input.TryGet(param, out int value) && value > 0 ? value : null;

    /// <summary>The per-generation VRAM lever overrides, or null when the request pinned none.</summary>
    /// <remarks>Null for an untouched group: the engine treats null as "follow the backend's policy", and an empty override would still count as a decision.
    /// Only runtime-scoped levers appear here; quantized compute and multi-GPU spill are fixed when the pipeline is built.</remarks>
    private static VramOverrides MapVramOverrides(T2IParamInput input)
    {
        VramOverrides overrides = new VramOverrides
        {
            Tier = ParseTierOverride(input, HartsyInference.VramModeParam),
            WeightStreaming = ParseLever(input, HartsyInference.VramWeightStreamingParam),
            KeepResident = ParseLever(input, HartsyInference.VramKeepResidentParam),
            PhaseUnload = ParseLever(input, HartsyInference.VramPhaseUnloadParam),
            ActivationOffload = ParseLever(input, HartsyInference.VramActivationOffloadParam),
            FreeAfterGeneration = ParseLever(input, HartsyInference.VramFreeAfterGenerationParam),
            Caches = ParseCachePrecision(input, HartsyInference.VramCachePrecisionParam),
            ChunkScale = input.TryGet(HartsyInference.VramChunkScaleParam, out double scale) && scale > 0
                ? (float)scale : null,
        };
        return overrides.IsEmpty ? null : overrides;
    }

    /// <summary>"Backend default" (and anything unrecognized) leaves the lever unset so the backend's policy stands.</summary>
    private static LeverState? ParseLever(T2IParamInput input, T2IRegisteredParam<string> param)
        => input.TryGet(param, out string value)
            ? value?.Trim().ToLowerInvariant() switch
            {
                "auto" => LeverState.Auto,
                "on" => LeverState.On,
                "off" => LeverState.Off,
                _ => null,
            }
            : null;

    private static VramTier? ParseTierOverride(T2IParamInput input, T2IRegisteredParam<string> param)
        => input.TryGet(param, out string value)
            ? value?.Trim().ToLowerInvariant() switch
            {
                "performance" => VramTier.Performance,
                "auto" => VramTier.Auto,
                "balanced" => VramTier.Balanced,
                "aggressive" => VramTier.Aggressive,
                "maximum" => VramTier.Maximum,
                _ => null,
            }
            : null;

    private static CachePrecision? ParseCachePrecision(T2IParamInput input, T2IRegisteredParam<string> param)
        => input.TryGet(param, out string value)
            ? value?.Trim().ToLowerInvariant() switch
            {
                "auto" => CachePrecision.Auto,
                "full" => CachePrecision.Full,
                "half" => CachePrecision.Half,
                _ => null,
            }
            : null;

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

    /// <summary>The annotator device: the Engine's backend when one exists (shared caches, no second CUDA context), else a lazily created device we own.</summary>
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
                    // Prefer Init's probed kind: Create would re-resolve 'auto' cheaply, which is the disagreement Init avoids.
                    // Otherwise the annotator could land on CPU, or on a card the probe rejected.
                    string selector = _resolvedComputeKind ?? Settings?.ComputeBackend?.ToLowerInvariant() ?? "auto";
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

    /// <summary>Drops the annotator device reference. Disposes it only when standalone, since disposing a borrowed Engine backend would evict its resident weights.</summary>
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

    /// <summary>Turns an out-of-VRAM failure into advice a user can act on. The engine's message is quoted verbatim, and only the advice is added here.</summary>
    /// <remarks>The engine's message is already specific (pre-flight geometry and byte requirement, or the allocation that failed), so it is not re-derived.</remarks>
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

    /// <summary>Refuses a sampler or sigma schedule the selected family cannot run, so Swarm can route the request to a Comfy backend.</summary>
    /// <remarks>Support is asked of the Engine, the same way <see cref="ModelSupport.SupportedFeatures"/> asks the recipe registry, so the answer cannot drift from the pipeline.
    /// Refusing here beats a <see cref="NotSupportedException"/> mid-generation, which some families (Wan UniPC, LTX, Ideogram 4, Lumina2, Lance image) throw from inside the pipeline.</remarks>
    private static bool ValidateSamplingChoice(T2IParamInput input, ModelSupport.Family family)
    {
        input.TryGet(ComfyUIBackendExtension.SamplerParam, out string sampler);
        input.TryGet(ComfyUIBackendExtension.SchedulerParam, out string schedule);
        bool wantsSampler = !string.IsNullOrWhiteSpace(sampler);
        // "normal" is the identity schedule: the Engine treats it as no selection, so this does too.
        bool wantsSchedule = !string.IsNullOrWhiteSpace(schedule)
            && !string.Equals(schedule, "normal", StringComparison.OrdinalIgnoreCase);
        if (!wantsSampler && !wantsSchedule)
        {
            return true;
        }

        SamplingCapabilities.SamplingSupport support = family.Kind == ModelSupport.Kind.Video
            ? SamplingCapabilities.ForVideo(family.Id)
            : SamplingCapabilities.ForImage(family.Id);
        if (support.Samplers.Count == 0 && support.Schedules.Count == 0)
        {
            input.RefusalReasons.Add(
                $"HartsyInference: {family.Id} samples with its own solver and takes no sampler or scheduler "
                + "selection. Remove the Sampler and Scheduler settings and try again.");
            return false;
        }
        if (wantsSampler && !support.Samplers.Contains(sampler, StringComparer.OrdinalIgnoreCase))
        {
            input.RefusalReasons.Add(
                $"HartsyInference: the '{sampler}' sampler isn't available on {family.Id} (it has "
                + $"{string.Join(", ", support.Samplers)}). Pick one of those.");
            return false;
        }
        if (wantsSchedule && !support.Schedules.Contains(schedule, StringComparer.OrdinalIgnoreCase))
        {
            string available = support.Schedules.Count == 0
                ? $"{family.Id} owns its own sigma spacing"
                : $"it has {string.Join(", ", support.Schedules)}";
            input.RefusalReasons.Add(
                $"HartsyInference: the '{schedule}' scheduler isn't available on {family.Id} ({available}). "
                + "Pick one it has, or leave the scheduler unset.");
            return false;
        }
        // A re-spaced schedule cannot combine with img2img or inpaint: the init latent is noised at the family's own sigma, so the sampler
        // would start at the wrong noise level. Refusing here keeps the request routable.
        if (wantsSchedule && (input.Get(T2IParamTypes.InitImage) is not null || input.Get(T2IParamTypes.MaskImage) is not null))
        {
            input.RefusalReasons.Add(
                $"HartsyInference: the '{schedule}' scheduler can't be combined with an init image or mask — the init "
                + "latent is noised at the model's own sigma, so a re-spaced schedule would start from the wrong "
                + "noise level. Drop the scheduler.");
            return false;
        }
        return true;
    }

    /// <summary>IDs of comfyui-tagged params this backend services, so the comfy-only guard in <see cref="IsValidForThisBackend"/> does not refuse them.</summary>
    /// <remarks>Refiner sampler and scheduler are absent: the refiner shares the base loop's scheduler (StepSwap), so there is no separate choice to honor.</remarks>
    private static readonly HashSet<string> HonoredComfyParams =
        ["sampler", "scheduler",
         // Tangential Damping CFG: Comfy's param drives ImageRequest.Tcfg, so no second control is registered.
         "usetcfg",
         // Refiner Upscale Method: picks the pixel-space engine when there is no refiner model; see ResolveUpscaleModel.
         "refinerupscalemethod",
         // Style-model (FLUX.1 Redux) strengths, mapped onto the engine's redux.* Extra keys.
         "stylemodelmergestrength", "stylemodelmultiplystrength", "stylemodelapplystart",
         // IP-Adapter scheduling knobs, mapped onto the engine's ipadapter.* Extra keys.
         "ipadapterweight", "ipadapterstart", "ipadapterend", "ipadapterweighttype"];

    /// <summary>Prompt-syntax tags the Engine's request contract cannot express. Feeding them to a text encoder would corrupt the output.</summary>
    /// <remarks><c>region</c> and <c>segment</c> are not here: they map to <see cref="Regional"/> and are gated per family.</remarks>
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

        // Refuse unsupported prompt syntax by name rather than feeding the tag text to the tokenizer. A clean refusal routes to Comfy.
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
                    + "Remove the tag and try again.");
                return false;
            }
        }

        // The image-then-animate flow (Comfy's ImageToVideoGenInfo, driven by the Video Model param) has no Engine equivalent.
        // Refuse upfront rather than return a still image.
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
        // Last, after every capability check: memory routing only matters among backends that can run the request.
        return ValidateComfyOnlyParams(input) && ModelFitGate.Allows(input, this);
    }

    /// <summary>Refuses image-only knobs on music models, plus any music param the selected family would silently ignore.</summary>
    /// <remarks>Music has no per-family feature model, so these checks are hard-coded by family id. YuE and MusicGen belong to AudioLab, which routes them through its own providers.</remarks>
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
        bool hasSourceAudio = input.Get(HartsyInference.AceStepSourceAudioParam) is not null;
        bool isAceStep = family.Id == "acestep";
        if (hasSourceAudio && !isAceStep)
        {
            input.RefusalReasons.Add(
                $"HartsyInference: music editing (continuation/repaint/cover) is ACE-Step only; '{family.Id}' has no "
                + "audio-conditioned edit path. Remove ACE-Step Source Audio or pick an ACE-Step model.");
            return false;
        }
        // Pre-empt the engine's src_latents-conflict throw with a routable refusal.
        if (hasSourceAudio && input.TryGet(HartsyInference.AceStepLmPlannerParam, out string lmModel) && lmModel != "none")
        {
            input.RefusalReasons.Add(
                "HartsyInference: the ACE-Step edit modes and the 5 Hz LM planner both occupy the src_latents slot — "
                + "set ACE-Step LM Planner to none, or remove the Source Audio.");
            return false;
        }
        if (hasSourceAudio && input.Get(HartsyInference.AceStepEditModeParam, "continuation") == "repaint"
            && input.Get(HartsyInference.AceStepRepaintEndParam, 0.0) <= input.Get(HartsyInference.AceStepRepaintStartParam, 0.0))
        {
            input.RefusalReasons.Add(
                $"HartsyInference: repaint needs Repaint End > Repaint Start (got {input.Get(HartsyInference.AceStepRepaintStartParam, 0.0)}"
                + $"..{input.Get(HartsyInference.AceStepRepaintEndParam, 0.0)} s).");
            return false;
        }
        if (!isAceStep)
        {
            (bool Set, string Name)[] aceOnly =
            [
                (input.TryGet(HartsyInference.AceStepLmPlannerParam, out string lm) && lm != "none", "ACE-Step LM Planner"),
                (input.TryGet(HartsyInference.AceStepSolverParam, out string im) && im != "ode", "ACE-Step Solver"),
                (input.TryGet(HartsyInference.AceStepGuidanceTypeParam, out string gt) && gt != "apg", "ACE-Step Guidance Type"),
                (input.TryGet(HartsyInference.AceStepCfgIntervalStartParam, out double cis) && cis > 0, "ACE-Step CFG Interval Start"),
                (input.TryGet(HartsyInference.AceStepCfgIntervalEndParam, out double cie) && cie < 1, "ACE-Step CFG Interval End"),
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
        if (family.Id != "minimaxmusic3"
            && input.TryGet(HartsyInference.MiniMaxMusicLmPrecisionParam, out string precision) && precision != "bf16")
        {
            input.RefusalReasons.Add(
                $"HartsyInference: MiniMax Music LM Precision is MiniMax Music 3 only; '{family.Id}' would silently ignore it. Unset it or pick a MiniMax Music 3 model.");
            return false;
        }
        return true;
    }

    /// <summary>Video model checks. LoRAs are refused because <see cref="VideoRequest"/> has no composition phase, and a refiner cannot run over an encoded clip.</summary>
    private static bool ValidateVideo(T2IParamInput input, string compat, ModelSupport.Family family)
    {
        if (!ValidateSamplingChoice(input, family))
        {
            return false;
        }
        // Init and end-frame conditioning are per family. Without this check the Engine would silently generate text-to-video.
        // Checkpoint-aware: Wan's Animate, VACE and S2V variants share compat classes, so the engine's header sniff decides.
        VideoFeatures videoSupported = ModelSupport.SupportedVideoFeatures(input.Get(T2IParamTypes.Model));
        // LTX-2.5 ships as four files and is handed to the Engine as a folder. Reference media rides the prompt box
        // (core's PromptImages/Audios/Videos carriers), so these checks read what the user attached there.
        bool hasRefImages = input.TryGet(T2IParamTypes.PromptImages, out List<Image> refImgs) && refImgs is { Count: > 0 };
        bool hasRefAudios = input.TryGet(T2IParamTypes.PromptAudios, out List<AudioFile> refAuds) && refAuds is { Count: > 0 };
        bool hasRefVideos = input.TryGet(T2IParamTypes.PromptVideos, out List<VideoFile> refVids) && refVids is { Count: > 0 };
        // On an Animate checkpoint a prompt-attached image is the character reference (BuildVideoRequest consumes
        // it as such, mirroring core's PromptImages-as-Wan-reference convention), not a ReferenceImages request.
        bool animateCheckpoint = compat.StartsWith("wan-2", StringComparison.Ordinal) && (videoSupported & VideoFeatures.DrivingVideo) != 0;
        (VideoFeatures Feature, string Name, bool Requested)[] videoChecks =
        [
            (VideoFeatures.InitImage, "image-to-video (Init Image)", input.Get(T2IParamTypes.InitImage) is not null),
            (VideoFeatures.EndFrame, "end-frame conditioning (Video End Image)", GetVideoEndImage(input) is not null),
            (VideoFeatures.ReferenceImages, "reference images (attached to the prompt)", hasRefImages && !animateCheckpoint),
            (VideoFeatures.ReferenceVideos, "reference videos (attached to the prompt)", hasRefVideos),
            (VideoFeatures.ReferenceAudios, "reference audio (attached to the prompt, or Video Audio Reference)",
                hasRefAudios || input.Get(HartsyInference.VideoAudioReferenceParam) is not null),
            (VideoFeatures.DrivingVideo, "Wan-Animate driving (Animate Reference Image / Pose Video / Face Video)",
                input.Get(HartsyInference.AnimateReferenceImageParam) is not null
                || input.Get(HartsyInference.AnimatePoseVideoParam) is not null
                || input.Get(HartsyInference.AnimateFaceVideoParam) is not null
                || (animateCheckpoint && hasRefImages)),
            (VideoFeatures.LongFormChain, "long-form chaining (H3 Chain Total Frames)",
                input.TryGet(HartsyInference.H3ChainTotalFramesParam, out int chainCheck) && chainCheck > 0),
        ];
        foreach ((VideoFeatures feature, string name, bool requested) in videoChecks)
        {
            if (requested && (videoSupported & feature) == 0)
            {
                // For H3 the limit is the checkpoint, not the architecture, so the hint points at the sibling checkpoint.
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
        // Feature-driven: MiniMax-H3 merges LoRAs, other video families do not read them, and silently dropping a LoRA is worse than refusing it.
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

    /// <summary>Checks each composition object against the features the family's Engine recipe declares, and refuses by name.</summary>
    /// <remarks>Driven by <see cref="IArchitectureRecipe.Supports"/>, so it never claims more than the Engine applies.</remarks>
    private static bool ValidateImageFeatures(T2IParamInput input, string compat, ModelSupport.Family family)
    {
        if (!ValidateSamplingChoice(input, family))
        {
            return false;
        }
        // Refiner Upscale without a refiner model is a pixel-space pass (Real-ESRGAN or SeedVR2). The latent methods need a
        // refiner's re-denoise, which this backend does not do, so refuse them and let Comfy serve.
        if (input.Get(T2IParamTypes.RefinerModel) is null
            && input.TryGet(T2IParamTypes.RefinerUpscale, out double upscale) && upscale > 1.0 + 1e-6
            && input.TryGet(ComfyUIBackendExtension.RefinerUpscaleMethod, out string upscaleMethod)
            && upscaleMethod is not null && upscaleMethod.StartsWith("latent", StringComparison.OrdinalIgnoreCase))
        {
            input.RefusalReasons.Add(
                $"HartsyInference: the '{upscaleMethod}' upscale method needs a refiner model's latent pass. Without a "
                + "refiner this backend upscales in pixel space — pick a Pixel, Model, Real-ESRGAN or SeedVR2 method.");
            return false;
        }
        // A refiner model must come from a family whose recipe can consume an init image (the PostApply hand-off).
        // Refusing here names the model; the engine-side check would fail mid-generation.
        if (input.Get(T2IParamTypes.RefinerModel) is T2IModel refModel && refModel is not null)
        {
            ModelSupport.Family refFamily = ModelSupport.Resolve(refModel.ModelClass?.CompatClass?.ID);
            // The classic SDXL refiner checkpoint is the internal pair; it has no Img2Img bit but is fully handled.
            bool isSdxlRefinerClass = refModel.ModelClass?.CompatClass?.ID == "stable-diffusion-xl-v1-refiner";
            if (!isSdxlRefinerClass)
            {
                if (refFamily is null || refFamily.Kind != ModelSupport.Kind.Image)
                {
                    input.RefusalReasons.Add(
                        $"HartsyInference: refiner model '{refModel.Name}' has no image-family mapping, so it can't run a refine pass.");
                    return false;
                }
                ImageFeatures refFeatures = ModelSupport.SupportedFeatures(refModel);
                if ((refFeatures & (ImageFeatures.Img2Img | ImageFeatures.RefEdit)) == 0)
                {
                    input.RefusalReasons.Add(
                        $"HartsyInference: '{refModel.Name}' ({refFamily.Id}) can't be a refiner — its recipe has no "
                        + "img2img path to consume the base image with. Pick a model whose family supports Init Image.");
                    return false;
                }
            }
        }
        ImageFeatures supported = ModelSupport.SupportedFeatures(input.Get(T2IParamTypes.Model));
        List<(ImageFeatures Feature, string Name, bool Requested)> checks =
        [
            (ImageFeatures.Lora, "LoRAs", input.TryGet(T2IParamTypes.Loras, out List<string> loras) && loras is not null && loras.Count > 0),
            (ImageFeatures.ControlNet, "ControlNet", AnyControlNetSelected(input)),
            // Prompt images are either adapter conditioning or edit references, depending on the family, so either bit satisfies the slot.
            (ImageFeatures.IpAdapter | ImageFeatures.RefEdit, "IP-Adapter / reference image prompting",
                input.TryGet(T2IParamTypes.PromptImages, out List<Image> imgs) && imgs is not null && imgs.Count > 0),
            (ImageFeatures.Refiner, "Refiners", input.Get(T2IParamTypes.RefinerModel) is not null),
            // An Init Image is satisfied by either mode: strength-based img2img, or reference editing on an edit model where Creativity does not apply.
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
        // An explicit init-image mode must match the family: reference needs RefEdit, denoise needs Img2Img.
        // RefEdit-only families ignore Mode, so "denoise" there would silently run as an edit.
        if (input.Get(T2IParamTypes.InitImage) is not null
            && input.TryGet(HartsyInference.InitImageModeParam, out string initMode))
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
        return ValidateImageInputs(input, compat, family, supported);
    }

    /// <summary>Refuses image inputs the resolved variant would drop: prompt images on a family that only edits the Init Image, or more images than it reads.</summary>
    /// <remarks>Limits are the engine's own <see cref="ImageInputs.LimitsFor"/>. A base Qwen-Image build needs no check here: its variant lacks
    /// <see cref="ImageFeatures.RefEdit"/>, so the feature checks already refuse reference editing.</remarks>
    private static bool ValidateImageInputs(T2IParamInput input, string compat, ModelSupport.Family family, ImageFeatures supported)
    {
        T2IModel model = input.Get(T2IParamTypes.Model);
        if (model is null)
        {
            return true;
        }
        bool hasInit = input.Get(T2IParamTypes.InitImage) is not null;
        // Prompt images count only where they become edit references; elsewhere they are IP-Adapter input, gated above.
        int promptImages = (supported & ImageFeatures.RefEdit) != 0
            && input.TryGet(T2IParamTypes.PromptImages, out List<Image> imgs) && imgs is not null
            ? imgs.Count(i => i is not null) : 0;
        ImageInputLimits limits = ImageInputs.LimitsFor(model, family);
        if (!hasInit && promptImages > 0 && limits.ReferencesRequireInitImage)
        {
            input.RefusalReasons.Add(
                $"HartsyInference: '{compat}' (engine family '{family.Id}') edits the Init Image and never reads prompt "
                + "images on their own. Attach the image as the Init Image instead.");
            return false;
        }
        int images = (hasInit ? 1 : 0) + promptImages;
        if (images > limits.MaxImages)
        {
            input.RefusalReasons.Add(
                $"HartsyInference: '{compat}' (engine family '{family.Id}') reads at most {limits.MaxImages} input "
                + $"image{(limits.MaxImages == 1 ? "" : "s")} (Init Image plus prompt images); this request has {images}. "
                + $"Remove {images - limits.MaxImages}.");
            return false;
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

    /// <summary>Refuses any comfyui-tagged param that is set and that this backend cannot service, routing the request to Comfy.</summary>
    /// <remarks>Driven by each param's own FeatureFlag, so new Comfy params are covered automatically. "comfyui" is advertised so such requests reach this check.</remarks>
    private static bool ValidateComfyOnlyParams(T2IParamInput input)
    {
        // Params actually set on the request, keyed by cleaned param ID.
        Dictionary<string, object> setParams = input.InternalSet.ValuesInput;

        // Explicit safety net: never accept a custom ComfyUI workflow, raw or via the stored-workflow selector.
        foreach (string wfKey in new[] { "comfyworkflowraw", "comfyuicustomworkflow" })
        {
            if (setParams.TryGetValue(wfKey, out object wfVal) && !string.IsNullOrWhiteSpace($"{wfVal}"))
            {
                input.RefusalReasons.Add(
                    "HartsyInference: custom workflows aren't supported by this engine. "
                    + "Remove the custom workflow and try again.");
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
            // Skip params left at their default or ignore value, as they have no effect.
            if (valStr == pType.Default || (pType.IgnoreIf is not null && valStr == pType.IgnoreIf))
            {
                continue;
            }
            input.RefusalReasons.Add(
                $"HartsyInference: the '{pType.Name}' setting isn't supported by this engine. "
                + "Remove it and try again.");
            return false;
        }
        return true;
    }

    // ─────────────────────────────── 6. Engine auto-update ───────────────────────────────

    /// <summary>Checks NuGet for a newer HartsyInference engine and, when <c>AutoUpdate</c> opts in, stages a rebuild of this extension against it.</summary>
    /// <remarks>The engine is an in-process library, so a staged update takes effect on the next SwarmUI start. 'aggressive' also clears NuGet caches
    /// and calls <see cref="Program.RequestRestart"/>. Returns true when a newer engine was staged and this backend must refuse to run on the old one.
    /// Returns false when up to date, when staging failed, when a restart is already underway, or when the boot-loop guard trips.</remarks>
    private async Task<bool> MaybeAutoUpdateEngine()
    {
        string mode = (Settings?.AutoUpdate ?? "false").Trim().ToLowerInvariant();
        if (mode is "false" or "" or "0" or "no" or "off")
        {
            return false;
        }
        bool aggressive = mode is "aggressive" or "force";
        // Boot-loop guard: 'aggressive' records the target version before restarting. Coming back still behind that exact
        // version means the restore is not advancing, so do not auto-restart again.
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
                // Up to date: any prior staged update applied, so clear the marker.
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
                Logs.Warning($"[HartsyInference] Auto-update: engine {latest} was already staged on a previous restart but the running engine is still {loaded}. "
                    + $"Not auto-restarting again (avoiding a boot loop). The rebuild's NuGet restore isn't resolving {latest}.");
                AddLoadStatus($"Auto-update: {latest} did not apply after a restart — auto-restart paused to avoid a loop (see logs).");
                return false; // run degraded on the old engine rather than bricking the backend
            }

            string csproj = ExtensionProjectPath();
            if (csproj is null)
            {
                Logs.Warning("[HartsyInference] Auto-update: the extension's '.csproj' file was not found next to the assembly; cannot rebuild.");
                return false;
            }
            AddLoadStatus($"Auto-update: verifying engine {latest} compiles against this extension (this can take a minute)...");
            string dir = Path.GetDirectoryName(csproj);
            // Verification build before touching anything: a new engine can have breaking API changes. This writes to the csproj's
            // default output, not Swarm's load folder. RestoreForceEvaluate re-resolves the floating alpha.* versions.
            (int code, string output) = await RunDotnet($"build \"{csproj}\" -c Release /p:RestoreForceEvaluate=true", dir);
            if (code != 0)
            {
                Logs.Warning($"[HartsyInference] Auto-update: engine {latest} does not compile against this extension; staying on the current engine. Build output:\n{output}");
                AddLoadStatus($"Auto-update: engine {latest} failed to compile — keeping the current engine (see logs).");
                return false;
            }

            // The engine is in-process and cannot hot-swap. SwarmUI's ExtensionsManager.BuildExtension skips rebuilds by the
            // extension's git commit hash, not the engine version, so the cached build must be invalidated.
            string selfDll = typeof(HartsyInferenceBackend).Assembly.Location;
            try
            {
                // Unlinking a loaded assembly is allowed on Linux: the process keeps its mapping, and the file is gone for the next start.
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
            Logs.Info($"[HartsyInference] Engine {latest} verified and the cached extension build was invalidated. SwarmUI will rebuild against the new engine on next start (an in-process library cannot be hot-swapped).");
            if (aggressive)
            {
                Logs.Info("[HartsyInference] AutoUpdate is set to 'aggressive', so requesting a SwarmUI restart to rebuild and load the new engine.");
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
            Logs.Warning($"[HartsyInference] Auto-update failed; continuing with the current engine: {ex.ReadableString()}");
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
            Logs.Warning($"[HartsyInference] Auto-update: NuGet version query failed; continuing with the current engine: {ex.ReadableString()}");
            return null;
        }
    }

    /// <summary>True if <paramref name="candidate"/> is a newer "1.0.0-alpha.N" than <paramref name="current"/>. Unparseable forms fall back to an ordinal compare.</summary>
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
