using System.IO;
using System.Net.Http;
using System.Reflection;
using FreneticUtilities.FreneticDataSyntax;
using Newtonsoft.Json.Linq;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Hartsy.Extensions.HartsyInferenceBackend.Generation;
using HartsyInference.Core.Backends;
using SiLogs = HartsyInference.Core.Logging.Logs;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Vulkan;

namespace Hartsy.Extensions.HartsyInferenceBackend.Backends;

/// <summary>
/// SwarmUI backend that performs diffusion inference in-process via HartsyInference.
/// See docs/06-Backend-Lifecycle.md for the full lifecycle contract.
/// </summary>
public class HartsyInferenceBackend : AbstractT2IBackend
{
    /// <summary>Settings persisted to Data/Backends.fds. Must be nested inside the
    /// backend class so BackendHandler.RegisterBackendType discovers it via reflection.</summary>
    public class HartsyInferenceBackendSettings : AutoConfiguration
    {
        [ConfigComment("Compute backend to use. 'auto' tries CUDA, then Vulkan, then CPU.")]
        public string ComputeBackend = "auto";

        [ConfigComment("Which GPU to use, if multiple are available.\nShould be a single number, like '0' (first GPU), '1' (second GPU), etc.\nIgnored for the CPU compute backend.\nHartsyInference uses a single GPU per backend — to run on multiple GPUs, add one backend per GPU (each with its own GPU_ID). A '0,1'-style list is accepted but only the first number is used.")]
        public string GPU_ID = "0";

        [ConfigComment("Maximum number of model pipelines to keep cached in VRAM/RAM at once.\nHigher values avoid reloading when switching between models, at the cost of memory. 1 is recommended for a single GPU.")]
        public int MaxCachedPipelines = 1;

        [ConfigComment("Path to PTX kernel directory (CUDA only). Empty = use bundled Ptx/ folder next to the extension DLL.")]
        public string PtxDirectory = "";

        [ConfigComment("How many extra requests may queue up on this backend while one is generating.\n0 means a single live generation with nothing waiting (the scheduler routes further requests to other backends/GPUs immediately).\n1 (default) means a live generation plus one extra waiting in line before further requests route elsewhere.\n-1 makes this a UI-only instance that cannot do actual generations.\nGenerations always run one at a time on this backend (the queue just lets requests wait here instead of being sent elsewhere).")]
        public int OverQueue = 1;

        [ConfigComment("Per-step progress previews. 'off' disables them; 'latent2rgb' (default) uses a fast model-free latent→RGB approximation (blurry but instant); 'taesd' uses a tiny per-architecture autoencoder for higher-fidelity previews when the TAESD weights ship.")]
        public string PreviewMethod = "latent2rgb";

        [ConfigComment("Native FP8 GEMM for fp8 checkpoints (Ada/Hopper, SM 8.9+). When on, fp8 weights are matrix-multiplied directly in fp8 (via cuBLASLt) instead of being upcast to fp16 — roughly half the transformer VRAM and faster on supported GPUs. 'auto' (default) enables it on SM 8.9+ GPUs; 'on' forces it; 'off' always upcasts to fp16. On older GPUs (Ampere and below) the engine falls back to fp16 automatically regardless of this setting.")]
        public string NativeFp8Gemm = "auto";

        [ConfigComment("Whether to auto-update the HartsyInference engine (the in-process NuGet library) when this backend starts.\n'false' (default): never check.\n'true': on start, check NuGet for a newer engine build and, if found, download + rebuild the extension against it.\n'aggressive': same as 'true' but also clears the NuGet caches first (fixes a stuck floating-version restore) and automatically restarts SwarmUI to load the new build.\nThe engine is loaded in-process, so a staged update applies on the NEXT SwarmUI restart (a loaded DLL can't hot-swap). With 'true' you'll get a log line telling you to restart; 'aggressive' restarts for you.")]
        public string AutoUpdate = "false";
    }

    public HartsyInferenceBackendSettings Settings => SettingsRaw as HartsyInferenceBackendSettings;

    /// <summary>The HartsyInference IBackend (CPU / Vulkan / CUDA). Constructed in <see cref="Init"/>.</summary>
    private IBackend _backend;

    /// <summary>Pipeline cache. One entry per loaded checkpoint.</summary>
    private PipelineCache _cache;

    /// <summary>Cancellation source for the in-flight generation.</summary>
    private CancellationTokenSource _cancelCts;

    /// <summary>Serializes generations so a backend with <c>OverQueue &gt; 0</c> (MaxUsages &gt; 1) accepts
    /// extra requests into a queue but still runs them ONE AT A TIME. HartsyInference shares
    /// <see cref="_backend"/> / <see cref="_cache"/> / <see cref="_cancelCts"/> across a generation, so
    /// concurrent execution would collide — this lock makes the over-queue safe by holding extra
    /// dispatched jobs here until the current one finishes.</summary>
    private readonly SemaphoreSlim _genLock = new(1, 1);

    public override IEnumerable<string> SupportedFeatures
    {
        get
        {
            // Advertise ONLY flags we genuinely service. Lying here means params we
            // can't handle (Sampler, Scheduler, RefinerUpscaleMethod, custom workflow,
            // etc. — all tagged "comfyui") would route to us and be silently dropped.
            //
            // Trade-off: with no "comfyui" flag, params tagged "comfyui" disappear from
            // the UI when HartsyInference is the only configured backend. For Flux that's
            // correct — we use FlowMatchEulerDiscrete unconditionally and don't expose
            // sampler choice. When both Comfy and HartsyInference are configured, the
            // UI shows comfyui params (their union) and T2IEngine routes them to Comfy.
            //
            // As we ship real capabilities (LoRA, refiner, ControlNet, img2img), add the
            // matching specific flag here. Mirror Comfy's NodeToFeatureMap: only declare
            // what's compiled in.
            yield return "hartsyinference";
            yield return "text2image";
            yield return "flux-dev";   // in DisregardedFeatureFlags — informational, but signals that FluxGuidanceScale is honored
            yield return "lora";       // SD 1.5 / SDXL / Flux supported via LoraStack; SD3 / Z-Image refused at validation time
            yield return "endstepsearly"; // honored by SamplingParamResolver across all architectures
            yield return "refiners";   // SDXL refiner: PostApply (any base) + StepSwap (SDXL base only); RefinerUpscale / RefinerVAE refused at validation time
            yield return "img2img";    // SD 1.5 / SDXL / Flux / SD3 / Z-Image supported via VaeEncoder
            yield return "inpaint";    // SDXL / Flux / SD3 supported via blend-on-vanilla pipeline path; SD 1.5 / Z-Image refused at validation time
            yield return "controlnet"; // SDXL-base only in v1, Canny preprocessor only; SD 1.5 / Flux ControlNet + Depth/OpenPose/etc refused at validation time
            yield return "ipadapter";  // SDXL standard + Plus + Plus-Face via blend-on-vanilla cross-attn injection; SD 1.5 / Flux IPA + FaceID variants refused at validation time
            yield return "variation_seed"; // SD 1.5 / SDXL via InitialNoise slerp (VariationSeedResolver); other archs refused at validation time
            yield return "video";      // Wan2.2 TI2V-5B + LTX-Video text-to-video. Exposes VideoFPS/VideoFormat/boomerang/trim params
                                       // ("text2video" itself is client-derived from the model's compat class and disregarded for routing).
                                       // Unsupported video extras (end frame, video-extend, audio) are refused at validation time.
        }
    }

    /// <summary>Bridges HartsyInference's internal logger into Swarm's logging system so
    /// diagnostics like the OOM probe in CudaMemory.Allocate appear in the main log file
    /// instead of falling into Console.Error (where Swarm captures stdout but routes it
    /// inconsistently). Idempotent — safe to call multiple times.</summary>
    private static int _loggerWired = 0;
    private static void EnsureLoggerWired()
    {
        if (Interlocked.Exchange(ref _loggerWired, 1) != 0) return;
        // Don't double-filter. Set HartsyInference's level to Verbose (the chattiest) so
        // every message reaches the bridge below; Swarm's own MinimumLevel filter then
        // decides what actually appears in the log/UI. Mirroring Swarm's MinimumLevel
        // here was fragile: Swarm's level can change at runtime (settings UI, debug flag),
        // and a stale snapshot meant Verbose progress was getting dropped at the
        // HartsyInference layer before it ever had a chance to be forwarded.
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

    public override async Task Init()
    {
        EnsureLoggerWired();
        await MaybeAutoUpdateEngine();
        try
        {
            string requested = Settings?.ComputeBackend?.ToLowerInvariant() ?? "auto";
            int ordinal = ParseGpuId(Settings?.GPU_ID);

            // Kernels ship in our extension's own output dir, NOT Swarm's main runtime
            // dir (which is what AppContext.BaseDirectory returns when running inside
            // the Swarm process). Resolve from this assembly's location instead.
            string extensionDir = Path.GetDirectoryName(typeof(HartsyInferenceBackend).Assembly.Location) ?? AppContext.BaseDirectory;
            string ptxDir = string.IsNullOrWhiteSpace(Settings?.PtxDirectory)
                ? Path.Combine(extensionDir, "Ptx")
                : Settings.PtxDirectory;
            string spvDir = Path.Combine(extensionDir, "Spirv");

            AddLoadStatus($"Resolving compute backend (requested='{requested}', ordinal={ordinal})...");
            AddLoadStatus($"Kernel paths: PTX={ptxDir} (exists={Directory.Exists(ptxDir)}), SPIR-V={spvDir} (exists={Directory.Exists(spvDir)})");
            _backend = ConstructBackend(requested, ordinal, ptxDir, spvDir);
            AddLoadStatus($"IBackend ready: {_backend.Capabilities.Name} (device={_backend.Device})");

            // Native FP8 GEMM (Ada/Hopper): compute fp8 checkpoints directly in fp8 instead of upcasting to
            // fp16 — ~half the transformer VRAM + faster. Opt-in on the engine (default off); we enable it
            // here per the backend setting. The engine still hard-gates the actual dispatch on Fp8Executor
            // .IsSupported (SM 8.9+) and the operands being fp8, so enabling on unsupported HW is a safe no-op.
            if (_backend is CudaBackend cudaBackend)
            {
                int sm = cudaBackend.Context.ComputeCapabilityMajor * 10 + cudaBackend.Context.ComputeCapabilityMinor;
                string fp8Mode = (Settings?.NativeFp8Gemm ?? "auto").Trim().ToLowerInvariant();
                bool enableFp8 = fp8Mode switch
                {
                    "on" => true,
                    "off" => false,
                    _ => sm >= 89, // auto: Ada (8.9) / Hopper (9.0) / Blackwell (10.0+)
                };
                cudaBackend.EnableNativeFp8Gemm = enableFp8;
                AddLoadStatus($"Native FP8 GEMM: {(enableFp8 ? "enabled" : "disabled")} " +
                    $"(mode={fp8Mode}, SM {cudaBackend.Context.ComputeCapabilityMajor}.{cudaBackend.Context.ComputeCapabilityMinor}).");
            }

            _cache = new PipelineCache(_backend, Settings?.MaxCachedPipelines ?? 1);
            // MaxUsages is what the scheduler checks to decide when to route a request to a different
            // backend (BackendHandler: in-use once Usages >= MaxUsages). Mirror ComfyUI's model:
            // MaxUsages = 1 (the live gen) + OverQueue (extra waiting slots). OverQueue = -1 → MaxUsages 0
            // → UI-only (no gens). The _genLock keeps the extra slots queued rather than concurrent.
            int overQueue = Math.Max(-1, Settings?.OverQueue ?? 0);
            MaxUsages = 1 + overQueue;

            // RUNNING = alive and ready to accept generations.
            // IDLE in Swarm means "alive but currently unavailable" (e.g. remote API
            // disconnected) — it would make BackendHandler skip us for routing.
            Status = BackendStatus.RUNNING;
            AddLoadStatus("Ready. Pick a Flux .safetensors checkpoint to generate.");
            Logs.Init($"[HartsyInference] Backend #{BackendData?.ID} live on {_backend.Capabilities.Name}");
        }
        catch (Exception ex)
        {
            Status = BackendStatus.ERRORED;
            AddLoadStatus($"Init failed: {ex.Message}");
            Logs.Error($"[HartsyInference] Backend #{BackendData?.ID} init failed: {ex}");
            throw;
        }

        await Task.CompletedTask;
    }

    /// <summary>Resolve the user's compute-backend choice into a live HartsyInference IBackend.</summary>
    /// <summary>Renders an ASCII progress bar for log lines: <c>[████████░░░░░░░░░░░░] 40.0%</c>.
    /// 20 cells wide so each cell represents exactly 5% — matches the 5%-threshold throttling
    /// in <c>progressBridge</c>, so the bar visibly grows by one filled cell per logged line.</summary>
    private static string RenderProgressBar(double fraction, int width = 20)
    {
        fraction = Math.Clamp(fraction, 0.0, 1.0);
        int filled = (int)Math.Round(fraction * width);
        if (filled > width) filled = width;
        return $"[{new string('█', filled)}{new string('░', width - filled)}] {fraction * 100:F1}%";
    }

    /// <summary>Maps the <c>PreviewMethod</c> setting string to the <see cref="PreviewEncoder.Method"/>
    /// enum. Unknown values silently fall through to <see cref="PreviewEncoder.Method.Latent2Rgb"/>
    /// so a typo doesn't disable previews entirely.</summary>
    private static PreviewEncoder.Method ParsePreviewMethod(string raw) => (raw?.ToLowerInvariant()) switch
    {
        "off" or "none" or "false" or "" => PreviewEncoder.Method.Off,
        "taesd" => PreviewEncoder.Method.Taesd,
        _ => PreviewEncoder.Method.Latent2Rgb,
    };

    /// <summary>Checks NuGet for a newer HartsyInference engine and, when the <c>AutoUpdate</c> setting opts in,
    /// rebuilds this extension against it. Mirrors the ComfyUI backend's "update on launch" toggle — but because the
    /// engine is an <b>in-process</b> library (not an external process Swarm can relaunch), a fetched update is
    /// <i>staged</i> into the extension's build output and only takes effect on the next SwarmUI start. 'aggressive'
    /// clears NuGet caches and calls <see cref="Program.RequestRestart"/> so the new build loads automatically.
    /// Best-effort: any failure is logged and the current engine keeps running.</summary>
    private async Task MaybeAutoUpdateEngine()
    {
        string mode = (Settings?.AutoUpdate ?? "false").Trim().ToLowerInvariant();
        if (mode is "false" or "" or "0" or "no" or "off") return;
        bool aggressive = mode is "aggressive" or "force";
        try
        {
            string loaded = LoadedEngineVersion();
            string latest = await LatestEnginePackageVersion();
            AddLoadStatus($"Auto-update: loaded engine={loaded ?? "unknown"}, latest published={latest ?? "unknown"}.");
            if (latest is null) { AddLoadStatus("Auto-update: could not query NuGet; skipping."); return; }
            if (loaded is not null && !IsNewerAlpha(latest, loaded))
            {
                AddLoadStatus("Auto-update: engine is already up to date.");
                return;
            }

            string csproj = ExtensionProjectPath();
            if (csproj is null)
            {
                Logs.Warning("[HartsyInference] Auto-update: extension .csproj not found next to the assembly; cannot rebuild.");
                return;
            }
            AddLoadStatus($"Auto-update: rebuilding extension against engine {latest} (this can take a minute)...");
            string dir = Path.GetDirectoryName(csproj);
            if (aggressive)
            {
                await RunDotnet("nuget locals http-cache --clear", dir);
            }
            // Force-evaluate so the floating "1.0.0-alpha.*" reference re-resolves to the newest published build.
            (int code, string output) = await RunDotnet(
                $"build \"{csproj}\" -c Release /p:RestoreForceEvaluate=true", dir);
            if (code != 0)
            {
                Logs.Error($"[HartsyInference] Auto-update build failed (exit {code}):\n{output}");
                AddLoadStatus("Auto-update: rebuild failed — continuing on the current engine.");
                return;
            }
            Logs.Warning($"[HartsyInference] Engine updated to {latest}. The new DLLs are staged; a SwarmUI RESTART is required to load them (an in-process library can't hot-swap).");
            AddLoadStatus($"Auto-update: engine {latest} staged — restart SwarmUI to apply.");
            if (aggressive)
            {
                Logs.Warning("[HartsyInference] AutoUpdate=aggressive — requesting a SwarmUI restart to load the new engine.");
                Program.RequestRestart();
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] Auto-update failed: {ex.ReadableString()}");
            AddLoadStatus("Auto-update: failed (continuing with the current engine).");
        }
    }

    /// <summary>The NuGet version baked into the loaded engine assembly (e.g. "1.0.0-alpha.11"), or null.</summary>
    private static string LoadedEngineVersion()
    {
        System.Reflection.Assembly asm = typeof(IBackend).Assembly;
        string info = asm.GetCustomAttribute<System.Reflection.AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        // Informational version may carry a build-metadata suffix ("1.0.0-alpha.11+abc123"); trim it.
        if (info is not null) { int plus = info.IndexOf('+'); if (plus >= 0) info = info[..plus]; }
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
            JArray versions = parsed["versions"] as JArray;
            if (versions is null) return null;
            string best = null;
            foreach (JToken v in versions)
            {
                string s = v.ToString();
                if (best is null || IsNewerAlpha(s, best)) best = s;
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
    /// (compares the trailing alpha number; non-alpha or unparseable forms fall back to ordinal string compare).</summary>
    private static bool IsNewerAlpha(string candidate, string current)
    {
        int Num(string v)
        {
            int dash = v.LastIndexOf("alpha.", StringComparison.OrdinalIgnoreCase);
            if (dash < 0) return -1;
            string tail = v[(dash + "alpha.".Length)..];
            int dot = tail.IndexOfAny(['.', '-', '+']);
            if (dot >= 0) tail = tail[..dot];
            return int.TryParse(tail, out int n) ? n : -1;
        }
        int a = Num(candidate), b = Num(current);
        if (a >= 0 && b >= 0) return a > b;
        return string.CompareOrdinal(candidate, current) > 0;
    }

    /// <summary>Locates this extension's <c>.csproj</c> by walking up from the loaded assembly's directory.</summary>
    private static string ExtensionProjectPath()
    {
        string dir = Path.GetDirectoryName(typeof(HartsyInferenceBackend).Assembly.Location);
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            string[] found = Directory.GetFiles(dir, "*.csproj", SearchOption.TopDirectoryOnly);
            if (found.Length > 0) return found[0];
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

    /// <summary>Parse the <c>GPU_ID</c> setting (mirrors Comfy's GPU_ID field) into a device ordinal.
    /// Accepts a single number ("0", "1", …). A Comfy-style "0,1" list is tolerated but only the
    /// first ordinal is used — HartsyInference runs one GPU per backend (add a backend per GPU for
    /// multi-GPU). Falls back to 0 on empty/garbage.</summary>
    private static int ParseGpuId(string gpuId)
    {
        if (string.IsNullOrWhiteSpace(gpuId)) return 0;
        string first = gpuId.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? "0";
        if (int.TryParse(first, out int ordinal) && ordinal >= 0)
        {
            if (gpuId.Contains(','))
            {
                Logs.Warning($"[HartsyInference] GPU_ID='{gpuId}' lists multiple GPUs, but HartsyInference uses one GPU per backend — using GPU {ordinal}. Add a separate backend per GPU for multi-GPU.");
            }
            return ordinal;
        }
        Logs.Warning($"[HartsyInference] GPU_ID='{gpuId}' isn't a valid GPU number — defaulting to GPU 0.");
        return 0;
    }

    private static IBackend ConstructBackend(string choice, int ordinal, string ptxDir, string spvDir)
    {
        if (choice == "cpu")
        {
            return new CpuBackend();
        }
        if (choice == "cuda")
        {
            return new CudaBackend(ordinal, ptxDir);
        }
        if (choice == "vulkan")
        {
            return new VulkanBackend(ordinal, spvDir);
        }

        var attempts = new List<string>();
        try { return new CudaBackend(ordinal, ptxDir); }
        catch (Exception ex) { attempts.Add($"CUDA: {ex.Message}"); }

        try { return new VulkanBackend(ordinal, spvDir); }
        catch (Exception ex) { attempts.Add($"Vulkan: {ex.Message}"); }

        Logs.Info($"[HartsyInference] Auto-backend falling through to CPU. Tried: {string.Join(" | ", attempts)}");
        return new CpuBackend();
    }

    public override async Task Shutdown()
    {
        Status = BackendStatus.DISABLED;
        _cancelCts?.Cancel();
        _cancelCts = null;

        _cache?.DisposeAll();
        _cache = null;

        (_backend as IDisposable)?.Dispose();
        _backend = null;

        CurrentModelName = null;
        await Task.CompletedTask;
    }

    public override async Task<bool> LoadModel(T2IModel model, T2IParamInput input)
    {
        if (model is null) return false;

        string compat = model.ModelClass?.CompatClass?.ID;

        // Already cached? Skip.
        if (CurrentModelName == model.Name && IsCached(compat, model.Name))
        {
            return true;
        }

        // If input is null we can't read encoder/VAE selections — defer the heavy
        // load to GenerateLive where input is guaranteed. Just confirm the architecture.
        if (input is null)
        {
            CurrentModelName = model.Name;
            return ModelSupport.IsArchitectureSupported(compat);
        }

        Status = BackendStatus.LOADING;
        try
        {
            if (compat == Sd15Loader.Sd15CompatClassId)
            {
                await Task.Run(() =>
                {
                    Sd15CacheEntry entry = Sd15Loader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutSd15(entry);
                });
            }
            else if (compat == SdxlLoader.SdxlCompatClassId)
            {
                await Task.Run(() =>
                {
                    SdxlCacheEntry entry = SdxlLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutSdxl(entry);
                });
            }
            else if (Sd3Loader.IsSd3Compat(compat))
            {
                await Task.Run(() =>
                {
                    Sd3CacheEntry entry = Sd3Loader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutSd3(entry);
                });
            }
            else if (compat == FluxLoader.Flux1CompatClassId)
            {
                await Task.Run(() =>
                {
                    FluxCacheEntry entry = FluxLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutFlux(entry);
                });
            }
            else if (Flux2Loader.IsFlux2Compat(compat))
            {
                await Task.Run(() =>
                {
                    Flux2CacheEntry entry = Flux2Loader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutFlux2(entry);
                });
            }
            else if (compat == ChromaLoader.ChromaCompatClassId)
            {
                await Task.Run(() =>
                {
                    ChromaCacheEntry entry = ChromaLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutChroma(entry);
                });
            }
            else if (compat == ChromaRadianceLoader.ChromaRadianceCompatClassId)
            {
                await Task.Run(() =>
                {
                    ChromaRadianceCacheEntry entry = ChromaRadianceLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutChromaRadiance(entry);
                });
            }
            else if (compat == ZetaChromaLoader.ZetaChromaCompatClassId)
            {
                await Task.Run(() =>
                {
                    ZetaChromaCacheEntry entry = ZetaChromaLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutZetaChroma(entry);
                });
            }
            else if (compat == AuraFlowLoader.AuraFlowCompatClassId)
            {
                await Task.Run(() =>
                {
                    AuraFlowCacheEntry entry = AuraFlowLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutAuraFlow(entry);
                });
            }
            else if (compat == FLiteLoader.FLiteCompatClassId)
            {
                await Task.Run(() =>
                {
                    FLiteCacheEntry entry = FLiteLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutFLite(entry);
                });
            }
            else if (compat == Ideogram4Loader.Ideogram4CompatClassId)
            {
                await Task.Run(() =>
                {
                    Ideogram4CacheEntry entry = Ideogram4Loader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutIdeogram4(entry);
                });
            }
            else if (compat == ErnieImageLoader.ErnieImageCompatClassId)
            {
                await Task.Run(() =>
                {
                    ErnieImageCacheEntry entry = ErnieImageLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutErnieImage(entry);
                });
            }
            else if (compat == ZImageLoader.ZImageCompatClassId)
            {
                await Task.Run(() =>
                {
                    ZImageCacheEntry entry = ZImageLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutZImage(entry);
                });
            }
            else if (compat == AnimaLoader.AnimaCompatClassId)
            {
                await Task.Run(() =>
                {
                    AnimaCacheEntry entry = AnimaLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutAnima(entry);
                });
            }
            else if (compat == HiDreamLoader.HiDreamI1CompatClassId)
            {
                await Task.Run(() =>
                {
                    HiDreamCacheEntry entry = HiDreamLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutHiDream(entry);
                });
            }
            else if (compat == QwenImageLoader.QwenImageCompatClassId)
            {
                await Task.Run(() =>
                {
                    QwenImageCacheEntry entry = QwenImageLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutQwenImage(entry);
                });
            }
            else if (compat == WanVideoLoader.Wan22_5BCompatClassId
                || compat == WanVideoLoader.Wan21_1_3BCompatClassId
                || compat == WanVideoLoader.Wan21_14BCompatClassId)
            {
                await Task.Run(() =>
                {
                    // Every Wan conditioning variant shares the plain-Wan compat class but drives a different engine
                    // pipeline; route off the detected variant (VACE control / Animate / S2V / plain T2V-I2V).
                    switch (WanModelVariants.Detect(model))
                    {
                        case WanModelVariants.Variant.Vace:
                            _cache.PutWanVace(WanVaceLoader.Load(_backend, model, input, msg => AddLoadStatus(msg)));
                            break;
                        case WanModelVariants.Variant.Animate:
                            _cache.PutWanAnimate(WanAnimateLoader.Load(_backend, model, input, msg => AddLoadStatus(msg)));
                            break;
                        case WanModelVariants.Variant.S2V:
                            _cache.PutWanS2V(WanS2VLoader.Load(_backend, model, input, msg => AddLoadStatus(msg)));
                            break;
                        default:
                            _cache.PutWanVideo(WanVideoLoader.Load(_backend, model, input, msg => AddLoadStatus(msg)));
                            break;
                    }
                });
            }
            else if (compat == LtxVideoLoader.LtxVideoCompatClassId)
            {
                await Task.Run(() =>
                {
                    LtxVideoCacheEntry entry = LtxVideoLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutLtxVideo(entry);
                });
            }
            else if (compat == LtxVideo2Loader.LtxVideo2CompatClassId)
            {
                await Task.Run(() =>
                {
                    LtxVideo2CacheEntry entry = LtxVideo2Loader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutLtxVideo2(entry);
                });
            }
            else if (compat == AceStepLoader.AceStepCompatClassId)
            {
                await Task.Run(() =>
                {
                    if (model.ModelClass?.ID == AceStepLoader.AceStepV1ClassId)
                    {
                        AceStepCacheEntry entry = AceStepLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                        _cache.PutAceStep(entry);
                    }
                    else
                    {
                        AceStep15CacheEntry entry = AceStep15Loader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                        _cache.PutAceStep15(entry);
                    }
                });
            }
            else if (compat == MusicGenLoader.MusicGenCompatClassId)
            {
                await Task.Run(() =>
                {
                    MusicGenCacheEntry entry = MusicGenLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutMusicGen(entry);
                });
            }
            else if (compat == YueLoader.YueCompatClassId)
            {
                await Task.Run(() =>
                {
                    YueCacheEntry entry = YueLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutYue(entry);
                });
            }
            else if (compat == LanceLoader.LanceCompatClassId || compat == LanceLoader.LanceVideoCompatClassId)
            {
                await Task.Run(() =>
                {
                    LanceCacheEntry entry = LanceLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutLance(entry);
                });
            }
            else if (compat == LensLoader.LensCompatClassId)
            {
                await Task.Run(() =>
                {
                    LensCacheEntry entry = LensLoader.Load(_backend, model, input, msg => AddLoadStatus(msg));
                    _cache.PutLens(entry);
                });
            }
            else
            {
                Logs.Warning($"[HartsyInference] LoadModel: architecture '{compat}' not supported. Returning false so another backend can handle it.");
                return false;
            }

            CurrentModelName = model.Name;
            return true;
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] LoadModel failed for '{model.Name}' ({compat}): {ex}");
            return false;
        }
        finally
        {
            Status = BackendStatus.RUNNING;
        }
    }

    public override async Task<Image[]> Generate(T2IParamInput input)
    {
        var images = new List<Image>();
        await GenerateLive(input, "single", obj =>
        {
            if (obj is Image img) images.Add(img);
            else if (obj is T2IEngine.ImageOutput o && o.File is Image of) images.Add(of);
        });
        return images.ToArray();
    }

    public override async Task GenerateLive(T2IParamInput input, string batchId, Action<object> takeOutput)
    {
        // Status stays RUNNING throughout — it's the resting healthy state, not a
        // "currently busy" flag. BackendData.Usages is what tracks active utilization.
        //
        // Serialize: when OverQueue > 0 the scheduler may dispatch multiple jobs to us; run them one at
        // a time. Extra jobs wait here (holding a Usage slot, exactly like ComfyUI's queue) until the
        // current generation releases the lock. Acquire BEFORE touching _cancelCts so the per-gen token
        // belongs to the job that actually holds the GPU.
        await _genLock.WaitAsync();
        // Link to the input's InterruptToken (session.SessInterrupt) so the gen-page "stop"
        // button actually cancels us — that's the token Swarm trips on stop, exactly like the
        // ComfyUI backend passes user_input.InterruptToken into its job loop. Without the link
        // _cancelCts only ever fired from Shutdown(), so stop did nothing mid-generation.
        _cancelCts = CancellationTokenSource.CreateLinkedTokenSource(input.InterruptToken);
        long startMs = Environment.TickCount64;

        try
        {
            T2IModel model = input.Get(T2IParamTypes.Model)
                ?? throw new InvalidOperationException("No model selected.");
            string compat = model.ModelClass?.CompatClass?.ID;
            if (!ModelSupport.IsArchitectureSupported(compat))
            {
                throw new InvalidOperationException($"HartsyInference doesn't yet support architecture '{compat}'.");
            }

            // Verbose: full job acceptance line with the parameters that drive timing /
            // memory. Mirrors Comfy backend's "Will await a job, do parse..." +
            // "Will use workflow: ..." pair — same density of information.
            int wantedSteps = input.Get(T2IParamTypes.Steps);
            int wantedW = input.Get(T2IParamTypes.Width);
            int wantedH = input.Get(T2IParamTypes.Height);
            long wantedSeed = input.Get(T2IParamTypes.Seed);
            string promptPreview = input.Get(T2IParamTypes.Prompt) ?? "";
            if (promptPreview.Length > 80) promptPreview = promptPreview[..80] + "…";
            Logs.Verbose($"[HartsyInference] Backend #{BackendData?.ID} accepted job batch='{batchId}' " +
                $"model='{model.Name}' compat='{compat}' {wantedW}x{wantedH} steps={wantedSteps} seed={wantedSeed} " +
                $"prompt=\"{promptPreview}\"");

            // Ensure model is loaded. Log cache hit/miss so the slow first-gen vs fast
            // repeat-gen pattern is obvious in the timeline.
            bool cacheHit = CurrentModelName == model.Name && IsCached(compat, model.Name);
            if (cacheHit)
            {
                Logs.Verbose($"[HartsyInference] Model cache HIT — '{model.Name}' already resident, skipping load.");
            }
            else
            {
                Logs.Verbose($"[HartsyInference] Model cache MISS — loading '{model.Name}' (current='{CurrentModelName ?? "<none>"}')");
                long loadStartMs = Environment.TickCount64;
                if (!await LoadModel(model, input))
                {
                    throw new InvalidOperationException($"Failed to load model '{model.Name}'.");
                }
                Logs.Verbose($"[HartsyInference] Model load complete in {Environment.TickCount64 - loadStartMs}ms.");
            }

            CancellationToken cancel = _cancelCts.Token;
            // The pipeline now fires progress at heartbeat cadence (~500 ms) plus on each
            // step / VAE-tile completion — that's hundreds of callbacks per generation.
            // For takeOutput we forward EVERY one (the UI bar wants smooth motion). For
            // the log line we throttle to 5%-boundary crossings so the developer sees
            // ~20 progress lines per gen instead of 200, with an ASCII bar attached.
            // The preview encoder is throttled internally (≤4/sec).
            int lastLoggedThreshold = -1;
            PreviewEncoder.Method previewMethod = ParsePreviewMethod(Settings?.PreviewMethod);
            PreviewEncoder previewEncoder = new(
                previewMethod,
                backend: previewMethod == PreviewEncoder.Method.Taesd ? _backend : null,
                taesdResolver: previewMethod == PreviewEncoder.Method.Taesd
                    ? (arch => TaesdResolver.Resolve(arch, msg => AddLoadStatus(msg)))
                    : null);
            Action<GenerationProgress> progressBridge = (p) =>
            {
                double overall = p.OverallPercent >= 0 ? p.OverallPercent : (double)p.Step / Math.Max(p.TotalSteps, 1);
                int threshold = (int)(overall * 20); // 0..20 buckets of 5%
                if (threshold != lastLoggedThreshold)
                {
                    lastLoggedThreshold = threshold;
                    Logs.Verbose($"[HartsyInference] Progress batch='{batchId}' {RenderProgressBar(overall)} step {p.Step}/{p.TotalSteps}");
                }
                JObject previewObj = previewEncoder.Enabled
                    ? previewEncoder.TryEncode(p, batchId, overall)
                    : null;
                // When we have a preview encoded, send the richer JObject (preview + percent
                // in one message — matches Comfy's contract). Otherwise just the percent.
                if (previewObj is not null)
                {
                    takeOutput(previewObj);
                }
                else
                {
                    takeOutput(new JObject
                    {
                        ["batch_index"] = batchId,
                        ["overall_percent"] = overall,
                        ["current_percent"] = overall,
                    });
                }
            };

            // Resolve LoRAs once on the calling thread (touches Swarm's model handler;
            // throws SwarmUserErrorException for missing files which Swarm displays cleanly).
            List<LoraResolver.LoraSpec> loras = LoraResolver.Resolve(input);

            // Resolve refiner spec early. For StepSwap on SDXL we need the refiner UNet
            // available DURING the base call (not after) — pre-load the refiner cache entry
            // here on the calling thread so AddLoadStatus diagnostics surface in the UI,
            // then pass the refiner UNet through to the SDXL pipeline.
            RefinerResolver.RefinerSpec refinerSpec = RefinerResolver.Resolve(input);
            RefinerSwapConfig refinerSwapForSdxl = null;
            bool refinerHandledInPipeline = false;
            if (refinerSpec is not null
                && refinerSpec.Method == "StepSwap"
                && compat == SdxlLoader.SdxlCompatClassId)
            {
                RefinerCacheEntry refinerEntryForSwap = _cache.TryGetRefiner(refinerSpec.Model.Name);
                if (refinerEntryForSwap is null)
                {
                    refinerEntryForSwap = RefinerLoader.Load(_backend, refinerSpec.Model, msg => AddLoadStatus(msg));
                    _cache.PutRefiner(refinerEntryForSwap);
                }
                refinerSwapForSdxl = new RefinerSwapConfig
                {
                    RefinerUnet = refinerEntryForSwap.RefinerUnet,
                    Strength = refinerSpec.Strength,
                };
                refinerHandledInPipeline = true;
            }

            // IP-Adapter (SDXL + SD 1.5). Resolve on the calling thread so AddLoadStatus
            // and the IPA + CLIP-Vision auto-download surface in the UI; image projection
            // runs once here (before the lambda) and the resulting tokens flow through the
            // arch-specific loader. `using` ensures the spec's owned image-token tensors
            // are disposed at the end of the try block.
            HartsyInference.Diffusion.Adapters.IpAdapterBaseModel? ipaBaseModel = null;
            if (compat == SdxlLoader.SdxlCompatClassId) ipaBaseModel = HartsyInference.Diffusion.Adapters.IpAdapterBaseModel.Sdxl;
            else if (compat == Sd15Loader.Sd15CompatClassId) ipaBaseModel = HartsyInference.Diffusion.Adapters.IpAdapterBaseModel.Sd15;
            using IpAdapterResolver.ResolvedSpec ipaSpec = ipaBaseModel.HasValue
                ? IpAdapterResolver.Resolve(input, _backend, ipaBaseModel.Value,
                    msg => AddLoadStatus(msg),
                    cacheLookup: path => _cache.TryGetIpAdapter(path),
                    cachePut: entry => _cache.PutIpAdapter(entry))
                : null;

            Image[] images = await Task.Run(() =>
            {
                if (compat == Sd15Loader.Sd15CompatClassId)
                {
                    Sd15CacheEntry entry = _cache.TryGetSd15(model.Name)
                        ?? throw new InvalidOperationException("SD 1.5 model loaded but not in cache.");
                    return loras.Count > 0
                        ? Sd15Loader.GenerateWithLoras(entry, loras, _backend, input, progressBridge, cancel, ipaSpec?.Conditionings)
                        : Sd15Loader.Generate(entry, _backend, input, progressBridge, cancel, ipaSpec?.Conditionings);
                }
                if (compat == SdxlLoader.SdxlCompatClassId)
                {
                    SdxlCacheEntry entry = _cache.TryGetSdxl(model.Name)
                        ?? throw new InvalidOperationException("SDXL model loaded but not in cache.");
                    return loras.Count > 0
                        ? SdxlLoader.GenerateWithLoras(entry, loras, _backend, input, progressBridge, cancel, refinerSwapForSdxl, ipaSpec?.Conditionings)
                        : SdxlLoader.Generate(entry, input, progressBridge, cancel, refinerSwapForSdxl, ipaSpec?.Conditionings);
                }
                if (Sd3Loader.IsSd3Compat(compat))
                {
                    Sd3CacheEntry entry = _cache.TryGetSd3(model.Name)
                        ?? throw new InvalidOperationException("SD3 model loaded but not in cache.");
                    // LoRA path for SD3 not implemented — refused upfront in IsValidForThisBackend.
                    return Sd3Loader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == FluxLoader.Flux1CompatClassId)
                {
                    FluxCacheEntry entry = _cache.TryGetFlux(model.Name)
                        ?? throw new InvalidOperationException("Flux model loaded but not in cache.");
                    return loras.Count > 0
                        ? FluxLoader.GenerateWithLoras(entry, loras, _backend, input, progressBridge, cancel)
                        : FluxLoader.Generate(entry, input, progressBridge, cancel);
                }
                if (Flux2Loader.IsFlux2Compat(compat))
                {
                    Flux2CacheEntry entry = _cache.TryGetFlux2(model.Name)
                        ?? throw new InvalidOperationException("Flux.2 model loaded but not in cache.");
                    // LoRA path for Flux.2 not implemented — refused upfront in IsValidForThisBackend.
                    return Flux2Loader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == ChromaLoader.ChromaCompatClassId)
                {
                    ChromaCacheEntry entry = _cache.TryGetChroma(model.Name)
                        ?? throw new InvalidOperationException("Chroma model loaded but not in cache.");
                    return ChromaLoader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == ChromaRadianceLoader.ChromaRadianceCompatClassId)
                {
                    ChromaRadianceCacheEntry entry = _cache.TryGetChromaRadiance(model.Name)
                        ?? throw new InvalidOperationException("Chroma Radiance model loaded but not in cache.");
                    return ChromaRadianceLoader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == ZetaChromaLoader.ZetaChromaCompatClassId)
                {
                    ZetaChromaCacheEntry entry = _cache.TryGetZetaChroma(model.Name)
                        ?? throw new InvalidOperationException("Zeta-Chroma model loaded but not in cache.");
                    return ZetaChromaLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == AuraFlowLoader.AuraFlowCompatClassId)
                {
                    AuraFlowCacheEntry entry = _cache.TryGetAuraFlow(model.Name)
                        ?? throw new InvalidOperationException("AuraFlow model loaded but not in cache.");
                    return AuraFlowLoader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == FLiteLoader.FLiteCompatClassId)
                {
                    FLiteCacheEntry entry = _cache.TryGetFLite(model.Name)
                        ?? throw new InvalidOperationException("F-Lite model loaded but not in cache.");
                    return FLiteLoader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == Ideogram4Loader.Ideogram4CompatClassId)
                {
                    Ideogram4CacheEntry entry = _cache.TryGetIdeogram4(model.Name)
                        ?? throw new InvalidOperationException("Ideogram 4 model loaded but not in cache.");
                    // LoRA / img2img / inpaint / IPA / ControlNet not implemented for Ideogram 4 —
                    // refused upfront in IsValidForThisBackend (not in any per-feature allow list).
                    return Ideogram4Loader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == ErnieImageLoader.ErnieImageCompatClassId)
                {
                    ErnieImageCacheEntry entry = _cache.TryGetErnieImage(model.Name)
                        ?? throw new InvalidOperationException("ERNIE-Image model loaded but not in cache.");
                    return ErnieImageLoader.Generate(entry, input, progressBridge, cancel);
                }
                if (compat == ZImageLoader.ZImageCompatClassId)
                {
                    ZImageCacheEntry entry = _cache.TryGetZImage(model.Name)
                        ?? throw new InvalidOperationException("Z-Image model loaded but not in cache.");
                    // LoRA path for Z-Image not implemented — refused upfront in IsValidForThisBackend.
                    return ZImageLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == AnimaLoader.AnimaCompatClassId)
                {
                    AnimaCacheEntry entry = _cache.TryGetAnima(model.Name)
                        ?? throw new InvalidOperationException("Anima model loaded but not in cache.");
                    // LoRA / img2img / inpaint not implemented for Anima — refused upfront in IsValidForThisBackend.
                    return AnimaLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == HiDreamLoader.HiDreamI1CompatClassId)
                {
                    HiDreamCacheEntry entry = _cache.TryGetHiDream(model.Name)
                        ?? throw new InvalidOperationException("HiDream model loaded but not in cache.");
                    // LoRA / img2img / inpaint not implemented for HiDream — refused upfront in IsValidForThisBackend.
                    return HiDreamLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == QwenImageLoader.QwenImageCompatClassId)
                {
                    QwenImageCacheEntry entry = _cache.TryGetQwenImage(model.Name)
                        ?? throw new InvalidOperationException("Qwen-Image model loaded but not in cache.");
                    // LoRA / img2img (edit) / inpaint not implemented for Qwen-Image — refused upfront in IsValidForThisBackend.
                    return QwenImageLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == WanVideoLoader.Wan22_5BCompatClassId
                    || compat == WanVideoLoader.Wan21_1_3BCompatClassId
                    || compat == WanVideoLoader.Wan21_14BCompatClassId)
                {
                    // Route to the loader matching the detected Wan conditioning variant (must match LoadModel).
                    switch (WanModelVariants.Detect(model))
                    {
                        case WanModelVariants.Variant.Vace:
                        {
                            WanVaceCacheEntry vaceEntry = _cache.TryGetWanVace(model.Name)
                                ?? throw new InvalidOperationException("Wan VACE model loaded but not in cache.");
                            return WanVaceLoader.Generate(vaceEntry, _backend, input, progressBridge, cancel);
                        }
                        case WanModelVariants.Variant.Animate:
                        {
                            WanAnimateCacheEntry animEntry = _cache.TryGetWanAnimate(model.Name)
                                ?? throw new InvalidOperationException("Wan Animate model loaded but not in cache.");
                            return WanAnimateLoader.Generate(animEntry, _backend, input, progressBridge, cancel);
                        }
                        case WanModelVariants.Variant.S2V:
                        {
                            WanS2VCacheEntry s2vEntry = _cache.TryGetWanS2V(model.Name)
                                ?? throw new InvalidOperationException("Wan S2V model loaded but not in cache.");
                            return WanS2VLoader.Generate(s2vEntry, _backend, input, progressBridge, cancel);
                        }
                        default:
                        {
                            WanVideoCacheEntry entry = _cache.TryGetWanVideo(model.Name)
                                ?? throw new InvalidOperationException("Wan video model loaded but not in cache.");
                            // Video-extend / end-frame not implemented for Wan — refused upfront in IsValidForThisBackend.
                            return loras.Count > 0
                                ? WanVideoLoader.GenerateWithLoras(entry, loras, _backend, input, progressBridge, cancel)
                                : WanVideoLoader.Generate(entry, _backend, input, progressBridge, cancel);
                        }
                    }
                }
                if (compat == LtxVideoLoader.LtxVideoCompatClassId)
                {
                    LtxVideoCacheEntry entry = _cache.TryGetLtxVideo(model.Name)
                        ?? throw new InvalidOperationException("LTX-Video model loaded but not in cache.");
                    // I2V / LoRA / audio not implemented for LTX — refused upfront in IsValidForThisBackend.
                    return LtxVideoLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == LtxVideo2Loader.LtxVideo2CompatClassId)
                {
                    LtxVideo2CacheEntry entry = _cache.TryGetLtxVideo2(model.Name)
                        ?? throw new InvalidOperationException("LTX-2 model loaded but not in cache.");
                    return LtxVideo2Loader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == AceStepLoader.AceStepCompatClassId)
                {
                    if (model.ModelClass?.ID == AceStepLoader.AceStepV1ClassId)
                    {
                        AceStepCacheEntry entry = _cache.TryGetAceStep(model.Name)
                            ?? throw new InvalidOperationException("ACE-Step model loaded but not in cache.");
                        // LoRA / reference-audio refused upfront in IsValidForThisBackend.
                        return AceStepLoader.Generate(entry, _backend, input, progressBridge, cancel);
                    }
                    AceStep15CacheEntry entry15 = _cache.TryGetAceStep15(model.Name)
                        ?? throw new InvalidOperationException("ACE-Step 1.5 model loaded but not in cache.");
                    // Turbo: fixed 8-step no-CFG; timbre/cover hooks are engine Phase-2 TODOs.
                    return AceStep15Loader.Generate(entry15, _backend, input, progressBridge, cancel);
                }
                if (compat == MusicGenLoader.MusicGenCompatClassId)
                {
                    MusicGenCacheEntry entry = _cache.TryGetMusicGen(model.Name)
                        ?? throw new InvalidOperationException("MusicGen model loaded but not in cache.");
                    return MusicGenLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == YueLoader.YueCompatClassId)
                {
                    YueCacheEntry entry = _cache.TryGetYue(model.Name)
                        ?? throw new InvalidOperationException("YuE model loaded but not in cache.");
                    return YueLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == LanceLoader.LanceCompatClassId || compat == LanceLoader.LanceVideoCompatClassId)
                {
                    LanceCacheEntry entry = _cache.TryGetLance(model.Name)
                        ?? throw new InvalidOperationException("Lance model loaded but not in cache.");
                    // LoRA / editing / I2V refused upfront in IsValidForThisBackend (T2I + T2V only).
                    return LanceLoader.Generate(entry, _backend, input, progressBridge, cancel);
                }
                if (compat == LensLoader.LensCompatClassId)
                {
                    LensCacheEntry entry = _cache.TryGetLens(model.Name)
                        ?? throw new InvalidOperationException("Lens model loaded but not in cache.");
                    // LoRA / img2img / inpaint not implemented for Lens — refused upfront in IsValidForThisBackend.
                    return LensLoader.Generate(entry, input, progressBridge, cancel);
                }
                throw new InvalidOperationException($"No generator wired for compat '{compat}'.");
            }, cancel);

            // Optional: SDXL refiner pass over each base image (PostApply).
            // The refiner accepts ANY base architecture's pixel output, so this works
            // regardless of which base pipeline produced `images`. Skipped when StepSwap
            // was already applied in-pipeline above (refinerHandledInPipeline=true).
            if (refinerSpec is not null && !refinerHandledInPipeline)
            {
                images = await Task.Run(() => RefinePass(images, refinerSpec, input, progressBridge, cancel), cancel);
            }

            // Segment refinement (<segment:yolo-...>): detect → mask → re-denoise the region with
            // the segment's prompt via the arch's existing img2img+inpaint path. Validation has
            // already guaranteed the arch is inpaint-capable and all targets are YOLO.
            if (SegmentRefiner.HasSegments(input))
            {
                Image[] preSegment = images;
                images = await Task.Run(() => SegmentRefiner.Apply(
                    _backend, preSegment, input,
                    reGenerate: segInput => RunSegmentRedenoise(model, compat, segInput, cancel),
                    log: msg => Logs.Verbose($"[HartsyInference] {msg}"),
                    cancel: cancel), cancel);
            }

            // Yield final images.
            long totalMs = Environment.TickCount64 - startMs;
            int idx = 0;
            foreach (Image img in images)
            {
                Logs.Verbose($"[HartsyInference] Yielding image {idx + 1}/{images.Length} batch='{batchId}' (genTime={totalMs}ms, bytes={img.RawData?.Length ?? 0})");
                takeOutput(new T2IEngine.ImageOutput
                {
                    File = img,
                    IsReal = true,
                    GenTimeMS = totalMs,
                });
                idx++;
            }
            Logs.Verbose($"[HartsyInference] Job batch='{batchId}' complete: {images.Length} image(s) in {totalMs}ms.");
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
            // No Status change here: we stay RUNNING (alive+ready) regardless of
            // whether the generation succeeded or failed. ERRORED would be wrong
            // because the *backend* is still healthy — only this one request errored.
        }
    }

    /// <summary>Run the SDXL refiner over each image produced by the base stage.
    /// Loads the refiner pipeline lazily (cached after first use), then runs
    /// <see cref="RefinerLoader.Refine"/> per image. Validation in
    /// <see cref="IsValidForThisBackend"/> guarantees the refiner model is the
    /// official SDXL refiner and the method is "PostApply" by the time we get here.</summary>
    private Image[] RefinePass(
        Image[] baseImages,
        RefinerResolver.RefinerSpec spec,
        T2IParamInput input,
        Action<GenerationProgress> progressBridge,
        CancellationToken cancel)
    {
        if (baseImages is null || baseImages.Length == 0) return baseImages;

        // Lazy-load the refiner pipeline (cached for repeat generations).
        RefinerCacheEntry refinerEntry = _cache.TryGetRefiner(spec.Model.Name);
        if (refinerEntry is null)
        {
            refinerEntry = RefinerLoader.Load(_backend, spec.Model, msg => AddLoadStatus(msg));
            _cache.PutRefiner(refinerEntry);
        }

        Image[] refined = new Image[baseImages.Length];
        for (int i = 0; i < baseImages.Length; i++)
        {
            cancel.ThrowIfCancellationRequested();
            Logs.Info($"[HartsyInference] Refining image {i + 1}/{baseImages.Length} (strength={spec.Strength}, steps={spec.Steps}, cfg={spec.CfgScale}).");
            refined[i] = RefinerLoader.Refine(
                refinerEntry, baseImages[i], input,
                steps: spec.Steps,
                strength: spec.Strength,
                cfgScale: spec.CfgScale,
                onProgress: progressBridge,
                cancel: cancel);
        }
        return refined;
    }

    /// <summary>Re-denoise one segment region by dispatching the cloned input (InitImage + MaskImage +
    /// segment prompt already set) through the active architecture's existing img2img+inpaint path.
    /// Only the inpaint-capable arches reach here (validation enforces it). Progress is swallowed —
    /// the base gen already drove the UI bar; segment passes are short.</summary>
    private Image[] RunSegmentRedenoise(T2IModel model, string compat, T2IParamInput segInput, CancellationToken cancel)
    {
        static void NoProgress(GenerationProgress _) { }
        if (compat == SdxlLoader.SdxlCompatClassId)
        {
            SdxlCacheEntry entry = _cache.TryGetSdxl(model.Name)
                ?? throw new InvalidOperationException("SDXL model not in cache for segment re-denoise.");
            return SdxlLoader.Generate(entry, segInput, NoProgress, cancel, refinerSwap: null, ipAdapters: null);
        }
        if (compat == FluxLoader.Flux1CompatClassId)
        {
            FluxCacheEntry entry = _cache.TryGetFlux(model.Name)
                ?? throw new InvalidOperationException("Flux model not in cache for segment re-denoise.");
            return FluxLoader.Generate(entry, segInput, NoProgress, cancel);
        }
        if (Sd3Loader.IsSd3Compat(compat))
        {
            Sd3CacheEntry entry = _cache.TryGetSd3(model.Name)
                ?? throw new InvalidOperationException("SD3 model not in cache for segment re-denoise.");
            return Sd3Loader.Generate(entry, segInput, NoProgress, cancel);
        }
        throw new InvalidOperationException($"Segment re-denoise not supported for compat '{compat}'.");
    }

    /// <summary>Swarm prompt-syntax tags that a backend must service via conditioning /
    /// regional machinery we don't have yet. Comfy handles these through
    /// SwarmClipTextEncodeAdvanced + mask nodes; if we let them through, the tag text
    /// would be fed RAW into the text encoder and silently corrupt the generation.
    /// Refuse instead. NOTE: <c>segment</c> is handled separately (YOLO segment refinement
    /// IS supported on inpaint-capable arches) so it's NOT in this regex.</summary>
    private static readonly System.Text.RegularExpressions.Regex UnsupportedPromptSyntax =
        new(@"<(object|region|clear|embed)\s*:|<break\s*>",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>Architectures whose pipelines have the inpaint blend path — the only ones that
    /// can run YOLO segment re-denoise (which is img2img + mask under the hood).</summary>
    private static bool IsInpaintCapable(string compat) =>
        compat == SdxlLoader.SdxlCompatClassId
        || compat == FluxLoader.Flux1CompatClassId
        || Sd3Loader.IsSd3Compat(compat);

    public override bool IsValidForThisBackend(T2IParamInput input)
    {
        T2IModel model = input.Get(T2IParamTypes.Model);
        if (model is null) return true; // let other validators speak

        string compat = model.ModelClass?.CompatClass?.ID;

        // Prompt-syntax features we can't service yet: refuse with the tag named rather
        // than silently feeding the tag text into the tokenizer (silent bad output is
        // worse than a clean refusal that routes the request to a Comfy backend if one
        // is configured).
        foreach (string promptText in new[] { input.Get(T2IParamTypes.Prompt), input.Get(T2IParamTypes.NegativePrompt) })
        {
            if (string.IsNullOrEmpty(promptText)) continue;
            System.Text.RegularExpressions.Match match = UnsupportedPromptSyntax.Match(promptText);
            if (match.Success)
            {
                string tag = match.Value.TrimStart('<').TrimEnd(':', '>', ' ');
                input.RefusalReasons.Add(
                    $"HartsyInference: the '<{tag}:...>' prompt syntax isn't supported yet "
                    + "(needs regional-conditioning machinery). Remove the tag, "
                    + "or use a ComfyUI backend for this generation.");
                return false;
            }
        }

        // Segment refinement (<segment:yolo-...>): supported on inpaint-capable arches with YOLO
        // targets. Refuse cleanly for non-inpaint arches or non-YOLO (text/CLIP-Seg) targets.
        if (SegmentRefiner.HasSegments(input))
        {
            if (!IsInpaintCapable(compat))
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: '<segment:>' refinement needs an inpaint-capable architecture "
                    + $"(SDXL, Flux, or SD3); got '{compat}'. Remove the segment tag or switch models.");
                return false;
            }
            if (!SegmentRefiner.AllSegmentsAreYolo(input))
            {
                input.RefusalReasons.Add(
                    "HartsyInference: only '<segment:yolo-MODEL>' targets are supported (the engine has YOLO "
                    + "detection but no CLIP-Seg/text segmentation head). Use a yolo- target, or a ComfyUI backend.");
                return false;
            }
        }

        // Variation seed: wired for SD 1.5 + SDXL (spatial [1,4,H/8,W/8]) and Flux
        // ([1,16,H/8,W/8] unpacked) via InitialNoise slerp. SD3 doesn't inject
        // InitialNoise in its pipeline yet (engine work), so it stays refused.
        if (input.TryGet(T2IParamTypes.VariationSeedStrength, out double varStrength) && varStrength > 0
            && input.TryGet(T2IParamTypes.VariationSeed, out long _))
        {
            bool varSeedSupported = compat == Sd15Loader.Sd15CompatClassId
                || compat == SdxlLoader.SdxlCompatClassId
                || compat == FluxLoader.Flux1CompatClassId;
            if (!varSeedSupported)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: Variation Seed is currently supported on SD 1.5, SDXL, and Flux (got '{compat}'). "
                    + "Disable the Variation Seed group or pick a supported model.");
                return false;
            }
        }
        if (!ModelSupport.IsArchitectureSupported(compat))
        {
            // Be specific about WHY this architecture is unsupported. HartsyInference itself
            // has pipelines for many more architectures than the SwarmUI extension currently
            // dispatches to — we just haven't wired the per-arch loader (text-encoder selection,
            // VAE auto-download, tokenizer setup, IsValidForThisBackend rules) yet. Distinguish
            // those cases from architectures HartsyInference doesn't support at all.
            string status = ModelSupport.WhyNotSupported(compat);
            input.RefusalReasons.Add($"HartsyInference: {status}");
            return false;
        }

        // The two-stage "generate an image, then animate it with a separate video model" flow
        // (Comfy's ImageToVideoGenInfo, driven by the Video Model param) isn't implemented for ANY
        // architecture — refuse upfront rather than silently returning a still image.
        if (input.Get(T2IParamTypes.VideoModel) is not null)
        {
            input.RefusalReasons.Add(
                "HartsyInference: the image-then-animate flow (Video Model param) isn't supported. "
                + "For image-to-video, select a Wan 2.2 TI2V model as the main model and set an Init Image instead.");
            return false;
        }

        // Audio architectures (ACE-Step v1 + v1.5, MusicGen, YuE): both ACE generations now have
        // engine pipelines — v1 checkpoints route to AceStepLoader, anything else under the
        // ace-step-1_5 compat routes to AceStep15Loader. Refuse image-only features that make no
        // sense for audio.
        if (compat == AceStepLoader.AceStepCompatClassId
            || compat == MusicGenLoader.MusicGenCompatClassId
            || compat == YueLoader.YueCompatClassId)
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
            if (input.TryGet(T2IParamTypes.Loras, out List<string> audioLoras) && audioLoras is not null && audioLoras.Count > 0)
            {
                input.RefusalReasons.Add("HartsyInference: LoRAs aren't supported for music models yet. Remove the LoRA selection.");
                return false;
            }
            return true;
        }

        // Video architectures: Wan2.2 TI2V-5B does T2V + I2V (init image → VAE-encoded first-frame
        // conditioning); LTX-Video is T2V only (no image hook in the transformer yet).
        //   - End frame / video-extend / audio params: Comfy-only flows we don't implement.
        //   - Refiner over a video output would feed mp4 bytes into the SDXL refiner — refuse.
        bool isVideoArch = compat == WanVideoLoader.Wan22_5BCompatClassId
            || compat == WanVideoLoader.Wan21_1_3BCompatClassId
            || compat == WanVideoLoader.Wan21_14BCompatClassId
            || compat == LtxVideoLoader.LtxVideoCompatClassId
            || compat == LanceLoader.LanceVideoCompatClassId;
        if (isVideoArch)
        {
            // Wan VACE (control-video mode): the Init Image slot is the control clip and is REQUIRED;
            // LoRA on the VACE control branch isn't wired yet.
            if (WanModelVariants.IsVace(model.ModelClass?.ID))
            {
                if (input.Get(T2IParamTypes.InitImage) is null)
                {
                    input.RefusalReasons.Add(
                        "HartsyInference: Wan VACE needs a control video (or image) in the Init Image slot — "
                        + "that's the pose/depth/edge/sketch sequence the generation follows.");
                    return false;
                }
                if (input.TryGet(T2IParamTypes.Loras, out List<string> vaceLoras) && vaceLoras is not null && vaceLoras.Count > 0)
                {
                    input.RefusalReasons.Add("HartsyInference: LoRAs aren't supported for Wan VACE yet. Remove the LoRA selection.");
                    return false;
                }
            }
            if (input.Get(T2IParamTypes.InitImage) is not null && compat == LtxVideoLoader.LtxVideoCompatClassId)
            {
                input.RefusalReasons.Add(
                    "HartsyInference: image-to-video isn't supported for LTX-Video (text-to-video only). "
                    + "Remove the Init Image, or use a Wan 2.2 TI2V model for image-to-video.");
                return false;
            }
            if (input.Get(T2IParamTypes.InitImage) is not null && compat == LanceLoader.LanceVideoCompatClassId)
            {
                input.RefusalReasons.Add(
                    "HartsyInference: image-to-video isn't supported for Lance yet (text-to-video only — the "
                    + "image-editing/I2V path needs the frozen Qwen2.5-VL ViT, which the engine defers). "
                    + "Remove the Init Image, or use a Wan 2.2 TI2V model for image-to-video.");
                return false;
            }
            if (input.Get(T2IParamTypes.VideoEndFrame) is not null)
            {
                input.RefusalReasons.Add("HartsyInference: video end-frame conditioning isn't supported yet. Remove the Video End Image.");
                return false;
            }
            if (input.Get(T2IParamTypes.VideoExtendModel) is not null)
            {
                input.RefusalReasons.Add("HartsyInference: video extending isn't supported yet. Remove the Video Extend Model selection.");
                return false;
            }
            if (input.Get(T2IParamTypes.VideoAudioInput) is not null || input.Get(T2IParamTypes.VideoAudioReference) is not null)
            {
                input.RefusalReasons.Add("HartsyInference: audio-conditioned video isn't supported (no supported model uses it). Remove the audio input.");
                return false;
            }
            if (input.Get(T2IParamTypes.RefinerModel) is not null)
            {
                input.RefusalReasons.Add("HartsyInference: refiners can't run over video outputs. Remove the Refiner Model selection.");
                return false;
            }
        }

        // Z-Image's Qwen3-4B encoder + Flux VAE used to be required-explicit; ZImageLoader
        // now auto-resolves them via ModelAutoDownloader (mirroring Comfy's GetQwen3_4bModel
        // + CommonModels "flux-ae" auto-download). User picks via T2IParamTypes.QwenModel /
        // VAE still take priority — the auto-download only fires when the user left them blank.
        // No refusal here — let validation pass and let the loader handle resolution at
        // model-load time, where progress is visible via AddLoadStatus.

        // Refiner support:
        //   - PostApply (default): pixel-space second pass via SdxlRefinerPipeline. Works
        //     against any base architecture.
        //   - StepSwap: latent-handoff mid-denoise via SdxlPipeline's RefinerSwapConfig
        //     hook. Currently SDXL-base only — the swap reuses the base pipeline's denoise
        //     loop and is wired only into SdxlPipeline.
        //   - StepSwapNoisy: not implemented (would re-noise at swap; minor variant).
        // Refuse upfront on:
        //   - non-SDXL-refiner model classes (other "refiner" models need pipelines we don't have)
        //   - StepSwap on a non-SDXL base model (only SdxlPipeline accepts RefinerSwapConfig)
        //   - StepSwapNoisy / non-StepSwap-non-PostApply methods
        //   - RefinerUpscale != 1.0 (high-res-fix flow not implemented yet)
        //   - RefinerVAE set (we use the refiner's bundled VAE; separate VAE replacement not yet wired)
        T2IModel refinerModel = input.Get(T2IParamTypes.RefinerModel);
        if (refinerModel is not null)
        {
            string refinerCompat = refinerModel.ModelClass?.CompatClass?.ID;
            if (refinerCompat != SdxlLoader.SdxlRefinerCompatClassId)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: refiner model '{refinerModel.Name}' has compat class '{refinerCompat ?? "unknown"}', " +
                    $"but only the official SDXL Refiner ('{SdxlLoader.SdxlRefinerCompatClassId}') is currently supported as a refiner.");
                return false;
            }
            string refinerMethod = input.Get(T2IParamTypes.RefinerMethod) ?? "PostApply";
            if (refinerMethod == "StepSwap")
            {
                if (compat != SdxlLoader.SdxlCompatClassId)
                {
                    input.RefusalReasons.Add(
                        $"HartsyInference: refiner method 'StepSwap' is currently SDXL-base only " +
                        $"(got base architecture '{compat}'). Switch the base model to SDXL or use 'PostApply' instead.");
                    return false;
                }
            }
            else if (refinerMethod != "PostApply")
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: refiner method '{refinerMethod}' isn't supported. " +
                    "Use 'PostApply' (any base) or 'StepSwap' (SDXL base only).");
                return false;
            }
            double refinerUpscale = input.Get(T2IParamTypes.RefinerUpscale);
            if (Math.Abs(refinerUpscale - 1.0) > 1e-6)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: Refiner Upscale != 1.0 (got {refinerUpscale}) isn't supported yet. " +
                    "Set Refiner Upscale to 1 or pick a different backend.");
                return false;
            }
            if (input.Get(T2IParamTypes.RefinerVAE) is not null)
            {
                input.RefusalReasons.Add(
                    "HartsyInference: Refiner VAE override isn't supported yet — we use the refiner checkpoint's bundled VAE. " +
                    "Clear the Refiner VAE selection or pick a different backend.");
                return false;
            }
        }

        // LoRA support: SD 1.5 / SDXL / Flux are wired; SD3 and Z-Image aren't yet
        // (no upstream LoraTarget for the SD3 transformer or Z-Image's ZImageTransformer
        // path). Refuse upfront with a clear message rather than silently dropping
        // the LoRA selection at generation time.
        if (input.TryGet(T2IParamTypes.Loras, out List<string> selectedLoras)
            && selectedLoras is not null && selectedLoras.Count > 0)
        {
            bool isLoraSupported =
                compat == Sd15Loader.Sd15CompatClassId
                || compat == SdxlLoader.SdxlCompatClassId
                || compat == FluxLoader.Flux1CompatClassId
                || compat == WanVideoLoader.Wan22_5BCompatClassId
                || compat == WanVideoLoader.Wan21_1_3BCompatClassId
                || compat == WanVideoLoader.Wan21_14BCompatClassId;
            if (!isLoraSupported)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: LoRAs aren't yet supported on architecture '{compat}'. " +
                    $"Currently supported: SD 1.5, SDXL, Flux, Wan 2.2. Remove the LoRA selection or pick a model from a supported architecture.");
                return false;
            }
        }

        // Img2img: SD 1.5 / SDXL / Flux / SD3 / Z-Image all load a VaeEncoder. Flux.2
        // pipeline supports it but the loader doesn't wire it yet — exclude until
        // the loader's Img2ImgResolver path is plumbed in.
        if (input.Get(T2IParamTypes.InitImage) is not null)
        {
            bool isImg2ImgSupported =
                compat == Sd15Loader.Sd15CompatClassId
                || compat == SdxlLoader.SdxlCompatClassId
                || compat == FluxLoader.Flux1CompatClassId
                || Sd3Loader.IsSd3Compat(compat)
                || compat == ZImageLoader.ZImageCompatClassId
                || Flux2Loader.IsFlux2Compat(compat);
            if (!isImg2ImgSupported)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: img2img isn't supported on architecture '{compat}' yet. " +
                    $"Currently supported: SD 1.5, SDXL, Flux, Flux.2, SD3, Z-Image. Remove the Init Image or pick a model from a supported architecture.");
                return false;
            }
        }

        // Inpaint (mask image): blend-on-vanilla path is wired upstream for SDXL,
        // Flux, and SD3 — the per-step latent blend + post-decode pixel recomposite
        // run inside each pipeline's GenerateFromTokens when ImageToImageRequest.Mask
        // is set. SD 1.5 + Z-Image have img2img but no mask wiring yet (mechanical
        // follow-up; pipelines need the same blend hooks). The dedicated 9-channel
        // SdxlInpaintPipeline (specialized inpaint checkpoints) remains a stub —
        // not used by this path, only matters if a user tries that variant directly.
        if (input.Get(T2IParamTypes.MaskImage) is not null)
        {
            bool isInpaintSupported =
                compat == SdxlLoader.SdxlCompatClassId
                || compat == FluxLoader.Flux1CompatClassId
                || Sd3Loader.IsSd3Compat(compat);
            if (!isInpaintSupported)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: inpainting (mask image) isn't wired for architecture '{compat}' yet. " +
                    $"Currently supported: SDXL, Flux, SD3. Remove the Mask Image or pick a model from a supported architecture.");
                return false;
            }
        }

        // IP-Adapter: SD 1.5 + SDXL base models. Flux IPA needs a separate adapter (DiT
        // cross-attn), other architectures don't have published IPA checkpoints.
        if (T2IParamTypes.TryGetType("useipadapter", out T2IParamType ipaType, input)
            && input.TryGetRaw(ipaType, out object ipaRaw)
            && ipaRaw is string ipaModel
            && !string.IsNullOrEmpty(ipaModel)
            && ipaModel != "None")
        {
            bool ipaSupported = compat == SdxlLoader.SdxlCompatClassId
                || compat == Sd15Loader.Sd15CompatClassId;
            if (!ipaSupported)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: IP-Adapter is currently supported on SD 1.5 and SDXL (got base architecture '{compat}'). " +
                    $"Other architectures (Flux, SD3, Z-Image, Flux.2, AuraFlow, Chroma, F-Lite, ERNIE) don't have IPA wired — Flux needs a DiT-specific adapter, the others don't have published IPA checkpoints. " +
                    $"Either disable IP-Adapter (set Use IP-Adapter to None) or pick a SD 1.5 / SDXL model.");
                return false;
            }
            // FaceID variants are refused at load time inside IpAdapterResolver — they need
            // an InsightFace ArcFace runtime which we don't link.
        }

        // ControlNet: SDXL-base only in v1 (Canny preprocessor only). Other base models
        // and other preprocessors are refused upfront so the user gets a clean error
        // instead of a runtime exception mid-generation.
        T2IParamTypes.ControlNetParamHolder[] cnHolders = T2IParamTypes.Controlnets;
        if (cnHolders is not null)
        {
            bool anyCnSelected = false;
            for (int i = 0; i < cnHolders.Length; i++)
            {
                if (cnHolders[i]?.Model is not null && input.Get(cnHolders[i].Model) is not null)
                {
                    anyCnSelected = true;
                    break;
                }
            }
            if (anyCnSelected && compat != SdxlLoader.SdxlCompatClassId)
            {
                input.RefusalReasons.Add(
                    $"HartsyInference: ControlNet is currently supported on SDXL only (got '{compat}'). " +
                    $"Either remove the ControlNet selection or pick an SDXL model.");
                return false;
            }
        }

        return true;
    }

    public override async Task<bool> FreeMemory(bool systemRam)
    {
        bool freed = _cache?.EvictAll() ?? false;

        if (systemRam)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        await Task.CompletedTask;
        return freed;
    }

    /// <summary>Triggered by Swarm when the user clicks Cancel.</summary>
    public void RequestCancel()
    {
        _cancelCts?.Cancel();
    }

    /// <summary>True if the cache holds an entry for the given (compat, name) pair.
    /// Centralizes the per-architecture lookup so LoadModel and GenerateLive don't drift.</summary>
    private bool IsCached(string compat, string modelName)
    {
        if (_cache is null) return false;
        if (Sd3Loader.IsSd3Compat(compat)) return _cache.TryGetSd3(modelName) is not null;
        if (Flux2Loader.IsFlux2Compat(compat)) return _cache.TryGetFlux2(modelName) is not null;
        return compat switch
        {
            Sd15Loader.Sd15CompatClassId => _cache.TryGetSd15(modelName) is not null,
            SdxlLoader.SdxlCompatClassId => _cache.TryGetSdxl(modelName) is not null,
            FluxLoader.Flux1CompatClassId => _cache.TryGetFlux(modelName) is not null,
            ChromaLoader.ChromaCompatClassId => _cache.TryGetChroma(modelName) is not null,
            ChromaRadianceLoader.ChromaRadianceCompatClassId => _cache.TryGetChromaRadiance(modelName) is not null,
            ZetaChromaLoader.ZetaChromaCompatClassId => _cache.TryGetZetaChroma(modelName) is not null,
            AuraFlowLoader.AuraFlowCompatClassId => _cache.TryGetAuraFlow(modelName) is not null,
            FLiteLoader.FLiteCompatClassId => _cache.TryGetFLite(modelName) is not null,
            Ideogram4Loader.Ideogram4CompatClassId => _cache.TryGetIdeogram4(modelName) is not null,
            ErnieImageLoader.ErnieImageCompatClassId => _cache.TryGetErnieImage(modelName) is not null,
            ZImageLoader.ZImageCompatClassId => _cache.TryGetZImage(modelName) is not null,
            AnimaLoader.AnimaCompatClassId => _cache.TryGetAnima(modelName) is not null,
            HiDreamLoader.HiDreamI1CompatClassId => _cache.TryGetHiDream(modelName) is not null,
            QwenImageLoader.QwenImageCompatClassId => _cache.TryGetQwenImage(modelName) is not null,
            WanVideoLoader.Wan22_5BCompatClassId or WanVideoLoader.Wan21_1_3BCompatClassId
                or WanVideoLoader.Wan21_14BCompatClassId => _cache.TryGetWanVideo(modelName) is not null,
            LtxVideoLoader.LtxVideoCompatClassId => _cache.TryGetLtxVideo(modelName) is not null,
            AceStepLoader.AceStepCompatClassId => _cache.TryGetAceStep(modelName) is not null || _cache.TryGetAceStep15(modelName) is not null,
            MusicGenLoader.MusicGenCompatClassId => _cache.TryGetMusicGen(modelName) is not null,
            YueLoader.YueCompatClassId => _cache.TryGetYue(modelName) is not null,
            LanceLoader.LanceCompatClassId => _cache.TryGetLance(modelName) is not null,
            LanceLoader.LanceVideoCompatClassId => _cache.TryGetLance(modelName) is not null,
            LensLoader.LensCompatClassId => _cache.TryGetLens(modelName) is not null,
            _ => false,
        };
    }
}
