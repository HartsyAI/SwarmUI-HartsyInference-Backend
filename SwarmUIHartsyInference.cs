using SwarmUI.Accounts;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Hartsy.Extensions.HartsyInferenceBackend.Backends;
using Hartsy.Extensions.HartsyInferenceBackend.WebAPI;

// NOTE: Namespace must NOT contain "SwarmUI" (reserved for built-ins).
// See docs/00-Overview.md for the broader plan.
namespace Hartsy.Extensions.HartsyInferenceBackend;

// NOTE: This extension used to install an AssemblyLoadContext.Default.Resolving hook
// ([ModuleInitializer]) to locate HartsyInference.*.dll next to the extension DLL. That's
// now handled by Swarm core's SwarmExtensionLoadContext (ExtensionsManager.cs), which
// probes the extension's folder for private deps after host resolution fails. Keeping
// the old hook caused every HartsyInference DLL to load into the DEFAULT context first,
// producing "ships X.dll but host already has it loaded" warnings at startup and a
// version-skew hazard (host copy silently wins over the extension's copy).

/// <summary>Permissions for the HartsyInference backend extension.</summary>
public static class HartsyInferencePermissions
{
    public static readonly PermInfoGroup Group = new(
        "HartsyInference",
        "Permissions for the pure-C# HartsyInference backend.");

    public static readonly PermInfo PermUseHartsyInference = Permissions.Register(new(
        "use_hartsyinference",
        "Use HartsyInference backend",
        "Allows generating images using the in-process HartsyInference backend.",
        PermissionDefault.POWERUSERS, Group));

    public static readonly PermInfo PermAdminHartsyInference = Permissions.Register(new(
        "admin_hartsyinference",
        "Administer HartsyInference",
        "Allows clearing the pipeline cache, probing models, and managing devices.",
        PermissionDefault.ADMINS, Group));
}

/// <summary>
/// Extension entry point. Registers the HartsyInferenceBackend backend type, custom
/// parameters, feature flags, and HTTP routes.
/// See docs/01-Architecture.md for the component diagram.
/// </summary>
public class SwarmUIHartsyInference : Extension
{
    // HartsyInference-specific param group (see docs/07-Parameters-And-Feature-Flags.md).
    public static T2IParamGroup HartsyInferenceParamGroup;

    // HartsyInference-specific params. Registered under feature flag "hartsyinference"
    // so they only show when our backend is the active target.
    public static T2IRegisteredParam<Image> AnimateReferenceImageParam;

    public static T2IRegisteredParam<SwarmUI.Media.AudioFile> VideoAudioReferenceParam;
    public static T2IRegisteredParam<bool> AnimateAutoPreprocessParam;
    public static T2IRegisteredParam<Image> AnimatePoseVideoParam;
    public static T2IRegisteredParam<Image> AnimateFaceVideoParam;

    // CFG-Rescale is registered here rather than reading Comfy's "Rescale CFG Multiplier" because the two
    // compute different things: CfgHelper.ApplyCfgRescale rescales the per-token last-dim L2 norm, Comfy's
    // RescaleCFG node reduces standard deviation over all non-batch dims. The same slider value would produce a
    // different-strength effect on each backend, so sharing one control would mislead.
    // Sampler and TCFG have no such difference and are read from Comfy's params instead (see the backend's
    // BuildImageRequest); this is the only member of that cluster left.
    public static T2IRegisteredParam<double> CfgRescaleParam;

    /// <summary>How the Init Image is consumed: denoise (classic img2img) vs reference (in-context edit).</summary>
    public static T2IRegisteredParam<string> InitImageModeParam;

    // IP-Adapter FaceID-PlusV2 shortcut strength (read by Generation.IpAdapterResolver).
    public static T2IRegisteredParam<double> FaceIdV2WeightParam;

    // Ideogram 4 magic-prompt params (see Generation.Ideogram4MagicPrompt).
    public static T2IParamGroup Ideogram4ParamGroup;
    public static T2IRegisteredParam<bool> Ideogram4MagicPromptParam;
    public static T2IRegisteredParam<string> Ideogram4MagicPromptModelParam;

    public static T2IParamGroup VideoRestoreParamGroup;
    public static T2IRegisteredParam<string> RestoreModelParam;
    public static T2IRegisteredParam<int> RestoreWidthParam;
    public static T2IRegisteredParam<int> RestoreHeightParam;
    public static T2IRegisteredParam<int> RestoreClipFramesParam;
    public static T2IRegisteredParam<int> RestoreOverlapParam;
    public static T2IRegisteredParam<double> RestoreStrengthParam;

    // Music params (ACE-Step edit modes + 5 Hz LM planner + advanced CFG; YuE sampling). Named for the model
    // family that honors each one, which is also what keeps them off AudioLab's names — it owns the bare
    // "Cover Strength"/"LM Top K"/"YuE Temperature" spellings, and a cleaned-name collision in
    // T2IParamTypes.Register crashes SwarmUI at init. "YuE Stage-1" because these drive the Stage-1 LM only.
    public static T2IParamGroup MusicParamGroup;
    public static T2IRegisteredParam<AudioFile> AceStepSourceAudioParam;
    public static T2IRegisteredParam<string> AceStepEditModeParam;
    public static T2IRegisteredParam<double> AceStepRepaintStartParam;
    public static T2IRegisteredParam<double> AceStepRepaintEndParam;
    public static T2IRegisteredParam<double> AceStepCoverStrengthParam;
    public static T2IRegisteredParam<string> AceStepLmPlannerParam;
    public static T2IRegisteredParam<bool> AceStepLmThinkingParam;
    public static T2IRegisteredParam<double> AceStepLmTemperatureParam;
    public static T2IRegisteredParam<double> AceStepLmCfgParam;
    public static T2IRegisteredParam<int> AceStepLmTopKParam;
    public static T2IRegisteredParam<double> AceStepLmTopPParam;
    public static T2IRegisteredParam<string> AceStepLmNegativePromptParam;
    public static T2IRegisteredParam<string> AceStepSolverParam;
    public static T2IRegisteredParam<bool> AceStepUseAdgParam;
    public static T2IRegisteredParam<double> AceStepCfgIntervalStartParam;
    public static T2IRegisteredParam<double> AceStepCfgIntervalEndParam;
    public static T2IRegisteredParam<double> YueTemperatureParam;
    public static T2IRegisteredParam<int> YueTopKParam;
    public static T2IRegisteredParam<double> YueTopPParam;
    public static T2IRegisteredParam<double> YueRepetitionPenaltyParam;

    public override void OnPreInit()
    {
        Logs.Init("HartsyInference extension pre-init");

        // Engine performance config: since engine 1.0.0-alpha.45, the standard performance profile
        // (cuDNN fused SDPA, fp8-native GEMM on SM 8.9+, F16 DiT activations, resident DiT weights, raised
        // mem-pool release threshold) is DEFAULT-ON inside the engine itself — nothing to set here, every
        // install reproduces the published benchmark times out of the box. Operators can disable any feature
        // with HARTSY_<FEATURE>=0 in the launcher environment; the engine logs the resolved set at backend
        // init ("[Cuda] perf flags: ..."). See the engine's docs/PERFORMANCE.md.

        // Register feature flags here if needed before settings load.
        // Most feature flags are advertised dynamically via HartsyInferenceBackend.SupportedFeatures.

        // Model classes core doesn't know: ACE-Step v1 (core only registers v1.5), Lance image + video
        // (core has no Lance classes at all — folder models with llm_config.json), MusicGen, YuE, F-Lite.
        // Must register before model folders are scanned.
        Generation.ModelClassRegistrations.RegisterAll();
        // Boogu: core now detects it (compat class "boogu") and excludes it from the OmniGen2 probe, so
        // the extension reuses core's classification — no registration needed.
    }

    /// <summary>Adds <c>tcd</c> to the shared Sampler dropdown. We read Comfy's Sampler param rather than
    /// registering a second one, but its stock list has no <c>tcd</c> entry while the engine's
    /// <c>SchedulerFactory</c> implements <c>TcdScheduler</c> — without this, adopting the shared param would
    /// quietly drop TCD-distilled checkpoint support. The list is a union across backends by design (core
    /// concatenates whatever a live Comfy reports from its own <c>object_info</c>), so contributing one value a
    /// running backend can service is the same pattern, not a special case.</summary>
    private static void PopulateSamplerValues()
    {
        try
        {
            T2IParamTypes.ConcatDropdownValsClean(
                ref SwarmUI.Builtin_ComfyUIBackend.ComfyUIBackendExtension.Samplers,
                ["tcd///TCD (for TCD-distilled models)"]);
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] Failed to add 'tcd' to the Sampler dropdown: {ex.Message}");
        }
    }

    /// <summary>Lists <c>&lt;ModelRoot&gt;/ipadapter/*.safetensors|*.bin</c> into the Comfy extension's IP-Adapter dropdown values.</summary>
    private static void PopulateIpAdapterModels()
    {
        try
        {
            string folder = System.IO.Path.Combine(Program.ServerSettings.Paths.ActualModelRoot, "ipadapter");
            if (!System.IO.Directory.Exists(folder))
            {
                return;
            }
            List<string> found = [];
            foreach (string file in System.IO.Directory.EnumerateFiles(folder, "*.*", System.IO.SearchOption.AllDirectories))
            {
                if (file.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
                {
                    found.Add(System.IO.Path.GetRelativePath(folder, file).Replace('\\', '/'));
                }
            }
            if (found.Count > 0)
            {
                T2IParamTypes.ConcatDropdownValsClean(ref SwarmUI.Builtin_ComfyUIBackend.ComfyUIBackendExtension.IPAdapterModels, found);
                Logs.Verbose($"[HartsyInference] IP-Adapter dropdown populated with {found.Count} local file(s).");
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[HartsyInference] Failed to populate IP-Adapter model list: {ex.Message}");
        }
    }

    public override void OnInit()
    {
        Logs.Init("HartsyInference extension init");

        // Model-aware param visibility. "hartsyinference" only means "our backend is running", so without
        // this every param below shows under every model. See Assets/hartsy-params.js.
        ScriptFiles.Add("Assets/hartsy-params.js");

        RegisterParameterRemaps();

        // The Comfy extension's Use IP-Adapter dropdown normally populates from a live ComfyUI
        // backend's object_info. When only HartsyInference backends run, populate it from the
        // ipadapter model folder directly (the same folder-listing approach the style-model
        // dropdown uses), so IPA works Comfy-free. Refresh keeps it current after downloads.
        PopulateIpAdapterModels();
        Program.ModelRefreshEvent += PopulateIpAdapterModels;
        PopulateSamplerValues();

        // 1. Param group + HartsyInference-specific params.
        HartsyInferenceParamGroup = new("HartsyInference", Toggles: false, Open: false, IsAdvanced: true);

        AnimateReferenceImageParam = T2IParamTypes.Register<Image>(new(
            "Animate Reference Image",
            "Wan-Animate: the character/identity image to animate.\nThe Init Image slot carries the driving (pose/motion) video; this image is who performs that motion.\nRequired for Wan-Animate generations on the HartsyInference backend.",
            null,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        // MiniMax-H3 ref2va reference media is NOT registered as params here. Core already carries
        // drag/paste-attached prompt-box media in T2IParamTypes.PromptImages/PromptAudios/PromptVideos, and the
        // model's own text encoder resolves the <Picture N>/<Audio N>/<Video N> tags the user types inline — Swarm
        // just supplies the ordered lists. HartsyInferenceBackend.BuildReference* reads those, index for index with
        // the reference node, so the media shows up where the rest of SwarmUI puts media instead of in four
        // bespoke controls that only appear for one model.
        // Wan-Animate driving preprocessing. When on (default), the backend auto-derives the pose skeleton +
        // cropped face from the Init-Image driving clip (the way the checkpoint was trained), instead of feeding
        // the raw clip. Toggle off (or supply the overrides below) to hand the backend pre-rendered inputs.
        AnimateAutoPreprocessParam = T2IParamTypes.Register<bool>(new(
            "Animate Auto-Preprocess",
            "Wan-Animate: auto-derive the pose skeleton + cropped face from the Init-Image driving video (the format the model was trained on).\nOn (default) = best motion fidelity; off = feed the raw clip (legacy). The pose/face override params below take precedence when set.",
            "true",
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        AnimatePoseVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Pose Video",
            "Wan-Animate: an already-rendered pose/skeleton driving video (OpenPose/DWPose colored limbs).\nOverrides auto-preprocessing for the pose branch — supply this when you have a pre-rendered skeleton.",
            null,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        AnimateFaceVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Face Video",
            "Wan-Animate: an already-cropped, face-centered driving video (square, ~512px) for the facial-motion branch.\nOverrides auto-preprocessing for the face branch.",
            null,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        // Extension-registered (NOT a Swarm core param — verified absent upstream as of 2026-08-05): the
        // audio-reference input for joint audio+video models (MiniMax-H3 voice/style reference). The backend
        // maps it to the engine's VideoRequest.VideoAudioReference beside core's Video Audio Input.
        VideoAudioReferenceParam = T2IParamTypes.Register<SwarmUI.Media.AudioFile>(new(
            "Video Audio Reference",
            "For audio+video models that support a reference audio clip (e.g. MiniMax-H3 voice reference): the generated soundtrack imitates this clip's voice/style rather than treating it as literal input audio.",
            null,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            ChangeWeight: 2));

        // Sampler and TCFG are NOT registered here — the backend reads Comfy's own "Sampler" and "Use TCFG"
        // params instead (both are in HonoredComfyParams). Registering our own would put a second, near-identical
        // control beside Comfy's for no gain.

        CfgRescaleParam = T2IParamTypes.Register<double>(new(
            "CFG Rescale",
            "Pulls a high-CFG-Scale guided prediction back toward the conditional prediction's magnitude, reducing the oversaturated/burnt-highlights look that high CFG Scale causes.\n0 (default) = off. 0.7 is a reasonable starting point at CFG Scale 10+. Only SDXL honors this today.\nNot the same math as ComfyUI's RescaleCFG node — this rescales per-token L2 norm, not per-sample standard deviation — so the same numeric value produces a different-strength effect.",
            "0", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: HartsyInferenceParamGroup,
            // "sdxl" is granted/removed by core per selected model and sits in T2IEngine.DisregardedFeatureFlags,
            // so it hides this for every other family without ever gating backend selection.
            FeatureFlag: "hartsyinference,sdxl",
            ViewType: ParamViewType.SLIDER,
            IsAdvanced: true));

        InitImageModeParam = T2IParamTypes.Register<string>(new(
            "Init Image Mode",
            "How the Init Image is consumed.\n'denoise' = classic img2img: Creativity picks how much is regenerated.\n'reference' = in-context reference editing: the image conditions every step and Creativity is ignored.\n'auto' lets the model family decide (families offering both, like Qwen-Image, prefer denoise).",
            "auto",
            Toggleable: true,
            IgnoreIf: "auto",
            Group: HartsyInferenceParamGroup,
            FeatureFlag: "hartsyinference",
            GetValues: _ => new List<string> { "auto", "denoise", "reference" }));

        // FaceID-PlusV2 shortcut strength: sits with the Comfy extension's IP-Adapter params in the
        // Image Prompting group (same "ipadapter" feature flag + dropdown dependency) so it appears
        // exactly when an IP-Adapter is selected. Only consumed for faceid-plusv2 checkpoints; the
        // 1.0 default matches the official IPAdapterFaceIDPlus pipeline (s_scale).
        FaceIdV2WeightParam = T2IParamTypes.Register<double>(new(
            "FaceID V2 Weight",
            "Strength of the FaceID-PlusV2 CLIP-face shortcut mix (the official pipeline's 's_scale').\nHigher = the CLIP appearance of the face crop contributes more on top of the ArcFace identity tokens.\nOnly used with ip-adapter-faceid-plusv2 models; 1.0 is the official default.",
            "1", Min: 0, Max: 2, Step: 0.05, IgnoreIf: "1",
            FeatureFlag: "ipadapter",
            Group: T2IParamTypes.GroupImagePrompting,
            ViewType: ParamViewType.SLIDER,
            OrderPriority: 19.5,
            IsAdvanced: true,
            Examples: ["0.5", "1", "1.5"],
            DependNonDefault: SwarmUI.Builtin_ComfyUIBackend.ComfyUIBackendExtension.UseIPAdapterForRevision.Type.ID));

        // Ideogram 4 magic prompt: optional LLM rewrite of a plain prompt into Ideogram's structured JSON
        // caption (Generation.Ideogram4MagicPrompt). Ideogram 4 is trained on structured captions; plain
        // text also works, so this is opt-in and only fires for Ideogram 4 generations.
        Ideogram4ParamGroup = new("Ideogram 4", Toggles: false, Open: false, IsAdvanced: false);

        Ideogram4MagicPromptParam = T2IParamTypes.Register<bool>(new(
            "Ideogram 4 Magic Prompt",
            "Rewrite your plain prompt into Ideogram 4's structured JSON caption using a running LLM backend, the way Ideogram's own stack does.\nRequires an LLM backend (Server > Backends — LlamaSharp, Anthropic, or remote). When off (default), your prompt is sent to the model as-is (which also works).",
            "false",
            // Toggleable so the frontend only sends it (and thus only requires the
            // "hartsyinference" flag) when actually enabled. A non-toggleable flagged param
            // is sent on every request, forcing "hartsyinference" onto unrelated generations
            // and refusing the Comfy backend (which lacks the flag) — see [[backend feature flags]].
            Toggleable: true,
            Group: Ideogram4ParamGroup,
            FeatureFlag: "hartsyinference,hartsy_ideogram4"));

        Ideogram4MagicPromptModelParam = T2IParamTypes.Register<string>(new(
            "Ideogram 4 Magic Prompt LLM",
            "Optional: which LLM model the magic prompt should use (must be available on a running LLM backend).\nLeave unset to use the running LLM backend's default model. Only used when 'Ideogram 4 Magic Prompt' is on.",
            "",
            Toggleable: true,
            Group: Ideogram4ParamGroup,
            FeatureFlag: "hartsyinference,hartsy_ideogram4"));

        // Restore (SeedVR2): optional one-step restoration/upscale pass over the generated frames
        // before muxing. The target is an AREA (aspect preserved by the model's bicubic area-resize) —
        // SeedVR2 has no scale factor. Enabled by toggling the model param on; all params are Toggleable
        // for the same feature-flag reason as Ideogram 4 above.
        VideoRestoreParamGroup = new("Video Restore", Toggles: false, Open: false, IsAdvanced: true);

        RestoreModelParam = T2IParamTypes.Register<string>(new(
            "Restore Model",
            "Restore/upscale the generated output with SeedVR2 — video frames before muxing, or a still image after generation.\nToggle ON to enable; seedvr2-3b is the catalog default.",
            "seedvr2-3b",
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference"));

        RestoreWidthParam = T2IParamTypes.Register<int>(new(
            "Restore Target Width",
            "Width component of the restore target AREA (aspect is preserved; this is not an output width). Applies to video frames and stills alike.",
            "1280",
            Min: 256, Max: 4096, Step: 16,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference"));

        RestoreHeightParam = T2IParamTypes.Register<int>(new(
            "Restore Target Height",
            "Height component of the restore target AREA.",
            "720",
            Min: 256, Max: 4096, Step: 16,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference"));

        RestoreClipFramesParam = T2IParamTypes.Register<int>(new(
            "Restore Clip Frames",
            "Frames per restore chunk (rounded to the model's (n-1)%4==0 contract). Lower on tight VRAM — fp32 720p-area needs ~5. Ignored for stills.",
            "5",
            Min: 1, Max: 121, Step: 4,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference"));

        RestoreOverlapParam = T2IParamTypes.Register<int>(new(
            "Restore Frame Overlap",
            "Frame overlap between restore chunks, cross-faded.",
            "1",
            Min: 0, Max: 16, Step: 1,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference"));

        RestoreStrengthParam = T2IParamTypes.Register<double>(new(
            "Restore Strength",
            "Restoration strength 0..1. 1.0 = pure model output; lower keeps the input's low-frequency band (guards oversharpening on clean input).",
            "1",
            Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference"));

        // Music group: ACE-Step editing + planner + advanced knobs, YuE sampling. Subgroups are faked with
        // OrderPriority bands (edit 1-5, LM 10-16, CFG 20-23, sampling 30-33); Swarm has no nested groups.
        MusicParamGroup = new("HartsyInference Music", Toggles: false, Open: false, IsAdvanced: false);

        AceStepSourceAudioParam = T2IParamTypes.Register<AudioFile>(new(
            "ACE-Step Source Audio",
            "Source audio clip for ACE-Step music editing. What happens to it is picked by ACE-Step Edit Mode:\ncontinuation extends it, repaint regenerates a time span inside it, cover re-renders it in the prompt's style.\nACE-Step models only. Cannot be combined with the ACE-Step LM Planner.",
            null,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 1,
            ChangeWeight: 2));

        AceStepEditModeParam = T2IParamTypes.Register<string>(new(
            "ACE-Step Edit Mode",
            "What the Source Audio is used for.\n'continuation' = generate Duration seconds continuing past the clip.\n'repaint' = regenerate only Repaint Start..Repaint End seconds inside the clip.\n'cover' = re-render the whole clip in the prompt's style at Cover Strength.",
            "continuation",
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 2,
            GetValues: _ => new List<string> { "continuation", "repaint", "cover" }));

        AceStepRepaintStartParam = T2IParamTypes.Register<double>(new(
            "ACE-Step Repaint Start",
            "Repaint mode: start of the regenerated span, in seconds from the start of the Source Audio.",
            "0", Min: 0, Max: 600, Step: 0.5,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 3));

        AceStepRepaintEndParam = T2IParamTypes.Register<double>(new(
            "ACE-Step Repaint End",
            "Repaint mode: end of the regenerated span, in seconds. Must be greater than Repaint Start.",
            "0", Min: 0, Max: 600, Step: 0.5,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 4));

        AceStepCoverStrengthParam = T2IParamTypes.Register<double>(new(
            "ACE-Step Cover Strength",
            "Cover mode: how much of the source is re-rendered (0 keeps it, 1 fully regenerates). Clamped to 0.05 minimum engine-side.",
            "0.5", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 5,
            ViewType: ParamViewType.SLIDER));

        AceStepLmPlannerParam = T2IParamTypes.Register<string>(new(
            "ACE-Step LM Planner",
            "ACE-Step 5 Hz LM planner: a language model that plans the song's structure before diffusion.\n'0.6b' or '4b' selects the planner size (auto-downloaded). Cannot be combined with Source Audio editing.",
            "none",
            Toggleable: true,
            IgnoreIf: "none",
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 10,
            GetValues: _ => new List<string> { "none", "0.6b", "4b" }));

        AceStepLmThinkingParam = T2IParamTypes.Register<bool>(new(
            "ACE-Step LM Thinking",
            "Planner thinking mode (also selects the matching guidance scalers).",
            "true",
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 11,
            IsAdvanced: true));

        AceStepLmTemperatureParam = T2IParamTypes.Register<double>(new(
            "ACE-Step LM Temperature",
            "Planner sampling temperature.",
            "0.85", Min: 0, Max: 2, Step: 0.05,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 12,
            IsAdvanced: true));

        AceStepLmCfgParam = T2IParamTypes.Register<double>(new(
            "ACE-Step LM CFG Scale",
            "Planner guidance scale.",
            "2", Min: 1, Max: 10, Step: 0.5,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 13,
            IsAdvanced: true));

        AceStepLmTopKParam = T2IParamTypes.Register<int>(new(
            "ACE-Step LM Top K",
            "Planner top-k sampling cutoff (0 = disabled).",
            "0", Min: 0, Max: 500, Step: 10,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 14,
            IsAdvanced: true));

        AceStepLmTopPParam = T2IParamTypes.Register<double>(new(
            "ACE-Step LM Top P",
            "Planner nucleus sampling cutoff.",
            "0.9", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 15,
            IsAdvanced: true));

        AceStepLmNegativePromptParam = T2IParamTypes.Register<string>(new(
            "ACE-Step LM Negative Prompt",
            "Planner negative prompt.",
            "",
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 16,
            IsAdvanced: true,
            ViewType: ParamViewType.PROMPT));

        AceStepSolverParam = T2IParamTypes.Register<string>(new(
            "ACE-Step Solver",
            "ACE-Step diffusion solver: 'ode' (Euler, default) or 'sde' (predict-clean + renoise).",
            "ode",
            Toggleable: true,
            IgnoreIf: "ode",
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 20,
            IsAdvanced: true,
            GetValues: _ => new List<string> { "ode", "sde" }));

        AceStepUseAdgParam = T2IParamTypes.Register<bool>(new(
            "ACE-Step Use ADG",
            "ACE-Step: use ADG guidance instead of the default APG blend (only matters when CFG > 1 on non-turbo checkpoints).",
            "false",
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 21,
            IsAdvanced: true));

        AceStepCfgIntervalStartParam = T2IParamTypes.Register<double>(new(
            "ACE-Step CFG Interval Start",
            "ACE-Step: CFG applies only while sigma is inside this 0..1 interval.",
            "0", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 22,
            IsAdvanced: true));

        AceStepCfgIntervalEndParam = T2IParamTypes.Register<double>(new(
            "ACE-Step CFG Interval End",
            "ACE-Step: upper edge of the CFG sigma interval.",
            "1", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 23,
            IsAdvanced: true));

        YueTemperatureParam = T2IParamTypes.Register<double>(new(
            "YuE Stage-1 Temperature",
            "YuE sampling temperature (ACE-Step and MusicGen ignore this).",
            "1", Min: 0, Max: 2, Step: 0.05,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_yue",
            OrderPriority: 30,
            IsAdvanced: true));

        YueTopKParam = T2IParamTypes.Register<int>(new(
            "YuE Stage-1 Top K",
            "YuE top-k sampling cutoff.",
            "50", Min: 0, Max: 500, Step: 10,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_yue",
            OrderPriority: 31,
            IsAdvanced: true));

        YueTopPParam = T2IParamTypes.Register<double>(new(
            "YuE Stage-1 Top P",
            "YuE nucleus sampling cutoff.",
            "0.93", Min: 0, Max: 1, Step: 0.01,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_yue",
            OrderPriority: 32,
            IsAdvanced: true));

        YueRepetitionPenaltyParam = T2IParamTypes.Register<double>(new(
            "YuE Stage-1 Repetition Penalty",
            "YuE repetition penalty (floored at 1 engine-side).",
            "1.1", Min: 1, Max: 2, Step: 0.05,
            Toggleable: true,
            Group: MusicParamGroup,
            FeatureFlag: "hartsyinference,hartsy_yue",
            OrderPriority: 33,
            IsAdvanced: true));

        // 2. Register the backend type (single type — no _selfstart vs _api split,
        //    we always run in-process).
        // Fully qualify Backends.HartsyInferenceBackend because the trailing namespace
        // segment of this file (`Hartsy.Extensions.HartsyInferenceBackend`) collides
        // with the unqualified class name.
        Program.Backends.RegisterBackendType<Backends.HartsyInferenceBackend>(
            "hartsyinference",
            "HartsyInference (Pure C# Inference)",
            "In-process pure-C# diffusion backend. CPU / Vulkan / CUDA support, no Python required.",
            isStandard: false);

        // 3. Register HTTP routes.
        HartsyInferenceWebAPI.Register();

        // 4. Pre-register all built-in architecture handlers.
        Generation.ModelSupport.RegisterBuiltins();

        // 5. Self-check the flags above against what the backend advertises.
        WarnOnUndeclaredFeatureFlags();
    }

    /// <summary>Keeps saved presets, preset links and "reuse parameters" from older images working across the
    /// 2026-08 rename that dropped the vendor prefixes. <c>T2IParamTypes.ParameterRemaps</c> is consulted inside
    /// <c>GetType</c>, so one entry covers both preset loading (<c>T2IPreset</c>) and metadata reuse
    /// (<c>T2IParamInput</c>). Only maps IDs that are no longer registered, per that dictionary's own contract.</summary>
    private static void RegisterParameterRemaps()
    {
        (string Old, string New)[] renames =
        [
            ("hartsyinferencecfgrescale", "cfgrescale"),
            ("hartsyinferenceinitimagemode", "initimagemode"),
            ("animateautopreprocessdriving", "animateautopreprocess"),
            ("videorestoremodel", "restoremodel"),
            ("videorestoretargetwidth", "restoretargetwidth"),
            ("videorestoretargetheight", "restoretargetheight"),
            ("videorestoreclipframes", "restoreclipframes"),
            ("videorestoreoverlap", "restoreframeoverlap"),
            ("videorestorestrength", "restorestrength"),
            ("hartsymusicsourceaudio", "acestepsourceaudio"),
            ("hartsymusiceditmode", "acestepeditmode"),
            ("hartsymusicrepaintstart", "acesteprepaintstart"),
            ("hartsymusicrepaintend", "acesteprepaintend"),
            ("hartsymusiccoverstrength", "acestepcoverstrength"),
            ("hartsymusiclmplanner", "acesteplmplanner"),
            ("hartsymusiclmthinking", "acesteplmthinking"),
            ("hartsymusiclmtemperature", "acesteplmtemperature"),
            ("hartsymusiclmcfgscale", "acesteplmcfgscale"),
            ("hartsymusiclmtopk", "acesteplmtopk"),
            ("hartsymusiclmtopp", "acesteplmtopp"),
            ("hartsymusiclmnegativeprompt", "acesteplmnegativeprompt"),
            ("hartsymusicinfermethod", "acestepsolver"),
            ("hartsymusicuseadg", "acestepuseadg"),
            ("hartsymusiccfgintervalstart", "acestepcfgintervalstart"),
            ("hartsymusiccfgintervalend", "acestepcfgintervalend"),
            ("hartsymusictemperature", "yuestagetemperature"),
            ("hartsymusictopk", "yuestagetopk"),
            ("hartsymusictopp", "yuestagetopp"),
            ("hartsymusicrepetitionpenalty", "yuestagerepetitionpenalty"),
            // Deleted in favour of Comfy's own param. Safe to point at it: same boolean, same meaning, so an old
            // preset's "true"/"false" carries over unchanged. The deleted Sampler param deliberately has NO entry
            // here — it spelled DPM++ 2M as "dpm++2m" where Comfy's dropdown says "dpmpp_2m", so remapping would
            // hand that dropdown a value it doesn't list. Losing the preference is better than an invalid value.
            ("hartsyinferencetcfg", "usetcfg"),
        ];
        foreach ((string old, string updated) in renames)
        {
            T2IParamTypes.ParameterRemaps[old] = updated;
        }
    }

    /// <summary>Fails loudly at startup if any param registered above carries a FeatureFlag the backend does not
    /// advertise. <c>T2IEngine</c> drops a backend whose SupportedFeatures don't cover a job's required flags, so
    /// such a param makes every generation that touches it refuse — with a message that names no param and no flag.
    /// Flags in <c>T2IEngine.DisregardedFeatureFlags</c> are UI-visibility-only and never gate a backend, so they
    /// are exempt (that is how core's own "text2video"/"text2audio" tags work without any backend declaring them).</summary>
    private static void WarnOnUndeclaredFeatureFlags()
    {
        HashSet<string> covered = [.. Backends.HartsyInferenceBackend.DeclaredFeatures, .. T2IEngine.DisregardedFeatureFlags];
        foreach (T2IParamType type in T2IParamTypes.Types.Values)
        {
            if (type.Group != HartsyInferenceParamGroup || string.IsNullOrEmpty(type.FeatureFlag))
            {
                continue;
            }
            foreach (string flag in type.FeatureFlag.Split(','))
            {
                if (!covered.Contains(flag))
                {
                    Logs.Error($"[HartsyInference] Param '{type.Name}' requires feature flag '{flag}', which this "
                        + "backend does not advertise — every generation using that param will be refused with no "
                        + "explanation. Add it to HartsyInferenceBackend.DeclaredFeatures, or drop the flag.");
                }
            }
        }
    }

    public override void OnPreLaunch()
    {
        // No special wiring needed — backend instances are created on demand by Swarm.
    }

    public override void OnShutdown()
    {
        Logs.Init("HartsyInference extension shutdown");
        // Per-instance shutdown is handled by the BackendHandler;
        // nothing extension-level to clean up here.
    }
}
