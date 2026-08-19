using SwarmUI.Accounts;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
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
    // Wan-Animate conditioning inputs. A model-named group is right here precisely BECAUSE it is model-gated:
    // hartsy_wan_animate is only granted for an Animate checkpoint, so the group is absent otherwise.
    public static T2IParamGroup WanAnimateParamGroup;

    // HartsyInference-specific params. Registered under feature flag "hartsyinference"
    // so they only show when our backend is the active target.
    public static T2IRegisteredParam<Image> AnimateReferenceImageParam;

    public static T2IRegisteredParam<SwarmUI.Media.AudioFile> VideoAudioReferenceParam;
    public static T2IRegisteredParam<bool> AnimateAutoPreprocessParam;
    public static T2IRegisteredParam<Image> AnimatePoseVideoParam;
    public static T2IRegisteredParam<Image> AnimateFaceVideoParam;
    public static T2IRegisteredParam<Image> AnimateBackgroundVideoParam;
    public static T2IRegisteredParam<Image> AnimateMaskVideoParam;

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
    // Real nested groups (T2IParamGroup.Parent), the way core nests Video Obscure Options under Advanced Video.
    public static T2IParamGroup AceStepParamGroup, AceStepEditingGroup, AceStepPlannerGroup, AceStepGuidanceGroup, MiniMaxMusicGroup;
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
    public static T2IRegisteredParam<string> AceStepGuidanceTypeParam;
    public static T2IRegisteredParam<bool> AceStepErgTagParam, AceStepErgLyricParam, AceStepErgDiffusionParam;
    public static T2IRegisteredParam<double> AceStepCfgIntervalStartParam;
    public static T2IRegisteredParam<double> AceStepCfgIntervalEndParam;
    public static T2IRegisteredParam<string> MiniMaxMusicLmPrecisionParam;

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

        // 1. Param groups. There is deliberately no group named after this extension: a group named for the
        //    backend can never be model-scoped, which is what left every param below permanently visible.
        //    Everything either lives in the matching core group or in a group named for the model family.
        WanAnimateParamGroup = new("Wan Animate", Toggles: true, Open: false,
            Description: "Wan-Animate conditioning: the character image to animate, plus optional pre-rendered "
                + "pose/face driving clips. The Init Image slot carries the driving video.");

        AnimateReferenceImageParam = T2IParamTypes.Register<Image>(new(
            "Animate Reference Image",
            "Wan-Animate: the character/identity image to animate.\nThe Init Image slot carries the driving (pose/motion) video; this image is who performs that motion.\nRequired for Wan-Animate generations on the HartsyInference backend.",
            null,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
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
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2));

        AnimatePoseVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Pose Video",
            "Wan-Animate: an already-rendered pose/skeleton driving video (OpenPose/DWPose colored limbs).\nOverrides auto-preprocessing for the pose branch — supply this when you have a pre-rendered skeleton.",
            null,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        AnimateFaceVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Face Video",
            "Wan-Animate: an already-cropped, face-centered driving video (square, ~512px) for the facial-motion branch.\nOverrides auto-preprocessing for the face branch.",
            null,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        AnimateBackgroundVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Background Video",
            "Wan-Animate replacement mode: the background video the character is composited into.\nThe conditioning carries these frames instead of the mid-gray placeholder, so unmasked regions keep this background.",
            null,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        AnimateMaskVideoParam = T2IParamTypes.Register<Image>(new(
            "Animate Character Mask",
            "Wan-Animate replacement mode: per-frame mask video (white = generate the character there, black = keep the background).\nA single image repeats across all frames.",
            null,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        // Extension-registered (NOT a Swarm core param — verified absent upstream as of 2026-08-05): the
        // audio-reference input for joint audio+video models (MiniMax-H3 voice/style reference). The backend
        // maps it to the engine's VideoRequest.VideoAudioReference beside core's Video Audio Input.
        VideoAudioReferenceParam = T2IParamTypes.Register<SwarmUI.Media.AudioFile>(new(
            "Video Audio Reference",
            "For audio+video models that support a reference audio clip (e.g. MiniMax-H3 voice reference): the generated soundtrack imitates this clip's voice/style rather than treating it as literal input audio.",
            null,
            Toggleable: true,
            Group: T2IParamTypes.GroupAdvancedVideo,
            FeatureFlag: "hartsyinference,hartsy_audio_ref",
            ChangeWeight: 2));

        // Sampler and TCFG are NOT registered here — the backend reads Comfy's own "Sampler" and "Use TCFG"
        // params instead (both are in HonoredComfyParams). Registering our own would put a second, near-identical
        // control beside Comfy's for no gain.

        CfgRescaleParam = T2IParamTypes.Register<double>(new(
            "CFG Rescale",
            "Pulls a high-CFG-Scale guided prediction back toward the conditional prediction's magnitude, reducing the oversaturated/burnt-highlights look that high CFG Scale causes.\n0 (default) = off. 0.7 is a reasonable starting point at CFG Scale 10+. Only SDXL honors this today.\nNot the same math as ComfyUI's RescaleCFG node — this rescales per-token L2 norm, not per-sample standard deviation — so the same numeric value produces a different-strength effect.",
            "0", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: T2IParamTypes.GroupAlternateGuidance,
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
            Group: T2IParamTypes.GroupInitImage,
            FeatureFlag: "hartsyinference,hartsy_refedit_choice",
            GetValues: _ => new List<string> { "auto", "denoise", "reference" },
            DependNonDefault: T2IParamTypes.InitImage.Type.ID));

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
            FeatureFlag: "hartsyinference,hartsy_ideogram4",
            DependNonDefault: Ideogram4MagicPromptParam.Type.ID));

        // Restore (SeedVR2): optional one-step restoration/upscale pass over the generated frames
        // before muxing. The target is an AREA (aspect preserved by the model's bicubic area-resize) —
        // SeedVR2 has no scale factor. Enabled by toggling the model param on; all params are Toggleable
        // for the same feature-flag reason as Ideogram 4 above.
        // Not "Video Restore": it runs over a still image just as well, and this is the group users toggle to
        // enable the pass at all, so it toggles as one.
        VideoRestoreParamGroup = new("Restore / Upscale", Toggles: true, Open: false, IsAdvanced: true,
            Description: "SeedVR2 restoration pass over the finished generation — video frames before muxing, or a "
                + "still image after generation.");

        RestoreModelParam = T2IParamTypes.Register<string>(new(
            "Restore Model",
            "Restore/upscale the generated output with SeedVR2 — video frames before muxing, or a still image after generation.\nToggle ON to enable; the weights are fetched on first use.",
            "seedvr2-3b",
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference",
            // The engine's own restore catalog, not the model folder: these resolve through ModelResolver by
            // catalog id and download themselves, so a free-text box here just invited typos that fail at
            // generate time.
            GetValues: _ => [.. ModelCatalog.ForModality(Modality.Restore)
                .Select(entry => $"{entry.Id}///{entry.DisplayName}")]));

        RestoreWidthParam = T2IParamTypes.Register<int>(new(
            "Restore Target Width",
            "Width component of the restore target AREA (aspect is preserved; this is not an output width). Applies to video frames and stills alike.",
            "1280",
            Min: 256, Max: 4096, Step: 16,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference",
            DependNonDefault: RestoreModelParam.Type.ID));

        RestoreHeightParam = T2IParamTypes.Register<int>(new(
            "Restore Target Height",
            "Height component of the restore target AREA.",
            "720",
            Min: 256, Max: 4096, Step: 16,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference",
            DependNonDefault: RestoreModelParam.Type.ID));

        RestoreClipFramesParam = T2IParamTypes.Register<int>(new(
            "Restore Clip Frames",
            "Frames per restore chunk (rounded to the model's (n-1)%4==0 contract). Lower on tight VRAM — fp32 720p-area needs ~5. Ignored for stills.",
            "5",
            Min: 1, Max: 121, Step: 4,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference",
            DependNonDefault: RestoreModelParam.Type.ID));

        RestoreOverlapParam = T2IParamTypes.Register<int>(new(
            "Restore Frame Overlap",
            "Frame overlap between restore chunks, cross-faded.",
            "1",
            Min: 0, Max: 16, Step: 1,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference",
            DependNonDefault: RestoreModelParam.Type.ID));

        RestoreStrengthParam = T2IParamTypes.Register<double>(new(
            "Restore Strength",
            "Restoration strength 0..1. 1.0 = pure model output; lower keeps the input's low-frequency band (guards oversharpening on clean input).",
            "1",
            Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: VideoRestoreParamGroup,
            FeatureFlag: "hartsyinference",
            DependNonDefault: RestoreModelParam.Type.ID));

        // One group per music family, with real nested subgroups. These used to be a single vendor-named group
        // whose subgroups were faked with OrderPriority bands, on the belief that Swarm had no nested groups —
        // it does: T2IParamGroup takes a Parent, which is how core nests Video Obscure Options under Advanced
        // Video. Each group is model-gated by its params' flags, so selecting MusicGen (which honours none of
        // these) collapses the whole tree: hideUnsupportableParams hides a group with no visible params.
        AceStepParamGroup = new("ACE-Step", Toggles: false, Open: false,
            Description: "ACE-Step editing, planning and guidance controls.");
        AceStepEditingGroup = new("ACE-Step Editing", Toggles: true, Open: false, OrderPriority: 1, Parent: AceStepParamGroup,
            Description: "Generate from an existing clip: continue past it, regenerate a span inside it, or "
                + "re-render it in the prompt's style.");
        AceStepPlannerGroup = new("ACE-Step Planner", Toggles: false, Open: false, OrderPriority: 2, Parent: AceStepParamGroup,
            Description: "The 5 Hz LM planner, which plans the song's structure before diffusion runs.");
        AceStepGuidanceGroup = new("ACE-Step Guidance", Toggles: false, Open: false, OrderPriority: 3, IsAdvanced: true,
            Parent: AceStepParamGroup, Description: "Solver and classifier-free-guidance shaping.");
        MiniMaxMusicGroup = new("MiniMax Music", Toggles: false, Open: false, IsAdvanced: true,
            Description: "MiniMax Music 3 controls.");

        AceStepSourceAudioParam = T2IParamTypes.Register<AudioFile>(new(
            "ACE-Step Source Audio",
            "Source audio clip for ACE-Step music editing. What happens to it is picked by ACE-Step Edit Mode:\ncontinuation extends it, repaint regenerates a time span inside it, cover re-renders it in the prompt's style.\nACE-Step models only. Cannot be combined with the ACE-Step LM Planner.",
            null,
            Toggleable: true,
            Group: AceStepEditingGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 1,
            ChangeWeight: 2));

        AceStepEditModeParam = T2IParamTypes.Register<string>(new(
            "ACE-Step Edit Mode",
            "What the Source Audio is used for.\n'continuation' = generate Duration seconds continuing past the clip.\n'repaint' = regenerate only Repaint Start..Repaint End seconds inside the clip.\n'cover' = re-render the whole clip in the prompt's style at Cover Strength.",
            "continuation",
            Toggleable: true,
            Group: AceStepEditingGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 2,
            GetValues: _ => new List<string> { "continuation", "repaint", "cover" }));

        AceStepRepaintStartParam = T2IParamTypes.Register<double>(new(
            "ACE-Step Repaint Start",
            "Repaint mode: start of the regenerated span, in seconds from the start of the Source Audio.",
            "0", Min: 0, Max: 600, Step: 0.5,
            Toggleable: true,
            Group: AceStepEditingGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 3,
            DependNonDefault: AceStepSourceAudioParam.Type.ID));

        AceStepRepaintEndParam = T2IParamTypes.Register<double>(new(
            "ACE-Step Repaint End",
            "Repaint mode: end of the regenerated span, in seconds. Must be greater than Repaint Start.",
            "0", Min: 0, Max: 600, Step: 0.5,
            Toggleable: true,
            Group: AceStepEditingGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 4,
            DependNonDefault: AceStepSourceAudioParam.Type.ID));

        AceStepCoverStrengthParam = T2IParamTypes.Register<double>(new(
            "ACE-Step Cover Strength",
            "Cover mode: how much of the source is re-rendered (0 keeps it, 1 fully regenerates). Clamped to 0.05 minimum engine-side.",
            "0.5", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: AceStepEditingGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 5,
            ViewType: ParamViewType.SLIDER,
            DependNonDefault: AceStepSourceAudioParam.Type.ID));

        AceStepLmPlannerParam = T2IParamTypes.Register<string>(new(
            "ACE-Step LM Planner",
            "ACE-Step 5 Hz LM planner: a language model that plans the song's structure before diffusion.\n'0.6b' or '4b' selects the planner size (auto-downloaded). Cannot be combined with Source Audio editing.",
            "none",
            Toggleable: true,
            IgnoreIf: "none",
            Group: AceStepPlannerGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 10,
            GetValues: _ => new List<string> { "none", "0.6b", "4b" }));

        AceStepLmThinkingParam = T2IParamTypes.Register<bool>(new(
            "ACE-Step LM Thinking",
            "Planner thinking mode (also selects the matching guidance scalers).",
            "true",
            Toggleable: true,
            Group: AceStepPlannerGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 11,
            IsAdvanced: true,
            DependNonDefault: AceStepLmPlannerParam.Type.ID));

        AceStepLmTemperatureParam = T2IParamTypes.Register<double>(new(
            "ACE-Step LM Temperature",
            "Planner sampling temperature.",
            "0.85", Min: 0, Max: 2, Step: 0.05,
            Toggleable: true,
            Group: AceStepPlannerGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 12,
            IsAdvanced: true,
            DependNonDefault: AceStepLmPlannerParam.Type.ID));

        AceStepLmCfgParam = T2IParamTypes.Register<double>(new(
            "ACE-Step LM CFG Scale",
            "Planner guidance scale.",
            "2", Min: 1, Max: 10, Step: 0.5,
            Toggleable: true,
            Group: AceStepPlannerGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 13,
            IsAdvanced: true,
            DependNonDefault: AceStepLmPlannerParam.Type.ID));

        AceStepLmTopKParam = T2IParamTypes.Register<int>(new(
            "ACE-Step LM Top K",
            "Planner top-k sampling cutoff (0 = disabled).",
            "0", Min: 0, Max: 500, Step: 10,
            Toggleable: true,
            Group: AceStepPlannerGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 14,
            IsAdvanced: true,
            DependNonDefault: AceStepLmPlannerParam.Type.ID));

        AceStepLmTopPParam = T2IParamTypes.Register<double>(new(
            "ACE-Step LM Top P",
            "Planner nucleus sampling cutoff.",
            "0.9", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: AceStepPlannerGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 15,
            IsAdvanced: true,
            DependNonDefault: AceStepLmPlannerParam.Type.ID));

        AceStepLmNegativePromptParam = T2IParamTypes.Register<string>(new(
            "ACE-Step LM Negative Prompt",
            "Planner negative prompt.",
            "",
            Toggleable: true,
            Group: AceStepPlannerGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 16,
            IsAdvanced: true,
            ViewType: ParamViewType.PROMPT,
            DependNonDefault: AceStepLmPlannerParam.Type.ID));

        AceStepSolverParam = T2IParamTypes.Register<string>(new(
            "ACE-Step Solver",
            "ACE-Step diffusion solver: 'ode' (Euler, default) or 'sde' (predict-clean + renoise).",
            "ode",
            Toggleable: true,
            IgnoreIf: "ode",
            Group: AceStepGuidanceGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 20,
            IsAdvanced: true,
            GetValues: _ => new List<string> { "ode", "sde" }));

        // Replaced the old "ACE-Step Use ADG" bool: upstream offers three guidance blends and a bool could
        // only reach two. No ParameterRemaps entry — a preset's "true"/"false" is not a valid enum value
        // (same reasoning as the deleted Sampler param).
        AceStepGuidanceTypeParam = T2IParamTypes.Register<string>(new(
            "ACE-Step Guidance Type",
            "ACE-Step: guidance blend when CFG > 1 on non-turbo checkpoints. apg (default, momentum-projected), cfg (plain classifier-free), adg.",
            "apg", GetValues: _ => ["apg", "cfg", "adg"],
            Toggleable: true,
            Group: AceStepGuidanceGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 21,
            IsAdvanced: true));

        AceStepErgTagParam = T2IParamTypes.Register<bool>(new(
            "ACE-Step V1 ERG Tag",
            "ACE-Step v1: entropy-rectifying guidance on the tag branch — the unconditional pass sees a weakened (not zeroed) text encoding. Upstream default is on. v1 checkpoints only.",
            "true",
            Toggleable: true,
            Group: AceStepGuidanceGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 25,
            IsAdvanced: true));

        AceStepErgLyricParam = T2IParamTypes.Register<bool>(new(
            "ACE-Step V1 ERG Lyric",
            "ACE-Step v1: the unconditional pass keeps the lyrics with a weakened lyric encoder instead of dropping them. Upstream default is on. v1 checkpoints only.",
            "true",
            Toggleable: true,
            Group: AceStepGuidanceGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 26,
            IsAdvanced: true));

        AceStepErgDiffusionParam = T2IParamTypes.Register<bool>(new(
            "ACE-Step V1 ERG Diffusion",
            "ACE-Step v1: the unconditional diffusion forwards run with weakened attention queries in the middle blocks. Upstream default is on. v1 checkpoints only.",
            "true",
            Toggleable: true,
            Group: AceStepGuidanceGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 27,
            IsAdvanced: true));

        AceStepCfgIntervalStartParam = T2IParamTypes.Register<double>(new(
            "ACE-Step CFG Interval Start",
            "ACE-Step: CFG applies only while sigma is inside this 0..1 interval.",
            "0", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: AceStepGuidanceGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 22,
            IsAdvanced: true));

        AceStepCfgIntervalEndParam = T2IParamTypes.Register<double>(new(
            "ACE-Step CFG Interval End",
            "ACE-Step: upper edge of the CFG sigma interval.",
            "1", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: AceStepGuidanceGroup,
            FeatureFlag: "hartsyinference,hartsy_acestep",
            OrderPriority: 23,
            IsAdvanced: true));

        MiniMaxMusicLmPrecisionParam = T2IParamTypes.Register<string>(new(
            "MiniMax Music LM Precision",
            "Precision for MiniMax Music 3's 8B language model stage: bf16 (checkpoint precision), q8 or q4 (GGUF-quantized, for smaller cards). The flow DiT always runs the selected checkpoint's weights.",
            "bf16", GetValues: _ => ["bf16", "q8", "q4"],
            Group: MiniMaxMusicGroup,
            FeatureFlag: "hartsyinference,hartsy_minimaxmusic",
            OrderPriority: 30,
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
    /// are exempt (that is how core's own "text2video"/"text2audio" tags work without any backend declaring them).
    /// <para>Checks every registered param rather than one group's: our params live in core groups now, and the
    /// group filter this used to apply covered only 9 of them. Any flag beginning "hartsy" is ours to declare no
    /// matter who registered the param.</para></summary>
    private static void WarnOnUndeclaredFeatureFlags()
    {
        HashSet<string> covered = [.. Backends.HartsyInferenceBackend.DeclaredFeatures, .. T2IEngine.DisregardedFeatureFlags];
        foreach (T2IParamType type in T2IParamTypes.Types.Values)
        {
            if (string.IsNullOrEmpty(type.FeatureFlag)
                || !type.FeatureFlag.Split(',').Any(f => f.StartsWith("hartsy", StringComparison.Ordinal)))
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
