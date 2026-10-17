using SwarmUI.Accounts;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Hartsy.Extensions.HartsyInferenceBackend.Backends;
using Hartsy.Extensions.HartsyInferenceBackend.Services;
using Hartsy.Extensions.HartsyInferenceBackend.WebAPI;

// Namespace must not contain "SwarmUI" (reserved for built-ins).
namespace Hartsy.Extensions.HartsyInferenceBackend;

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

/// <summary>Entry point. Registers the backend type, parameters, feature flags and HTTP routes.</summary>
public class HartsyInference : Extension
{
    // Wan-Animate conditioning inputs. Model-gated: hartsy_wan_animate is granted only for Animate checkpoints.
    public static T2IParamGroup WanAnimateParamGroup;

    // Gated on the "hartsyinference" feature flag, so these only show when this backend is active.
    public static T2IRegisteredParam<Image> AnimateReferenceImageParam;

    public static T2IRegisteredParam<SwarmUI.Media.AudioFile> VideoAudioReferenceParam;
    public static T2IRegisteredParam<bool> AnimateAutoPreprocessParam;
    public static T2IRegisteredParam<Image> AnimatePoseVideoParam;
    public static T2IRegisteredParam<int> AnimateTotalFramesParam;
    public static T2IRegisteredParam<int> AnimateMotionContextFramesParam;
    public static T2IRegisteredParam<double> AnimateColorCorrectionParam;
    public static T2IRegisteredParam<double> AnimatePoseStrengthParam;
    public static T2IRegisteredParam<double> AnimateReferenceStrengthParam;
    public static T2IRegisteredParam<Image> AnimateFaceVideoParam;
    public static T2IRegisteredParam<Image> AnimateBackgroundVideoParam;
    public static T2IRegisteredParam<Image> AnimateMaskVideoParam;

    // MiniMax-H3 long-form chaining: each segment holds the previous segment's tail fixed. Model-gated like WanAnimateParamGroup.
    public static T2IParamGroup MiniMaxH3ParamGroup;
    public static T2IRegisteredParam<int> H3ChainTotalFramesParam;
    public static T2IRegisteredParam<int> H3ChainContextFramesParam;

    /// <summary>Not shared with Comfy's Rescale CFG Multiplier: that node uses a different norm, so one slider would mean different strengths per backend.</summary>
    public static T2IRegisteredParam<double> CfgRescaleParam;

    /// <summary>How the Init Image is consumed: denoise (classic img2img) vs reference (in-context edit).</summary>
    public static T2IRegisteredParam<string> InitImageModeParam;

    // IP-Adapter FaceID-PlusV2 shortcut strength (read by Generation.IpAdapterResolver).
    public static T2IRegisteredParam<double> FaceIdV2WeightParam;

    // Ideogram 4 magic-prompt params (see Generation.Ideogram4MagicPrompt).
    public static T2IParamGroup Ideogram4ParamGroup;
    public static T2IRegisteredParam<bool> Ideogram4MagicPromptParam;
    public static T2IRegisteredParam<string> Ideogram4MagicPromptModelParam;

    // Per-generation overrides of the backend's VRAM posture. Only the levers that bind at generation time are
    // here: quantized compute and multi-GPU spill are baked into a pipeline when it is constructed and cached, so
    // varying them per request would force a rebuild and a multi-GB re-upload. Those stay backend settings.
    public static T2IParamGroup VramParamGroup;
    public static T2IRegisteredParam<string> VramModeParam;
    public static T2IRegisteredParam<string> VramWeightStreamingParam;
    public static T2IRegisteredParam<string> VramKeepResidentParam;
    public static T2IRegisteredParam<string> VramPhaseUnloadParam;
    public static T2IRegisteredParam<string> VramCachePrecisionParam;
    public static T2IRegisteredParam<string> VramActivationOffloadParam;
    public static T2IRegisteredParam<string> VramFreeAfterGenerationParam;
    public static T2IRegisteredParam<double> VramChunkScaleParam;

    public static T2IParamGroup VideoRestoreParamGroup;
    public static T2IRegisteredParam<string> RestoreModelParam;
    public static T2IRegisteredParam<int> RestoreWidthParam;
    public static T2IRegisteredParam<int> RestoreHeightParam;
    public static T2IRegisteredParam<int> RestoreClipFramesParam;
    public static T2IRegisteredParam<int> RestoreOverlapParam;
    public static T2IRegisteredParam<double> RestoreStrengthParam;

    // Music params, named for the model family that honors each one. AudioLab owns the bare "Cover Strength" and
    // "LM Top K" spellings, and a duplicate cleaned name crashes SwarmUI at init in T2IParamTypes.Register.
    // "YuE Stage-1" because these drive the Stage-1 LM only. Nested via T2IParamGroup.Parent.
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
        Logs.Init("[HartsyInference] Starting up.");
        // Model classes must be registered before model folders are scanned.
        Generation.ModelClassRegistrations.RegisterAll();
    }

    /// <summary>Adds <c>tcd</c> to the Sampler dropdown. Comfy's stock list lacks it, but the engine's <c>SchedulerFactory</c>
    /// implements <c>TcdScheduler</c>, so TCD-distilled checkpoints need it selectable.</summary>
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
            Logs.Warning($"[HartsyInference] Could not add 'tcd' to the Sampler dropdown, so TCD-distilled models cannot be selected: {ex.Message}");
        }
    }

    /// <summary>Adds pixel-space upscalers to Comfy's Refiner Upscale Method dropdown, which core validates values against.
    /// <c>pixel-*</c> and the default map onto Real-ESRGAN by factor. <c>latent-*</c> needs a refiner model (see
    /// <c>HartsyInferenceBackend.ValidateImageFeatures</c>).</summary>
    private static void PopulateUpscaleMethodValues()
    {
        try
        {
            T2IParamTypes.ConcatDropdownValsClean(
                ref SwarmUI.Builtin_ComfyUIBackend.ComfyUIBackendExtension.UpscalerModels,
                ["real-esrgan-x4plus///Real-ESRGAN x4plus (HartsyInference)",
                 "real-esrgan-x2plus///Real-ESRGAN x2plus (HartsyInference)",
                 "real-esrgan-anime6b///Real-ESRGAN x4plus anime 6B (HartsyInference)",
                 "seedvr2///SeedVR2 restore (HartsyInference)"]);
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference] Could not add the Real-ESRGAN and SeedVR2 entries to the Refiner Upscale Method dropdown: {ex.Message}");
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
                Logs.Debug($"[HartsyInference] Found {found.Count} IP-Adapter model file(s) for the IP-Adapter dropdown.");
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference] Could not list IP-Adapter models, so the IP-Adapter dropdown may be empty: {ex.Message}");
        }
    }

    public override void OnInit()
    {
        Logs.Init("[HartsyInference] Registering parameters and web routes.");

        // Model-aware param visibility. "hartsyinference" only means "our backend is running", so without
        // this every param below shows under every model. See Assets/hartsy-params.js.
        ScriptFiles.Add("Assets/hartsy-params.js");

        RegisterParameterRemaps();

        // Comfy's Use IP-Adapter dropdown normally fills from a live ComfyUI backend. With only HartsyInference backends
        // running, list the ipadapter model folder instead, as the style-model dropdown does. Refresh keeps it current.
        PopulateIpAdapterModels();
        Program.ModelRefreshEvent += PopulateIpAdapterModels;

        // Memory-aware routing: freeze a per-request GPU fit answer before Swarm picks a backend, so our backends'
        // validator can steer each model to a card it fits on (see Services/ModelFitGate).
        T2IEngine.PreGenerateEvent += ModelFitGate.Prepare;
        PopulateSamplerValues();
        PopulateUpscaleMethodValues();

        // 1. Param groups. No group is named after this extension, since a backend-named group can never be model-scoped.
        //    Params live in the matching core group or in a group named for the model family.
        WanAnimateParamGroup = new("Wan Animate", Toggles: true, Open: false,
            Description: "Wan-Animate conditioning: the character image to animate, plus optional pre-rendered "
                + "pose/face driving clips. The Init Image slot carries the driving video. The character image can "
                + "also just be attached to the prompt box (the same carrier core uses for Wan reference images).");

        MiniMaxH3ParamGroup = new("MiniMax-H3 Chain", Toggles: true, Open: false,
            Description: "MiniMax-H3 long-form generation: output longer than one pass, produced as successive "
                + "segments that each hold the previous segment's tail fixed while denoising only their own new "
                + "frames.");

        AnimateReferenceImageParam = T2IParamTypes.Register<Image>(new(
            "Animate Reference Image",
            "Wan-Animate: the character/identity image to animate.\nThe Init Image slot carries the driving (pose/motion) video; this image is who performs that motion.\nAn image attached to the prompt box works too, matching how core's ComfyUI backend carries Wan reference images; this param wins if both are set.\nRequired (one way or the other) for Wan-Animate generations on the HartsyInference backend.",
            null,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2));

        // MiniMax-H3 ref2va media is not registered as params. It comes from core's prompt-box lists
        // (T2IParamTypes.PromptImages/PromptAudios/PromptVideos), which BuildReference* reads in order.
        // The model resolves the inline <Picture N>/<Audio N>/<Video N> tags itself.
        // Wan-Animate driving preprocessing. On (default), derives the pose skeleton and cropped face from the driving clip,
        // as the checkpoint was trained. Off, or with the overrides below, takes pre-rendered inputs instead.
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

        // Chunked extension (ComfyUI WanAnimateToVideo continue_motion): the video is generated as successive
        // Video Frames-long chunks, each conditioned on the tail of the one before it. Total Frames is the only
        // control a user needs; the context length is the fidelity/cost knob and stays at the reference default.
        AnimateTotalFramesParam = T2IParamTypes.Register<int>(new(
            "Animate Total Frames",
            "Wan-Animate: total length of the finished video, generated as back-to-back chunks of 'Video Frames' each.\nLeave off (or at or below Video Frames) for a single chunk. Every chunk after the first re-renders and discards a short motion-context prefix, so the driving video must be at least this long — a shorter one freezes on its last frame.",
            "0", Min: 0, Max: 2048, Step: 4,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        AnimateMotionContextFramesParam = T2IParamTypes.Register<int>(new(
            "Animate Motion Context Frames",
            "Wan-Animate: how many frames of the previous chunk each new chunk sees as motion context, which is what keeps the seams continuous.\n5 (default) is the reference value. Higher holds continuity better and costs that many re-rendered frames per chunk; values are snapped down onto the 4n+1 grid (1, 5, 9, 13, ...).",
            "5", Min: 1, Max: 61, Step: 4,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        AnimateColorCorrectionParam = T2IParamTypes.Register<double>(new(
            "Animate Color Correction",
            "Wan-Animate chunked generation: strength of the per-chunk color correction. Each chunk after the first is color-matched (Lab mean/std) to the reference image, so color drift cannot compound across chunks.\n1 (default) = fully matched, 0 = off. Only multi-chunk (Total Frames) generations are affected — the first chunk is never touched.",
            "1", Min: 0, Max: 1, Step: 0.05,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        AnimatePoseStrengthParam = T2IParamTypes.Register<double>(new(
            "Animate Pose Strength",
            "Wan-Animate-2: how strongly the driving video's motion steers the generation (ComfyUI pose_strength) — above 1 over-drives the motion, below 1 under-drives it, and 0 does not fully mute the driving branch.\n1 (default) is the trained behaviour. This is a deliberate stylization knob, NOT a fix for washed-out or hazy output — that comes from running a checkpoint at the wrong steps/CFG, not from strength.\nAnimate-2 checkpoints only; Wan-Animate V1 refuses a non-default value.",
            "1", Min: 0, Max: 2, Step: 0.05,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        AnimateReferenceStrengthParam = T2IParamTypes.Register<double>(new(
            "Animate Reference Strength",
            "Wan-Animate-2: how strongly the character reference image is attended (ComfyUI reference_image_strength) — above 1 weights identity harder, below 1 loosens it.\n1 (default) is the trained behaviour. This is a deliberate stylization knob, NOT a fix for washed-out or hazy output — that comes from running a checkpoint at the wrong steps/CFG, not from strength.\nAnimate-2 checkpoints only; Wan-Animate V1 refuses a non-default value.",
            "1", Min: 0, Max: 2, Step: 0.05,
            Toggleable: true,
            Group: WanAnimateParamGroup,
            FeatureFlag: "hartsyinference,hartsy_wan_animate",
            ChangeWeight: 2,
            DependNonDefault: AnimateReferenceImageParam.Type.ID));

        // Extension-registered (no core equivalent): audio reference for joint audio+video models (MiniMax-H3 voice/style).
        // Maps to the engine's VideoRequest.VideoAudioReference beside core's Video Audio Input.
        VideoAudioReferenceParam = T2IParamTypes.Register<SwarmUI.Media.AudioFile>(new(
            "Video Audio Reference",
            "For audio+video models that support a reference audio clip (e.g. MiniMax-H3 voice reference): the generated soundtrack imitates this clip's voice/style rather than treating it as literal input audio.",
            null,
            Toggleable: true,
            Group: T2IParamTypes.GroupAdvancedVideo,
            FeatureFlag: "hartsyinference,hartsy_audio_ref",
            ChangeWeight: 2));

        // Extension-registered (no core equivalent): MiniMax-H3 long-form chaining. The backend maps these to the
        // engine's VideoRequest.ChainTotalFrames/ChainContextFrames.
        H3ChainTotalFramesParam = T2IParamTypes.Register<int>(new(
            "H3 Chain Total Frames",
            "MiniMax-H3: total length of the finished video, generated as back-to-back segments that each hold the previous segment's tail fixed while denoising only their own new frames.\nLeave off (or at or below the normal frame count) for a single segment.",
            "0", Min: 0, Max: 4096, Step: 17,
            Toggleable: true,
            Group: MiniMaxH3ParamGroup,
            FeatureFlag: "hartsyinference,hartsy_h3_chain",
            ChangeWeight: 2));

        H3ChainContextFramesParam = T2IParamTypes.Register<int>(new(
            "H3 Chain Context Frames",
            "MiniMax-H3: how many frames of the previous segment are carried into the next as fixed context, which is what keeps the seams continuous.\nSnapped onto the chain's own grid (39, 90, 141, ...), the 17k+5 grid intersected with a whole number of 40 Hz audio rows; the default is the engine's own reference value. Refused (not silently adjusted) if the result leaves no room for new frames at the current frame count. Longer holds continuity better and costs that many re-denoised frames per segment.",
            global::HartsyInference.Engine.Recipes.Video.MiniMaxH3ChainPlanner.DefaultContextFrames.ToString(), Min: 39, Max: 498, Step: 51,
            Toggleable: true,
            Group: MiniMaxH3ParamGroup,
            FeatureFlag: "hartsyinference,hartsy_h3_chain",
            ChangeWeight: 2));

        // Sampler and TCFG are not registered here. The backend reads Comfy's "Sampler" and "Use TCFG" instead
        // (see HonoredComfyParams), since a second near-identical control adds nothing.

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

        // FaceID-PlusV2 shortcut strength. Sits in the Image Prompting group with Comfy's IP-Adapter params, so it shows
        // whenever an IP-Adapter is selected. Only faceid-plusv2 checkpoints read it; 1.0 matches IPAdapterFaceIDPlus's s_scale.
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

        // Ideogram 4 magic prompt: optional LLM rewrite of a plain prompt into the structured JSON caption Ideogram 4 is trained on.
        // Plain text also works, so this is opt-in and only fires for Ideogram 4 generations.
        Ideogram4ParamGroup = new("Ideogram 4", Toggles: false, Open: false, IsAdvanced: false);

        Ideogram4MagicPromptParam = T2IParamTypes.Register<bool>(new(
            "Ideogram 4 Magic Prompt",
            "Rewrite your plain prompt into Ideogram 4's structured JSON caption using a running LLM backend, the way Ideogram's own stack does.\nRequires an LLM backend (Server > Backends — LlamaSharp, Anthropic, or remote). When off (default), your prompt is sent to the model as-is (which also works).",
            "false",
            // Toggleable: a flagged param that is always sent would force "hartsyinference" onto every generation
            // and refuse the Comfy backend, which lacks the flag.
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

        // Restore (SeedVR2): one-step restoration/upscale over the generated frames before muxing. The target is an area,
        // not a scale factor. Enabled by toggling the model param on; params are Toggleable for the same flag reason as Ideogram 4.
        // Every lever defaults to "follow the backend", so this group only matters when overriding one knob per generation.
        VramParamGroup = new("VRAM & Memory", Toggles: false, Open: false, IsAdvanced: true,
            Description: "Per-generation overrides of this backend's VRAM mode. Every setting here defaults to "
                + "following the backend, so leave them alone unless you are targeting one specific behaviour.\n"
                + "Levers that bake into a loaded model (quantized compute, spilling onto a second GPU) are backend "
                + "settings instead — changing those per generation would reload the model.");

        VramModeParam = T2IParamTypes.Register<string>(new(
            "VRAM Mode",
            "Override the backend's VRAM mode for this generation.\nThe individual levers below still win over whatever this picks.",
            "Backend default",
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 1,
            GetValues: _ => ["Backend default", "Performance", "Auto", "Balanced", "Aggressive", "Maximum"]));

        VramWeightStreamingParam = T2IParamTypes.Register<string>(new(
            "VRAM Weight Streaming",
            "Stream the denoiser's weights through VRAM a few blocks at a time instead of holding all of them.\n"
                + "'On' forces it even when everything would fit (typically 5-8x slower, but it is what makes a large model run on a small card).\n"
                + "'Off' forbids it — an oversized model fails instead.\nOnly models with a block-structured denoiser can honour this.",
            "Backend default",
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 2,
            GetValues: _ => ["Backend default", "Auto", "On", "Off"]));

        VramKeepResidentParam = T2IParamTypes.Register<string>(new(
            "VRAM Keep Models Resident",
            "Keep this model's weights on the GPU after the generation finishes, so the next one skips the re-upload.\n"
                + "'Off' frees them every time — slower back-to-back, but it leaves the card free for other work.",
            "Backend default",
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 3,
            GetValues: _ => ["Backend default", "Auto", "On", "Off"]));

        VramPhaseUnloadParam = T2IParamTypes.Register<string>(new(
            "VRAM Phase Unload",
            "Release each stage's weights at its boundary — the text encoder before denoising, the denoiser before "
                + "the VAE decode — so the next stage gets the space.\nCosts a re-upload per stage; often the difference "
                + "between fitting and an out-of-VRAM error.",
            "Backend default",
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 4,
            GetValues: _ => ["Backend default", "Auto", "On", "Off"]));

        VramCachePrecisionParam = T2IParamTypes.Register<string>(new(
            "VRAM Cache Precision",
            "Storage precision for the caches that persist across steps (KV caches, Wan-Animate's driving cache).\n"
                + "'Half' roughly halves their size but diverges from the reference maths — it changes the output slightly.\n"
                + "'Full' keeps exact numerics whatever it costs, which is what a parity comparison needs.",
            "Backend default",
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 5,
            GetValues: _ => ["Backend default", "Auto", "Full", "Half"]));

        VramActivationOffloadParam = T2IParamTypes.Register<string>(new(
            "VRAM Activation Offload",
            "Page cross-step activations out to system RAM as they are produced.\nTrades a PCIe round trip per step for "
                + "their full size on the card. A last resort — it is suppressed automatically while a captured CUDA graph is live.",
            "Backend default",
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 6,
            GetValues: _ => ["Backend default", "Auto", "On", "Off"]));

        VramFreeAfterGenerationParam = T2IParamTypes.Register<string>(new(
            "VRAM Free After Generation",
            "Release this model's device memory as soon as the generation finishes, rather than leaving it loaded.\n"
                + "Use when something else needs the card immediately afterwards.",
            "Backend default",
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 7,
            GetValues: _ => ["Backend default", "Auto", "On", "Off"]));

        VramChunkScaleParam = T2IParamTypes.Register<double>(new(
            "VRAM Chunk Scale",
            "Scales the chunk/tile size a model decodes in. 1.0 leaves the model's own default alone; 0.5 halves it.\n"
                + "This is the lever for work whose peak is activations rather than weights — VAE decodes, vocoders, "
                + "3D grid decodes — where streaming weights does nothing at all.",
            "1.0",
            Min: 0.1, Max: 1.0, Step: 0.05,
            Toggleable: true,
            Group: VramParamGroup,
            FeatureFlag: "hartsyinference",
            OrderPriority: 8,
            ViewType: ParamViewType.SLIDER));

        // Not "Video Restore": the pass also runs on stills, and this group toggles the whole pass.
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
            // Engine restore catalog, not the model folder: ids resolve through ModelResolver and download on use,
            // so free text would only produce typos that fail at generate time.
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

        // One group per music family, nested via T2IParamGroup.Parent. Each is model-gated by its params' flags, so a model
        // that honours none of them collapses the whole tree (hideUnsupportableParams hides groups with no visible params).
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

        // Enum, not a bool: upstream has three guidance blends and a bool reaches only two.
        // No ParameterRemaps entry, since a preset's "true"/"false" is not a valid value.
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

        // 2. Register the backend type. Always in-process, so there is no separate API variant.
        // Fully qualify Backends.HartsyInferenceBackend: the namespace's last segment (Hartsy.Extensions.HartsyInferenceBackend)
        // shadows the bare name.
        Program.Backends.RegisterBackendType<Backends.HartsyInferenceBackend>(
            "hartsyinference",
            "HartsyInference (Pure C# Inference)",
            "In-process pure-C# diffusion backend. CPU / Vulkan / CUDA support, no Python required.",
            isStandard: false);

        // 3. Register HTTP routes.
        HartsyInferenceWebAPI.Register();
        HartsyQueueWebAPI.Register();

        // 4. Pre-register all built-in architecture handlers.
        Generation.ModelSupport.RegisterBuiltins();

        // 5. Self-check the flags above against what the backend advertises.
        WarnOnUndeclaredFeatureFlags();
    }

    /// <summary>Remaps retired param IDs so old presets, preset links and reused parameters still load. One entry covers T2IPreset and T2IParamInput, since ParameterRemaps is consulted in GetType. Only IDs that are no longer registered belong here.</summary>
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
            // Points at Comfy's own param: same boolean, same meaning. No entry for Sampler: its "dpm++2m"
            // value is not in Comfy's dropdown ("dpmpp_2m"), and losing that preference beats an invalid value.
            ("hartsyinferencetcfg", "usetcfg"),
        ];
        foreach ((string old, string updated) in renames)
        {
            T2IParamTypes.ParameterRemaps[old] = updated;
        }
    }

    /// <summary>Fails at startup if a registered param carries a "hartsy" FeatureFlag this backend does not advertise.
    /// T2IEngine would refuse every generation using that param, with no message naming the param or flag. Checks every
    /// registered param, not one group. Flags in <c>T2IEngine.DisregardedFeatureFlags</c> are UI-only and exempt.</summary>
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
                        + "backend does not advertise. Every generation using that param will be refused. "
                        + "Add the flag to HartsyInferenceBackend.DeclaredFeatures, or remove it from the param.");
                }
            }
        }
    }

    public override void OnPreLaunch()
    {
        // Backend instances are created on demand by Swarm, so nothing to wire here.
    }

    public override void OnShutdown()
    {
        Logs.Info("[HartsyInference] Shutting down.");
        T2IEngine.PreGenerateEvent -= ModelFitGate.Prepare;
        // Per-instance shutdown is handled by the BackendHandler.
    }
}
