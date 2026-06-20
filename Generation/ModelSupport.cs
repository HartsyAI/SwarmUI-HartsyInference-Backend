namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Maps SwarmUI's <c>T2IModel.ModelClass.CompatClass.ID</c> → HartsyInference architecture.
/// CompatClass IDs are registered in <c>src/Text2Image/T2IModelClassSorter.cs</c>; we
/// dispatch off them in the backend's LoadModel/GenerateLive.
///
/// See docs/05-Pipeline-Translation.md §Architecture detection table.
/// </summary>
public static class ModelSupport
{
    /// <summary>Architectures with a fully wired loader in this extension.</summary>
    private static readonly HashSet<string> _supportedArchs = new()
    {
        Sd15Loader.Sd15CompatClassId,             // "stable-diffusion-v1"
        SdxlLoader.SdxlCompatClassId,             // "stable-diffusion-xl-v1"
        Sd3Loader.Sd3MediumCompatClassId,         // "stable-diffusion-v3-medium"
        Sd3Loader.Sd35MediumCompatClassId,        // "stable-diffusion-v3.5-medium"
        Sd3Loader.Sd35LargeCompatClassId,         // "stable-diffusion-v3.5-large"
        FluxLoader.Flux1CompatClassId,            // "flux-1"
        Flux2Loader.Flux2BaseCompatClassId,       // "flux-2"
        Flux2Loader.Flux2Klein4BCompatClassId,    // "flux-2-klein-4b"
        Flux2Loader.Flux2Klein9BCompatClassId,    // "flux-2-klein-9b" — loader refuses at runtime if Qwen3-8B preset missing
        ChromaLoader.ChromaCompatClassId,         // "chroma"
        ChromaRadianceLoader.ChromaRadianceCompatClassId, // "chroma-radiance" — pixel-space (no VAE); validation-gated numerics
        ZetaChromaLoader.ZetaChromaCompatClassId, // "zeta-chroma" — pixel-space Qwen3-4B; validation-gated numerics
        AuraFlowLoader.AuraFlowCompatClassId,     // "auraflow-v1"
        FLiteLoader.FLiteCompatClassId,           // "f-lite" — wired-untested (no E2E test in HartsyInference yet)
        Ideogram4Loader.Ideogram4CompatClassId,   // "ideogram-4" — dual 9.3B DiT; ≥22 GB VRAM gate at load time; non-commercial license
        ErnieImageLoader.ErnieImageCompatClassId, // "ernie-image" — Baidu ~8B single-stream DiT + Ministral-3-3B TE + Flux.2 VAE (Apache-2.0)
        ZImageLoader.ZImageCompatClassId,         // "z-image"
        AnimaLoader.AnimaCompatClassId,           // "anima" — Cosmos-Predict2-2B family + LlmAdapter
        HiDreamLoader.HiDreamI1CompatClassId,     // "hidream-i1" — MMDiT + 4 text encoders (CLIP-L/G, T5-XXL, Llama-3.1)
        QwenImageLoader.QwenImageCompatClassId,   // "qwen-image" — 20B MMDiT + Qwen2.5-VL-7B encoder
        WanVideoLoader.Wan22_5BCompatClassId,     // "wan-22-5b" — Wan2.2 TI2V-5B text/image-to-video
        WanVideoLoader.Wan21_1_3BCompatClassId,   // "wan-21-1_3b" — Wan2.1 1.3B T2V + VACE-1.3B (model-class ID routes VACE → WanVaceLoader)
        WanVideoLoader.Wan21_14BCompatClassId,    // "wan-21-14b" — Wan2.1 14B (T2V + CLIP-I2V) + VACE-14B + Wan2.2 A14B (single-expert); VACE routes via WanModelVariants.IsVace
        LtxVideoLoader.LtxVideoCompatClassId,     // "lightricks-ltx-video" — LTX-Video 0.9 single-file text-to-video
        LtxVideo2Loader.LtxVideo2CompatClassId,   // "lightricks-ltx-video-2" — LTX-2.3 22B dual-stream text-to-video+audio (validation-pending)
        AceStepLoader.AceStepCompatClassId,       // "ace-step-1_5" — v1 checkpoints route to AceStepLoader, real v1.5
                                                  // checkpoints to AceStep15Loader (2B turbo, validation-pending numerics)
        // MusicGenLoader.MusicGenCompatClassId — moved to _pendingArchs until the engine ships
        // EnCodec-32kHz + T5-Base presets and the converter's text-encoder path (loader is otherwise complete)
        // YueLoader.YueCompatClassId — moved to _pendingArchs until HartsyInference ships YueTokenizer (loader is otherwise complete)
        LanceLoader.LanceCompatClassId,           // "lance" — ByteDance Lance 3B folder-checkpoint T2I (validation-pending numerics)
        LanceLoader.LanceVideoCompatClassId,      // "lance-video" — Lance 3B Video T2V (validation-pending numerics)
        LensLoader.LensCompatClassId,             // "lens" — Microsoft Lens 3.8B MMDiT + GPT-OSS-20B encoder (Comfy split files)
        // TODO: SdxlLoader.SdxlRefinerCompatClassId once we wire the refiner two-pass flow.
    };

    /// <summary>Architectures where HartsyInference HAS a working pipeline but the SwarmUI
    /// extension hasn't wired a loader yet. Refusing one of these isn't "we can't do this";
    /// it's "the backend code exists but the SwarmUI integration glue (text-encoder selection,
    /// VAE auto-download, tokenizer wiring, parameter rules) is a TODO." The error message
    /// for these should suggest using ComfyUI in the meantime, not "this won't work."</summary>
    private static readonly Dictionary<string, string> _pendingArchs = new()
    {
        // Kandinsky 5 / OmniGen 2 / Lumina 2: the HartsyInference pipelines exist and pass structural
        // tests, but their conditioning encoders are NOT faithfully implemented — the upstream E2E
        // tests feed PRE-COMPUTED embeddings from .bin dumps (Kandinsky: dual Qwen2.5-VL + CLIP-L with
        // an unverified prompt template; OmniGen 2: Qwen2.5-VL; Lumina 2: Gemma-2-2B, and HartsyInference
        // has no Gemma tokenizer at all). Wiring a loader with guessed templates would produce
        // semantically-wrong conditioning — same reason HunyuanImage is refused below.
        ["kandinsky5-imglite"] = "Kandinsky 5 Image Lite (engine pipeline needs pre-computed Qwen2.5-VL + CLIP-L embeddings — live encode path unverified)",
        ["omnigen-2"] = "OmniGen 2 (engine pipeline needs pre-computed Qwen2.5-VL embeddings — live encode path unverified)",
        ["lumina-2"] = "Lumina-Image-2.0 (no Gemma-2 tokenizer/encoder path in HartsyInference yet)",
        // ErnieImage: WIRED 2026-06-17 (ErnieImageLoader) — engine shipped ErnieTokenizer in alpha.8, so it
        // moved to _supportedArchs above. (Was blocked on the missing real Ernie tokenizer.)
        // YuE: the extension loader (YueLoader.cs) is fully written, but HartsyInference has no
        // YueTokenizer (the mm SentencePiece wrapper) — lyrics can't be encoded. Same class of
        // blocker as Ernie. Lift by restoring the TODO(engine-blocked) lines in YueLoader.cs and
        // re-adding YueCompatClassId to _supportedArchs.
        [YueLoader.YueCompatClassId] = "YuE (no YuE mm tokenizer in HartsyInference yet — lyrics can't be encoded)",
        // MusicGen: extension loader (MusicGenLoader.cs) fully written; engine is missing the
        // EnCodec-32kHz preset, T5-Base preset, and the converter's bundled-text-encoder path.
        [MusicGenLoader.MusicGenCompatClassId] = "MusicGen (engine missing EnCodec-32kHz/T5-Base presets + text-encoder converter path)",
        // HunyuanImage: the HartsyInference pipeline exists and the E2E test runs, BUT it substitutes
        // T5-XXL for the real Qwen2.5-VL MLLM primary encoder (and drops the byT5 glyph stream), so
        // output wouldn't match the real model. Refuse until the engine wires the correct encoders.
        ["hunyuan-image-2_1"] = "HunyuanImage (HartsyInference pipeline uses T5-XXL as a stand-in for the real Qwen2.5-VL encoder — not faithful yet)",
    };

    public static bool IsArchitectureSupported(string compatClass)
    {
        return !string.IsNullOrEmpty(compatClass) && _supportedArchs.Contains(compatClass);
    }

    /// <summary>Human-readable explanation of why a given compat class isn't supported.
    /// Distinguishes "we have the engine but not the loader" (pending) from "this isn't
    /// implemented anywhere" (genuinely unsupported).</summary>
    public static string WhyNotSupported(string compatClass)
    {
        if (string.IsNullOrEmpty(compatClass))
        {
            return "Model has no architecture compat class set — HartsyInference can't dispatch.";
        }
        if (_pendingArchs.TryGetValue(compatClass, out string friendlyName))
        {
            return $"{friendlyName} ('{compatClass}') is not yet wired into the SwarmUI extension. " +
                   "HartsyInference itself has the pipeline + checkpoint converter, but the per-architecture " +
                   "loader (text-encoder selection, VAE auto-download, tokenizer setup) is a TODO. " +
                   "Use the ComfyUI backend for this architecture in the meantime.";
        }
        return $"Architecture '{compatClass}' is not implemented in HartsyInference. " +
               $"Supported today: {string.Join(", ", _supportedArchs)}.";
    }

    public static IReadOnlyCollection<string> SupportedArchitectures => _supportedArchs;

    /// <summary>Architectures the engine has a pipeline for but the extension refuses today,
    /// mapped to the human-readable blocker reason. Surfaced by the WebAPI for admin UX.</summary>
    public static IReadOnlyDictionary<string, string> PendingArchitectures => _pendingArchs;

    /// <summary>Stub from earlier scaffolding; legacy step-priority registration is unused
    /// while we're flat-dispatching from the backend's GenerateLive.</summary>
    public static void RegisterBuiltins()
    {
        // No-op; loaders are invoked directly from HartsyInferenceBackend.
    }
}
