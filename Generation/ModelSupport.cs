namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Maps SwarmUI's <c>T2IModel.ModelClass.CompatClass.ID</c> → SharpInference architecture.
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
        AuraFlowLoader.AuraFlowCompatClassId,     // "auraflow-v1"
        FLiteLoader.FLiteCompatClassId,           // "f-lite" — wired-untested (no E2E test in SharpInference yet)
        Ideogram4Loader.Ideogram4CompatClassId,   // "ideogram-4" — dual 9.3B DiT; ≥22 GB VRAM gate at load time; non-commercial license
        ZImageLoader.ZImageCompatClassId,         // "z-image"
        AnimaLoader.AnimaCompatClassId,           // "anima" — Cosmos-Predict2-2B family + LlmAdapter
        HiDreamLoader.HiDreamI1CompatClassId,     // "hidream-i1" — MMDiT + 4 text encoders (CLIP-L/G, T5-XXL, Llama-3.1)
        QwenImageLoader.QwenImageCompatClassId,   // "qwen-image" — 20B MMDiT + Qwen2.5-VL-7B encoder
        WanVideoLoader.Wan22_5BCompatClassId,     // "wan-22-5b" — Wan2.2 TI2V-5B text-to-video (I2V pending VAE encoder)
        LtxVideoLoader.LtxVideoCompatClassId,     // "lightricks-ltx-video" — LTX-Video 0.9 single-file text-to-video
        AceStepLoader.AceStepCompatClassId,       // "ace-step-1_5" compat shared by our ACE-Step v1 class; validation
                                                  // refuses actual v1.5 checkpoints (engine implements v1)
        // TODO: SdxlLoader.SdxlRefinerCompatClassId once we wire the refiner two-pass flow.
    };

    /// <summary>Architectures where SharpInference HAS a working pipeline but the SwarmUI
    /// extension hasn't wired a loader yet. Refusing one of these isn't "we can't do this";
    /// it's "the backend code exists but the SwarmUI integration glue (text-encoder selection,
    /// VAE auto-download, tokenizer wiring, parameter rules) is a TODO." The error message
    /// for these should suggest using ComfyUI in the meantime, not "this won't work."</summary>
    private static readonly Dictionary<string, string> _pendingArchs = new()
    {
        ["chroma-radiance"] = "Chroma Radiance",
        ["zeta-chroma"] = "Zeta-Chroma",
        // ErnieImage: pipeline + Ministral3B encoder preset both exist, but the SharpInference
        // upstream test uses HARDCODED token IDs ([1,2,3,...]) — there is no real Ernie tokenizer
        // implementation. Wiring a loader without a tokenizer means the user's prompt would
        // never reach the model. Refuse cleanly until SharpInference ships an Ernie tokenizer.
        ["ernie-image"] = "ErnieImage (no tokenizer in SharpInference yet — prompts can't be encoded)",
        // HunyuanImage: the SharpInference pipeline exists and the E2E test runs, BUT it substitutes
        // T5-XXL for the real Qwen2.5-VL MLLM primary encoder (and drops the byT5 glyph stream), so
        // output wouldn't match the real model. Refuse until the engine wires the correct encoders.
        ["hunyuan-image-2_1"] = "HunyuanImage (SharpInference pipeline uses T5-XXL as a stand-in for the real Qwen2.5-VL encoder — not faithful yet)",
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
            return "Model has no architecture compat class set — SharpInference can't dispatch.";
        }
        if (_pendingArchs.TryGetValue(compatClass, out string friendlyName))
        {
            return $"{friendlyName} ('{compatClass}') is not yet wired into the SwarmUI extension. " +
                   "SharpInference itself has the pipeline + checkpoint converter, but the per-architecture " +
                   "loader (text-encoder selection, VAE auto-download, tokenizer setup) is a TODO. " +
                   "Use the ComfyUI backend for this architecture in the meantime.";
        }
        return $"Architecture '{compatClass}' is not implemented in SharpInference. " +
               $"Supported today: {string.Join(", ", _supportedArchs)}.";
    }

    public static IReadOnlyCollection<string> SupportedArchitectures => _supportedArchs;

    /// <summary>Stub from earlier scaffolding; legacy step-priority registration is unused
    /// while we're flat-dispatching from the backend's GenerateLive.</summary>
    public static void RegisterBuiltins()
    {
        // No-op; loaders are invoked directly from SharpInferenceBackend.
    }
}
