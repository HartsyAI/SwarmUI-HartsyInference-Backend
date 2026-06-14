using SwarmUI.Text2Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Central registry of every text-encoder / VAE / shared component that the per-architecture
/// loaders may need to fetch on demand. One source of truth for canonical filename, download
/// URL, SHA-256 hash, target folder, and the SwarmUI parameter the user can set to override.
///
/// <para>This mirrors ComfyUI's <c>WorkflowGeneratorModelSupport.GetXxxModel()</c> family —
/// every <c>Get</c> accessor in Comfy maps to one entry here. Keeping the catalog centralized
/// means: (1) URLs live in one place to update when upstream re-hosts; (2) two architectures
/// that share a side-model (e.g. Z-Image and Flux.2 Klein 4B both want Qwen3-4B) point at the
/// SAME canonical filename so the file gets reused, not redundantly downloaded; (3) loaders
/// can be terse — <c>SideModels.EnsureClip(SideModels.Qwen3_4B, ...)</c> instead of repeating
/// URL/hash strings.</para>
///
/// <para>For new architectures: add an <see cref="Entry"/> here, then the loader just calls
/// <see cref="ModelAutoDownloader.EnsureSideModel(Entry, ...)"/>. Don't re-hardcode URLs.</para>
/// </summary>
public static class SideModels
{
    /// <summary>One side-model: where to save it, where to fetch it from, how to verify it,
    /// and which SwarmUI parameter the user can use to override it.</summary>
    public sealed record Entry(
        string CanonicalName,
        string FolderType,
        string Url,
        string Hash,
        string DisplayName)
    {
        /// <summary>Returns true if this entry's hash field is empty / null. Hash should be
        /// non-empty for production entries — a registry without hashes silently lets the
        /// downloader skip verification, which has bitten us during slow-link mid-flight
        /// corruption (the file is the right SIZE but bytes are wrong, model loads with NaN).</summary>
        public bool HasHash => !string.IsNullOrEmpty(Hash);
    }

    // ── Text encoders (folder = "Clip" — same convention as ComfyUI's text_encoders/) ──

    /// <summary>T5-XXL encoder-only fp8 (used by Flux.1 Dev, SD3 with T5, Chroma).</summary>
    public static readonly Entry T5XxlEnconly = new(
        CanonicalName: "t5xxl_enconly.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        Hash: "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",
        DisplayName: "T5-XXL (encoder-only fp8)");

    /// <summary>Pile-T5-XL — used by AuraFlow. Bundled inside the AuraFlow checkpoint already,
    /// so this entry is mostly here as a fallback; AuraFlowLoader doesn't auto-download it.</summary>
    public static readonly Entry PileT5XlAuraFlow = new(
        CanonicalName: "pile_t5xl_auraflow.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/fal/AuraFlow-v0.2/resolve/main/text_encoder/model.safetensors",
        Hash: "0a07449cf1141c0ec86e653c00465f6f0d79c6e58a2c60c8bcf4203d0e4ec4f6",
        DisplayName: "Pile-T5-XL (AuraFlow)");

    /// <summary>CLIP-L (SD1.5/SDXL/SD3/Flux text encoder).</summary>
    public static readonly Entry ClipL = new(
        CanonicalName: "clip_l.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors",
        Hash: "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
        DisplayName: "CLIP-L");

    /// <summary>CLIP-G (SDXL / SD3 second text encoder).</summary>
    public static readonly Entry ClipG = new(
        CanonicalName: "clip_g.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors",
        Hash: "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
        DisplayName: "CLIP-G");

    /// <summary>HiDream long-CLIP-L. Architecturally a standard SDXL CLIP-L (SdxlClipL config), but a
    /// HiDream-specific long-context fine-tune — distinct weights from the plain <see cref="ClipL"/>.
    /// Canonical filename matches Comfy's local name so a side-by-side install shares the file.</summary>
    public static readonly Entry HiDreamClipL = new(
        CanonicalName: "long_clip_l_hi_dream.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_l_hidream.safetensors",
        Hash: "706fdb88e22e18177b207837c02f4b86a652abca0302821f2bfa24ac6aea4f71",
        DisplayName: "HiDream long-CLIP-L");

    /// <summary>HiDream long-CLIP-G. Standard SDXL CLIP-G (SdxlClipG config) with HiDream-tuned weights.</summary>
    public static readonly Entry HiDreamClipG = new(
        CanonicalName: "long_clip_g_hi_dream.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_g_hidream.safetensors",
        Hash: "3771e70e36450e5199f30bad61a53faae85a2e02606974bcda0a6a573c0519d5",
        DisplayName: "HiDream long-CLIP-G");

    /// <summary>Llama-3.1-8B-Instruct (fp8 scaled) — HiDream's fourth text encoder. Run as a feature
    /// extractor; HiDream harvests hidden states from every layer. Pairs with the embedded Llama-3.1
    /// tokenizer in HartsyInference.Tokenizers.</summary>
    public static readonly Entry Llama31_8B = new(
        CanonicalName: "llama_3.1_8b_instruct_fp8_scaled.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors",
        Hash: "9f86897bbeb933ef4fd06297740edb8dd962c94efcd92b373a11460c33765ea6",
        DisplayName: "Llama-3.1-8B-Instruct (fp8 scaled)");

    /// <summary>Qwen2.5-VL-7B (fp8 scaled) — Qwen-Image's text encoder, run as a feature extractor.
    /// Pairs with the embedded Qwen3 tokenizer (shared base BPE merges; raw-text IDs match).</summary>
    public static readonly Entry Qwen2_5_VL_7B = new(
        CanonicalName: "qwen_2.5_vl_7b.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        Hash: "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4",
        DisplayName: "Qwen2.5-VL-7B (fp8 scaled)");

    /// <summary>Qwen3-Embedding-0.6B — ACE-Step 1.5's style + lyric conditioner (1024-d states).
    /// Official Qwen single-file release. TODO: pin the SHA-256 (entry ships hash-less; the
    /// downloader logs a no-verification warning until it's added).</summary>
    public static readonly Entry Qwen3Embedding06B = new(
        CanonicalName: "qwen3_embedding_0.6b.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/model.safetensors",
        Hash: "",
        DisplayName: "Qwen3-Embedding-0.6B (ACE-Step 1.5)");

    /// <summary>Qwen3-4B fp8-mixed — used by Z-Image and Flux.2 Klein 4B. Same file, both
    /// architectures share it once it's downloaded.</summary>
    public static readonly Entry Qwen3_4B = new(
        CanonicalName: "qwen_3_4b.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        Hash: "72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15",
        DisplayName: "Qwen3-4B (fp8 mixed)");

    /// <summary>Qwen3-8B for Flux.2 Klein 9B. NOTE: Comfy's canonical URL points at an fp4-mixed
    /// quant (`qwen_3_8b_fp4mixed.safetensors`). HartsyInference doesn't yet have FP4 GEMM
    /// support — Flux2Loader refuses Klein 9B at runtime regardless. Once HartsyInference
    /// either (a) supports FP4 cuBLAS / Vulkan, or (b) we point this entry at a fp16/fp8
    /// alternative, Klein 9B will work end-to-end.</summary>
    public static readonly Entry Qwen3_8B_Fp4Mixed = new(
        CanonicalName: "qwen_3_8b.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp4mixed.safetensors",
        Hash: "bbf16f981d98e16d080c566134814c4e9f6aadd0d0e1383c60bc44ba939d760d",
        DisplayName: "Qwen3-8B (fp4 mixed) — Klein 9B");

    /// <summary>Mistral 3 Small for Flux.2 Dev. Same FP4 caveat as Qwen3-8B above.</summary>
    public static readonly Entry MistralSmallFlux2 = new(
        CanonicalName: "mistral_3_small_flux2.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors",
        Hash: "1ee1ff334d78228d73049ef0ee4fcd21c1700536b5a45c06547af057f92463a7",
        DisplayName: "Mistral 3 Small (fp4 mixed) — Flux.2 Dev");

    /// <summary>Ministral 3.3B for Ernie Image. HartsyInference has the encoder preset
    /// (LlamaStyleEncoderConfig.Ministral3B), but no Ernie tokenizer — see
    /// <see cref="ModelSupport"/> notes.</summary>
    public static readonly Entry Ministral_3_3B = new(
        CanonicalName: "ministral-3-3b.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/text_encoders/ministral-3-3b.safetensors",
        Hash: "49a750a128863854eac7d85e1a277a7b44bf6ec3646405b84686dfeeca3708ca",
        DisplayName: "Ministral 3.3B (Ernie)");

    /// <summary>GPT-OSS-20B (NVFP4) — Microsoft Lens' text encoder (MoE feature extractor with
    /// multi-layer tap). Canonical name/URL/hash match <c>Comfy-Org/Lens</c> so the 13 GB file is
    /// shared with a Comfy install; the engine's NVFP4 codec dequants at load.</summary>
    public static readonly Entry LensGptOss20b = new(
        CanonicalName: "gpt_oss_20b_nvfp4.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/Lens/resolve/main/text_encoders/gpt_oss_20b_nvfp4.safetensors",
        Hash: "103d7759c720627e5ffdcb0d885595695085dad4201fa6a522a84d4b86335ca0",
        DisplayName: "GPT-OSS-20B (nvfp4) — Lens");

    /// <summary>UMT5-base — ACE-Step v1's style/genre text encoder (768-d hidden states). The official
    /// repo bundles it under <c>umt5-base/model.safetensors</c>; pairs with the embedded umT5 SentencePiece
    /// (same 256k vocab as umT5-XXL). SHA-256 from the HF LFS metadata.</summary>
    public static readonly Entry Umt5Base = new(
        CanonicalName: "umt5_base.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/umt5-base/model.safetensors",
        Hash: "779cec0d210b2123e21d0a9cd8128f02b4d412627355028965a8be0b241cc3b6",
        DisplayName: "UMT5-base (ACE-Step)");

    /// <summary>umT5-XXL (fp8 e4m3 scaled) — Wan-Video's text encoder. Canonical filename, URL, and
    /// hash match SwarmUI ComfyUI's <c>GetUniMaxT5XXLModel()</c> exactly so the file is shared with a
    /// Comfy install. The fp8 scale_weight tensors are folded at load time via
    /// <c>CheckpointConvertUtils.ApplyFp8ScaledDequant</c>. Pairs with the embedded umT5 SentencePiece
    /// in HartsyInference.Tokenizers (the base T5 spiece is NOT compatible — 32k vs 256k vocab).</summary>
    public static readonly Entry Umt5Xxl = new(
        CanonicalName: "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        Hash: "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
        DisplayName: "umT5-XXL (fp8 scaled) — Wan video");

    // ── VAEs (folder = "VAE") ──

    /// <summary>Qwen3-0.6B Base — Anima's text encoder. The Anima HF repo hosts this file as
    /// <c>split_files/text_encoders/qwen_3_06b_base.safetensors</c>; we save it locally with the
    /// canonical name <c>qwen_3_600m.safetensors</c> to match SwarmUI ComfyUI's
    /// <c>GetQwen3_600mModel()</c> filename so the file is reused if/when ComfyUI auto-downloads it.</summary>
    public static readonly Entry Qwen3_0_6B = new(
        CanonicalName: "qwen_3_600m.safetensors",
        FolderType: "Clip",
        Url: "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
        Hash: "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
        DisplayName: "Qwen3-0.6B Base (Anima)");

    /// <summary>Flux.1 VAE — shared by Flux.1, Z-Image, and Chroma.</summary>
    public static readonly Entry FluxAe = new(
        CanonicalName: "Flux/ae.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/mcmonkey/swarm-vaes/resolve/main/flux_ae.safetensors",
        Hash: "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        DisplayName: "Flux.1 ae");

    /// <summary>Flux.2 VAE — distinct from Flux.1's ae (32-channel latent + BatchNorm stats).</summary>
    public static readonly Entry Flux2Vae = new(
        CanonicalName: "Flux2/flux2-vae.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/Comfy-Org/Flux2_repackaged/resolve/main/split_files/vae/flux2_vae.safetensors",
        Hash: "",  // NOTE: hash not yet captured from Comfy registry — verify on first download
        DisplayName: "Flux.2 VAE");

    /// <summary>Qwen-Image VAE — 16-channel autoencoder used by Anima and Qwen Image. Matches the
    /// hash registered in SwarmUI's <c>CommonModels</c> ("qwen-image-vae") so a previously-downloaded
    /// copy is reused.</summary>
    public static readonly Entry QwenImageVae = new(
        CanonicalName: "QwenImage/qwen_image_vae.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        Hash: "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
        DisplayName: "Qwen-Image VAE");

    /// <summary>ACE-Step v1 Music-DCAE (Sana-style mel autoencoder, decode side used at inference).
    /// SHA-256 from the HF LFS metadata. Saved under <c>VAE/AceStep/</c> — it's the audio analog of a VAE.</summary>
    public static readonly Entry AceStepDcae = new(
        CanonicalName: "AceStep/music_dcae_f8c8.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_dcae_f8c8/diffusion_pytorch_model.safetensors",
        Hash: "2b0cb469307ac50659d1880db2a99bae47d0df335cbb36853964662d4b80e8ee",
        DisplayName: "ACE-Step Music-DCAE");

    /// <summary>ACE-Step v1 ADaMoS HiFi-GAN vocoder (mel → 44.1 kHz waveform). SHA-256 from the HF LFS metadata.</summary>
    public static readonly Entry AceStepVocoder = new(
        CanonicalName: "AceStep/music_vocoder.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/diffusion_pytorch_model.safetensors",
        Hash: "c92c9b46e28ab7b37b777780cf4308ad7ddac869636bb77aa61599358c4bc1c0",
        DisplayName: "ACE-Step Music Vocoder");

    /// <summary>Wan2.2 VAE — 48-channel 3D video autoencoder used by Wan2.2 TI2V-5B. Canonical path,
    /// URL, and hash match SwarmUI core's <c>CommonModels.Known["wan22-vae"]</c> so a copy downloaded
    /// by the ComfyUI backend is reused (and vice versa).</summary>
    public static readonly Entry Wan22Vae = new(
        CanonicalName: "Wan/wan2.2_vae.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
        Hash: "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
        DisplayName: "Wan 2.2 VAE");

    /// <summary>ACE-Step 1.5 Oobleck audio VAE (48 kHz stereo ↔ 64-ch 25 Hz latents). Canonical
    /// path, URL, and hash match SwarmUI core's <c>CommonModels.Known["ace-step-15-vae"]</c> so a
    /// copy downloaded by the ComfyUI backend is reused (and vice versa).</summary>
    public static readonly Entry AceStep15Vae = new(
        CanonicalName: "AceStep/ace_1.5_vae.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files/vae/ace_1.5_vae.safetensors",
        Hash: "6de92e3a862acd287e08b024ac90f0783a8635451b728721a33ff03565bcb2bb",
        DisplayName: "ACE-Step 1.5 VAE (Oobleck)");

    // ── TAESD preview decoders (folder = "VAE", subfolder "Taesd/") ──
    //
    // Tiny autoencoders by Ollin Boer Bohan (madebyollin/taesd). ~10 MB each, one per latent
    // family. Used by PreviewEncoder when PreviewMethod=taesd. Hashes are intentionally
    // empty — the upstream repos don't publish a canonical sha256 and we don't want to gate
    // first-download on a hash we can't verify in advance; the SafeTensorsLoader will fail
    // loudly if the file is truncated/corrupt anyway. Fill these in once a known-good copy
    // is on disk if drift protection becomes important.

    public static readonly Entry TaesdSd15 = new(
        CanonicalName: "Taesd/sd_decoder.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/madebyollin/taesd/resolve/main/taesd_decoder.safetensors",
        Hash: "",
        DisplayName: "TAESD (SD 1.5 preview decoder)");

    public static readonly Entry TaesdSdxl = new(
        CanonicalName: "Taesd/sdxl_decoder.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/madebyollin/taesdxl/resolve/main/taesdxl_decoder.safetensors",
        Hash: "",
        DisplayName: "TAESD (SDXL preview decoder)");

    public static readonly Entry TaesdSd3 = new(
        CanonicalName: "Taesd/sd3_decoder.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/madebyollin/taesd3/resolve/main/taesd3_decoder.safetensors",
        Hash: "",
        DisplayName: "TAESD (SD3 preview decoder)");

    public static readonly Entry TaesdFlux = new(
        CanonicalName: "Taesd/flux_decoder.safetensors",
        FolderType: "VAE",
        Url: "https://huggingface.co/madebyollin/taef1/resolve/main/taef1_decoder.safetensors",
        Hash: "",
        DisplayName: "TAESD (Flux.1 preview decoder)");

    // ── CLIP-Vision (folder = "ClipVision") ──

    /// <summary>CLIP-ViT-H/14 image encoder used by IP-Adapter SDXL standard + Plus and several
    /// other image-conditioned components. Auto-downloaded when the user enables IP-Adapter
    /// without explicitly setting <c>ClipVisionModel</c>. The h94/IP-Adapter repo's bundled
    /// <c>image_encoder/model.safetensors</c> is the canonical version trained against (LAION
    /// OpenCLIP weights — same architecture as Stability's, same normalization constants).</summary>
    public static readonly Entry ClipVisionH14 = new(
        CanonicalName: "clip-vision-h-14.safetensors",
        FolderType: "ClipVision",
        Url: "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        Hash: "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
        DisplayName: "CLIP-Vision H/14 (IP-Adapter compatible)");
}
