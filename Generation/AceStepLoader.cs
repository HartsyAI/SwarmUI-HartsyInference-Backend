using System.IO;
using Newtonsoft.Json.Linq;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Music;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;
using Image = SwarmUI.Utils.Image;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Loads ACE-Step v1 (3.5B flow-matching music DiT; <c>ACE-Step/ACE-Step-v1-3.5B</c>). The user's
/// main model is the DiT safetensors (<c>ace_step_transformer/diffusion_pytorch_model.safetensors</c>
/// from the HF repo, any filename — detected by the <c>lyric_embs.weight</c> key); the Music-DCAE,
/// ADaMoS vocoder, and UMT5-base style encoder auto-download as side models, and the XTTS-style
/// lyric tokenizer vocab is fetched + exported once from the official GitHub repo.
///
/// <para>SwarmUI core only registers the v1.5 model class, so <see cref="RegisterModelClass"/> adds a
/// v1 class under the SAME <c>ace-step-1_5</c> compat class — that lights up the Text2Audio param
/// group and audio output handling without core changes. Validation distinguishes the two by model
/// class ID and refuses actual v1.5 checkpoints (the engine implements v1).</para>
///
/// <para>Param mapping mirrors Comfy's audio flow: Prompt = lyrics (empty → instrumental),
/// <c>Text2AudioStyle</c> = the style/genre tags (with BPM / key scale / time signature appended as
/// free text — v1 reads those from the style prompt), <c>Text2AudioDuration</c> = length,
/// <c>Text2AudioLanguage</c> = lyric language override. Output is MP3 V0 like Comfy's
/// <c>SaveAudioMP3</c>.</para>
/// </summary>
public static class AceStepLoader
{
    /// <summary>Swarm's audio compat class — reused so the Text2Audio params appear. The loader keys
    /// off <see cref="AceStepV1ClassId"/> to tell v1 checkpoints from real v1.5 ones.</summary>
    public const string AceStepCompatClassId = "ace-step-1_5";

    /// <summary>The extension-registered model class for v1 checkpoints.</summary>
    public const string AceStepV1ClassId = "ace-step-v1";

    /// <summary>UMT5 style-prompt context length (the reference pipeline's max_len=256).</summary>
    private const int TextTokenLength = 256;

    private const string LyricVocabUrl = "https://raw.githubusercontent.com/ace-step/ACE-Step/main/acestep/models/lyrics_utils/vocab.json";

    /// <summary>Registers the ACE-Step v1 model class (core only knows v1.5). Call once at extension init,
    /// before model folders are scanned.</summary>
    public static void RegisterModelClass()
    {
        T2IModelClassSorter.Register(new T2IModelClass
        {
            ID = AceStepV1ClassId,
            CompatClass = T2IModelClassSorter.CompatAceStep15,
            Name = "ACE-Step v1",
            IsThisModelOfClass = (model, header) =>
                header.ContainsKey("lyric_embs.weight") || header.ContainsKey("model.lyric_embs.weight"),
        });
    }

    public static AceStepCacheEntry Load(
        IBackend backend,
        T2IModel model,
        T2IParamInput input,
        Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(model?.RawFilePath))
            throw new InvalidOperationException("ACE-Step model has no file path.");
        if (!File.Exists(model.RawFilePath))
            throw new FileNotFoundException($"ACE-Step checkpoint not found: {model.RawFilePath}");

        T2IModel dcaeModel = ModelAutoDownloader.EnsureSideModel(
            userPick: null, entry: SideModels.AceStepDcae, log: log);
        T2IModel vocoderModel = ModelAutoDownloader.EnsureSideModel(
            userPick: null, entry: SideModels.AceStepVocoder, log: log);
        T2IModel umt5Model = ModelAutoDownloader.EnsureSideModel(
            userPick: input?.Get(T2IParamTypes.T5XXLModel), entry: SideModels.Umt5Base, log: log);
        (string vocabPath, string mergesPath) = EnsureLyricTokenizerFiles(log);

        // ── 1. DiT ──
        log($"Loading ACE-Step DiT: {model.Name}");
        var (ditWeights, ditLoader) = AceStepCheckpointConverter.LoadTransformer(model.RawFilePath);
        if (ditWeights.Count == 0)
        {
            ditLoader.Dispose();
            throw new InvalidOperationException(
                $"ACE-Step checkpoint '{model.Name}' has no recognized transformer weights.");
        }
        AceStepConfig config = AceStepConfig.V1;
        AceStepDit dit = new AceStepDit(config);
        dit.LoadWeights(ditWeights);

        // ── 2. Music-DCAE decoder + ADaMoS vocoder (published-model default configs) ──
        log($"Loading Music-DCAE: {dcaeModel.Name}");
        var (dcaeWeights, dcaeLoader) = AceStepCheckpointConverter.LoadDcae(dcaeModel.RawFilePath);
        MusicDcaeDecoder dcae = new MusicDcaeDecoder();
        dcae.LoadWeights(dcaeWeights);
        log($"Loading ADaMoS vocoder: {vocoderModel.Name}");
        var (vocoderWeights, vocoderLoader) = AceStepCheckpointConverter.LoadVocoder(vocoderModel.RawFilePath);
        AdaMosHiFiGanV1 vocoder = new AdaMosHiFiGanV1();
        vocoder.LoadWeights(vocoderWeights);

        // ── 3. UMT5-base style encoder (768-d; embedded umT5 tokenizer) ──
        log($"Loading UMT5-base: {umt5Model.Name}");
        SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
        umt5Loader.Load(umt5Model.RawFilePath);
        T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Base);
        umt5.LoadWeights(umt5Loader.GetAllTensors());
        T5Tokenizer textTokenizer = T5Tokenizer.CreateUmt5(maxLength: TextTokenLength);

        // ── 4. Lyric tokenizer (XTTS VoiceBpe export) ──
        log("Loading ACE lyric tokenizer...");
        AceStepLyricTokenizer lyricTokenizer = new AceStepLyricTokenizer(vocabPath, mergesPath);

        log("Building ACE-Step pipeline...");
        AceStepPipeline pipeline = new AceStepPipeline(backend, dit, dcae, vocoder, config);

        log("ACE-Step v1 ready (text/lyrics-to-music).");
        return new AceStepCacheEntry
        {
            ModelName = model.Name,
            CompatClass = AceStepCompatClassId,
            Pipeline = pipeline,
            Config = config,
            Dit = dit,
            Umt5 = umt5,
            TextTokenizer = textTokenizer,
            LyricTokenizer = lyricTokenizer,
            CheckpointLoader = ditLoader,
            DcaeLoader = dcaeLoader,
            VocoderLoader = vocoderLoader,
            Umt5Loader = umt5Loader,
        };
    }

    public static Image[] Generate(
        AceStepCacheEntry entry,
        IBackend backend,
        T2IParamInput input,
        Action<GenerationProgress> onProgress,
        CancellationToken cancel)
    {
        string lyrics = input.Get(T2IParamTypes.Prompt) ?? "";
        string style = ComposeStyleText(input);
        string language = input.Get(T2IParamTypes.Text2AudioLanguage, "en");
        double duration = Math.Clamp(input.Get(T2IParamTypes.Text2AudioDuration, 120d), 1d, 600d);
        int steps = SamplingParamResolver.ResolveSteps(input, fallback: entry.Config.NumInferenceSteps);
        long seedLong = input.Get(T2IParamTypes.Seed);
        double cfgRaw = input.Get(T2IParamTypes.CFGScale);
        float? guidance = cfgRaw <= 0 ? null : (float)cfgRaw;

        // Style prompt → UMT5 features [T, 768]; drop the encoder's GPU weights before the 3.5B DiT
        // preload (mirrors the video loaders). The pipeline zero-fills its own unconditional context.
        int[] textTokens = entry.TextTokenizer.Encode(style);
        Tensor batch = entry.Umt5.Encode(backend, [textTokens], [T5Tokenizer.CreateAttentionMask(textTokens)]);
        Tensor textEmbeds = CfgHelper.SliceBatchElement(batch, 0, TextTokenLength, entry.Config.TextDim);
        batch.Dispose();
        backend.Sync();
        backend.FreeWeights(entry.Umt5.EnumerateWeights());

        int[] lyricIds = string.IsNullOrWhiteSpace(lyrics)
            ? []
            : entry.LyricTokenizer.TokenizeLyrics(lyrics, languageOverride: language);

        long start = Environment.TickCount64;
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            onProgress(p);
        };

        try
        {
            var (left, right, sampleRate, _) = entry.Pipeline.Generate(
                textEmbeds, lyricIds, duration,
                steps: steps, guidance: guidance,
                guidanceMode: AceStepPipeline.GuidanceMode.Apg,
                sampler: AceStepPipeline.SamplerMode.Euler,
                seed: seedLong < 0 ? null : (int?)(int)(seedLong & 0x7FFFFFFF),
                onProgress: bridge);
            Logs.Verbose($"[SharpInference][ACE] Pipeline returned {left.Length} samples/channel @ {sampleRate} Hz " +
                $"({duration:0}s requested, {lyricIds.Length} lyric tokens) in {Environment.TickCount64 - start}ms.");
            return new[] { AudioOutputEncoder.EncodeMp3(left, right, sampleRate, cancel) };
        }
        finally
        {
            textEmbeds.Dispose();
        }
    }

    /// <summary>Builds the UMT5 style prompt: the user's tags plus BPM / key scale / time signature as
    /// free text — ACE-Step v1 has no dedicated encoders for those, it reads them from the prompt.</summary>
    private static string ComposeStyleText(T2IParamInput input)
    {
        string style = input.Get(T2IParamTypes.Text2AudioStyle, "pop");
        long bpm = input.Get(T2IParamTypes.Text2AudioBPM, 120);
        string keyScale = input.Get(T2IParamTypes.Text2AudioKeyScale, "C");
        string timeSig = input.Get(T2IParamTypes.Text2AudioTimeSignature, "4");
        return $"{style}, {bpm} bpm, key of {keyScale}, {timeSig}/4 time";
    }

    /// <summary>Downloads the official lyric tokenizer JSON once and exports the <c>vocab.json</c> +
    /// <c>merges.txt</c> pair <c>AceStepLyricTokenizer</c> consumes, under <c>Clip/AceStep/</c> (non-safetensors
    /// files there don't appear in Swarm's model lists).</summary>
    private static (string VocabPath, string MergesPath) EnsureLyricTokenizerFiles(Action<string> log)
    {
        string dir = Path.Combine(Program.T2IModelSets["Clip"].DownloadFolderPath, "AceStep");
        string vocabPath = Path.Combine(dir, "lyric_vocab.json");
        string mergesPath = Path.Combine(dir, "lyric_merges.txt");
        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            return (vocabPath, mergesPath);
        }
        Directory.CreateDirectory(dir);
        string rawPath = Path.Combine(dir, "lyric_tokenizer_raw.json");
        log("Downloading ACE-Step lyric tokenizer vocab (one-time)...");
        Utilities.DownloadFile(LyricVocabUrl, rawPath, null).Wait();
        JObject tokenizer = JObject.Parse(File.ReadAllText(rawPath));
        JObject vocab = (JObject)tokenizer["model"]!["vocab"]!;
        File.WriteAllText(vocabPath, vocab.ToString());
        IEnumerable<string> merges = ((JArray)tokenizer["model"]!["merges"]!)
            .Select(m => m is JArray pair ? $"{pair[0]} {pair[1]}" : m.ToString());
        File.WriteAllLines(mergesPath, merges);
        File.Delete(rawPath);
        log("  Lyric tokenizer vocab + merges exported.");
        return (vocabPath, mergesPath);
    }
}

public sealed class AceStepCacheEntry : IDisposable
{
    public required string ModelName { get; init; }
    public required string CompatClass { get; init; }
    public required AceStepPipeline Pipeline { get; init; }
    public required AceStepConfig Config { get; init; }
    public required AceStepDit Dit { get; init; }
    public required T5TextEncoder Umt5 { get; init; }
    public required T5Tokenizer TextTokenizer { get; init; }
    public required AceStepLyricTokenizer LyricTokenizer { get; init; }
    public required SafeTensorsLoader CheckpointLoader { get; init; }
    public required SafeTensorsLoader DcaeLoader { get; init; }
    public required SafeTensorsLoader VocoderLoader { get; init; }
    public required SafeTensorsLoader Umt5Loader { get; init; }

    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;
    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (Pipeline as IDisposable)?.Dispose();
        TextTokenizer?.Dispose();
        (Dit as IDisposable)?.Dispose();
        Umt5?.Dispose();
        CheckpointLoader?.Dispose();
        DcaeLoader?.Dispose();
        VocoderLoader?.Dispose();
        Umt5Loader?.Dispose();
    }
}
