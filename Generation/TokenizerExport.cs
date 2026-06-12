using System.IO;
using Newtonsoft.Json.Linq;
using SwarmUI.Utils;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// One-time download + export of a HuggingFace <c>tokenizer.json</c> into the
/// <c>vocab.json</c> + <c>merges.txt</c> pair SharpInference's file-based BPE tokenizers consume
/// (<c>AceStepLyricTokenizer</c>, <c>GptOssTokenizer</c>). The exported files are cached on disk —
/// subsequent loads skip the network entirely. Non-safetensors files under the Clip folder don't
/// appear in Swarm's model lists, so subfolders there are a safe home.
/// </summary>
public static class TokenizerExport
{
    /// <summary>Ensures <c>{dir}/{baseName}_vocab.json</c> and <c>{dir}/{baseName}_merges.txt</c> exist,
    /// downloading and splitting <paramref name="tokenizerJsonUrl"/> on first use.</summary>
    public static (string VocabPath, string MergesPath) EnsureVocabMerges(
        string tokenizerJsonUrl, string dir, string baseName, Action<string> log)
    {
        string vocabPath = Path.Combine(dir, $"{baseName}_vocab.json");
        string mergesPath = Path.Combine(dir, $"{baseName}_merges.txt");
        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            return (vocabPath, mergesPath);
        }
        Directory.CreateDirectory(dir);
        string rawPath = Path.Combine(dir, $"{baseName}_tokenizer_raw.json");
        log($"Downloading {baseName} tokenizer vocab (one-time)...");
        Utilities.DownloadFile(tokenizerJsonUrl, rawPath, null).Wait();
        JObject tokenizer = JObject.Parse(File.ReadAllText(rawPath));
        JObject vocab = (JObject)tokenizer["model"]!["vocab"]!;
        File.WriteAllText(vocabPath, vocab.ToString());
        IEnumerable<string> merges = ((JArray)tokenizer["model"]!["merges"]!)
            .Select(m => m is JArray pair ? $"{pair[0]} {pair[1]}" : m.ToString());
        File.WriteAllLines(mergesPath, merges);
        File.Delete(rawPath);
        log($"  {baseName} tokenizer vocab + merges exported.");
        return (vocabPath, mergesPath);
    }
}
