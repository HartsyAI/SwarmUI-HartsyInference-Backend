using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tokenizers;
using SwarmUI.Core;
using SwarmUI.Text2Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves SwarmUI textual-inversion <c>&lt;embed:name&gt;</c> syntax for CLIP pipelines. SwarmUI core rewrites
/// <c>&lt;embed:name&gt;</c> into the control marker <c>\0swarmembed:NAME\0end</c> in the prompt text; this class
/// finds those markers, loads each embedding's learned vectors (<see cref="TextualInversion.Load"/>, which handles
/// A1111 / diffusers / SDXL-dual layouts) and builds a CLIP token sequence where each embedding's N vectors occupy
/// N sequential placeholder token ids (past the tokenizer vocab). The caller encodes those tokens with the
/// per-hidden-size <see cref="Plan.InlineMap"/> passed to <c>ClipTextEncoder.Encode(..., inlineEmbeddings)</c>, so the
/// engine substitutes the learned vectors at the placeholder positions.
///
/// <para>Returns null when the prompt carries no embed markers — callers keep their plain token path unchanged.</para>
/// </summary>
public static class EmbeddingResolver
{
    private static readonly Regex MarkerRx = new("\u0000swarmembed:(.*?)\u0000end", RegexOptions.Compiled);

    /// <summary>The token layout + loaded embeddings for one prompt. Owns the loaded embedding tensors; dispose after encoding.</summary>
    public sealed class Plan : IDisposable
    {
        /// <summary>CLIP token ids <c>[MaxLength]</c> — SOT, interleaved text/placeholder ids, EOT, EOT-pad.</summary>
        public int[] TokenIds = Array.Empty<int>();

        /// <summary>Position of the EOT token (for pooled-output extraction).</summary>
        public int EosPosition;

        /// <summary>Per embed occurrence: its first placeholder id + the loaded <c>[N, hidden]</c> tensor keyed by hidden size.</summary>
        internal readonly List<(int StartId, Dictionary<int, Tensor> ByHidden)> Occurrences = new();

        private readonly List<Tensor> _owned = new();
        internal void AddOwned(Tensor t) => _owned.Add(t);

        /// <summary>Builds the <c>{placeholderId → [hidden] vector}</c> map for one encoder's hidden size. Empty when no
        /// occurrence has a tensor at that size (e.g. an SD1.5-only embed under an SDXL CLIP-G request).</summary>
        public Dictionary<int, Tensor> InlineMap(int hiddenSize)
        {
            Dictionary<int, Tensor> merged = new();
            foreach ((int startId, Dictionary<int, Tensor> byHidden) in Occurrences)
            {
                if (byHidden.TryGetValue(hiddenSize, out Tensor emb))
                {
                    (Dictionary<int, Tensor> map, _) = TextualInversion.BuildInlineMap(emb, startId);
                    foreach (KeyValuePair<int, Tensor> kv in map)
                    {
                        merged[kv.Key] = kv.Value;
                    }
                }
            }
            return merged;
        }

        public void Dispose()
        {
            foreach (Tensor t in _owned)
            {
                t.Dispose();
            }
            _owned.Clear();
        }
    }

    /// <summary>Builds the embed <see cref="Plan"/> for <paramref name="prompt"/>, or null if it has no embed markers.
    /// <paramref name="hiddenSizes"/> are the encoder hidden sizes to load per embed ([768] for SD1.5, [768, 1280] for
    /// SDXL). Unresolvable/incompatible embeds are skipped (SwarmUI core already warned about missing files).</summary>
    public static Plan Resolve(string prompt, ClipTokenizer tokenizer, int[] hiddenSizes)
    {
        if (string.IsNullOrEmpty(prompt) || prompt.IndexOf('\u0000') < 0)
        {
            return null;
        }
        MatchCollection matches = MarkerRx.Matches(prompt);
        if (matches.Count == 0)
        {
            return null;
        }

        Plan plan = new Plan();
        List<int> tokens = new(ClipTokenizer.MaxLength) { ClipTokenizer.StartOfTextId };
        int nextPlaceholder = ClipTokenizer.VocabSize; // 49408+, past the real vocab
        int cursor = 0;
        foreach (Match m in matches)
        {
            AppendRaw(tokens, tokenizer, prompt.Substring(cursor, m.Index - cursor));
            cursor = m.Index + m.Length;

            string path = ResolveEmbedPath(m.Groups[1].Value);
            if (path is null)
            {
                continue;
            }
            Dictionary<int, Tensor> byHidden = new();
            int n = 0;
            foreach (int h in hiddenSizes)
            {
                try
                {
                    Tensor emb = TextualInversion.Load(path, h); // [N, h]
                    plan.AddOwned(emb);
                    byHidden[h] = emb;
                    n = (int)emb.Shape[0];
                }
                catch
                {
                    // This hidden size isn't present in the file (e.g. an SD1.5 embed has no CLIP-G tensor) — skip it.
                }
            }
            // SAFETY: only inject if the embed loaded for EVERY requested hidden size. Otherwise a placeholder id
            // would be missing from one encoder's inline map, and EmbedTokens would fall through to the normal
            // token-embedding lookup at id >= VocabSize → out-of-bounds read (native crash). Skip partial embeds.
            if (n <= 0 || byHidden.Count != hiddenSizes.Length)
            {
                continue;
            }
            for (int r = 0; r < n; r++)
            {
                tokens.Add(nextPlaceholder + r);
            }
            plan.Occurrences.Add((nextPlaceholder, byHidden));
            nextPlaceholder += n;
        }
        AppendRaw(tokens, tokenizer, prompt.Substring(cursor));

        // Reserve the final slot for EOT, then pad with EOT (CLIP pads with EOT, not zero).
        int limit = ClipTokenizer.MaxLength - 1;
        if (tokens.Count > limit)
        {
            tokens.RemoveRange(limit, tokens.Count - limit);
        }
        plan.EosPosition = tokens.Count;
        tokens.Add(ClipTokenizer.EndOfTextId);
        while (tokens.Count < ClipTokenizer.MaxLength)
        {
            tokens.Add(ClipTokenizer.EndOfTextId);
        }
        plan.TokenIds = tokens.ToArray();

        if (plan.Occurrences.Count == 0)
        {
            plan.Dispose();
            return null;
        }
        return plan;
    }

    /// <summary>Removes the <c>\0swarmembed:…\0end</c> markers from a prompt (used for the plain token path that
    /// feeds the pooled vector — the embedding's cross-attention effect comes from the schedule, not the pooled).</summary>
    public static string StripMarkers(string prompt)
        => string.IsNullOrEmpty(prompt) ? prompt : MarkerRx.Replace(prompt, "");

    /// <summary>Builds an SDXL dual-CLIP <c>[2, S, 2048]</c> conditioning schedule (uncond, cond) from an embed
    /// <paramref name="plan"/>: penultimate CLIP-L (768) + CLIP-G (1280), each encoded with its inline-embedding map so
    /// the learned <c>&lt;embed&gt;</c> vectors occupy the placeholder positions. The negative is encoded plainly (embeds
    /// in a negative are unusual). Matches <c>SdxlPipeline</c>'s plain textEmbeddings shape → passed as a
    /// <c>conditioningSchedule</c> override; the pooled vector stays on the pipeline's own plain path.</summary>
    public static ConditioningSchedule BuildDualClipSchedule(
        IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, ClipTokenizer tokenizer,
        Plan plan, string negative, int layersFromEnd)
    {
        int[] negTokens = tokenizer.Encode(negative ?? "");
        int[][] batch = new[] { negTokens, plan.TokenIds };            // [uncond, cond]
        int[] eos = new[] { ClipTokenizer.FindEosPosition(negTokens), plan.EosPosition };

        Dictionary<int, Tensor> mapL = plan.InlineMap(768);
        Dictionary<int, Tensor> mapG = plan.InlineMap(1280);

        (Tensor lHidden, Tensor lPooled) = clipL.EncodePenultimate(backend, batch, eos, layersFromEnd, mapL); // [2,S,768]
        lPooled?.Dispose();
        (Tensor gHidden, Tensor gPooled) = clipG.EncodePenultimate(backend, batch, eos, layersFromEnd, mapG); // [2,S,1280]
        gPooled?.Dispose();

        Tensor concat = CfgHelper.ConcatLastDim(lHidden, gHidden);      // [2, S, 2048]
        lHidden.Dispose();
        gHidden.Dispose();
        return new ConditioningSchedule { Variants = new[] { concat }, IndexForStep = static (_, _) => 0 };
    }

    private static void AppendRaw(List<int> tokens, ClipTokenizer tokenizer, string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return;
        }
        foreach (int id in tokenizer.EncodeRaw(text))
        {
            tokens.Add(id);
        }
    }

    /// <summary>Resolves an embedding name (as SwarmUI matched it) to a file path via the Embedding model set.</summary>
    private static string ResolveEmbedPath(string name)
    {
        if (!Program.T2IModelSets.TryGetValue("Embedding", out T2IModelHandler set) || set is null)
        {
            return null;
        }
        T2IModel model = set.GetModel(name) ?? set.GetModel(name + ".safetensors");
        string path = model?.RawFilePath;
        return path is not null && File.Exists(path) ? path : null;
    }
}
