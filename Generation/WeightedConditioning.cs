using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Tokenizers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Builds the engine's <see cref="ConditioningSchedule"/> for ComfyUI-style prompt weighting and
/// <c>&lt;break&gt;</c> chunking, for CLIP-based pipelines whose denoise loop consumes a batched
/// <c>[2, seqLen, hidden]</c> (negative, positive) conditioning tensor (SD 1.5 today; SDXL/SD3 will
/// add dual-CLIP variants).
///
/// <para>The engine's <c>ConditioningSchedule.Variants</c> are pre-encoded tensors, so the consumer
/// assembles them: tokenize+weight each prompt (<see cref="WeightedPromptTokenizer"/>), pad positive
/// and negative to equal <c>&lt;break&gt;</c>-chunk counts, encode each via
/// <see cref="ClipTextEncoder.EncodeWeighted"/>, and stack into the <c>[2, …]</c> batch the loop slices
/// (index 0 = uncond/negative, index 1 = cond/positive).</para>
///
/// <para>Returns <c>null</c> when the prompt has no weighting/break syntax, so callers keep the
/// byte-identical plain-encode path for ordinary prompts (zero behavior change, zero cost).</para>
/// </summary>
public static class WeightedConditioning
{
    /// <summary>Cheap pre-check: does either prompt contain weighting <c>( )</c>, alternation/scheduling
    /// brackets <c>[ ]</c>, or a <c>&lt;break&gt;</c>? Structural tags are already stripped by
    /// <see cref="PromptConditioningResolver.BaseText"/> before this is called, so a remaining <c>&lt;</c>
    /// is effectively only <c>&lt;break&gt;</c>.</summary>
    public static bool HasWeightingSyntax(params string[] prompts)
    {
        foreach (string p in prompts)
        {
            if (string.IsNullOrEmpty(p)) continue;
            if (p.IndexOf('(') >= 0 || p.IndexOf('[') >= 0
                || p.Contains("<break>", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Single-CLIP (SD 1.5) weighted conditioning. Returns a one-variant schedule whose tensor
    /// is <c>[2, 77*chunks, hidden]</c> (negative, positive), or null when there's no weighting syntax.
    /// <paramref name="layersFromEnd"/> is the CLIP-skip value (1 = last layer).</summary>
    public static ConditioningSchedule BuildSingleClip(
        IBackend backend, ClipTextEncoder encoder, ClipTokenizer tokenizer,
        string positive, string negative, int layersFromEnd)
    {
        if (!HasWeightingSyntax(positive, negative))
        {
            return null;
        }
        (IReadOnlyList<int[]> posIds, IReadOnlyList<float[]> posW) = WeightedPromptTokenizer.Tokenize(tokenizer, positive ?? "");
        (IReadOnlyList<int[]> negIds, IReadOnlyList<float[]> negW) = WeightedPromptTokenizer.Tokenize(tokenizer, negative ?? "");
        EqualizeChunkCount(tokenizer, ref posIds, ref posW, ref negIds, ref negW);

        Tensor posCond = encoder.EncodeWeighted(backend, posIds, posW, layersFromEnd);   // [1, S, H]
        Tensor negCond = encoder.EncodeWeighted(backend, negIds, negW, layersFromEnd);   // [1, S, H]
        Tensor batched;
        try
        {
            batched = StackBatch2(negCond, posCond);                                     // [2, S, H]
        }
        finally
        {
            posCond.Dispose();
            negCond.Dispose();
        }
        return new ConditioningSchedule
        {
            Variants = [batched],
            IndexForStep = static (_, _) => 0,
        };
    }

    /// <summary>Pads the shorter of (positive, negative) chunk lists with empty SOT..EOT chunks so both
    /// have the same number of 77-token chunks — required before stacking into one <c>[2, …]</c> tensor.</summary>
    private static void EqualizeChunkCount(
        ClipTokenizer tokenizer,
        ref IReadOnlyList<int[]> posIds, ref IReadOnlyList<float[]> posW,
        ref IReadOnlyList<int[]> negIds, ref IReadOnlyList<float[]> negW)
    {
        int target = Math.Max(posIds.Count, negIds.Count);
        if (posIds.Count == negIds.Count) return;

        // An empty prompt tokenizes to exactly one bare SOT..EOT pad chunk — the neutral filler.
        (IReadOnlyList<int[]> emptyIds, IReadOnlyList<float[]> emptyW) = WeightedPromptTokenizer.Tokenize(tokenizer, "");
        int[] padIds = emptyIds[0];
        float[] padW = emptyW[0];

        posIds = Pad(posIds, padIds, target);
        posW = Pad(posW, padW, target);
        negIds = Pad(negIds, padIds, target);
        negW = Pad(negW, padW, target);
    }

    private static IReadOnlyList<T> Pad<T>(IReadOnlyList<T> list, T pad, int target)
    {
        if (list.Count >= target) return list;
        List<T> result = new(list);
        while (result.Count < target) result.Add(pad);
        return result;
    }

    /// <summary>Stacks two <c>[1, S, H]</c> F32 tensors into a single <c>[2, S, H]</c> (row 0 = first,
    /// row 1 = second). Matches the (uncond, cond) batch the CFG loop slices.</summary>
    private static unsafe Tensor StackBatch2(Tensor first, Tensor second)
    {
        if (first.Shape.Rank != 3 || second.Shape.Rank != 3)
            throw new ArgumentException("StackBatch2 expects rank-3 [1,S,H] tensors.");
        if (!first.Shape.Equals(second.Shape))
            throw new ArgumentException($"StackBatch2 shape mismatch: {first.Shape} vs {second.Shape}.");
        if (first.DType != DType.F32 || second.DType != DType.F32)
            throw new ArgumentException("StackBatch2 expects F32 tensors.");

        long s = first.Shape[1], h = first.Shape[2];
        Tensor result = new(new TensorShape(2, s, h), DType.F32);
        long rowBytes = s * h * sizeof(float);
        byte* dst = (byte*)result.DataPointer;
        Buffer.MemoryCopy((void*)first.DataPointer, dst, rowBytes, rowBytes);
        Buffer.MemoryCopy((void*)second.DataPointer, dst + rowBytes, rowBytes, rowBytes);
        return result;
    }
}
