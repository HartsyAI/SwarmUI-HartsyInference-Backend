using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Translates SwarmUI prompt syntax into the inputs the HartsyInference engine consumes. This is the
/// extension-side counterpart to the engine's prompt-engineering primitives — the engine owns the
/// universal SD <c>( )</c>/<c>[ ]</c> grammar and structured conditioning types; this class owns the
/// Swarm-specific <c>&lt;tag&gt;</c> translation.
///
/// <para><b>Phase 0 (current):</b> only <see cref="BaseText"/> — strip the structural region/stage tags
/// (<c>&lt;region:&gt;</c>, <c>&lt;object:&gt;</c>, <c>&lt;segment:&gt;</c>, <c>&lt;base&gt;</c>,
/// <c>&lt;refiner&gt;</c>, …) from the text that reaches the tokenizer, so their literal characters
/// don't leak into the base conditioning. Segment/region handling is done separately (SegmentRefiner
/// reads the RAW prompt), so this only governs what the base stage encodes.</para>
///
/// <para>Later phases grow this into the full resolver (weighting → engine <c>EncodeWeighted</c>,
/// <c>[a|b]</c> → <c>ConditioningSchedule</c>, <c>&lt;region:&gt;</c> → <c>RegionalPlan</c>,
/// <c>&lt;embed:&gt;</c> → textual inversion) as the engine publishes those public APIs.</para>
/// </summary>
public static class PromptConditioningResolver
{
    /// <summary>Returns the base-stage text for encoding: the prompt's global text with SwarmUI
    /// structural tags removed (via <see cref="PromptRegion.GlobalPrompt"/>). A tagless prompt is
    /// returned unchanged. Encoder-level syntax the engine will later honor — weighting <c>(x:1.3)</c>,
    /// alternation <c>[a|b]</c>, <c>&lt;break&gt;</c>, <c>\0swarmembed:</c> — is intentionally left
    /// intact for the engine to process; only the <c>&lt;tag&gt;</c> structure is stripped here.</summary>
    public static string BaseText(string raw)
    {
        if (string.IsNullOrEmpty(raw))
        {
            return raw ?? "";
        }
        // No structural tags → avoid the parse entirely (the 99% case, and keeps text byte-identical).
        if (!raw.Contains('<'))
        {
            return raw;
        }
        return new PromptRegion(raw).GlobalPrompt ?? "";
    }
}
