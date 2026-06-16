using System.IO;
using System.Reflection;
using SwarmUI.Accounts;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.LLMs;
using SwarmUI.Utils;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Ideogram 4 "magic prompt": rewrites a plain user prompt into the structured JSON caption Ideogram 4
/// was trained on, using a SwarmUI LLM backend (LlamaSharp / Anthropic / remote — whichever is running).
///
/// <para>Faithful port of upstream <c>ideogram-oss/ideogram4</c>'s <c>magic_prompt.py</c> +
/// <c>ClaudeOpusMagicPromptV1</c> flow (Ideogram's own hosted API is not reproduced — we use the user's
/// configured LLM instead):</para>
/// <list type="number">
///   <item>Parse the embedded <c>v1.txt</c> system-prompt file into its <c>[SYSTEM]</c> / <c>[USER]</c> blocks.</item>
///   <item>Build a 2-message chat: system = <c>[SYSTEM]</c>; user = <c>[USER]</c> template with
///         <c>{{aspect_ratio}}</c> (reduced W:H) and <c>{{original_prompt}}</c> substituted.</item>
///   <item>Run one non-streaming completion (temperature 1.0, like upstream) and strip any ```json fence.</item>
///   <item>Round-trip through <see cref="Ideogram4Dialect"/> (Parse → Serialize) to validate and emit the
///         exact canonical key order the model expects.</item>
/// </list>
/// The returned string is the canonical JSON caption, ready to tokenize and feed to the pipeline.
/// </summary>
public static class Ideogram4MagicPrompt
{
    /// <summary>Upstream uses temperature 1.0 for the Claude-based expanders (<c>magic_prompt.py</c>).</summary>
    private const double ExpandTemperature = 1.0;

    /// <summary>Upstream <c>openrouter_chat</c> default; the structured caption can be long.</summary>
    private const int ExpandMaxTokens = 16384;

    private static readonly object _sectionsLock = new();
    private static (string System, string UserTemplate)? _sections;

    /// <summary>True if <paramref name="prompt"/> is already an Ideogram 4 structured JSON caption (e.g. authored
    /// by a prompt-builder UI like SwarmUI-IdeogramPromptBuilder, or pasted by hand). Such a prompt must NOT be
    /// re-expanded through the LLM — that would double-process it and strip its user-drawn bboxes. Cheap shape
    /// check: a JSON object carrying one of the schema's top-level keys.</summary>
    public static bool LooksLikeStructuredCaption(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return false;
        }
        string trimmed = prompt.TrimStart();
        return trimmed.StartsWith('{')
            && (prompt.Contains("compositional_deconstruction") || prompt.Contains("high_level_description"));
    }

    /// <summary>Expands <paramref name="plainPrompt"/> into the canonical Ideogram 4 JSON caption via a
    /// running LLM backend. Throws <see cref="SwarmReadableErrorException"/> if no LLM backend is available.
    /// If the LLM returns something that won't parse as an Ideogram caption, logs a warning and returns the
    /// original plain prompt (Ideogram 4 also accepts plain text), so a flaky LLM degrades rather than fails.</summary>
    public static string Expand(
        string plainPrompt,
        int width,
        int height,
        string modelOverride,
        Session session,
        Action<string> log)
    {
        (string systemPrompt, string userTemplate) = LoadSections();
        string aspectRatio = AspectRatioFromSize(width, height);
        string userMessage = BuildUserMessage(userTemplate, aspectRatio, plainPrompt);

        AbstractLLMBackend backend = PickBackend(modelOverride);
        log($"Magic prompt: expanding via LLM backend '{backend.GetType().Name}'" +
            (string.IsNullOrWhiteSpace(modelOverride) ? "" : $" (model '{modelOverride}')") + "...");

        LLMParamInput llmInput = new()
        {
            SystemPrompt = systemPrompt,
            UserMessage = userMessage,
            Model = string.IsNullOrWhiteSpace(modelOverride) ? null : modelOverride,
            Temperature = ExpandTemperature,
            MaxTokens = ExpandMaxTokens,
            Stream = false,
            RequestSession = session,
        };

        string raw;
        try
        {
            raw = backend.Generate(llmInput).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            throw new SwarmReadableErrorException(
                $"Ideogram 4 magic prompt failed to reach the LLM backend: {ex.Message}. " +
                "Disable the magic prompt or fix the LLM backend.");
        }

        string json = StripCodeFences(raw);
        try
        {
            string caption = StripAspectRatioAndBboxes(json);
            log("Magic prompt: expanded to structured caption.");
            return caption;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[HartsyInference][Ideogram4] Magic prompt output was not valid JSON " +
                $"({ex.Message}); falling back to the plain prompt.");
            return plainPrompt;
        }
    }

    /// <summary>Mirrors upstream <c>strip_aspect_ratio_and_bboxes(strip_bboxes=True)</c> — the post-step the
    /// Claude-based (i.e. LLM, not hosted-API) expanders run: drop the top-level <c>aspect_ratio</c> key and
    /// every element's <c>bbox</c>, then re-emit minified JSON (no spaces, literal Unicode). We strip bboxes
    /// because the engine's regional-attention coupling is not yet implemented, so per-element boxes can't be
    /// honored — feeding them would only dilute the caption.</summary>
    private static string StripAspectRatioAndBboxes(string caption)
    {
        JObject data = JObject.Parse(caption);
        data.Remove("aspect_ratio");
        if (data["compositional_deconstruction"] is JObject comp && comp["elements"] is JArray elements)
        {
            foreach (JToken element in elements)
            {
                if (element is JObject elementObj)
                {
                    elementObj.Remove("bbox");
                }
            }
        }
        return data.ToString(Formatting.None);
    }

    /// <summary>Picks an LLM backend the same way <c>LLMDispatcher</c> does: the only running one, else (when a
    /// specific model is named) the backend that owns that model, else the first running backend.</summary>
    private static AbstractLLMBackend PickBackend(string modelOverride)
    {
        List<AbstractLLMBackend> backends = Program.Backends.RunningBackendsOfType<AbstractLLMBackend>().ToList();
        if (backends.Count == 0)
        {
            throw new SwarmReadableErrorException(
                "Ideogram 4 magic prompt is enabled, but no LLM backend is running. " +
                "Add one in Server > Backends, or turn off the 'Ideogram 4 Magic Prompt' parameter.");
        }
        if (backends.Count == 1 || string.IsNullOrWhiteSpace(modelOverride))
        {
            return backends[0];
        }
        foreach (AbstractLLMBackend backend in backends)
        {
            try
            {
                if (backend.ListModels().GetAwaiter().GetResult().Any(m => m.Id == modelOverride || m.Name == modelOverride))
                {
                    return backend;
                }
            }
            catch (Exception)
            {
                // Skip backends that fail to list models.
            }
        }
        return backends[0];
    }

    /// <summary>Reduces pixel <paramref name="width"/>x<paramref name="height"/> to a "W:H" string. Mirrors
    /// upstream <c>aspect_ratio_from_size</c> (used to fill the <c>{{aspect_ratio}}</c> placeholder).</summary>
    private static string AspectRatioFromSize(int width, int height)
    {
        int divisor = Gcd(Math.Abs(width), Math.Abs(height));
        if (divisor == 0) divisor = 1;
        return $"{width / divisor}:{height / divisor}";
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }
        return a;
    }

    /// <summary>Builds the user message from the <c>[USER]</c> template. Mirrors upstream <c>build_messages</c>:
    /// substitute <c>{{aspect_ratio}}</c>, then <c>{{original_prompt}}</c> if present, else append the prompt
    /// after a blank line.</summary>
    private static string BuildUserMessage(string userTemplate, string aspectRatio, string plainPrompt)
    {
        string template = string.IsNullOrEmpty(userTemplate)
            ? "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height)."
            : userTemplate;
        string user = template.Replace("{{aspect_ratio}}", aspectRatio);
        return user.Contains("{{original_prompt}}")
            ? user.Replace("{{original_prompt}}", plainPrompt)
            : $"{user}\n\n{plainPrompt}";
    }

    /// <summary>Drops a surrounding ```json … ``` fence a model may add. Mirrors upstream <c>_strip_code_fences</c>.</summary>
    private static string StripCodeFences(string text)
    {
        text = text.Trim();
        if (!text.StartsWith("```"))
        {
            return text;
        }
        string[] lines = text.Split('\n');
        int start = 0, end = lines.Length;
        if (start < end && lines[start].TrimStart().StartsWith("```")) start++;
        if (end > start && lines[end - 1].Trim() == "```") end--;
        return string.Join('\n', lines[start..end]).Trim();
    }

    /// <summary>Loads and caches the embedded <c>v1.txt</c> system prompt, parsed into its <c>[SYSTEM]</c> and
    /// <c>[USER]</c> blocks. Mirrors upstream <c>_load_sections</c> ([NAME] markers alone on a line).</summary>
    private static (string System, string UserTemplate) LoadSections()
    {
        if (_sections is { } cached)
        {
            return cached;
        }
        lock (_sectionsLock)
        {
            if (_sections is { } cached2)
            {
                return cached2;
            }
            string raw = ReadEmbeddedSystemPrompt();
            Dictionary<string, string> sections = ParseSections(raw);
            if (!sections.TryGetValue("system", out string system) || string.IsNullOrWhiteSpace(system))
            {
                throw new InvalidOperationException("Embedded Ideogram 4 magic-prompt file has no [SYSTEM] section.");
            }
            sections.TryGetValue("user", out string userTemplate);
            _sections = (system, userTemplate ?? "");
            return _sections.Value;
        }
    }

    private static Dictionary<string, string> ParseSections(string raw)
    {
        Dictionary<string, string> sections = new();
        string current = null;
        List<string> lines = new();
        foreach (string line in raw.Split('\n'))
        {
            string stripped = line.Trim();
            if (stripped.StartsWith("[") && stripped.EndsWith("]") && !stripped.Contains(' '))
            {
                if (current is not null)
                {
                    sections[current] = string.Join('\n', lines).Trim();
                }
                current = stripped[1..^1].Trim().ToLowerInvariant();
                lines = new();
            }
            else
            {
                lines.Add(line);
            }
        }
        if (current is not null)
        {
            sections[current] = string.Join('\n', lines).Trim();
        }
        return sections;
    }

    private static string ReadEmbeddedSystemPrompt()
    {
        Assembly asm = typeof(Ideogram4MagicPrompt).Assembly;
        string resourceName = Array.Find(asm.GetManifestResourceNames(),
            n => n.EndsWith("ideogram4_magic_prompt_v1.txt", StringComparison.Ordinal));
        if (resourceName is null)
        {
            throw new InvalidOperationException(
                "Embedded resource 'ideogram4_magic_prompt_v1.txt' not found in the extension assembly.");
        }
        using Stream stream = asm.GetManifestResourceStream(resourceName);
        using StreamReader reader = new(stream);
        return reader.ReadToEnd();
    }
}
