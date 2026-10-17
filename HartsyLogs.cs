using System.Collections.Generic;
using SwarmUI.Utils;

namespace Hartsy.Extensions.HartsyInferenceBackend;

/// <summary>Writes this extension's log lines with the [HartsyInference] prefix, skipping hidden levels before formatting.</summary>
public static class HartsyLogs
{
    /// <summary>Prefix on every line this extension logs.</summary>
    private const string Prefix = "[HartsyInference]";

    /// <summary>Keys already seen by <see cref="WarnOnce"/>.</summary>
    private static readonly HashSet<string> _warnedKeys = [];

    /// <summary>Logs an initialization milestone.</summary>
    public static void Init(string message, string tag = null)
    {
        if (Logs.MinimumLevel <= Logs.LogLevel.Init)
        {
            Logs.Init(Format(message, tag));
        }
    }

    /// <summary>Logs an operator-visible event.</summary>
    public static void Info(string message, string tag = null)
    {
        if (Logs.MinimumLevel <= Logs.LogLevel.Info)
        {
            Logs.Info(Format(message, tag));
        }
    }

    /// <summary>Logs a degraded or skipped operation that generation survives.</summary>
    public static void Warning(string message, string tag = null)
    {
        if (Logs.MinimumLevel <= Logs.LogLevel.Warning)
        {
            Logs.Warning(Format(message, tag));
        }
    }

    /// <summary>Logs a real failure. Also sets the CI exit code, so use it only for failures.</summary>
    public static void Error(string message, string tag = null)
    {
        if (Logs.MinimumLevel <= Logs.LogLevel.Error)
        {
            Logs.Error(Format(message, tag));
        }
    }

    /// <summary>Logs diagnostic detail for one problem.</summary>
    public static void Debug(string message, string tag = null)
    {
        if (Logs.MinimumLevel <= Logs.LogLevel.Debug)
        {
            Logs.Debug(Format(message, tag));
        }
    }

    /// <summary>Logs per-step or per-chunk detail.</summary>
    public static void Verbose(string message, string tag = null)
    {
        if (Logs.MinimumLevel <= Logs.LogLevel.Verbose)
        {
            Logs.Verbose(Format(message, tag));
        }
    }

    /// <summary>Warns the first time <paramref name="key"/> is seen in this process, then logs at Debug.</summary>
    public static void WarnOnce(string key, string message, string tag = null)
    {
        bool first;
        lock (_warnedKeys)
        {
            first = _warnedKeys.Add(key);
        }
        if (first)
        {
            Warning(message, tag);
        }
        else
        {
            Debug(message, tag);
        }
    }

    /// <summary>Joins the prefix, optional tag, and message. A message starting with "[" is a subsystem tag and joins without a space.</summary>
    private static string Format(string message, string tag)
    {
        string head = tag is null ? Prefix : $"{Prefix}[{tag}]";
        string text = message ?? "";
        return text.StartsWith('[') ? head + text : $"{head} {text}";
    }
}
