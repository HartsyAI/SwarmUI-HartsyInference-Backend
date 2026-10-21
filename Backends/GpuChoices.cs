using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Vulkan;
using SwarmUI.Core;

namespace Hartsy.Extensions.HartsyInferenceBackend.Backends;

/// <summary>The GPUs the backend card lists, named by the engine, so each GPU field shows real devices instead of a bare number.</summary>
/// <remarks>Values are the ordinals GPU_ID has always saved, passed to the engine as <see cref="HartsyInferenceBackend"/> passes them:
/// under CUDA a value is the CUDA ordinal, and under Vulkan 0 lets the engine pick while any other value is the loader's raw index.
/// Each list is built once per process, since a probe opens a real device.</remarks>
public static class GpuChoices
{
    /// <summary>One dropdown entry: the value saved to Backends.fds, and the label shown for it.</summary>
    public readonly record struct Choice(string Value, string Name);

    /// <summary>The kind Auto runs on here, from the same checks as the Compute Backend list: the CUDA probe, then whether Vulkan sees a GPU.</summary>
    /// <remarks>Not <see cref="BackendFactory.ResolveProbed"/>: its Vulkan probe builds a backend with the default kernel folder before any
    /// backend applies KernelDirectory, and the engine caches that answer for Init.</remarks>
    private static readonly Lazy<string> _autoKind = new(() => Guard(() => BackendFactory.ProbeCuda() ? "cuda" : VulkanContext.IsAvailable() ? "vulkan" : "cpu", "Auto probe") ?? "cpu");

    /// <summary>CUDA GPUs that passed the engine's test computation.</summary>
    private static readonly Lazy<IReadOnlyList<Choice>> _cuda = new(BuildCuda);

    /// <summary>Vulkan GPUs, led by 0 for "let the engine pick".</summary>
    private static readonly Lazy<IReadOnlyList<Choice>> _vulkan = new(BuildVulkan);

    /// <summary>The kind Auto runs on here: "cuda", "vulkan" or "cpu".</summary>
    public static string AutoKind => _autoKind.Value;

    /// <summary>The GPU_ID choices for one Compute Backend value, with the kind they belong to.</summary>
    /// <remarks>Never empty: every list offers "0", the value GPU_ID saves by default.</remarks>
    public static (string Kind, IReadOnlyList<Choice> Choices) ForCompute(string computeBackend)
    {
        string kind = BackendFactory.Kind(computeBackend);
        bool auto = kind is not ("cuda" or "vulkan" or "cpu");
        if (auto)
        {
            kind = AutoKind;
        }
        IReadOnlyList<Choice> choices = kind switch
        {
            "cuda" => _cuda.Value,
            "vulkan" => _vulkan.Value,
            _ => Array.Empty<Choice>(),
        };
        if (choices.Count == 0)
        {
            string name = kind != "cpu" ? "Default GPU" : auto ? "No usable GPU found" : "Not used on CPU";
            choices = [new Choice("0", name)];
        }
        return (kind, choices);
    }

    /// <summary>GPU_ID's dropdown: the GPUs Auto runs on here. The card script relists them when Compute Backend changes.</summary>
    public class MainGpuOptions : SettingsOptionsAttribute.AbstractImpl
    {
        public override string[] GetOptions => [.. ForCompute("auto").Choices.Select(c => c.Value)];

        public override string[] Names => [.. ForCompute("auto").Choices.Select(c => c.Name)];
    }

    /// <summary>The second-GPU dropdowns: None, then each CUDA GPU, since the backend places these parts with CUDA selectors only.</summary>
    public class SecondGpuOptions : SettingsOptionsAttribute.AbstractImpl
    {
        public override string[] GetOptions => ["", .. _cuda.Value.Select(c => c.Value)];

        public override string[] Names => ["None", .. _cuda.Value.Select(c => c.Name)];
    }

    /// <summary>Each CUDA GPU the engine's test computation passes on, by CUDA ordinal, which is fastest-first rather than nvidia-smi's order.</summary>
    /// <remarks>A GPU that fails is left out: Auto would quietly run its number on Vulkan instead, which can be a different card.</remarks>
    private static IReadOnlyList<Choice> BuildCuda()
    {
        List<Choice> choices = [];
        foreach (GpuTopologyInfo gpu in Guard(() => CudaTopology.Probe(), "CUDA topology probe") ?? [])
        {
            int ordinal = gpu.Ordinal;
            if (!Guard(() => BackendFactory.ProbeCuda(ordinal), $"CUDA probe of GPU {ordinal}"))
            {
                string why = Guard(() => BackendFactory.CudaProbeFailureReason, "CUDA probe reason") ?? "no reason reported";
                // GPU 0 failing already warns from the Compute Backend list, as its CUDA entry disappears.
                string message = $"GPU list: leaving out CUDA GPU {ordinal} ({gpu.Name}): {why}";
                if (ordinal == 0)
                {
                    HartsyLogs.Debug(message);
                }
                else
                {
                    HartsyLogs.Warning(message);
                }
                continue;
            }
            choices.Add(new Choice($"{ordinal}", $"GPU {ordinal}: {Describe(gpu.Name, gpu.TotalMemoryBytes)}"));
        }
        return choices;
    }

    /// <summary>The Vulkan choices: 0 lets the engine pick, then every other GPU by the loader's raw index.</summary>
    /// <remarks>The loader's device 0 cannot be pinned, since GPU_ID 0 means "let the engine pick". Software rasterizers are left out, as
    /// they are slower than the CPU backend, and so is any device the engine cannot open. Each device is opened briefly to read its name.</remarks>
    private static IReadOnlyList<Choice> BuildVulkan()
    {
        List<Choice> choices = [];
        Guard(() =>
        {
            using VulkanInstance instance = new();
            int count = instance.EnumeratePhysicalDevices().Length;
            VulkanCapabilities best = Guard(() => Open(instance, null), "Vulkan best-device probe");
            choices.Add(new Choice("0", best is null ? "Auto (the engine picks the best GPU)" : $"Auto: {Describe(best.DeviceName, best.TotalVramBytes)}"));
            for (int ordinal = 1; ordinal < count; ordinal++)
            {
                int index = ordinal;
                VulkanCapabilities caps = Guard(() => Open(instance, index), $"Vulkan device {index} probe");
                if (caps is null || caps.DeviceType == VkPhysicalDeviceType.Cpu)
                {
                    continue;
                }
                choices.Add(new Choice($"{index}", $"GPU {index}: {Describe(caps.DeviceName, caps.TotalVramBytes)}"));
            }
            return true;
        }, "Vulkan device list");
        return choices;
    }

    /// <summary>Opens one Vulkan device, or the one the engine would pick when <paramref name="ordinal"/> is null, just long enough to read what it is.</summary>
    private static VulkanCapabilities Open(VulkanInstance instance, int? ordinal)
    {
        using VulkanDevice device = VulkanDevice.Create(instance, ordinal);
        return device.Capabilities;
    }

    /// <summary>"Name (8 GB)", or just the name when the size is unknown.</summary>
    private static string Describe(string name, double bytes) => bytes > 0 ? $"{name} ({bytes / 1073741824.0:0.#} GB)" : name;

    /// <summary>Runs a probe that must never break backend registration. A throw (no engine, no driver) returns default.</summary>
    private static T Guard<T>(Func<T> probe, string what)
    {
        try
        {
            return probe();
        }
        catch (Exception ex)
        {
            HartsyLogs.Debug($"GPU list: {what} failed ({ex.Message}).");
            return default;
        }
    }
}
