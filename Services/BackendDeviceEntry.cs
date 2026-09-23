using HartsyInference.Engine.Services;

namespace Hartsy.Extensions.HartsyInferenceBackend.Services;

/// <summary>One live HartsyInference backend: the physical device it bound to, the engine that answers memory
/// questions for it, and the fit profile it shares with every backend configured the same way.</summary>
/// <param name="DeviceKey">What the engine reports it bound to (<c>cuda:0</c>, or a Vulkan device UUID).</param>
/// <param name="DeviceName">The device's human name, for messages a user reads.</param>
/// <param name="FitProfileKey">Device plus <paramref name="FitSettings"/>. Two backends with one key always get the same
/// verdict, so a request is assessed once per key rather than once per backend.</param>
/// <param name="FitSettings">Every setting besides the device that changes a memory verdict (VRAM tier and the
/// text-encoder/VAE/CFG/shard placements). Out-of-memory evidence only transfers between cards with equal settings.</param>
/// <param name="Engine">The backend's engine, which assesses fit against its own device, placement and policy.</param>
public sealed record BackendDeviceEntry(string DeviceKey, string DeviceName, string FitProfileKey, string FitSettings,
    IInferenceEngine Engine);
