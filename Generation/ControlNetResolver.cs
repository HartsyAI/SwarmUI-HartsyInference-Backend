using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Adapters;
using SharpInference.Diffusion.Models.Denoisers;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's ControlNet params (registered by the ComfyUI extension under
/// the <c>controlnet</c> feature flag) into a list of
/// <see cref="ControlNetConditioning"/>s ready to hand to a SharpInference
/// pipeline. Loads each selected ControlNet checkpoint, runs the appropriate
/// preprocessor on the input image, and bundles everything into a disposable
/// <see cref="ResolvedSpec"/>.
///
/// <para><b>v1 scope:</b> SDXL-base ControlNets only (Canny preprocessor only).
/// SD1.5 and Flux ControlNets are refused at this layer with a clear message —
/// the upstream <see cref="ControlNet"/> class supports SD1.5 architecturally,
/// but we haven't yet wired SD1.5 into <see cref="Sd15Loader"/> + pipeline
/// (mechanical follow-up). Other preprocessors (Depth, OpenPose, etc.) require
/// ONNX runtime + bundled models — separate phase.</para>
///
/// <para>The Swarm UI exposes up to 3 ControlNet slots
/// (<c>T2IParamTypes.Controlnets[0..2]</c>). This resolver iterates all 3 and
/// includes every slot the user has populated — pipelines support stacking via
/// summed residuals.</para>
/// </summary>
public static class ControlNetResolver
{
    /// <summary>One generation's worth of resolved ControlNet state. Owns both
    /// the loaded adapters and the preprocessed condition images, disposed
    /// together at end of generation.</summary>
    public sealed class ResolvedSpec : IDisposable
    {
        public required List<ControlNetConditioning> Conditionings { get; init; }
        public required List<ControlNetCacheEntry> Adapters { get; init; }
        public required List<Tensor> ConditionImages { get; init; }

        public void Dispose()
        {
            foreach (Tensor img in ConditionImages) img.Dispose();
            foreach (ControlNetCacheEntry a in Adapters) a.Dispose();
        }
    }

    /// <summary>Resolve the ControlNet param holders for this generation.
    /// Returns null when no ControlNets are configured (caller falls back to
    /// plain pipeline call).</summary>
    public static ResolvedSpec Resolve(
        T2IParamInput input,
        UNetConfig baseConfig,
        int targetW,
        int targetH,
        Action<string> log)
    {
        if (input is null) return null;
        T2IParamTypes.ControlNetParamHolder[] cnHolders = T2IParamTypes.Controlnets;
        if (cnHolders is null) return null;

        List<ControlNetConditioning> conditionings = new();
        List<ControlNetCacheEntry> adapters = new();
        List<Tensor> images = new();

        try
        {
            for (int i = 0; i < cnHolders.Length; i++)
            {
                T2IParamTypes.ControlNetParamHolder holder = cnHolders[i];
                if (holder?.Model is null) continue; // params not registered (Comfy extension absent)
                T2IModel cnModel = input.Get(holder.Model);
                if (cnModel is null) continue; // user hasn't selected a model in this slot
                if (string.IsNullOrWhiteSpace(cnModel.RawFilePath))
                {
                    throw new InvalidOperationException($"ControlNet[{i}] '{cnModel.Name}' has no file path.");
                }

                log($"Loading ControlNet[{i}]: {cnModel.Name}");
                ControlNetCacheEntry entry = ControlNetWeightLoader.Load(cnModel, baseConfig);
                adapters.Add(entry);

                // Input image: dedicated CN image, falling back to InitImage if missing
                // (matches Comfy's behavior when "ControlNet Image Input" is empty but
                // "Init Image" is present).
                Image cnImage = input.Get(holder.Image) ?? input.Get(T2IParamTypes.InitImage);
                if (cnImage is null)
                {
                    throw new InvalidOperationException(
                        $"ControlNet[{i}] '{cnModel.Name}' is selected but no ControlNet Image Input or Init Image was provided.");
                }

                // Preprocess based on the file's auto-detected mode (filename heuristic).
                // v1 supports Canny only; refuse other modes with a clear message.
                Tensor condTensor = entry.File.Mode switch
                {
                    ControlNetMode.Canny => CannyPreprocessor.Process(cnImage, targetW, targetH),
                    _ => throw new NotSupportedException(
                        $"ControlNet[{i}] '{cnModel.Name}' detected as mode '{entry.File.Mode}'. " +
                        $"Currently supported preprocessors: Canny. Other modes (Depth, OpenPose, etc.) need ONNX runtime + bundled models — see Phase B follow-ups."),
                };
                images.Add(condTensor);

                double strength = input.Get(holder.Strength);
                conditionings.Add(new ControlNetConditioning
                {
                    Adapter = entry.Adapter,
                    ConditionImage = condTensor,
                    Scale = (float)strength,
                });
            }
        }
        catch
        {
            // Roll back all partial state on failure (no leaks of GPU weights / tensors).
            foreach (Tensor img in images) img.Dispose();
            foreach (ControlNetCacheEntry a in adapters) a.Dispose();
            throw;
        }

        if (conditionings.Count == 0) return null;
        log($"ControlNet enabled: {conditionings.Count} adapter(s).");
        return new ResolvedSpec
        {
            Conditionings = conditionings,
            Adapters = adapters,
            ConditionImages = images,
        };
    }
}

/// <summary>Loaded ControlNet checkpoint kept around for the duration of one
/// generation. Owns the safetensors-backed file (mmap'd) and the constructed
/// adapter; both are disposed together.</summary>
public sealed class ControlNetCacheEntry : IDisposable
{
    public required string FilePath { get; init; }
    public required ControlNetFile File { get; init; }
    public required ControlNet Adapter { get; init; }

    private bool _disposed;
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Adapter.Dispose();
        File.Dispose();
    }
}

/// <summary>Wraps SharpInference's <see cref="ControlNetLoader"/> + <see cref="ControlNet"/>
/// construction into a single Swarm-side helper. Keeps the cache-entry shape
/// consistent with other side-model loaders in this extension.</summary>
public static class ControlNetWeightLoader
{
    /// <summary>Loads a ControlNet checkpoint and constructs an adapter against
    /// the given base UNet config. Throws if the checkpoint is for a base model
    /// other than SDXL (the only family wired through to a pipeline in v1).</summary>
    public static ControlNetCacheEntry Load(T2IModel model, UNetConfig baseConfig)
    {
        ControlNetFile file = ControlNetLoader.Load(model.RawFilePath);
        try
        {
            // v1: SDXL-only. SharpInference's ControlNet adapter handles SD15
            // architecturally but the Swarm side hasn't wired SD15 + pipeline
            // through yet — refuse with a clear message instead of silently
            // running with a mismatched base UNet config.
            if (file.BaseModel != ControlNetBaseModel.Sdxl)
            {
                throw new InvalidOperationException(
                    $"ControlNet '{model.Name}' detected as base={file.BaseModel}. " +
                    $"This extension currently supports SDXL ControlNets only. " +
                    $"SD 1.5 ControlNets need the SD 1.5 pipeline wired to accept the adapter — follow-up.");
            }
            ControlNet adapter = new ControlNet(file.Config, baseConfig);
            adapter.LoadWeights(file.Weights);
            return new ControlNetCacheEntry
            {
                FilePath = model.RawFilePath,
                File = file,
                Adapter = adapter,
            };
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }
}
