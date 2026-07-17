using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's ControlNet params (registered by the ComfyUI extension under
/// the <c>controlnet</c> feature flag) into a list of
/// <see cref="ControlNetConditioning"/>s ready to hand to a HartsyInference
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
/// <para><b>Union checkpoints</b> (xinsir controlnet-union-sdxl, detected via
/// <see cref="ControlNetConfig.UnionControlTypeCount"/>) take their control type
/// from the per-slot "ControlNet Union Type" param and run the matching
/// preprocessor; the type index is passed through
/// <see cref="ControlNetConditioning.UnionControlType"/> to the engine's union
/// fusion path. Stack the same union model in multiple slots with different
/// types to combine controls.</para>
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
        IBackend backend,
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

                // Union checkpoints (xinsir controlnet-union-sdxl): one file covers all modes.
                // The control type comes from the user's "ControlNet Union Type" param (the
                // ComfyUI extension registers it per slot); the preprocessor follows the type.
                // Single-mode files keep the existing filename-mode dispatch untouched.
                bool isUnion = entry.File.Config.UnionControlTypeCount > 0;
                SdxlUnionControlType? unionType = null;
                Tensor condTensor;
                if (isUnion)
                {
                    unionType = ResolveUnionType(input, i, entry.File.Config.UnionControlTypeCount, cnModel.Name, log);
                    condTensor = PreprocessForUnionType(unionType.Value, cnImage, targetW, targetH, backend, log);
                }
                else
                {
                    // Preprocess based on the file's auto-detected mode (filename heuristic).
                    // Canny is algorithmic; Depth runs the in-engine Depth-Anything-V2 annotator.
                    condTensor = entry.File.Mode switch
                    {
                        ControlNetMode.Canny => CannyPreprocessor.Process(cnImage, targetW, targetH),
                        ControlNetMode.Depth => DepthPreprocessor.Process(cnImage, targetW, targetH, backend,
                            msg => log($"[Depth] {msg}")),
                        ControlNetMode.OpenPose => OpenPoseControlPreprocessor.Process(cnImage, targetW, targetH, backend,
                            msg => log($"[OpenPose] {msg}")),
                        ControlNetMode.SoftEdge => AnnotatorControlPreprocessors.ProcessSoftEdge(cnImage, targetW, targetH, backend,
                            msg => log($"[SoftEdge] {msg}")),
                        ControlNetMode.Scribble => AnnotatorControlPreprocessors.ProcessScribble(cnImage, targetW, targetH, backend,
                            msg => log($"[Scribble] {msg}")),
                        ControlNetMode.LineArt => AnnotatorControlPreprocessors.ProcessLineart(cnImage, targetW, targetH, backend,
                            msg => log($"[Lineart] {msg}")),
                        ControlNetMode.Normal => AnnotatorControlPreprocessors.ProcessNormal(cnImage, targetW, targetH, backend,
                            msg => log($"[Normal] {msg}")),
                        ControlNetMode.Segmentation => AnnotatorControlPreprocessors.ProcessSegment(cnImage, targetW, targetH, backend,
                            msg => log($"[Segment] {msg}")),
                        _ => throw new NotSupportedException(
                            $"ControlNet[{i}] '{cnModel.Name}' detected as mode '{entry.File.Mode}'. " +
                            $"Currently supported preprocessors: Canny, Depth, OpenPose, SoftEdge/HED, Scribble, Lineart, Normal, Segmentation. Tile/Inpaint modes are follow-ups."),
                    };
                }
                images.Add(condTensor);

                double strength = input.Get(holder.Strength);
                double startFrac = holder.Start is not null ? input.Get(holder.Start) : 0.0;
                double endFrac = holder.End is not null ? input.Get(holder.End) : 1.0;
                conditionings.Add(new ControlNetConditioning
                {
                    Adapter = entry.Adapter,
                    ConditionImage = condTensor,
                    Scale = (float)strength,
                    StartFraction = (float)Math.Clamp(startFrac, 0.0, 1.0),
                    EndFraction = (float)Math.Clamp(endFrac, 0.0, 1.0),
                    UnionControlType = unionType,
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

    /// <summary>Maps the user's per-slot "ControlNet Union Type" param (registered by the ComfyUI
    /// extension; values follow the xinsir training list) onto the checkpoint's control-type index.
    /// Untoggled/"auto" defaults to the thin-line (canny) type — algorithmic, no annotator download.
    /// Tile/Repaint need the 8-type ProMax revision; the 6-type standard union is rejected early
    /// with a clear message instead of an out-of-range engine error.</summary>
    private static SdxlUnionControlType ResolveUnionType(T2IParamInput input, int slot, int numControlTypes, string modelName, Action<string> log)
    {
        string typeStr = "auto";
        T2IRegisteredParam<string> param = ComfyUIBackendExtension.ControlNetUnionTypeParams != null && slot < ComfyUIBackendExtension.ControlNetUnionTypeParams.Length
            ? ComfyUIBackendExtension.ControlNetUnionTypeParams[slot] : null;
        if (param is not null && input.TryGet(param, out string val) && !string.IsNullOrWhiteSpace(val))
        {
            typeStr = val.Trim().ToLowerInvariant();
        }
        SdxlUnionControlType type = typeStr switch
        {
            "openpose" => SdxlUnionControlType.OpenPose,
            "depth" => SdxlUnionControlType.Depth,
            "hed/pidi/scribble/ted" => SdxlUnionControlType.SoftEdge,
            "canny/lineart/anime_lineart/mlsd" => SdxlUnionControlType.Canny,
            "normal" => SdxlUnionControlType.Normal,
            "segment" => SdxlUnionControlType.Segment,
            "tile" => SdxlUnionControlType.Tile,
            "repaint" => SdxlUnionControlType.Repaint,
            "auto" => SdxlUnionControlType.Canny,
            _ => throw new InvalidOperationException(
                $"ControlNet[{slot}] Union Type '{typeStr}' is not a recognized union control type."),
        };
        if (typeStr == "auto")
        {
            log($"ControlNet[{slot}] '{modelName}' is a union checkpoint and no Union Type was chosen — defaulting to canny (thin line). Set 'ControlNet Union Type' to pick another mode.");
        }
        if ((int)type >= numControlTypes)
        {
            throw new InvalidOperationException(
                $"ControlNet[{slot}] '{modelName}' has {numControlTypes} control types (standard union revision) — '{type}' needs the 8-type ProMax revision (controlnet-union-sdxl promax).");
        }
        return type;
    }

    /// <summary>Runs the preprocessor matching the union control type. Segment/Tile/Repaint pass the
    /// user's image through raw (0..1 RGB) — the Comfy convention of supplying a pre-made map (tile
    /// conditions on the raw image itself; repaint's masked-inpaint image assembly is a follow-up).</summary>
    private static Tensor PreprocessForUnionType(SdxlUnionControlType type, Image cnImage, int targetW, int targetH, IBackend backend, Action<string> log)
    {
        return type switch
        {
            SdxlUnionControlType.OpenPose => OpenPoseControlPreprocessor.Process(cnImage, targetW, targetH, backend,
                msg => log($"[OpenPose] {msg}")),
            SdxlUnionControlType.Depth => DepthPreprocessor.Process(cnImage, targetW, targetH, backend,
                msg => log($"[Depth] {msg}")),
            SdxlUnionControlType.SoftEdge => AnnotatorControlPreprocessors.ProcessSoftEdge(cnImage, targetW, targetH, backend,
                msg => log($"[SoftEdge] {msg}")),
            SdxlUnionControlType.Canny => CannyPreprocessor.Process(cnImage, targetW, targetH),
            SdxlUnionControlType.Normal => AnnotatorControlPreprocessors.ProcessNormal(cnImage, targetW, targetH, backend,
                msg => log($"[Normal] {msg}")),
            _ => FluxControlNetResolver.RawImageZeroOne(cnImage, targetW, targetH),
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

/// <summary>Wraps HartsyInference's <see cref="ControlNetLoader"/> + <see cref="ControlNet"/>
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
            // SDXL + SD1.5 are wired through their pipelines; refuse the rest (Flux DiT
            // ControlNets need the separate Flux adapter class — Wave 2).
            if (file.BaseModel is not (ControlNetBaseModel.Sdxl or ControlNetBaseModel.Sd15))
            {
                throw new InvalidOperationException(
                    $"ControlNet '{model.Name}' detected as base={file.BaseModel}. " +
                    $"This extension currently supports SDXL and SD 1.5 ControlNets; Flux DiT ControlNets are a follow-up.");
            }
            // Guard base/config mismatch (e.g. an SD15 CN selected on an SDXL gen).
            bool baseIsSdxl = baseConfig.CrossAttentionDim == 2048;
            if ((file.BaseModel == ControlNetBaseModel.Sdxl) != baseIsSdxl)
            {
                throw new InvalidOperationException(
                    $"ControlNet '{model.Name}' is a {file.BaseModel} ControlNet but the current generation uses a {(baseIsSdxl ? "SDXL" : "SD 1.5")} base model. Pick a matching ControlNet.");
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
