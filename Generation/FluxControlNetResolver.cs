using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Resolves Swarm's ControlNet param slots into Flux DiT ControlNet conditionings
/// (<see cref="FluxControlNetConditioning"/>) for Flux.1 generations. Mirrors
/// <see cref="ControlNetResolver"/> (which serves the UNet families): per-slot model load with
/// cache, preprocessor dispatch by detected mode, strength + start/end wiring. Union checkpoints
/// without a mode embedder (Union-Pro-2.0) take the preprocessed map as-is; single-mode files get
/// their matching preprocessor. Unknown/union-with-embedder modes currently pass the reference
/// image through RAW (the user supplies a pre-made map, the Comfy convention).
/// </summary>
public static class FluxControlNetResolver
{
    /// <summary>Owns the loaded adapters + control tensors for one generation.</summary>
    public sealed class ResolvedSpec : IDisposable
    {
        public required List<FluxControlNetConditioning> Conditionings { get; init; }
        public required List<FluxControlNetCacheEntry> Adapters { get; init; }
        public required List<Tensor> ControlImages { get; init; }

        public void Dispose()
        {
            foreach (Tensor img in ControlImages) img.Dispose();
            foreach (FluxControlNetCacheEntry a in Adapters) a.Dispose();
        }
    }

    /// <summary>Returns null when no ControlNet slots are populated with Flux DiT checkpoints.</summary>
    public static ResolvedSpec Resolve(T2IParamInput input, int targetW, int targetH, IBackend backend, Action<string> log)
    {
        if (input is null) return null;
        T2IParamTypes.ControlNetParamHolder[] cnHolders = T2IParamTypes.Controlnets;
        if (cnHolders is null) return null;

        List<FluxControlNetConditioning> conditionings = new();
        List<FluxControlNetCacheEntry> adapters = new();
        List<Tensor> images = new();
        try
        {
            for (int i = 0; i < cnHolders.Length; i++)
            {
                T2IParamTypes.ControlNetParamHolder holder = cnHolders[i];
                if (holder?.Model is null) continue;
                T2IModel cnModel = input.Get(holder.Model);
                if (cnModel is null) continue;
                if (string.IsNullOrWhiteSpace(cnModel.RawFilePath))
                {
                    throw new InvalidOperationException($"ControlNet[{i}] '{cnModel.Name}' has no file path.");
                }

                log($"Loading Flux ControlNet[{i}]: {cnModel.Name}");
                ControlNetFile file = ControlNetLoader.Load(cnModel.RawFilePath);
                if (file.FluxConfig is null)
                {
                    file.Dispose();
                    throw new InvalidOperationException(
                        $"ControlNet '{cnModel.Name}' is a {file.BaseModel} (UNet) ControlNet, but the current generation uses a Flux model. Pick a Flux DiT ControlNet.");
                }
                FluxControlNet adapter = new FluxControlNet(file.FluxConfig);
                adapter.LoadWeights(file.Weights);
                FluxControlNetCacheEntry entry = new FluxControlNetCacheEntry { File = file, Adapter = adapter };
                adapters.Add(entry);

                Image cnImage = input.Get(holder.Image) ?? input.Get(T2IParamTypes.InitImage);
                if (cnImage is null)
                {
                    throw new InvalidOperationException(
                        $"ControlNet[{i}] '{cnModel.Name}' is selected but no ControlNet Image Input or Init Image was provided.");
                }

                Tensor control = BuildControlTensor(file, cnImage, targetW, targetH, backend, log);
                images.Add(control);

                double strength = input.Get(holder.Strength);
                double startFrac = holder.Start is not null ? input.Get(holder.Start) : 0.0;
                double endFrac = holder.End is not null ? input.Get(holder.End) : 1.0;
                conditionings.Add(new FluxControlNetConditioning
                {
                    Adapter = adapter,
                    ControlImage = control,
                    Scale = (float)strength,
                    StartFraction = (float)Math.Clamp(startFrac, 0.0, 1.0),
                    EndFraction = (float)Math.Clamp(endFrac, 0.0, 1.0),
                });
            }
        }
        catch
        {
            foreach (Tensor img in images) img.Dispose();
            foreach (FluxControlNetCacheEntry a in adapters) a.Dispose();
            throw;
        }

        if (conditionings.Count == 0) return null;
        log($"Flux ControlNet enabled: {conditionings.Count} adapter(s).");
        return new ResolvedSpec { Conditionings = conditionings, Adapters = adapters, ControlImages = images };
    }

    /// <summary>Builds the [-1,1] control tensor: single-mode files run their matching preprocessor; union/unknown modes take the user's image as the pre-made map.</summary>
    private static Tensor BuildControlTensor(ControlNetFile file, Image cnImage, int targetW, int targetH, IBackend backend, Action<string> log)
    {
        Tensor zeroOne = file.Mode switch
        {
            ControlNetMode.Canny => CannyPreprocessor.Process(cnImage, targetW, targetH),
            ControlNetMode.Depth => DepthPreprocessor.Process(cnImage, targetW, targetH, backend, msg => log($"[Depth] {msg}"), fluxScaling: true),
            ControlNetMode.OpenPose => OpenPoseControlPreprocessor.Process(cnImage, targetW, targetH, backend, msg => log($"[OpenPose] {msg}")),
            ControlNetMode.SoftEdge => AnnotatorControlPreprocessors.ProcessSoftEdge(cnImage, targetW, targetH, backend, msg => log($"[SoftEdge] {msg}")),
            ControlNetMode.LineArt => AnnotatorControlPreprocessors.ProcessLineart(cnImage, targetW, targetH, backend, msg => log($"[Lineart] {msg}")),
            ControlNetMode.Normal => AnnotatorControlPreprocessors.ProcessNormal(cnImage, targetW, targetH, backend, msg => log($"[Normal] {msg}")),
            _ => RawImageZeroOne(cnImage, targetW, targetH),
        };
        try
        {
            return ScaleToMinusOneOne(zeroOne);
        }
        finally
        {
            zeroOne.Dispose();
        }
    }

    private static unsafe Tensor RawImageZeroOne(Image input, int targetW, int targetH)
    {
        byte[] rgb = Img2ImgResolver.LoadResizedRgb(input, targetW, targetH);
        Tensor output = new Tensor(new TensorShape(1, 3, targetH, targetW), DType.F32);
        float* dp = (float*)output.DataPointer;
        int spatial = targetW * targetH;
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < spatial; i++)
            {
                dp[(long)c * spatial + i] = rgb[i * 3 + c] / 255f;
            }
        }
        return output;
    }

    private static unsafe Tensor ScaleToMinusOneOne(Tensor zeroOne)
    {
        Tensor output = new Tensor(zeroOne.Shape, DType.F32);
        float* sp = (float*)zeroOne.DataPointer;
        float* dp = (float*)output.DataPointer;
        long count = zeroOne.ElementCount;
        for (long i = 0; i < count; i++) dp[i] = sp[i] * 2f - 1f;
        return output;
    }
}

/// <summary>Holds one loaded Flux ControlNet + its backing file (mmap keep-alive).</summary>
public sealed class FluxControlNetCacheEntry : IDisposable
{
    public required ControlNetFile File { get; init; }
    public required FluxControlNet Adapter { get; init; }

    public void Dispose()
    {
        Adapter?.Dispose();
        File?.Dispose();
    }
}
