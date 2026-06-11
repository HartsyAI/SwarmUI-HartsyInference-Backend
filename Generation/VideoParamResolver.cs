using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Hartsy.Extensions.SharpInferenceBackend.Generation;

/// <summary>
/// Shared SwarmUI-param → video-pipeline value mapping for the video loaders (Wan, LTX).
/// Mirrors the ComfyUI workflow generator's reads exactly: <c>Text2VideoFrames</c> with a
/// per-model default, <c>VideoFPS</c> defaulting to 24 (Comfy's <c>Text2VideoFPS()</c>),
/// <c>VideoFormat</c> defaulting to h264-mp4, and snap-don't-reject rounding for the
/// model's frame-count and resolution constraints (Comfy nodes round internally; we must
/// round before calling the pipeline or it throws).
/// </summary>
public static class VideoParamResolver
{
    /// <summary>Frame count from <c>Text2VideoFrames</c>, snapped to the model's <c>step·n + 1</c>
    /// rule (Wan: 4n+1, LTX: 8n+1). Logs when the user's value was adjusted.</summary>
    public static int ResolveFrames(T2IParamInput input, int modelDefault, int step)
    {
        int requested = input.Get(T2IParamTypes.Text2VideoFrames, modelDefault);
        int snapped = 1 + (int)Math.Round((requested - 1) / (double)step) * step;
        snapped = Math.Max(1, snapped);
        if (snapped != requested)
        {
            Logs.Info($"[SharpInference] Video frame count {requested} rounded to {snapped} (model requires {step}n+1 frames).");
        }
        return snapped;
    }

    /// <summary>Target FPS from <c>VideoFPS</c>, defaulting to 24 like Comfy's <c>Text2VideoFPS()</c>.</summary>
    public static int ResolveFps(T2IParamInput input) => input.Get(T2IParamTypes.VideoFPS, 24);

    /// <summary>Output container from <c>VideoFormat</c> (default h264-mp4, matching Comfy's save node).</summary>
    public static string ResolveFormat(T2IParamInput input) => input.Get(T2IParamTypes.VideoFormat, "h264-mp4");

    /// <summary>Width/height snapped down-or-up to the nearest multiple of the model's VAE spatial
    /// compression (Wan: 16, LTX: 32). Logs when adjusted.</summary>
    public static (int Width, int Height) ResolveResolution(T2IParamInput input, int multiple)
    {
        int width = SnapToMultiple(input.Get(T2IParamTypes.Width), multiple);
        int height = SnapToMultiple(input.Get(T2IParamTypes.Height), multiple);
        if (width != input.Get(T2IParamTypes.Width) || height != input.Get(T2IParamTypes.Height))
        {
            Logs.Info($"[SharpInference] Video resolution {input.Get(T2IParamTypes.Width)}x{input.Get(T2IParamTypes.Height)} rounded to {width}x{height} (model requires multiples of {multiple}).");
        }
        return (width, height);
    }

    /// <summary>Runs trim + boomerang frame edits, then muxes to the user's chosen format. The single
    /// final step every video loader's Generate ends with.</summary>
    public static SwarmUI.Utils.Image FinishVideo(byte[][] frames, int width, int height, T2IParamInput input, CancellationToken cancel)
    {
        frames = VideoOutputEncoder.ApplyFrameEdits(
            frames,
            trimStart: input.Get(T2IParamTypes.TrimVideoStartFrames, 0),
            trimEnd: input.Get(T2IParamTypes.TrimVideoEndFrames, 0),
            boomerang: input.Get(T2IParamTypes.VideoBoomerang, false));
        return VideoOutputEncoder.Encode(frames, width, height, ResolveFps(input), ResolveFormat(input), cancel);
    }

    /// <summary>Image-to-video target resolution per the <c>VideoResolution</c> param's documented modes
    /// (mirrors Comfy's ImageToVideo sizing): "Image Aspect, Model Res" fits the init image's aspect into
    /// the model's standard pixel count via <see cref="Utilities.ResToModelFit(int, int, int, int)"/>;
    /// "Model Preferred" uses the model's standard WxH; "Image" keeps the init image's own size. All modes
    /// snap to the model's VAE multiple.</summary>
    public static (int Width, int Height) ResolveI2VResolution(T2IParamInput input, T2IModel model, int imageWidth, int imageHeight, int multiple)
    {
        string resFormat = input.Get(T2IParamTypes.VideoResolution, "Image Aspect, Model Res");
        int stdW = model is not null && model.StandardWidth > 0 ? model.StandardWidth : input.Get(T2IParamTypes.Width);
        int stdH = model is not null && model.StandardHeight > 0 ? model.StandardHeight : input.Get(T2IParamTypes.Height);
        (int width, int height) = resFormat switch
        {
            "Model Preferred" => (stdW, stdH),
            "Image" => (imageWidth, imageHeight),
            _ => Utilities.ResToModelFit(imageWidth, imageHeight, stdW * stdH, multiple),
        };
        return (SnapToMultiple(width, multiple), SnapToMultiple(height, multiple));
    }

    private static int SnapToMultiple(int value, int multiple)
    {
        int snapped = (int)Math.Round(value / (double)multiple) * multiple;
        return Math.Max(multiple, snapped);
    }
}
