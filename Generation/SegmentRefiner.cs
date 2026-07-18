using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using HartsyInference.Core.Backends;
using ISImage = SixLabors.ImageSharp.Image;

namespace Hartsy.Extensions.HartsyInferenceBackend.Generation;

/// <summary>
/// Implements Swarm's <c>&lt;segment:yolo-...&gt;</c> auto-refinement (Comfy parity) for the
/// HartsyInference backend. After the base (and refiner) stage produces an image, each segment
/// part is detected with YOLO / RT-DETR / Grounding DINO (<see cref="SegmentResolver"/>) or CLIPSeg,
/// turned into a mask, and the masked region is re-denoised with the segment's own prompt via the
/// architecture's existing img2img + inpaint-blend path — fed back through the
/// <paramref name="reGenerate"/> delegate.
///
/// <para><b>Crop-to-bbox optimization.</b> This is the in-process equivalent of Comfy's
/// SwarmYoloDetection → <c>SwarmMaskBounds</c> crop → KSampler → <c>ImageCompositeMasked</c>
/// recomposite chain. Rather than re-denoising the entire canvas (the mask localizes the change,
/// but the diffusion cost is paid over every pixel), we crop the base image + mask to the mask's
/// bounding box (grown by <see cref="T2IParamTypes.SegmentMaskOversize"/> for inpaint context),
/// run the img2img refine on just that smaller crop, then composite the refined crop back into the
/// full image using the mask as the alpha. A small object (e.g. a face) that occupies a fraction of
/// a 1024² canvas is refined over a fraction of the pixels — a large wall-clock win on the segment
/// pass with no visible quality change. The crop is sampled at (near) its native resolution rather
/// than up-scaled to the model's full megapixel target (Comfy's detail-maximizing default): that is
/// the deliberate divergence that produces the speed-up, floored at a working resolution so tiny
/// crops still denoise coherently, and overridable via <see cref="T2IParamTypes.SegmentTargetResolution"/>.</para>
///
/// <para><b>Correctness.</b> The recomposite writes the refined pixels only inside the crop
/// rectangle, blended by the (feathered) mask alpha: mask==0 keeps the base pixel byte-for-byte,
/// mask==255 takes the refined pixel, intermediate values blend proportionally. Every pixel outside
/// the crop rectangle is copied unchanged from the base image, so non-masked pixels are guaranteed
/// byte-identical to the input. The crop bounds are derived from the mask's non-zero extent, so the
/// entire feathered blend region is always contained by the crop.</para>
/// </summary>
public static class SegmentRefiner
{
    /// <summary>Diffusion models degrade badly below this working resolution, so a crop whose long
    /// side is smaller is scaled up (aspect-preserving) to at least this before sampling. Still far
    /// fewer pixels than a full-canvas refine for a typical small detection.</summary>
    private const int MinWorkingSide = 512;

    /// <summary>True if the prompt contains any <c>&lt;segment:&gt;</c> parts.</summary>
    public static bool HasSegments(T2IParamInput input)
    {
        string prompt = input.Get(T2IParamTypes.Prompt) ?? "";
        if (!prompt.Contains("<segment:", StringComparison.OrdinalIgnoreCase)) return false;
        return new PromptRegion(prompt).Parts.Any(p => p.Type == PromptRegion.PartType.Segment);
    }

    /// <summary>Runs every segment part over each base image, returning the refined images.
    /// <paramref name="reGenerate"/> takes a (cloned) input with InitImage + MaskImage + segment
    /// prompt set and returns the re-denoised image(s) using the active architecture's loader.</summary>
    public static Image[] Apply(
        IBackend backend,
        Image[] baseImages,
        T2IParamInput input,
        Func<T2IParamInput, Image[]> reGenerate,
        Action<string> log,
        CancellationToken cancel)
    {
        if (baseImages is null || baseImages.Length == 0) return baseImages;
        PromptRegion region = new(input.Get(T2IParamTypes.Prompt) ?? "");
        PromptRegion.Part[] parts = region.Parts.Where(p => p.Type == PromptRegion.PartType.Segment).ToArray();
        if (parts.Length == 0) return baseImages;

        PromptRegion negativeRegion = new(input.Get(T2IParamTypes.NegativePrompt) ?? "");

        Image[] result = new Image[baseImages.Length];
        for (int i = 0; i < baseImages.Length; i++)
        {
            Image current = baseImages[i];
            int segIdx = 0;
            foreach (PromptRegion.Part part in parts)
            {
                cancel.ThrowIfCancellationRequested();
                // "yolo-..." → closed-set YOLO; "rtdetr[-class]" → closed-set RT-DETR (COCO, transformer);
                // "dino-..." → open-vocabulary Grounding DINO text detection; any other target text →
                // free-text CLIPSeg heatmap. YOLO/RT-DETR/DINO boxes are refined to pixel masks by SAM 2
                // when a checkpoint is installed.
                Image mask = SegmentResolver.IsYoloTarget(part)
                    ? SegmentResolver.BuildYoloMask(backend, current, part, input, log)
                    : SegmentResolver.IsRtDetrTarget(part)
                        ? SegmentResolver.BuildRtDetrMask(backend, current, part, input, log)
                        : SegmentResolver.IsDinoTarget(part)
                            ? SegmentResolver.BuildDinoMask(backend, current, part, input, log)
                            : ClipSegResolver.BuildTextMask(backend, current, part, input, log);
                if (mask is null)
                {
                    segIdx++;
                    continue; // nothing detected for this segment — leave the image as-is
                }

                // Match this segment's negative to a same-target negative part if present, else the global negative.
                string segNegative = negativeRegion.Parts
                    .FirstOrDefault(p => p.Type == PromptRegion.PartType.Segment && p.DataText == part.DataText)?.Prompt
                    ?? negativeRegion.GlobalPrompt;

                T2IParamInput clone = input.Clone();
                clone.Set(T2IParamTypes.Prompt, string.IsNullOrWhiteSpace(part.Prompt) ? region.GlobalPrompt : part.Prompt);
                clone.Set(T2IParamTypes.NegativePrompt, segNegative ?? "");
                // Strength2 (default 0.6) is the segment's denoise amount → img2img creativity.
                clone.Set(T2IParamTypes.InitImageCreativity, part.Strength2);
                // Per-segment step/cfg overrides if the user set them.
                if (input.TryGet(T2IParamTypes.SegmentSteps, out int segSteps) && segSteps > 0)
                {
                    clone.Set(T2IParamTypes.Steps, segSteps);
                }
                if (input.TryGet(T2IParamTypes.SegmentCFGScale, out double segCfg) && segCfg > 0)
                {
                    clone.Set(T2IParamTypes.CFGScale, segCfg);
                }

                // Decode the base + mask once, then decide between the crop-to-bbox fast path and the
                // whole-canvas fallback (used when the mask is empty / spans the whole frame).
                (byte[] baseRgb, int bw, int bh) = RgbToImage.ToHwcRgb(current);
                byte[] maskGray = LoadMaskGray(mask, bw, bh);
                int oversize = input.Get(T2IParamTypes.SegmentMaskOversize, 16);

                long t0 = Environment.TickCount64;
                Image nextCurrent = null;
                if (TryComputeCropBounds(maskGray, bw, bh, oversize, out int cx, out int cy, out int cw, out int ch)
                    && (cw < bw || ch < bh)) // only worth cropping if it's actually smaller than the canvas
                {
                    ComputeSampleDims(cw, ch, bw, bh, input, out int sw, out int sh);
                    clone.Set(T2IParamTypes.InitImage, CropRgb(baseRgb, bw, cx, cy, cw, ch));
                    clone.Set(T2IParamTypes.MaskImage, CropGray(maskGray, bw, cx, cy, cw, ch));
                    clone.Set(T2IParamTypes.Width, sw);
                    clone.Set(T2IParamTypes.Height, sh);

                    Image[] refined = reGenerate(clone);
                    if (refined is not null && refined.Length > 0 && refined[0] is not null)
                    {
                        nextCurrent = Recomposite(baseRgb, bw, bh, refined[0], maskGray, cx, cy, cw, ch);
                        log($"[Segment] Applied segment {segIdx + 1}/{parts.Length} (creativity={part.Strength2:F2}) " +
                            $"cropped to {cw}x{ch}@({cx},{cy}) sampled@{sw}x{sh} " +
                            $"(vs {bw}x{bh} full) in {Environment.TickCount64 - t0}ms.");
                    }
                }
                else
                {
                    // Fallback: mask empty or effectively full-frame — re-denoise the whole canvas
                    // (original behaviour; still correct, just no crop speed-up to be had).
                    clone.Set(T2IParamTypes.InitImage, current);
                    clone.Set(T2IParamTypes.MaskImage, mask);
                    Image[] refined = reGenerate(clone);
                    if (refined is not null && refined.Length > 0 && refined[0] is not null)
                    {
                        nextCurrent = refined[0];
                        log($"[Segment] Applied segment {segIdx + 1}/{parts.Length} (creativity={part.Strength2:F2}) " +
                            $"full-frame {bw}x{bh} in {Environment.TickCount64 - t0}ms.");
                    }
                }

                if (nextCurrent is not null)
                {
                    current = nextCurrent;
                }
                segIdx++;
            }
            result[i] = current;
        }
        return result;
    }

    /// <summary>Decodes a mask <see cref="Image"/> to a base-resolution single-channel byte buffer
    /// (L8, 255 = refine). Resizes to (<paramref name="bw"/>,<paramref name="bh"/>) if the mask was
    /// produced at a different resolution, so its coordinates line up with the base image.</summary>
    private static byte[] LoadMaskGray(Image mask, int bw, int bh)
    {
        using var frame = ISImage.Load<L8>(mask.RawData);
        if (frame.Width != bw || frame.Height != bh)
        {
            frame.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new SixLabors.ImageSharp.Size(bw, bh),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Bicubic,
            }));
        }
        byte[] bytes = new byte[bw * bh];
        frame.CopyPixelDataTo(bytes);
        return bytes;
    }

    /// <summary>Computes the crop rectangle: the bounding box of the mask's non-zero pixels, grown by
    /// <paramref name="grow"/> px on each side and clamped to the image (mirrors Comfy's
    /// <c>SwarmMaskBounds</c>). Returns false when the mask has no non-zero pixel (nothing to refine).
    /// Deriving bounds from the (already grown + feathered) mask guarantees the whole blend region is
    /// inside the crop, so the recomposite never has to touch a pixel outside it.</summary>
    private static bool TryComputeCropBounds(byte[] gray, int bw, int bh, int grow, out int cx, out int cy, out int cw, out int ch)
    {
        int minX = bw, minY = bh, maxX = -1, maxY = -1;
        for (int y = 0; y < bh; y++)
        {
            int off = y * bw;
            for (int x = 0; x < bw; x++)
            {
                if (gray[off + x] != 0)
                {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0)
        {
            cx = cy = cw = ch = 0;
            return false;
        }
        int x1 = Math.Clamp(minX - grow, 0, bw - 1);
        int x2 = Math.Clamp(maxX + grow, 0, bw - 1);
        int y1 = Math.Clamp(minY - grow, 0, bh - 1);
        int y2 = Math.Clamp(maxY + grow, 0, bh - 1);
        cx = x1;
        cy = y1;
        cw = x2 - x1 + 1;
        ch = y2 - y1 + 1;
        return cw > 0 && ch > 0;
    }

    /// <summary>Chooses the sampling resolution for the crop. Honors an explicit
    /// <see cref="T2IParamTypes.SegmentTargetResolution"/> ("WxH") if set; otherwise samples at the
    /// crop's native size (aspect-preserving), scaled up only when the crop's long side is below
    /// <see cref="MinWorkingSide"/> so tiny detections still denoise coherently, capped at the base
    /// canvas dimensions, and rounded to a multiple of 64 (the safe latent-grid alignment for all
    /// supported arches).</summary>
    private static void ComputeSampleDims(int cw, int ch, int bw, int bh, T2IParamInput input, out int sw, out int sh)
    {
        string targetRes = input.Get(T2IParamTypes.SegmentTargetResolution, "0x0") ?? "0x0";
        int xPos = targetRes.IndexOf('x');
        if (xPos > 0
            && int.TryParse(targetRes[..xPos].Trim(), out int tx)
            && int.TryParse(targetRes[(xPos + 1)..].Trim(), out int ty)
            && tx > 0 && ty > 0)
        {
            sw = Round64(tx);
            sh = Round64(ty);
            return;
        }
        double scale = 1.0;
        int longSide = Math.Max(cw, ch);
        if (longSide < MinWorkingSide && longSide > 0)
        {
            scale = (double)MinWorkingSide / longSide;
        }
        int capW = Math.Max(64, (bw / 64) * 64);
        int capH = Math.Max(64, (bh / 64) * 64);
        sw = Math.Clamp(Round64((int)Math.Round(cw * scale)), 64, capW);
        sh = Math.Clamp(Round64((int)Math.Round(ch * scale)), 64, capH);
    }

    /// <summary>Rounds up to the nearest multiple of 64 (minimum 64).</summary>
    private static int Round64(int v) => Math.Max(64, (int)Math.Round(v / 64.0) * 64);

    /// <summary>Crops an HWC-RGB byte buffer to (x,y,w,h) and wraps it as a PNG <see cref="Image"/>.</summary>
    private static Image CropRgb(byte[] rgb, int bw, int x, int y, int w, int h)
    {
        byte[] outb = new byte[w * h * 3];
        for (int yy = 0; yy < h; yy++)
        {
            int srcOff = ((y + yy) * bw + x) * 3;
            int dstOff = yy * w * 3;
            Array.Copy(rgb, srcOff, outb, dstOff, w * 3);
        }
        return RgbToImage.FromHwcRgb(outb, w, h);
    }

    /// <summary>Crops a single-channel (L8) byte buffer to (x,y,w,h) and wraps it as a PNG <see cref="Image"/>.</summary>
    private static Image CropGray(byte[] gray, int bw, int x, int y, int w, int h)
    {
        byte[] outb = new byte[w * h];
        for (int yy = 0; yy < h; yy++)
        {
            Array.Copy(gray, (y + yy) * bw + x, outb, yy * w, w);
        }
        using var img = ISImage.LoadPixelData<L8>(outb, w, h);
        return new Image(img);
    }

    /// <summary>Composites the refined crop back onto the base image. The refined crop (returned at
    /// the sampling resolution) is resized to the crop rectangle, then blended over the base inside
    /// that rectangle using the mask as alpha (0 = keep base, 255 = take refined). Every pixel
    /// outside the rectangle — and every pixel inside it where the mask is 0 — is byte-identical to
    /// the base image.</summary>
    private static Image Recomposite(byte[] baseRgb, int bw, int bh, Image refined, byte[] maskGray, int cx, int cy, int cw, int ch)
    {
        // Refined comes back at the sampling resolution; map it 1:1 onto the crop rectangle.
        (int rw, int rh) = RgbToImage.GetDimensions(refined);
        byte[] refRgb = (rw == cw && rh == ch)
            ? RgbToImage.ToHwcRgb(refined).rgbData
            : RgbToImage.ToHwcRgbResized(refined, cw, ch);

        byte[] outRgb = (byte[])baseRgb.Clone();
        for (int yy = 0; yy < ch; yy++)
        {
            int by = cy + yy;
            int baseRow = by * bw;
            int refRow = yy * cw;
            for (int xx = 0; xx < cw; xx++)
            {
                int bx = cx + xx;
                int m = maskGray[baseRow + bx];
                if (m == 0)
                {
                    continue; // mask fully preserves the base pixel
                }
                int baseIdx = (baseRow + bx) * 3;
                int refIdx = (refRow + xx) * 3;
                if (m == 255)
                {
                    outRgb[baseIdx] = refRgb[refIdx];
                    outRgb[baseIdx + 1] = refRgb[refIdx + 1];
                    outRgb[baseIdx + 2] = refRgb[refIdx + 2];
                }
                else
                {
                    int inv = 255 - m;
                    outRgb[baseIdx] = (byte)((baseRgb[baseIdx] * inv + refRgb[refIdx] * m + 127) / 255);
                    outRgb[baseIdx + 1] = (byte)((baseRgb[baseIdx + 1] * inv + refRgb[refIdx + 1] * m + 127) / 255);
                    outRgb[baseIdx + 2] = (byte)((baseRgb[baseIdx + 2] * inv + refRgb[refIdx + 2] * m + 127) / 255);
                }
            }
        }
        return RgbToImage.FromHwcRgb(outRgb, bw, bh);
    }
}
