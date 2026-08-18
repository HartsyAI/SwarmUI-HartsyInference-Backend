# Known Issues & TODO

The working list: what is currently broken or unverified, and what is queued to fix it.
Engine-side work is tracked in the engine repo's
[`ROADMAP.md`](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/ROADMAP.md);
Comfy-parity gaps are in [`11-Comfy-Parity-Punchlist.md`](./11-Comfy-Parity-Punchlist.md).
Benchmark numbers live in the engine's
[benchmark scoreboards](https://github.com/HartsyAI/HartsyInference/tree/main/benchmarks/scoreboards),
not here — they move too fast for a hand-maintained table to stay honest.

## Known limitations

- **VAE decode is slow** when memory is tight (tens of seconds for 1024²) because of
  the OOM-retry pressure. Each retry costs a stream sync. A pre-flight memory budget
  check (see TODO below) would eliminate most retries.
- **F16 VAE produces black output on Flux Schnell** — pipeline runs without error but
  values come out NaN/saturated. F32 VAE works fine.
- **Single-batch only.** Batch > 1 not validated; several pipelines are explicitly
  sized for B=1 in their activation buffers.

## TODO

Tracked as `// TODO: ...` comments in the code where applicable.

### Memory + performance
- [ ] **F16 VAE precision**. Black output at F16 on Flux Schnell — debug whether the
  F16 GroupNorm kernel accumulates in F32 internally, and whether the F16 softmax
  subtracts max-before-exp. If both are clean, the issue is somewhere in the
  ResNetBlock / VaeAttention chain. F16 VAE is needed for 2K+ resolutions where even
  tiled F32 won't fit.
- [ ] **Pre-flight memory check before each VAE tile**. Currently we allocate
  optimistically and recover via OOM retry, which costs ~600 ms per retry. A
  pre-flight `cuMemGetInfo` + mempool trim would catch the tight cases and drain
  the pool before the alloc is even attempted, eliminating most retries.
- [ ] **`VramStrategy` foundation.** A single source of truth for VRAM budget, used by
  every pipeline phase to plan load/evict decisions, doesn't exist yet — budget
  decisions are made locally per-pipeline (e.g. Chroma's T5-vs-DiT eviction).
- [ ] **Img2img with `VaeEncoder` on the tiled path.** `VaeDecoder.DecodeTiled` exists
  and every pipeline routes through it; there is no `EncodeTiled` sibling, so large
  img2img/inpaint inputs go through the encoder untiled.

### Quality / correctness
- [ ] **Tile seam visibility audit.** 64-pixel RGB overlap with tent blending should
  be smooth, but worth a side-by-side vs an un-tiled F32 reference at a few
  resolutions to confirm.
- [ ] **Numerical comparison against ComfyUI** for the same prompt + seed at the
  same model. Identifies any silent precision drift in the F16 transformer path.

### Long-term (defer)
- [ ] **CPU-offloaded activations** for >24 GB models. Currently we only offload
  weights; activations always stay on GPU. Real "lowvram" mode would page activation
  tensors out too.
