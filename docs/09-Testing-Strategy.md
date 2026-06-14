# 09 — Testing Strategy

How we validate that this extension actually works and produces correct output.

## Three layers

| Layer | Tooling | What it catches |
|-------|---------|-----------------|
| Unit | xUnit (or whatever Swarm uses) | Pure logic: parameter mapping, sampler-name translation, prefix partitioning, LoRA fingerprinting |
| Component | xUnit + real models | Backend init, model load, single-image generation per architecture |
| Parity | Custom harness | Same prompt + seed + model on Comfy and HartsyInference; diff outputs |

Performance / regression / soak tests live alongside the parity harness.

## Unit tests

Live under `tests/SwarmUI-HartsyInference.Tests/` (separate csproj). Don't pull in
HartsyInference compute — they should run on any machine in <1s.

What to cover:

- **`SamplerMap`:** every Swarm sampler name maps to an expected HartsyInference
  scheduler string (or null with a clear default).
- **Architecture detection:** given a tensor-key list, the right `CompatClass` is
  inferred. Use snapshot fixtures of real model key lists (recorded once via the
  probe route).
- **Checkpoint partitioning:** given a flat `Dictionary<string, Tensor>`, the
  per-component dicts have the expected key sets.
- **LoRA fingerprinting:** the same LoRA list (in any order) produces the same
  fingerprint; different lists produce different.
- **`IsValidForThisBackend`:** returns false for unknown architectures, returns
  false for img2img inputs in pre-phase-4 builds, returns true for SDXL t2i.
- **Param-input → TextToImageRequest mapping:** edge cases (seed=-1 → null, missing
  optional params, etc.).

Mock HartsyInference dependencies behind a tiny `IInferenceEngine` interface in
`Generation/InferenceEngineStub.cs` so tests don't need real models.

## Component tests

Live under `tests/SwarmUI-HartsyInference.Component.Tests/`. Require:

- One real SD 1.5 model (smallest reasonable: ~2 GB)
- One real SDXL model
- One real Flux schnell model
- A test runner with at least 12 GB VRAM (skip on CI without GPU)

Marked `[Fact(Skip = "Requires GPU + models")]` by default; enabled via
`HARTSYINFERENCE_RUN_COMPONENT_TESTS=1` env var.

What to cover:

- **Init / Shutdown:** create backend, init, shutdown, verify no leaks
  (`IBackend.GetVramUsage()` returns to zero — needs upstream).
- **Single-shot t2i per architecture:** SD1.5, SDXL, Flux schnell. Compare output
  hash against a snapshot from a previous successful run on the same machine
  (machine-dependent — not portable).
- **Model swap:** load SD1.5, generate, load SDXL, generate, swap back to SD1.5,
  generate. Verify cache reused (no reload from disk on the third gen).
- **LoRA application:** generate without LoRA → snapshot. Generate with the same
  prompt + seed + a known-strength LoRA → image is visibly different (some tolerance).
- **Cancellation:** start a 30-step gen, cancel at step 5, verify it stops within
  ~1 step.
- **Memory release:** after 50 alternating SD1.5 / SDXL gens, VRAM stays steady
  (no monotonic growth).

## Parity tests

The most important and the slowest.

**Setup:** A driver (`tests/parity/run_parity.py` — yes Python, ironic) that:

1. Reads a fixture file `parity_cases.yaml`:
   ```yaml
   - name: sdxl_simple_castle
     model: sd_xl_base_1.0.safetensors
     prompt: "A castle on a mountain at sunset"
     negative_prompt: "blurry"
     steps: 20
     cfg: 7.0
     seed: 42
     width: 1024
     height: 1024
     sampler: euler
   ```
2. Generates each case once via Swarm + ComfyUI backend (HTTP API).
3. Generates each case once via Swarm + HartsyInference backend (HTTP API).
4. Computes:
   - **Pixel-space MAE** between the two outputs
   - **CLIP similarity** (via a third pure-Python CLIP implementation — we're not eating our own dogfood here, that defeats the purpose)
   - **LPIPS** for perceptual similarity
5. Reports per-case pass/fail against tolerances:
   - `MAE < 0.05` (in [0,1] normalized RGB)
   - `CLIP_sim > 0.95`
   - `LPIPS < 0.10`

These tolerances are loose initially — we're not aiming for bit-exact, we're aiming
for "the user couldn't tell the difference." They tighten as HartsyInference's numerical
stability improves.

**A few cases will diverge intentionally** — different scheduler implementations
(particularly `dpm++ 2m karras` vs Comfy's variant) can produce visibly different
outputs at the same seed. Those cases get a `tolerance: looser` flag in the YAML.

## Manual / smoke testing

Before each phase milestone:

- [ ] Fresh `update-linux.sh` build, no errors
- [ ] Fresh Swarm install, extension installed via Server → Extensions, restart succeeds
- [ ] Configure HartsyInference backend, save, restart backend, status reaches IDLE
- [ ] Generate a 512×512 SD1.5 image; appears in gallery
- [ ] Open browser devtools, generate again, no console errors
- [ ] Click Cancel during a 30-step generation; UI returns to idle
- [ ] Switch to a different model; generate; verify model loaded (logs)
- [ ] Restart Swarm; verify backend re-inits cleanly

Run on Linux (Ubuntu) and Windows separately — CUDA path quirks differ.

## Performance tracking

Per phase, run a fixed benchmark and log results:

| Bench | Hardware | Backend | Latency target |
|-------|----------|---------|----------------|
| SDXL 1024×1024 30 steps | RTX 4090 | HartsyInference CUDA | within 1.5× of Comfy |
| SDXL 1024×1024 30 steps | RTX 4090 | HartsyInference Vulkan | within 2× of Comfy |
| SD1.5 512×512 20 steps | M1 Max | HartsyInference CPU | <60s |
| Flux schnell 1024×1024 4 steps | RTX 4090 | HartsyInference CUDA | within 1.5× of Comfy |

Numbers go into `tests/perf/results-<date>.md`. Regressions against the previous
phase's numbers are blockers for the next phase.

## CI

GitHub Actions or whatever Swarm uses. CI runs:

- Unit tests on every push
- Component + parity tests once per week (manual trigger), on a self-hosted runner
  with a GPU + models pre-cached on the file system
- Lint + style on every push (matches Swarm's own conventions)

CI does **not** run the full parity sweep on PR — too slow, too expensive. PRs get
unit + a tiny smoke generation.

## Continuous validation: parity dashboard

A static HTML page generated by `run_parity.py` that shows side-by-side images,
metric scores, and pass/fail indicators. Regenerated after each parity run.
Linked from the README under "Quality Status."

## Bugs found in HartsyInference

If a parity test fails because HartsyInference has a numerical bug, that's an
**upstream issue**, not an extension bug. We file it against
`/home/kalebbroo/Desktop/Projects/HartsyInference/` and either work around it
(suppress that case in our parity harness) or wait for the fix. Bugs found in
*us* (parameter wiring, partitioning, LoRA logic) are extension issues.
