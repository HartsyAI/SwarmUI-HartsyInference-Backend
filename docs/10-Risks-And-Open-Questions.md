# 10 — Risks and Open Questions

Known unknowns. Risks ordered by severity. Open questions that need a human decision
before phase 1 starts.

## R1 — TFM mismatch (CRITICAL, blocking phase 1)

**Risk:** HartsyInference targets `net10.0`. SwarmUI extensions target `net8.0` (per
`src/SwarmUI.extension.props`). A net8.0 assembly cannot reference a net10.0
assembly at compile time. Without resolving this, **the extension cannot build**.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| (A) Multi-target HartsyInference (`net8.0;net10.0`) | Lowest cross-project disruption; this extension stays net8.0 | HartsyInference team has to verify no net10-only API usage; may not be free |
| (B) Bump Swarm to net10 | Cleanest long-term; .NET 10 is GA Nov 2025 | Affects every Swarm extension; needs upstream PR + coordination |
| (C) Out-of-process bridge | Unblocks immediately | Defeats the in-process goal; reintroduces the IPC overhead we're trying to remove |
| (D) Use HartsyInference.Server when it's done | Punts the problem to a hosted endpoint | Server isn't built yet; HTTP back to ourselves is silly when we wanted in-process |

**Decision needed before phase 1:** Pick (A) — multi-target HartsyInference. PR upstream.

**Open question:** Does HartsyInference use net10-only APIs (`nuint` improvements,
collection expressions everywhere, ref-struct enhancements, `field` keyword)? If yes,
multi-targeting requires `#if NET8_0` shims for those uses. Needs an audit.

## R2 — LoRA cache strategy (HIGH, phase 2 design)

**Risk:** `LoraStack.ApplyToWeights` mutates weight dicts in place. If we apply
LoRAs to the cached UNet, we contaminate it for the next generation that doesn't
want those LoRAs. The naïve fix (deep-copy the UNet weights every gen) costs ~3 GB
of memory copy per SDXL gen — unacceptable.

**Options recapped from [`05-Pipeline-Translation.md`](./05-Pipeline-Translation.md):**

- (A) Deep-copy: simple, correct, slow
- (B) Pristine + working dict: complicated to keep in sync; reset cost is real
- (C) Upstream non-destructive merge API: best long-term; needs upstream design
- (D) Reload from disk on LoRA change: simplest; very slow on disk-bound systems

**Plan:** ship (D) in phase 2; pursue (C) upstream in parallel. Not a phase-1 blocker
(phase 1 has no LoRA support).

**Open question:** does HartsyInference have any concept of "weight delta" / "bias
offset" that could let us implement non-destructive merges purely in this extension
without an upstream change?

## R3 — No CancellationToken (MEDIUM, phase 3)

**Risk:** HartsyInference pipelines have no cancellation support. Workaround
(throw from progress callback) only granular to per-step boundaries. For Flux at
4 steps, that means up to 25% of generation time before cancel takes effect; for
SD1.5 at 20 steps it's <5% which is fine.

**Plan:** ship workaround in phase 3, file upstream for proper support.

**Open question:** can we monkey-patch a cancellation flag into the inner loops via
a HartsyInference progress hook that already runs every step? Need to read the pipeline
source. Probably yes.

## R4 — VaeEncoder doesn't exist (MEDIUM, phase 4 blocker)

**Risk:** Without `VaeEncoder`, no img2img and no proper inpainting. Implementing
a VAE encoder from scratch is non-trivial — it's a real model (~80M params for SDXL),
not a wrapper.

**Plan:** This is upstream work. We file it as a HartsyInference issue during phase 1
to maximize lead time. Img2img / inpaint stays disabled in our `IsValidForThisBackend`
until the encoder lands.

**Open question:** can we ship a VAE-encode helper as part of *this* extension as a
stopgap (loading the encoder weights from the same checkpoint, running them on
HartsyInference's IBackend ops)? Probably yes — it'd be a duplicated implementation
that gets deleted once upstream lands.

## R5 — Pipeline ctors aren't uniform (LOW, design tax)

**Risk:** Every HartsyInference pipeline has a different constructor. Our dispatcher
is a switch-on-architecture. If a new pipeline class adds a parameter (e.g. an
optional `RefinerUNet`), we have to update both upstream and our dispatcher.

**Plan:** Treat each pipeline class as an explicit `IPipelineHandler` implementation
in our `Generation/ModelSupport.cs`. Adding a new architecture is N-line code with
no surprise. We monitor HartsyInference upstream for ctor changes.

**Open question:** would it be useful to PR upstream a thin builder pattern
(`new SdxlPipelineBuilder().WithBackend(b).WithText(t)...`)? Maybe, but not
load-bearing.

## R6 — All-in-one checkpoint partitioning (MEDIUM)

**Risk:** Swarm checkpoints are typically all-in-one .safetensors with text-encoder,
UNet, and VAE under conventional but model-family-specific prefixes. Our partition
function has to know each family's prefix conventions. A prefix change in a
community fine-tune (rare but possible) could break loading.

**Plan:** maintain partition rules per architecture in `Generation/ModelSupport.cs`.
Ship a `HartsyInferenceProbeModel` route to surface partition results to users (helps
diagnose "my model didn't load" issues).

**Open question:** does the Comfy backend have an existing prefix-detection
implementation we can mirror? Yes — see
`src/BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorModelSupport.cs` for what it
expects to find. We'd port the recognition heuristics, not the workflow building.

## R7 — Live previews need scheduler hooks (MEDIUM, phase 3)

**Risk:** Mid-generation latent decode for preview images requires HartsyInference to
expose the in-flight latent. Currently `GenerationProgress` carries only
`(Step, TotalSteps, ElapsedMs)`.

**Plan:** Phase 3 upstream PR — extend `GenerationProgress` with an opaque
`Tensor? CurrentLatent` field, populated every N steps (configurable per pipeline).

**Open question:** could we run the partial decode at the application layer by
intercepting the scheduler? Probably not without HartsyInference exposing scheduler
callbacks. Cleanest path is upstream.

## R8 — Threading model (LOW)

**Risk:** HartsyInference pipelines are synchronous. We wrap them in `Task.Run`. If
the same backend is used for two concurrent generations (`MaxUsages > 1`),
HartsyInference may not be thread-safe.

**Plan:** Default `MaxUsages = 1` on our backend. Document that increasing it is at
the user's risk. PR upstream once we know what's safe.

**Open question:** is it safe to run two pipelines simultaneously *if they share an
IBackend*? On CUDA, this means two concurrent kernel queues — likely safe but needs
upstream confirmation.

## R9 — Native dependency packaging (LOW, phase 1)

**Risk:** HartsyInference's CUDA path needs PTX files; Vulkan needs SPIR-V. These
have to be present in the Swarm output directory at runtime. If HartsyInference's
build doesn't auto-copy, our csproj has to.

**Plan:** Verify in phase 1 that `dotnet build` of our csproj copies the assets to
the right place. Add `<Content>` rules if not.

**Open question:** what happens when HartsyInference NuGet packages exist? NuGet
content-files have specific copy semantics that may differ from `<ProjectReference>`
behaviour. Test before phase 2 swap.

## R10 — Model loading time (LOW)

**Risk:** First-time SDXL load might be ~10–30s (parsing safetensors, casting fp16
→ fp32 if needed). Worse on CPU. The Swarm UI may time out before LoadModel returns.

**Plan:** verify load times in phase 2. If too slow, add progress reporting via
`AddLoadStatus()` during `LoadModel`. Swarm shows that text on the backends UI.

**Open question:** can HartsyInference's safetensors loader memory-map without
parsing? Yes (per CLAUDE.md docs — mmap is a design pillar). Confirm cast-on-demand
rather than cast-at-load is the path.

## R11 — Numerical parity will not be perfect (KNOWN)

Different schedulers, different attention implementations, different fp accumulation
orders → different images at the same seed. This is normal for a backend swap.

**Plan:** parity tolerances in [`09-Testing-Strategy.md`](./09-Testing-Strategy.md)
explicitly accept this. Users who want bit-exact reproducibility against Comfy keep
using Comfy.

**Open question:** does HartsyInference promise scheduler-by-scheduler parity with
diffusers? Per their VALIDATION_STRATEGY.md they target Python references — yes,
roughly, within tolerance.

## R12 — User confusion: which backend does what (LOW, UX)

**Risk:** A user with both ComfyUI and HartsyInference backends configured may not
understand which one ran a given generation, especially when output differs.

**Plan:** Swarm core already shows backend ID in image metadata. We ensure our
backend tags itself clearly ("HartsyInference (CUDA, device 0)"). Documentation in
the README explains side-by-side use.

## Open questions list (consolidated)

1. Does HartsyInference use any net10-only APIs that would block multi-targeting?
2. Should LoRA non-destructive merge be done upstream or as a stopgap in this extension?
3. Can we monkey-patch cancellation into HartsyInference's per-step loops without an upstream change?
4. Should VaeEncoder be implemented in this extension as a stopgap or strictly upstream?
5. Are we OK shipping phase 1 with `"comfyui"` feature flag advertised even though it's a slight white-lie? (Plan: yes.)
6. What's HartsyInference's threading guarantee for shared IBackend? Concurrent use safe or not?
7. PTX / SPIR-V copy semantics under `<PackageReference>` (phase 2 swap) — needs validation.
8. Do we add a "Stored Custom Pipelines" feature analogous to Comfy's stored workflows? (Plan: no, out of scope. But user might ask.)
9. Should we support the HartsyInference.Server out-of-process variant as a second backend type later? (Plan: yes, post-v1.)
10. Audio / video / vision modalities — separate extension or expand this one? (Plan: separate.)

## Decisions needed before phase 1 starts

- [ ] Phase 0 path: confirm option (A) — multi-target HartsyInference. PR opened upstream.
- [ ] LoRA cache strategy first cut: confirm option (D) reload-from-disk as phase-2 starting point.
- [ ] Permission default: `POWERUSERS` for use, `ADMINS` for admin routes — confirm with Swarm conventions.
- [ ] Backend type ID: `hartsyinference` (single type, no `_selfstart` / `_api` split). Confirm.
- [ ] Pipeline cache size default: 2 entries. Confirm.
