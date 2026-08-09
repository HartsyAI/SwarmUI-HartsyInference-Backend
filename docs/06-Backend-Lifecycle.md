# 06 — Backend Lifecycle

How `HartsyInferenceBackend` (subclass of `AbstractT2IBackend`, in
`Backends/HartsyInferenceBackend.cs`) handles its lifetime and per-request flow. The
class doc comment states its role plainly: it is a **thin mapper**. `HartsyInference.Engine`
owns architecture detection, per-family recipes, pipeline caching, side-model download, and
composition (LoRA / ControlNet / IP-Adapter / refiner / img2img / inpaint / regional). This
class only does four things: resolve a `T2IModel` to a checkpoint + `ModelSpec`, map the
Swarm request onto an Engine request record, bridge progress onto `takeOutput`, and map the
Engine's result back onto a SwarmUI `Image`.

The file is organized under six numbered banners (`// ── 1. Lifecycle ──` through
`// ── 6. Engine auto-update ──`) — search for those rather than line numbers, since line
numbers drift and the banners don't.

## The backend contract (recap)

Real overrides on `AbstractT2IBackend` / `AbstractBackend`, all implemented in this file:

| Member | Purpose |
|--------|---------|
| `Task Init()` | Construct the Engine, validate GPU config, set `Status` |
| `Task Shutdown()` | Tear down the Engine and release device registration |
| `IEnumerable<string> SupportedFeatures` | Returns `DeclaredFeatures` |
| `Task<Image[]> Generate(T2IParamInput)` | Thin wrapper over `GenerateLive` |
| `Task GenerateLive(T2IParamInput, batchId, takeOutput)` | Dispatch, stream progress, yield output |
| `Task<bool> LoadModel(T2IModel, T2IParamInput)` | Architecture check only — no separate load step |
| `bool IsValidForThisBackend(T2IParamInput)` | Refuse requests this backend can't honestly service |
| `Task<bool> FreeMemory(bool systemRam)` | Release the Engine's cached pipelines/weights |
| `volatile string CurrentModelName` | Set on load, cleared on shutdown/free |
| `BackendStatus Status` | See the state-machine section below |

## Settings (`HartsyInferenceBackendSettings`)

Nested `AutoConfiguration` class, persisted to `Data/Backends.fds`. The real fields today:

`ComputeBackend`, `GPU_ID`, `LowVram`, `TextEncoderGpuId`, `VaeGpuId`, `CfgParallelGpuId`,
`DitShardGpuId`, `LmShardGpuId`, `KernelDirectory`, `OverQueue` (int), `Previews` (bool),
`AutoUpdate`.

There is no dtype field and no tile-VAE-threshold field — those existed early on and were
removed because the engine never read them (dtype is per-model engine policy; VAE tiling is
decided from measured VRAM, not a pixel threshold). See
[`07-Parameters-And-Feature-Flags.md`](./07-Parameters-And-Feature-Flags.md) for that removal
note and for how params surface in the Swarm UI generally.

The four GPU-placement fields (`TextEncoderGpuId`, `VaeGpuId`, `CfgParallelGpuId`,
`DitShardGpuId`/`LmShardGpuId`) are documented in depth in
[`15-Two-GPU-Setups.md`](./15-Two-GPU-Setups.md) — what each one buys you, the VRAM-vs-latency
tradeoff, and quality deltas across mixed GPU architectures. This doc covers only how `Init`
turns them into a `PlacementConfig` (next section).

## Init()

1. `EnsureLoggerWired()` bridges the Engine's internal logger into Swarm's `Logs` (idempotent,
   guarded by an `Interlocked.Exchange` flag) so engine diagnostics land in the main log file.
2. `MaybeAutoUpdateEngine()` runs first. If it staged a newer engine build, `Init` sets
   `Status = ERRORED` with a "restart SwarmUI to load it" message and returns immediately —
   the engine is an in-process library, so a fetched update can't hot-swap into the running
   process. See section 6 of the source file for the update/restart-loop-guard mechanics.
3. Resolves `ComputeBackend` and `GPU_ID` into a requested backend string and an optional
   device ordinal, and resolves `LowVram` into a `LowVramMode?` override.
4. Calls `BackendFactory.Validate(...)` on the main device **eagerly, before constructing the
   Engine**. The Engine builds its device lazily on first use, so without this eager check a
   bad `GPU_ID` would report `Status = RUNNING` and only fail mid-generation; validating here
   means a bad config shows up as `ERRORED` at startup instead.
5. Repeats that same eager `BackendFactory.Validate` for each optional placement selector that
   differs from the main ordinal — text-encoder GPU, VAE GPU, CFG-parallel GPU, DiT/LM shard
   GPU — building up a `PlacementConfig` as it goes. `DitShardGpuId` and `LmShardGpuId` share
   one shard-device list; if both are set and point at different ordinals, `Init` sets
   `Status = ERRORED` and refuses rather than picking one silently.
6. Refuses to start if `ComputeBackend` resolves to `cpu` when something other than `cpu` was
   requested — i.e., `'auto'` silently degraded because CUDA wasn't available. The CPU kernels
   are F32-only; loading fp8/bf16 checkpoint weights through them reads past the end of the
   allocation and corrupts the heap. `Init` treats that as a start-up refusal (`ERRORED`, with
   `CudaUnavailableReason()` in the message) rather than a slow-but-working CPU backend. An
   operator who explicitly sets `ComputeBackend = cpu` is exempt — that's an intentional
   override, not a silent degrade.
7. Constructs `_engine = new InferenceEngine(requested, deviceOrdinal, engineOptions)` with the
   assembled `PlacementConfig` and the `LowVram` override.
8. Sets `MaxUsages = 1 + max(-1, OverQueue)` — this is what `BackendHandler` checks to decide
   when this backend is "in use" for routing purposes (mirrors ComfyUI's model: one live
   generation plus `OverQueue` extra requests allowed to queue here rather than route
   elsewhere). `OverQueue = -1` makes `MaxUsages` 0, i.e. a UI-only instance that never
   generates.
9. Calls `RegisterDeviceUsage`, which tracks a live count per `"cuda:N"` / `"vulkan:N"` key
   across all `HartsyInferenceBackend` instances in the process and logs a warning (not a
   refusal) when more than one backend instance shares a device — same-GPU generations run
   serially and both models must co-fit in that GPU's VRAM.
10. Sets `Status = BackendStatus.RUNNING` and logs the ready line. Any exception anywhere in
    `Init` is caught by an outer `try`/`catch` that sets `Status = ERRORED`, records the
    message via `AddLoadStatus`, and rethrows.

## Shutdown()

Sets `Status = DISABLED`, cancels and clears `_cancelCts` if a generation is in flight,
disposes `_engine`, disposes the standalone preprocess/annotator backend if this instance owns
one, decrements this instance's entry in the live-device-count table, and clears
`CurrentModelName`.

## FreeMemory(systemRam)

Calls `_engine.FreeMemory()` to drop cached pipelines/weights, then drops the reference to the
preprocess (ControlNet annotator) backend so it re-borrows fresh from the Engine next use —
that borrow-not-own relationship exists because a second backend on the same device would
otherwise collide with the Engine's per-context CUDA state, and disposing it would evict the
Engine's own resident weights. When `systemRam` is true it also forces a blocking
`GC.Collect()` + `GC.WaitForPendingFinalizers()`. Clears `CurrentModelName`. Returns `false`
only when there was no engine to free (backend never finished `Init`).

## LoadModel(model, input)

Does not perform a load. It checks that `model.ModelClass.CompatClass.ID` is one
`ModelSupport.IsArchitectureSupported` recognizes; if not, it logs a warning and returns
`false` so Swarm can route the pre-warm to a different backend. If supported, it just sets
`CurrentModelName` and returns `true`. There is no separate load step to drive because the
Engine loads (and caches) the pipeline lazily inside its own generate call, keyed on the
checkpoint path plus the request's LoRA/component identity — a separate eager load here would
only fight that cache, not warm it usefully.

## Generate(input) and GenerateLive(input, batchId, takeOutput)

`Generate` is `GenerateLive` with a collector callback that appends any `Image` (or
`T2IEngine.ImageOutput.File`) it receives, then returns the accumulated array.

`GenerateLive` is where everything happens:

- **Serialization.** It first awaits `_genLock` (a `SemaphoreSlim(1,1)`). When `OverQueue > 0`
  the scheduler may dispatch more than one job to this backend at a time; the Engine's caches
  and device are shared across a generation, so concurrent execution would collide. The lock
  makes the over-queue safe by holding extra dispatched jobs here rather than running them
  concurrently — Swarm sees them as queued, not rejected.
- **Cancellation.** Only *after* acquiring the lock does it create
  `_cancelCts = CancellationTokenSource.CreateLinkedTokenSource(input.InterruptToken)` — so the
  per-generation token belongs to whichever job is actually holding the GPU. `input.InterruptToken`
  is the token Swarm's gen-page "stop" button trips (`session.SessInterrupt`); the linked token
  is threaded into every `_engine.*.GenerateAsync`/`RestoreAsync` call, and an
  `OperationCanceledException` bubbling back out is caught and logged as
  `"Generation cancelled by user."` rather than surfaced as an error. There is also a
  `public void RequestCancel()` method that cancels `_cancelCts` directly, but it is a plain
  method (not an override of any base member) and nothing in the extension calls it — the
  actual cancel path is the linked `InterruptToken`, not this method.
- **Dispatch.** Resolves the model's `CompatClass.ID` to a `ModelSupport.Family`, builds a
  `ModelSpec` via `ModelSupport.BuildSpec`, and switches on `family.Kind` to call one of three
  private helpers: `GenerateImage` (→ `_engine.Images.GenerateAsync`), `GenerateVideo`
  (→ `_engine.Video.GenerateAsync`, then frame collection + optional boomerang + muxing), or
  `GenerateMusic` (→ `_engine.Music.GenerateAsync`, wrapped as a WAV `Image` with
  `MediaType.AudioWav`). The optional SeedVR2 restore pass (Video Restore param group) runs
  inside `GenerateImage` and `GenerateVideo` only, when the user selected a restore model —
  `GenerateMusic` has no restore path. Engine result metadata is folded into
  `input.ExtraMeta["hartsy_*"]` inside `GenerateImage` (plus `hartsy_engine_seed`) and
  `GenerateMusic`; `GenerateVideo` does not touch `ExtraMeta`.
- **Out-of-VRAM retry.** If `Dispatch()` throws `OutOfVramException` and cancellation was not
  requested, it logs a warning, calls `FreeMemory(false)`, and retries `Dispatch()` exactly
  once — the assumption being a pipeline cached from an earlier, larger model is holding VRAM
  this request needs. A second `OutOfVramException` (or the first one, if cancellation-excluded
  retry didn't apply) is wrapped in `SwarmReadableErrorException` via `DescribeVramFailure`,
  which reports the requested/available byte counts when known and suggests concrete levers
  (lower resolution/frame-count/batch, set `DitShardGpuId` to pool a second card's VRAM, or set
  `LowVram` to stream instead of holding weights resident) — and states plainly that the retry
  already failed, so this isn't a stale-cache problem.
- **Output.** For each resulting `Image`, calls
  `takeOutput(new T2IEngine.ImageOutput { File = img, IsReal = true, GenTimeMS = totalMs })`.
- **`finally`.** Disposes and clears `_cancelCts`, releases `_genLock` (handing the GPU to the
  next queued job, if any), and deliberately does **not** touch `Status` — the backend stays
  `RUNNING` whether the generation succeeded, failed, or was cancelled, because the backend
  itself is still healthy; only `BackendData.Usages` reflects busy/idle.

## Progress bridge

`BuildProgressBridge` wraps a `PreviewEncoder` (gated by the `Previews` setting) and returns a
custom `InlineProgress<StepPreview>` — not the framework's `Progress<T>` — because `Progress<T>`
posts callbacks to a captured `SynchronizationContext`, which would reorder or drop ticks fired
from the sampler thread. `InlineProgress` just invokes the handler synchronously on whatever
thread reports.

Each tick computes `overall = step / totalSteps`, sends `takeOutput` either the richer preview
`JObject` (when `PreviewEncoder` produced one) or a plain
`{ batch_index, overall_percent, current_percent }` object, and — only on crossing one of 20
5%-wide buckets — also logs an ASCII bar (`RenderProgressBar`) at `Verbose`.

## IsValidForThisBackend(input)

Returns `true` immediately if no model is selected (lets other validators speak). Otherwise it
runs checks in this order, each adding to `input.RefusalReasons` and returning `false` on the
first hit (so the request routes to a Comfy backend if one exists):

1. Architecture support (`ModelSupport.IsArchitectureSupported`).
2. A regex (`UnsupportedPromptSyntax`) over prompt and negative prompt for Comfy-only prompt
   tags (`<object:…>`, `<clear:…>`, `<embed:…>`, `<break>`) that have no Engine conditioning
   equivalent — refused by name rather than fed raw into a tokenizer.
3. The two-stage "generate then animate with a separate video model" flow (Comfy's
   `VideoModel` param on a non-video family) — refused; there's no Engine equivalent.
4. A per-`family.Kind` validator: `ValidateMusic`, `ValidateVideo`, or `ValidateImageFeatures`.
   Each checks the specific composition the request would need (LoRA, ControlNet, IP-Adapter,
   img2img, inpaint, regional, refiner, per-family video features like init/end-frame or
   reference media) against what that family's Engine recipe actually declares supporting, and
   refuses by feature name when the request asks for something the recipe doesn't have.
5. `ValidateComfyOnlyParams` — refuses any request-set param flagged `"comfyui"` that isn't in
   the small `HonoredComfyParams` allow-list (sampler/scheduler/refiner selection, style-model
   and IP-Adapter scheduling knobs), and explicitly refuses raw custom-workflow IR.

Why this backend advertises `"comfyui"` at all, and the honesty-guard design behind step 5, is
covered in [`07-Parameters-And-Feature-Flags.md`](./07-Parameters-And-Feature-Flags.md) — this
section only describes the mechanics of the check itself, not the routing rationale.

## SupportedFeatures / DeclaredFeatures

`SupportedFeatures` just returns the static `DeclaredFeatures` list. It's `static` (and public)
so the extension's own startup self-check can compare it against the feature flags the
registered params actually carry — a param whose `FeatureFlag` isn't covered by this list gets
silently refused by `T2IEngine` with no error naming the param, so keeping the two in sync
matters. What each flag gates and the coexistence design around `"comfyui"` are covered in
[`07-Parameters-And-Feature-Flags.md`](./07-Parameters-And-Feature-Flags.md); this doc only
notes that the list exists and why it's static.

## Backend status state machine

`BackendStatus` has six values (`WAITING`, `LOADING`, `IDLE`, `RUNNING`, `ERRORED`,
`DISABLED`), but this backend's own code only ever sets three of them: `RUNNING` (end of a
successful `Init`), `ERRORED` (any `Init` failure), and `DISABLED` (`Shutdown`). `WAITING` is
`BackendHandler`'s own default before `Init` runs; `LOADING` and `IDLE` are never set by this
class at all — there's no separate load phase to occupy `LOADING` (see `LoadModel` above), and
no recoverable-but-unavailable condition that would call for `IDLE`.

The distinction matters because the two are easy to conflate and Swarm's routing treats them
very differently (confirmed against `src/Backends/BackendHandler.cs`):

- **`RUNNING` = alive and ready to accept generations.** This is the resting healthy state.
  `BackendHandler`'s dispatch-candidate filter and its model-load filter both require
  `Status == BackendStatus.RUNNING` (`BackendHandler.cs`, the `EnumerateT2IBackends`/candidate
  selection around lines 761 and 975) — a non-`RUNNING` backend is never picked for a new job.
  A live generation does **not** move status away from `RUNNING` — utilization is tracked
  separately via `BackendData.Usages` against `MaxUsages` (the `CheckIsInUse*` predicates gate
  "is this backend already full" on `Usages`/`MaxUsages`, separately from that dispatch filter).
- **`IDLE` = alive but currently unavailable.** `BackendHandler.FeaturesSupported` explicitly
  excludes `IDLE` backends when aggregating which feature flags are currently servicable, and
  `IDLE` is checked as its own "some backend is present but not usable" bucket for status
  reporting. A backend sitting at `IDLE` is skipped for routing even though it still exists —
  this fits a backend with a remote endpoint that can go down and come back, which this
  in-process Engine backend has no equivalent of, hence it never sets `IDLE`.

A failed generation inside `GenerateLive` leaves `Status` at `RUNNING` on purpose: the backend
itself is still healthy, only that one request failed (see the `finally` block's comment in the
source). `ERRORED` is reserved for conditions that make the whole backend unusable — a bad GPU
config, an unavailable CUDA device, or a staged engine update needing a restart — all of which
are detected in `Init`, not per-generation.
