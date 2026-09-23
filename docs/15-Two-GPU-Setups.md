# Two-GPU setups

Which knob to reach for depends on what you are short of. There are two different problems and they
want opposite configurations, so picking by "I have two GPUs" rather than by symptom is how people
end up slower than with one card.

**Every GPU number in this extension is a CUDA ordinal, not an `nvidia-smi` index.** CUDA orders
fastest-first by default; `nvidia-smi` orders by PCI bus. On a 4090 + 3060 box they disagree — CUDA
ordinal 0 is the 4090 while `nvidia-smi` index 0 is the 3060. Check with
`nvidia-smi --query-gpu=index,name --format=csv` and then invert it in your head, or pin
`CUDA_DEVICE_ORDER=PCI_BUS_ID` on the service to make them agree.

## Posture A — one generation that doesn't fit

Symptom: out-of-VRAM refusals, or a model that streams weights and crawls.

- `DitShardGpuId` = the second ordinal. The denoiser's block loop is **split** across both cards, so
  their VRAM pools. This is a capacity win, not a speed win — the two halves run in sequence, and if
  the cards have no P2P path between them the hand-off stages through host RAM, which costs real
  time per step.
- Leave the second backend **disabled**. A second backend on the shard card fights the shard for the
  same VRAM.
- Not combinable with `CfgParallelGpuId` — those are two different uses of a second card (pooling
  vs replication) and the backend refuses to start with both set.

## Posture B — several small jobs

Symptom: jobs queue behind each other while a card sits idle.

- Two backends, one per GPU, **no placement knobs set**.
- `OverQueue = 0`, so a second job spills to the idle backend immediately instead of queuing behind
  the first. Among backends that can take a job, SwarmUI's scheduler prefers the least-busy one, and
  loads a new model on the one idle longest; it never looks at VRAM itself. This extension adds that
  (next section), so on unequal cards a big model is not sent to the small one just because it was free.

## Memory-aware routing (automatic)

Before SwarmUI picks a backend, the extension asks each GPU's engine whether the request fits, from
the checkpoint header alone (read once per model, then cached). The answer follows each backend's
`LowVram` tier, with a per-generation **VRAM Mode** taking precedence:

| Effective tier | What routing does |
|---|---|
| Performance | Nothing. You size your own workloads; the backend takes whatever it is sent. |
| Auto / Balanced | Prefer a card that holds the model fully resident. If none can, use the largest card that can stream it. A smaller card waits rather than streaming slowly. |
| Aggressive / Maximum | You chose streaming, so any card that can stream the model takes it. |

If the estimate says no card fits, the largest card is still tried; the engine's own pre-flight then
gives the exact refusal. Only backends SwarmUI can currently dispatch to count, and a job never waits
on a preferred card that has since been disabled or reloaded. If a card actually runs out of memory
(after the free-and-retry), the job is redirected to a larger card when one exists. For the next 30
minutes, jobs for that model with the same shape (frame count left default or set, refiner upscale,
LoRA and ControlNet counts, per-generation VRAM levers) at that size or larger skip that card and any
equally configured card no larger. Changing the backend's settings clears its records.
Pinning a job with **Exact Backend ID** bypasses routing entirely.
The routing decision for each GPU is logged at Verbose level as `Fit '<model>' on <GPU>`.
To check routing on a two-card box, queue two jobs at once, with one deliberately too big for the
small card: the small-card refusal, the wait and any redirect only show up under queue load, never on
a single manual run. A Verbose `has no answer for a request` line means a request reached a backend
without being prepared, which would turn routing off for it.
Wan video is sized with its pipeline's own formulas; other families use header weights plus a
generic allowance, so their routing is coarser.

## Component placement (composes with either posture)

`TextEncoderGpuId` and `VaeGpuId` move one component off the main card. Useful when a large DiT
should stay resident and the encoder or the full-res decode would otherwise force an evict and
re-upload every generation. When a component is placed off the primary card its weights **stay
resident between generations** — you deliberately gave it a card, so nothing there is competing with
the DiT for the space. They are released on model switch, `FreeMemory`, and shutdown.

Wired for Wan and MiniMax-H3. On H3 the encoder is ~15 GB and the two VAEs about 5.5 GB together,
so on a 12 GB second card the VAE fits comfortably and the encoder does not (it will load, but it
streams).

**Mixed-architecture caveat.** Moving a component to a different GPU architecture changes its
output. Measured on a 4090 + 3060 pair with MiniMax-H3:

| Moved | Effect on the denoise | Effect on pixels |
|---|---|---|
| VAE (`VaeGpuId`) | none — latents cross as host tensors, relL2 `0.000e+00` | a handful of 1/255 steps from the decode kernels |
| Text encoder (`TextEncoderGpuId`) | different embeddings → a different sample | same scene and prompt adherence, different frame |

Both are deterministic run-to-run. Neither is a bug — it is the same arithmetic on different silicon.
If you need a seed to reproduce exactly, keep the component on the card that produced it.

## When it refuses

An out-of-VRAM refusal from this backend names the geometry, what it needed, what was free, which
card fell short, and the longest clip that *would* fit at that resolution. It has already freed
cached models and retried once, so a refusal is not a stale-cache problem — believe the numbers.

## For contributors

Prefer core's existing params, groups and feature flags over inventing extension-local ones, and
read `src/BuiltinExtensions/ComfyUIBackend` for the canonical pattern before adding UI surface for a
new model family. Two concrete examples of that principle in this extension:

- Reference images/video/audio ride core's `PromptImages`/`PromptAudios`/`PromptVideos` — the
  prompt box's own drag/paste carriers — rather than bespoke controls, which is exactly what the
  reference workflow consumes.
- `T2IEngine.DisregardedFeatureFlags` (`text2video`, `text2audio`, …) are UI-visibility flags only
  and never gate a backend. Flags outside that set **do** gate: a param tagged with one the backend
  doesn't advertise makes every job using it refuse, naming neither the param nor the flag. A
  startup self-check in `SwarmUIHartsyInference` catches that case for our own params.
