# 00 — Overview

## What we're building

A SwarmUI extension that registers a backend type (`hartsyinference_selfstart`) which
performs Stable Diffusion / SDXL / Flux / SD3 / etc. inference **in-process, in pure C#**
by calling the [HartsyInference](https://github.com/Hartsy/HartsyInference) library.

The user clicks **Generate** in Swarm. The request flows from Swarm's `T2IAPI` →
`BackendHandler` → our `HartsyInferenceBackend.GenerateLive()` → HartsyInference pipeline
→ image bytes → back to the browser. **No subprocess. No HTTP. No Python. No ComfyUI.**

## Why we're building it

Today Swarm's default backend is ComfyUI: a separate Python process that Swarm shells
out to via HTTP + WebSocket. That works, but it carries:

- A ~3 GB Python install (Comfy + torch + xformers + custom node packs)
- Cross-process JSON serialization for every generation
- A node graph IR that Swarm has to translate to and Comfy has to interpret
- Custom-node sprawl (IPAdapter, ControlNet aux, was-node-suite, etc.) that drift independently of Swarm
- Startup time dominated by venv + torch initialization
- A second model loader / VRAM owner that Swarm can't introspect cleanly

A pure-C# backend collapses all of that into one process: Swarm and inference share
the same heap, the same model cache, the same image objects. Swarm's `T2IParamInput`
becomes a HartsyInference `TextToImageRequest` directly, with no JSON in the middle.

## Goals

1. **Functional parity with the ComfyUI backend** for the common path: text-to-image,
   img2img, inpaint, LoRA, ControlNet, refiners — across SD1.5, SDXL, SD3, Flux on
   models the HartsyInference team has already brought up.
2. **In-process execution** — no Python, no subprocess, no IPC.
3. **Match Swarm's existing UX** — the user sees the same parameters, the same
   `Generate` button, the same image gallery. The backend swap should be invisible.
4. **Multi-backend coexistence** — users can run HartsyInference and ComfyUI side-by-side
   on the same Swarm instance and pick per-generation. (Same way the API Backends
   extension coexists with Comfy today.)
5. **Capability advertisement via feature flags** so Swarm's existing UI param-gating
   works without modification.

## Non-goals (for v1)

- **Replacing the workflow editor tab.** ComfyUI's workflow editor is a graph UI. We
  don't need one to start. Power users who want graph editing can keep using the
  ComfyUI backend in parallel.
- **Replacing every esoteric Comfy feature.** TensorRT compilation, custom node
  packs, the `Stored Custom Workflows` system — all out of scope until parity on the
  common path lands.
- **Building HartsyInference itself.** This extension consumes it; it doesn't extend it.
  Bugs and missing primitives in HartsyInference are upstream issues, not extension
  issues.
- **Audio / video / vision modalities.** HartsyInference supports these (Whisper, Kokoro,
  YOLO, LTX-Video) but Swarm's backend abstraction is `AbstractT2IBackend` — image
  generation. Audio/video can be a separate extension later.

## Definition of done for v1

The extension is "done" enough to ship when:

- [ ] A user with a clean Swarm install can pick "HartsyInference" as their backend
- [ ] Pick an SDXL .safetensors checkpoint from their `Models/` folder
- [ ] Type a prompt, hit Generate, and get an image
- [ ] LoRAs from `Models/Lora/` apply correctly
- [ ] Live progress ticks in the Swarm UI (% complete, current step)
- [ ] Cancel-mid-generation works
- [ ] Memory is reclaimed cleanly on model swap
- [ ] At least SDXL + SD1.5 + Flux schnell pass a parity test against the same model
      run through ComfyUI (within a documented numerical tolerance)

Everything past that is incremental milestones — see
[`03-Implementation-Roadmap.md`](./03-Implementation-Roadmap.md).

## Out-of-scope but adjacent

- **Audio/video extensions.** Separate extension repos that mount HartsyInference's
  Audio/Vision/Video pipelines onto Swarm's eventual audio/video tabs. Not this repo.
- **Hosted HartsyInference.Server backend.** HartsyInference will ship an
  OpenAI-compatible HTTP server eventually. When that lands, we *could* add a second
  backend type (`hartsyinference_remote`) that talks to it over HTTP — useful for
  someone running HartsyInference on a separate inference box. Not v1.
- **Quantized / GGUF support.** HartsyInference has a GGUF loader; routing GGUF
  checkpoints into Flux / SDXL pipelines is a HartsyInference milestone, not a
  Swarm-side concern.

## How to read these docs

- Start with [`01-Architecture.md`](./01-Architecture.md) for the component diagram.
- Skim [`02-Comfy-Feature-Parity-Matrix.md`](./02-Comfy-Feature-Parity-Matrix.md) to
  see the feature surface we're targeting.
- [`03-Implementation-Roadmap.md`](./03-Implementation-Roadmap.md) is the milestone
  plan.
- [`04-HartsyInference-Integration.md`](./04-HartsyInference-Integration.md) is the
  contract with the upstream library — read this before writing any consumption code.
- [`10-Risks-And-Open-Questions.md`](./10-Risks-And-Open-Questions.md) is the list of
  known unknowns. The .NET 10 vs .NET 8 TFM issue is the biggest one.
