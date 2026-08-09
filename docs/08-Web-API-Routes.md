# 08 — Web API Routes

Extra HTTP routes the extension adds to Swarm's WebAPI. Registered in
`WebAPI/HartsyInferenceWebAPI.cs`. Pattern mirrors `ComfyUIBackend/ComfyUIWebAPI.cs`.

## Why we need so few

The Comfy extension has many extra routes because it's *bridging two processes*:
saving/reading/listing/deleting workflows, redirecting to the Comfy process
(`ComfyBackendDirect/{*Path}`), returning generated workflow JSON, installing
custom node packs, extracting LoRAs, compiling TensorRT engines. We're
in-process and have no workflow IR — most of those concerns vanish.

Our routes are about diagnostics, cache management, and architecture coverage.
All of them are read-only except `HartsyInferenceClearCache`.

## Registration

```csharp
public static void Register()
{
    API.RegisterAPICall(HartsyInferenceGetSupportedArchs, false, HartsyInferencePermissions.PermUseHartsyInference);
    API.RegisterAPICall(HartsyInferenceProbeModel, false, HartsyInferencePermissions.PermUseHartsyInference);
    API.RegisterAPICall(HartsyInferenceListLoadedPipelines, false, HartsyInferencePermissions.PermAdminHartsyInference);
    API.RegisterAPICall(HartsyInferenceGetDeviceInfo, false, HartsyInferencePermissions.PermAdminHartsyInference);
    API.RegisterAPICall(HartsyInferenceClearCache, true, HartsyInferencePermissions.PermAdminHartsyInference);
}
```

The boolean is `isModifying` (true marks the call as a mutation).
`GetSupportedArchs` and `ProbeModel` need only `use_hartsyinference`
(default: power users); the other three need `admin_hartsyinference` (default:
admins).

## Routes

### `HartsyInferenceGetSupportedArchs`

`Task<JObject> HartsyInferenceGetSupportedArchs(Session session)`

Lists every compat class `ModelSupport` knows about, split into what the
engine can drive today and what's mapped but not yet registered.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceGetSupportedArchs` |
| Permission | `use_hartsyinference` |
| Mutating | false |

**Returns:**
```json
{
  "success": true,
  "supported": ["flux-1", "stable-diffusion-xl-v1", "z-image"],
  "pending": {
    "hunyuan-video": "Architecture 'hunyuan-video' maps to HartsyInference family 'hunyuan-video', but no video recipe is registered for it in this engine build. Currently drivable: wan-22-5b, wan-21-1_3b, ... Use the ComfyUI backend for this architecture in the meantime."
  }
}
```

`supported` holds compat-class IDs (`ModelSupport.SupportedArchitectures`),
not the engine's internal family IDs — a mapped family only counts as
supported once `RecipeRegistry`/`VideoRecipeRegistry` actually has a recipe
for it, checked live at call time. `pending` is an object, compat class ID to
the `WhyNotSupported` reason string, for classes that are mapped to a family
but have no registered recipe yet. The exact contents of both depend on which
recipes are registered in the running build; see `Generation/ModelSupport.cs`.

### `HartsyInferenceProbeModel`

`Task<JObject> HartsyInferenceProbeModel(Session session, string model_name)`

Answers "will HartsyInference run this model, and how?" for a model already
known to Swarm's `Stable-Diffusion` model set, without loading it.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceProbeModel` |
| Permission | `use_hartsyinference` |
| Mutating | false |

**Inputs:**
```json
{
  "session_id": "...",
  "model_name": "sd_xl_base_1.0.safetensors"
}
```

**Returns (success):**
```json
{
  "success": true,
  "model_name": "sd_xl_base_1.0.safetensors",
  "arch_id": "stable-diffusion-xl-v1-base",
  "compat_class": "stable-diffusion-xl-v1",
  "state": "supported",
  "reason": "HartsyInference can generate with this model."
}
```

`state` is one of `supported`, `pending`, `unsupported`. When not supported,
`reason` is `ModelSupport.WhyNotSupported(compat_class)` — the same string
that ends up in `GetSupportedArchs`'s `pending` map.

**Returns (failure)** — `model_name` missing, the `Stable-Diffusion` model
set isn't registered, or the named model doesn't exist:
```json
{ "success": false, "error": "Model 'nonexistent.safetensors' not found." }
```

### `HartsyInferenceListLoadedPipelines`

`Task<JObject> HartsyInferenceListLoadedPipelines(Session session)`

Snapshots every live `HartsyInferenceBackend` instance: its status and
configured device settings, plus current model and Swarm's own usage counter.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceListLoadedPipelines` |
| Permission | `admin_hartsyinference` |
| Mutating | false |

**Returns:**
```json
{
  "success": true,
  "backends": [
    {
      "backend_id": 0,
      "status": "RUNNING",
      "current_model": "sd_xl_base_1.0.safetensors",
      "compute_backend": "auto",
      "gpu_id": "0",
      "text_encoder_gpu_id": "",
      "vae_gpu_id": "",
      "cfg_parallel_gpu_id": "",
      "dit_shard_gpu_id": "",
      "usages": 3
    }
  ]
}
```

`status` is the backend's `.Status.ToString()`, one of Swarm's `BackendStatus`
enum values: `DISABLED`, `ERRORED`, `WAITING`, `LOADING`, `IDLE`, `RUNNING`.
`usages` is Swarm's own `T2IBackendData.Usages` counter, passed through
verbatim.

### `HartsyInferenceGetDeviceInfo`

`Task<JObject> HartsyInferenceGetDeviceInfo(Session session)`

Reports each live backend's *configured* compute target. This is not a
hardware scan — the engine doesn't expose device enumeration ahead of backend
construction — so it's the same settings surface as
`HartsyInferenceListLoadedPipelines` minus `current_model`/`usages`.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceGetDeviceInfo` |
| Permission | `admin_hartsyinference` |
| Mutating | false |

**Returns:**
```json
{
  "success": true,
  "devices": [
    {
      "backend_id": 0,
      "compute_backend": "auto",
      "gpu_id": "0",
      "text_encoder_gpu_id": "",
      "vae_gpu_id": "",
      "cfg_parallel_gpu_id": "",
      "dit_shard_gpu_id": "",
      "status": "RUNNING"
    }
  ]
}
```

### `HartsyInferenceClearCache`

`Task<JObject> HartsyInferenceClearCache(Session session, int backend_id = -1, bool free_system_ram = false)`

Frees each matching backend's resident pipeline via `backend.FreeMemory(...)`.
The only mutating route.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceClearCache` |
| Permission | `admin_hartsyinference` |
| Mutating | true |

**Inputs:**
```json
{
  "session_id": "...",
  "backend_id": -1,
  "free_system_ram": false
}
```

`backend_id = -1` (the default) clears every HartsyInference backend;
otherwise only the backend with that ID. `free_system_ram` is passed through
to `FreeMemory` to also drop CPU-side copies. There's no per-model eviction —
`FreeMemory` clears a backend's whole resident pipeline, not one entry. A
`backend_id` that matches nothing still returns success with
`backends_cleared: 0`.

**Returns:**
```json
{ "success": true, "backends_cleared": 1 }
```

## Routes we explicitly do NOT add

Comparing against `ComfyUIBackend/ComfyUIWebAPI.cs`'s registrations:

- `ComfySaveWorkflow`, `ComfyReadWorkflow`, `ComfyListWorkflows`,
  `ComfyDeleteWorkflow` — we have no workflow IR to save/read/list/delete
- `ComfyBackendDirect/{*Path}` passthrough — no separate Comfy process to
  redirect to
- `ComfyGetGeneratedWorkflow` — no JSON IR to return
- `ComfyInstallFeatures` — no custom node packs
- `ComfyListTorchInstalls`, `ComfyUpdateTorch`, `ComfyGetNodeTypesForBackend`
  — no Python/torch install and no node types to enumerate
- `DoLoraExtractionWS` (diff two checkpoints, write a LoRA) — not yet; open
  item in [`11-Comfy-Parity-Punchlist.md`](./11-Comfy-Parity-Punchlist.md)
  (Tier 3)
- `DoTensorRTCreateWS` — not yet; same punchlist tier

## Architecture coverage

`GetSupportedArchs` and `ProbeModel` both read from
`Generation/ModelSupport.cs`, the compat-class-to-engine-family mapping table.
It's the whole of the extension's architecture knowledge: whether a mapped
family is actually drivable is asked of the engine's `RecipeRegistry` /
`VideoRecipeRegistry` at call time, not hard-coded here.
