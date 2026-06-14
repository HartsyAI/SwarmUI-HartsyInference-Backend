# 08 — Web API Routes

Extra HTTP routes the extension adds to Swarm's WebAPI. Registered in
`WebAPI/HartsyInferenceWebAPI.cs`. Pattern mirrors `ComfyUIBackend/ComfyUIWebAPI.cs`.

## Why we need so few

The Comfy extension has many extra routes because it's *bridging two processes*:
saving workflows, listing nodes, proxying to Comfy, installing custom nodes,
running TensorRT compiles. We're in-process and have no workflow IR — most of
those concerns vanish.

Our routes are about diagnostics, cache management, and device discovery.

## Registration

```csharp
public static class HartsyInferenceWebAPI
{
    public static void Register()
    {
        API.RegisterAPICall<JObject>(HartsyInferenceProbeModel,            true,  HartsyInferencePermissions.PermAdminHartsyInference);
        API.RegisterAPICall<JObject>(HartsyInferenceListLoadedPipelines,   false, HartsyInferencePermissions.PermAdminHartsyInference);
        API.RegisterAPICall<JObject>(HartsyInferenceClearCache,            true,  HartsyInferencePermissions.PermAdminHartsyInference);
        API.RegisterAPICall<JObject>(HartsyInferenceGetDeviceInfo,         false, HartsyInferencePermissions.PermAdminHartsyInference);
        API.RegisterAPICall<JObject>(HartsyInferenceGetSupportedArchs,     false, HartsyInferencePermissions.PermUseHartsyInference);
    }
}
```

The boolean is `isModifying` (true = mutation; affects cache invalidation). The
`PermInfo` is the permission required to call.

## Routes

### `HartsyInferenceProbeModel`

Diagnose a checkpoint without loading it.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceProbeModel` |
| Permission | `admin_hartsyinference` |
| Mutating | true (touches disk) |

**Inputs:**
```json
{
  "session_id": "...",
  "model_path": "Models/Stable-Diffusion/myCheckpoint.safetensors"
}
```

**Returns:**
```json
{
  "success": true,
  "detected_arch": "stable-diffusion-xl-v1-base",
  "tensor_count": 2738,
  "fp_dtype": "fp16",
  "approx_size_mb": 6938,
  "components_found": ["text_encoder_l", "text_encoder_g", "unet", "vae"],
  "components_missing": [],
  "warnings": []
}
```

Used by the Server → Backends UI to validate a model before the user picks it.
Implementation: open with `SafeTensorsLoader`, inspect tensor key prefixes,
match against the architecture detection table from
[`05-Pipeline-Translation.md`](./05-Pipeline-Translation.md).

### `HartsyInferenceListLoadedPipelines`

Show what's currently in the pipeline cache (per backend instance).

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceListLoadedPipelines` |
| Permission | `admin_hartsyinference` |
| Mutating | false |

**Returns:**
```json
{
  "backends": [
    {
      "backend_id": 0,
      "compute": "cuda",
      "device_ordinal": 0,
      "loaded_pipelines": [
        {
          "model_name": "sd_xl_base_1.0.safetensors",
          "arch": "stable-diffusion-xl-v1-base",
          "vram_estimate_mb": 7300,
          "last_used_unix": 1714328400,
          "lora_fingerprint": "sha256:..."
        }
      ]
    }
  ]
}
```

### `HartsyInferenceClearCache`

Drop the pipeline cache, free VRAM. Useful when debugging or before a model swap.

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
  "backend_id": 0,
  "model_name": null
}
```

If `model_name` is null, clears the entire cache for the given backend; otherwise
evicts just that entry.

**Returns:**
```json
{ "success": true, "evicted_count": 2 }
```

### `HartsyInferenceGetDeviceInfo`

Enumerate available compute devices. Used to populate the "Compute Backend" dropdown
in the Backend settings UI.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceGetDeviceInfo` |
| Permission | `admin_hartsyinference` |
| Mutating | false |

**Returns:**
```json
{
  "cuda_available": true,
  "cuda_devices": [
    { "ordinal": 0, "name": "NVIDIA GeForce RTX 4090", "vram_mb": 24576, "compute_cap": "8.9" }
  ],
  "vulkan_available": true,
  "vulkan_devices": [
    { "ordinal": 0, "name": "NVIDIA GeForce RTX 4090", "vram_mb": 24576, "fp16_support": true }
  ],
  "cpu_available": true
}
```

### `HartsyInferenceGetSupportedArchs`

Used by the model dropdown to grey out architectures we don't yet handle. Lower
permission level so any user with `use_hartsyinference` can call it.

| | |
|---|---|
| Method | POST |
| Path | `/API/HartsyInferenceGetSupportedArchs` |
| Permission | `use_hartsyinference` |
| Mutating | false |

**Returns:**
```json
{
  "supported": [
    "stable-diffusion-v1",
    "stable-diffusion-xl-v1-base",
    "Flux.1-dev",
    "Flux.1-schnell"
  ],
  "planned": [
    "stable-diffusion-v3-medium",
    "Flux.2-dev"
  ]
}
```

## Routes we explicitly do NOT add

- `SaveWorkflow`, `LoadWorkflow`, `ListWorkflows`, `DeleteWorkflow` — we have no
  workflow IR
- `ComfyBackendDirect/{*Path}` passthrough — no Comfy to passthrough to
- `GetGeneratedWorkflow` — no JSON IR to return
- `InstallFeatures` — no node packs
- `DoLoraExtractionWS` — out of scope (training)
- `DoTensorRTCreateWS` — out of scope

## How tabs / UI consume these

If we ever add a Server → Tab for "HartsyInference Diagnostics", it lives at
`Tabs/Server/HartsyInference Diagnostics.html` and calls these routes via Swarm's
`genericRequest('HartsyInferenceListLoadedPipelines', ...)` JS helper.

For v1, the routes are mostly there for command-line debugging (curl / browser-tab)
and for the standard Backend-config UI to query device info. No custom tab is required.
