/**
 * HartsyInference parameter visibility.
 *
 * The "hartsyinference" flag every param carries means "a HartsyInference backend is running", NOT
 * "this model can use this" — so on its own it shows every param under every model. This adds the
 * missing model-aware layer via SwarmUI's featureSetChangers hook (genpage/main.js), the same
 * mechanism core uses for 'sdxl'/'text2video' and that SwarmUI-AudioLab uses for its own params.
 * Core calls reviseBackendFeatureSet() on model selection (gentab/models.js), so no extra wiring.
 *
 * Rules this file follows:
 *  1. Only hartsy_* flags are added/removed. Removing a core flag would undo core's own grant —
 *     removes are applied after adds (see AudioLab's note about 'text2audio').
 *  2. Compat classes are matched with === ; startsWith is used ONLY for the genuinely-prefixed
 *     families. SwarmUI-API-Backends reports compat_class "stable-diffusion" for unclassified API
 *     models, so a loose prefix test would light our params up on those.
 *  3. curCompatClass / curModel are null before the first model selection.
 */

const HartsyParamConfig = {
    /** compat class (exact) -> flag granted for models of that class. */
    exactCompatFlags: {
        'ideogram-4': 'hartsy_ideogram4',
        'ace-step-1_5': 'hartsy_acestep',
        'yue': 'hartsy_yue',
        // Only family declaring BOTH Img2Img and RefEdit, so the only one where picking a mode is a real
        // choice. RefEdit-only families (Boogu, Mage-Flow, OmniGen2) have nothing to choose between.
        'qwen-image': 'hartsy_refedit_choice',
    },

    /** Flags that additionally depend on the checkpoint FILE, not just its compat class. */
    fileFlags: [
        {
            flag: 'hartsy_wan_animate',
            // Core registers no distinct model class for Animate, so the filename is the only signal the
            // browser has. The backend still checks properly via WanVideoRecipe.SupportsFor(checkpointPath),
            // so a renamed checkpoint gets a clean refusal rather than a wrong generation.
            test: (compat, model) => compat.startsWith('wan-2') && model.includes('animate'),
        },
        {
            flag: 'hartsy_audio_ref',
            // MiniMax-H3 ships fl2va (first/last frame) and ref2va (references) as separate checkpoints with
            // byte-identical key sets, so ModelSupport.MiniMaxH3TaskFeatures also has to go by name.
            test: (compat, model) => compat === 'minimax-h3' && !model.includes('fl2va'),
        },
    ],

    /** Every flag this file controls. Anything not granted this pass is removed. */
    get allFlags() {
        return [...Object.values(this.exactCompatFlags), ...this.fileFlags.map(r => r.flag)];
    },

    /** Flags the current model should have. */
    activeFlags(compatClass, modelName) {
        if (!compatClass) {
            return [];
        }
        let model = (modelName || '').toLowerCase();
        let flags = [];
        let exact = this.exactCompatFlags[compatClass];
        if (exact) {
            flags.push(exact);
        }
        for (let rule of this.fileFlags) {
            if (rule.test(compatClass, model)) {
                flags.push(rule.flag);
            }
        }
        return flags;
    },
};

/**
 * Core params that only work on some families, keyed by the engine ImageFeatures/VideoFeatures flag that
 * has to be present. Core shows every one of these for every model; the backend then refuses at generate
 * time (IsValidForThisBackend), which is far too late. The per-architecture feature map comes from
 * HartsyInferenceGetSupportedArchs, so this list never needs to know WHICH families have what.
 *
 * These are core's params, not ours, so they can't be handled by adding/removing a flag — the AudioLab /
 * API-Backends approach of swapping param.feature_flag and restoring it is used instead.
 */
const HartsyCoreGating = {
    /** param id -> engine feature flag it needs. Ids verified against a live ListT2IParams response. */
    requires: {
        'loras': 'lora',
        'lorasectionconfinement': 'lora',
        'loratencweights': 'lora',
        'loraweights': 'lora',
        'controlnetend': 'controlnet',
        'controlnetimageinput': 'controlnet',
        'controlnetmodel': 'controlnet',
        'controlnetpreprocessor': 'controlnet',
        'controlnetpreviewonly': 'controlnet',
        'controlnetstart': 'controlnet',
        'controlnetstrength': 'controlnet',
        'controlnetuniontype': 'controlnet',
        'controlnettwoend': 'controlnet',
        'controlnettwoimageinput': 'controlnet',
        'controlnettwomodel': 'controlnet',
        'controlnettwopreprocessor': 'controlnet',
        'controlnettwostart': 'controlnet',
        'controlnettwostrength': 'controlnet',
        'controlnettwouniontype': 'controlnet',
        'controlnetthreeend': 'controlnet',
        'controlnetthreeimageinput': 'controlnet',
        'controlnetthreemodel': 'controlnet',
        'controlnetthreepreprocessor': 'controlnet',
        'controlnetthreestart': 'controlnet',
        'controlnetthreestrength': 'controlnet',
        'controlnetthreeuniontype': 'controlnet',
        'refinercfgscale': 'refiner',
        'refinercontrolpercentage': 'refiner',
        'refinerdotiling': 'refiner',
        'refinerhypertile': 'refiner',
        'refinermethod': 'refiner',
        'refinermodel': 'refiner',
        'refinersampler': 'refiner',
        'refinerscheduler': 'refiner',
        'refinersteps': 'refiner',
        'refinerupscale': 'refiner',
        'refinerupscalemethod': 'refiner',
        'refinervae': 'refiner',
        'seamlesstileable': 'seamlesstiling',
        'variationseed': 'variationseed',
        'variationseedstrength': 'variationseed',
        'initimagerecompositemask': 'inpaint',
        'maskbehavior': 'inpaint',
        'maskblur': 'inpaint',
        'maskcompositeunthresholded': 'inpaint',
        'maskgrow': 'inpaint',
        'maskimage': 'inpaint',
        'maskshrinkgrow': 'inpaint',
        'savesegmentmask': 'inpaint',
        'useinpaintingencode': 'inpaint',
    },

    /** compat class -> array of lowercase feature names. Populated from the backend, empty until it answers. */
    featuresByArch: null,

    /** Marker flag parked on a param to hide it; nothing ever grants it. */
    BLOCKED: '__hartsy_unsupported__',

    /** True when we know this arch and it lacks the feature. Unknown arch => don't touch anything. */
    lacks(compatClass, feature) {
        let map = this.featuresByArch;
        if (!map || !compatClass || !(compatClass in map)) {
            return false;
        }
        return !map[compatClass].includes(feature);
    },

    apply(compatClass) {
        if (typeof gen_param_types == 'undefined' || !gen_param_types) {
            return;
        }
        for (let param of gen_param_types) {
            let needed = this.requires[param.id];
            if (!needed) {
                continue;
            }
            if (this.lacks(compatClass, needed)) {
                if (!param.hasOwnProperty('original_feature_flag_hartsy')) {
                    param.original_feature_flag_hartsy = param.feature_flag;
                }
                param.feature_flag = this.BLOCKED;
            }
            else if (param.hasOwnProperty('original_feature_flag_hartsy')) {
                param.feature_flag = param.original_feature_flag_hartsy;
                delete param.original_feature_flag_hartsy;
            }
        }
    },

    /** Fetch the per-architecture feature map once the backend can answer. */
    load() {
        genericRequest('HartsyInferenceGetSupportedArchs', {}, data => {
            if (!data || !data.features) {
                return;
            }
            this.featuresByArch = data.features;
            reviseBackendFeatureSet();
        }, 0, () => { /* backend not up yet; the next backends-revised callback retries */ });
    },
};

featureSetChangers.push(() => {
    let compat = currentModelHelper.curCompatClass;
    // Only gate core params while a HartsyInference backend is the one that would serve this. With a Comfy
    // backend running too, Comfy can service LoRAs/ControlNet on families our engine can't, so hiding them
    // would be wrong.
    let hartsyOnly = currentBackendFeatureSet.includes('hartsyinference') && !hasAnyComfyBackend();
    HartsyCoreGating.apply(hartsyOnly ? compat : null);
    let active = HartsyParamConfig.activeFlags(compat, currentModelHelper.curModel);
    let inactive = HartsyParamConfig.allFlags.filter(f => !active.includes(f));
    return [active, inactive];
});

/** True when a real ComfyUI backend is loaded (ours advertises "comfyui" too, so the flag can't answer this). */
function hasAnyComfyBackend() {
    if (typeof backends_loaded == 'undefined' || !backends_loaded) {
        return false;
    }
    return Object.values(backends_loaded).some(b => b.enabled && `${b.type}`.startsWith('comfyui'));
}

backendsRevisedCallbacks.push(() => {
    if (!HartsyCoreGating.featuresByArch) {
        HartsyCoreGating.load();
    }
});
