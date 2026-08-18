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

    /**
     * Our music compat classes. Core does not act on the compat class's IsAudioModel flag in JS at all — only
     * AudioLab's own hide-list does, and that is keyed on ITS virtual model classes (acestep_music, yue_music),
     * not on a checkpoint file's compat class. So picking an ACE-Step/YuE/MusicGen checkpoint still shows
     * Width, Height, Init Image, ControlNet and the rest of the image-only surface.
     */
    audioArchs: ['ace-step-1_5', 'yue', 'musicgen'],

    /**
     * Image-only core params to hide for those. Deliberately shorter than AudioLab's equivalent list:
     * BuildMusicRequest genuinely reads Steps, CFG Scale, Seed and Sigma Shift for ACE-Step, so hiding those
     * would remove working controls.
     */
    audioHideParams: [
        'width', 'height', 'sidelength', 'aspectratio', 'batchsize',
        'initimage', 'initimagecreativity', 'initimageresettonorm', 'initimagenoise',
        'maskimage', 'maskblur', 'maskgrow', 'maskshrinkgrow', 'useinpaintingencode',
        'initimagerecompositemask', 'maskbehavior', 'seamlesstileable', 'clipstopatlayer',
        'vaetilesize', 'vaetileoverlap', 'removebackground', 'automaticvae',
        'modelspecificenhancements', 'fluxguidancescale', 'fluxdisableguidance', 'zeronegative',
    ],

    /** Image/video-only groups to hide for those, matched including inherited parents. */
    audioHideGroups: [
        'resolution', 'refineupscale', 'refinerparamoverrides', 'controlnet', 'controlnettwo', 'controlnetthree',
        'imageprompting', 'initimage', 'freeu', 'regionalprompting', 'segmentrefining', 'segmentparamoverrides',
        'texttovideo', 'imagetovideo', 'advancedvideo', 'videoobscureoptions', 'videoextend', 'seedvr',
        'alternateguidance', 'variationseed', 'restoreupscale', 'wananimate', 'ideogram',
    ],

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

    /** True when this param sits in (or under) one of the groups hidden for audio models. */
    inHiddenGroup(param) {
        for (let group = param.group; group; group = group.parent) {
            if (this.audioHideGroups.includes(group.id)) {
                return true;
            }
        }
        return false;
    },

    /**
     * Whether this param should be hidden.
     *
     * The two halves have different conditions on purpose. Width/height/init-image on a MUSIC checkpoint are
     * meaningless whichever backend runs it, so that half applies whenever our backend is up. The
     * feature-map half is different: it says "our engine's recipe for this family can't do LoRAs", which is
     * only a reason to hide the control when our backend is the one that would serve it — a ComfyUI backend
     * can service LoRAs and ControlNet on families our engine cannot, and Swarm would route there.
     */
    shouldHide(compatClass, param, hartsyIsOnlyOption) {
        if (this.audioArchs.includes(compatClass)
            && (this.audioHideParams.includes(param.id) || this.inHiddenGroup(param))) {
            return true;
        }
        if (!hartsyIsOnlyOption) {
            return false;
        }
        let needed = this.requires[param.id];
        return needed ? this.lacks(compatClass, needed) : false;
    },

    /** One-shot guard for the deferred re-run scheduled when a foreign marker blocks a param we want. */
    revisePending: false,

    apply(compatClass, hartsyIsOnlyOption) {
        if (typeof gen_param_types == 'undefined' || !gen_param_types) {
            return;
        }
        let sawForeignMarker = false;
        for (let param of gen_param_types) {
            if (this.shouldHide(compatClass, param, hartsyIsOnlyOption)) {
                // Never touch a param currently carrying another extension's marker. AudioLab and
                // API-Backends rewrite feature_flag on these same core params with their own save/restore
                // keys, and OUR changer runs FIRST (extension prep order) — so if we blocked it now, the
                // foreign extension's restore would clobber our marker later in this same pass, and if we
                // saved the marker as our "original" we'd restore garbage. Leave it alone and schedule ONE
                // deferred re-run: by then the foreign extension has restored the true original and we can
                // gate it cleanly.
                if (`${param.feature_flag}`.startsWith('__') && param.feature_flag != this.BLOCKED) {
                    sawForeignMarker = true;
                    continue;
                }
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
        if (sawForeignMarker && !this.revisePending) {
            this.revisePending = true;
            setTimeout(() => {
                this.revisePending = false;
                reviseBackendFeatureSet();
            }, 1);
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
    let compat = currentBackendFeatureSet.includes('hartsyinference') ? currentModelHelper.curCompatClass : null;
    HartsyCoreGating.apply(compat, !hasAnyComfyBackend());
    let active = HartsyParamConfig.activeFlags(currentModelHelper.curCompatClass, currentModelHelper.curModel);
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
