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
        'minimax-music-3': 'hartsy_minimaxmusic',
        // Long-form chaining (VideoFeatures.LongFormChain). Unlike hartsy_audio_ref below, this is NOT stripped
        // by ModelSupport.MiniMaxH3TaskFeatures for either fl2va or ref2va, so a plain compat-class match is
        // enough: no filename test needed.
        'minimax-h3': 'hartsy_h3_chain',
    },

    /**
     * Flags that depend on the checkpoint, not just its compat class. `features` is the selected model's own
     * feature list from HartsyInferenceGetModelFeatures (its resolved variant), or null until that answers; each
     * rule falls back to the class/name guess it used before so the control does not flicker on first load.
     */
    fileFlags: [
        {
            flag: 'hartsy_refedit_choice',
            // Picking a mode is a real choice only where the variant does BOTH (a Qwen-Image-Edit build). A base
            // Qwen-Image has no reference editing, and RefEdit-only families have nothing to choose between.
            test: (compat, model, features) => features
                ? features.includes('img2img') && features.includes('refedit')
                : compat === 'qwen-image',
        },
        {
            flag: 'hartsy_wan_animate',
            // Animate shares Wan's compat classes; the engine's variant resolver tells it apart from the weights
            // and metadata. The filename guess only covers the moment before that answer arrives.
            test: (compat, model, features) => features
                ? features.includes('drivingvideo')
                : compat.startsWith('wan-2') && model.includes('animate'),
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
    activeFlags(compatClass, modelName, features) {
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
            if (rule.test(compatClass, model, features)) {
                flags.push(rule.flag);
            }
        }
        return flags;
    },
};

/**
 * Per-model features from HartsyInferenceGetModelFeatures. Checkpoints sharing a compat class can do different
 * things (Qwen-Image base vs Edit, Wan vs Wan-Animate), which the per-class map cannot express, so the selected
 * model's own list is fetched once and preferred wherever it is known.
 */
const HartsyModelFeatures = {
    /** "model|class" -> array of lowercase feature names. Keyed on the class too, so re-classing a model in the
     * model editor (base -> Qwen Image Edit) fetches its new answer instead of reusing the old one. */
    byModel: {},

    /** Keys with a request in flight, so a burst of revisions sends one. */
    pending: {},

    key(modelName, arch) {
        return `${modelName}|${arch}`;
    },

    /** The model's features, or null when not fetched yet (callers fall back to the per-class map). */
    get(modelName, arch) {
        let key = this.key(modelName, arch);
        return modelName && key in this.byModel ? this.byModel[key] : null;
    },

    /** Fetch the model's features once, then re-run the feature-set changers with them. */
    request(modelName, arch) {
        let key = this.key(modelName, arch);
        if (!modelName || key in this.byModel || this.pending[key]) {
            return;
        }
        this.pending[key] = true;
        genericRequest('HartsyInferenceGetModelFeatures', { model_name: modelName }, data => {
            delete this.pending[key];
            if (!data || !data.success) {
                return;
            }
            this.byModel[key] = data.features;
            reviseBackendFeatureSet();
        }, 0, () => { delete this.pending[key]; });
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
/** Feature name no model advertises: a param gated on it hides while Hartsy is the only backend option. */
const HARTSY_UNIMPLEMENTED = 'unimplemented';

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
        // Init Image has THREE features that can satisfy it, and the list is not optional: it mirrors the
        // (Img2Img | RefEdit) check IsValidForThisBackend makes on the image side, plus VideoFeatures.InitImage
        // on the video side, because featuresByArch mixes both vocabularies in one map. Dropping any one of the
        // three hides Init Image on a family that serves it — 'refedit' alone for the edit-only image families
        // (Boogu, Mage-Flow, OmniGen2, where the init image IS the reference), 'initimage' alone for every
        // image-to-video family (Wan, MiniMax-H3, Kandinsky5). Measured against a live
        // HartsyInferenceGetSupportedArchs before this list was written.
        //
        // What it hides today: Qwen-Image 2.1 (text-to-image only, while SwarmUI core advertises Init Image for
        // the class), HunyuanVideo, LTX 0.9/2, Lance video — all of which the backend already refuses at
        // generate time, so this only moves the refusal to where the user can see it.
        //
        // Prompt Images is deliberately NOT gated. It would need (IpAdapter | RefEdit | ReferenceImages), and
        // even then Wan-Animate would lose it: Animate shares plain Wan's compat class, whose feature row says
        // nothing about references, and before HartsyModelFeatures answers that row is all there is. Hiding a
        // control that works is worse than a late refusal.
        'initimage': ['img2img', 'refedit', 'initimage'],
        'initimagecreativity': ['img2img', 'refedit', 'initimage'],
        'initimagenoise': ['img2img', 'refedit', 'initimage'],
        'initimageresettonorm': ['img2img', 'refedit', 'initimage'],
        'initimagerecompositemask': 'inpaint',
        // Not read by this backend: Mask Behavior (always plain latent blending), Use Inpainting Encode, Save Segment Mask.
        'maskbehavior': HARTSY_UNIMPLEMENTED,
        'maskblur': 'inpaint',
        'maskcompositeunthresholded': 'inpaint',
        'maskgrow': 'inpaint',
        'maskimage': 'inpaint',
        'maskshrinkgrow': 'inpaint',
        'savesegmentmask': HARTSY_UNIMPLEMENTED,
        'useinpaintingencode': HARTSY_UNIMPLEMENTED,
    },

    /** compat class -> array of lowercase feature names. Populated from the backend, empty until it answers. */
    featuresByArch: null,

    /**
     * compat class -> { samplers: [...], schedulers: [...] }. Populated from the backend alongside featuresByArch.
     * Empty arrays mean the family samples with its own solver and refuses any selection (Wan's UniPC, LTX,
     * Ideogram 4, Lumina2, Lance image), so offering the dropdown would offer only values that get refused.
     */
    samplingByArch: null,

    /**
     * Our music compat classes. Core does not act on the compat class's IsAudioModel flag in JS at all — only
     * AudioLab's own hide-list does, and that is keyed on ITS virtual model classes (acestep_music, yue_music),
     * not on a checkpoint file's compat class. So picking an ACE-Step/MiniMax-Music-3 checkpoint still shows
     * Width, Height, Init Image, ControlNet and the rest of the image-only surface.
     */
    audioArchs: ['ace-step-1_5', 'minimax-music-3'],

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

    /**
     * True when we know this arch and it has none of the listed features. Unknown arch => don't touch anything.
     * A list means "any one of these satisfies the param", mirroring the server-side checks in
     * IsValidForThisBackend: Prompt Images is satisfied by ipadapter OR refedit, Init Image by img2img OR refedit.
     * Hiding on a single feature would hide Init Image on every edit-only family (Boogu, Mage-Flow, OmniGen2),
     * where the init image IS the reference.
     */
    lacks(compatClass, features, modelFeatures) {
        let map = this.featuresByArch;
        let have = modelFeatures || (map && compatClass && compatClass in map ? map[compatClass] : null);
        if (!have) {
            return false;
        }
        return !(Array.isArray(features) ? features : [features]).some(f => have.includes(f));
    },

    /**
     * True when this arch takes no sampler/scheduler selection at all. Unknown arch => don't touch anything, same
     * rule as lacks(). Only the "takes nothing" case hides: a family with a RESTRICTED list (Wan-Animate's unipc)
     * keeps the control visible, because the server-side refusal names the value it does accept, which is more
     * useful than a missing control.
     */
    takesNoSampler(compatClass) {
        let map = this.samplingByArch;
        if (!map || !compatClass || !(compatClass in map)) {
            return false;
        }
        let entry = map[compatClass];
        return entry.samplers.length == 0 && entry.schedulers.length == 0;
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
    shouldHide(compatClass, param, hartsyIsOnlyOption, modelFeatures) {
        if (this.audioArchs.includes(compatClass)
            && (this.audioHideParams.includes(param.id) || this.inHiddenGroup(param))) {
            return true;
        }
        if (!hartsyIsOnlyOption) {
            return false;
        }
        if (param.id == 'sampler' || param.id == 'scheduler') {
            return this.takesNoSampler(compatClass);
        }
        let needed = this.requires[param.id];
        return needed ? this.lacks(compatClass, needed, modelFeatures) : false;
    },

    /** One-shot guard for the deferred re-run scheduled when a foreign marker blocks a param we want. */
    revisePending: false,

    apply(compatClass, hartsyIsOnlyOption, modelFeatures) {
        if (typeof gen_param_types == 'undefined' || !gen_param_types) {
            return;
        }
        let sawForeignMarker = false;
        for (let param of gen_param_types) {
            if (this.shouldHide(compatClass, param, hartsyIsOnlyOption, modelFeatures)) {
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
            this.samplingByArch = data.sampling || null;
            reviseBackendFeatureSet();
        }, 0, () => { /* backend not up yet; the next backends-revised callback retries */ });
    },
};

featureSetChangers.push(() => {
    let hartsyUp = currentBackendFeatureSet.includes('hartsyinference');
    let compat = hartsyUp ? currentModelHelper.curCompatClass : null;
    let modelFeatures = hartsyUp ? HartsyModelFeatures.get(currentModelHelper.curModel, currentModelHelper.curArch) : null;
    if (hartsyUp && compat && !modelFeatures) {
        HartsyModelFeatures.request(currentModelHelper.curModel, currentModelHelper.curArch);
    }
    HartsyCoreGating.apply(compat, !hasAnyComfyBackend(), modelFeatures);
    let active = HartsyParamConfig.activeFlags(currentModelHelper.curCompatClass, currentModelHelper.curModel, modelFeatures);
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
