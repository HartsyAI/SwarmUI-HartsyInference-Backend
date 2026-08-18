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
    },

    /** Every flag this file controls. Anything not granted this pass is removed. */
    get allFlags() {
        return Object.values(this.exactCompatFlags);
    },

    /** Flags the current model should have. */
    activeFlags(compatClass) {
        if (!compatClass) {
            return [];
        }
        let flag = this.exactCompatFlags[compatClass];
        return flag ? [flag] : [];
    },
};

featureSetChangers.push(() => {
    let active = HartsyParamConfig.activeFlags(currentModelHelper.curCompatClass);
    let inactive = HartsyParamConfig.allFlags.filter(f => !active.includes(f));
    return [active, inactive];
});
