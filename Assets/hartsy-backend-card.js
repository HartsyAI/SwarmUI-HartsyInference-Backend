/**
 * Organizes the HartsyInference backend card that core renders under Server > Backends, without replacing core code:
 * - Groups the fields into Basics, Multi-GPU and Advanced sections, each with a link to its guide.
 * - Shows plain labels, with the saved name and a guide link in each tooltip. Saved names never change.
 * - Relists GPU when Compute Backend changes, from HartsyInferenceListGpus.
 * - Keeps a saved value a dropdown does not list, so the next Save does not overwrite it.
 * - Marks the main GPU in the second-GPU lists, and blocks and explains the pair the backend refuses.
 * Core rebuilds a card when it is saved or its settings change, so a MutationObserver re-runs this. An organized card gets no writes.
 */
class HartsyBackendCard {

    /** Sets up the card layout. */
    constructor() {
        this.typeId = 'hartsyinference';
        this.guideRoot = 'https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend/blob/main/docs/guides/';
        /** Sections in card order. Basics starts open; the others open by themselves when a field in them is not at its default. */
        this.sections = [
            { key: 'basics', title: 'Basics', guide: '03-Backend-Settings.md#basics', fields: ['ComputeBackend', 'GPU_ID', 'LowVram', 'Previews'] },
            { key: 'multigpu', title: 'Multi-GPU', guide: '05-Multi-GPU.md', fields: ['TextEncoderGpuId', 'VaeGpuId', 'DitShardGpuId', 'CfgParallelGpuId'] },
            { key: 'advanced', title: 'Advanced', guide: '03-Backend-Settings.md#advanced', fields: ['OverQueue', 'AutoUpdate', 'KernelDirectory'] }
        ];
        /** Label, guide section and default value per saved field name. */
        this.fields = {
            ComputeBackend: { label: 'Compute Backend', guide: '03-Backend-Settings.md#compute-backend', default: 'auto' },
            GPU_ID: { label: 'GPU', guide: '03-Backend-Settings.md#gpu', default: '0' },
            LowVram: { label: 'VRAM Mode', guide: '03-Backend-Settings.md#vram-mode', default: 'Auto' },
            Previews: { label: 'Previews', guide: '03-Backend-Settings.md#previews', default: 'true' },
            TextEncoderGpuId: { label: 'Text Encoder GPU', guide: '05-Multi-GPU.md#text-encoder-gpu', default: '' },
            VaeGpuId: { label: 'VAE GPU', guide: '05-Multi-GPU.md#vae-gpu', default: '' },
            DitShardGpuId: { label: 'Model Split GPU', guide: '05-Multi-GPU.md#model-split-gpu', default: '' },
            CfgParallelGpuId: { label: 'CFG Parallel GPU', guide: '05-Multi-GPU.md#cfg-parallel-gpu', default: '' },
            OverQueue: { label: 'Queue Size', guide: '03-Backend-Settings.md#queue-size', default: '1' },
            AutoUpdate: { label: 'Engine Updates', guide: '03-Backend-Settings.md#engine-updates', default: 'false' },
            KernelDirectory: { label: 'Kernel Folder', guide: '03-Backend-Settings.md#kernel-folder', default: '' }
        };
        /** Fields that place part of a generation on a second CUDA GPU. */
        this.secondGpuFields = ['TextEncoderGpuId', 'VaeGpuId', 'DitShardGpuId', 'CfgParallelGpuId'];
        /** HartsyInferenceListGpus answers by Compute Backend value. Null marks a failed request, so it is not retried on every poll. */
        this.gpuLists = {};
        /** Compute Backend values with a list request in flight. */
        this.gpuPending = {};
        /** Sections the user opened or closed, by backend id and section key, so a rebuilt card keeps them. */
        this.openState = {};
        /** Reentrancy guard so our own DOM writes don't recurse through the MutationObserver. */
        this.enhancing = false;
    }

    /** Hooks core's re-render points. Called once the session is ready. */
    init() {
        if (typeof backendsRevisedCallbacks == 'undefined') {
            return;
        }
        backendsRevisedCallbacks.push(() => this.enhanceAllCards());
        let list = document.getElementById('backends_list');
        if (list) {
            new MutationObserver(() => this.onListMutated()).observe(list, { childList: true, subtree: true });
        }
        document.addEventListener('change', e => this.onFieldChanged(e));
        this.enhanceAllCards();
    }

    /** Re-runs enhancement after core changes the backends list, guarding against our own writes. */
    onListMutated() {
        if (this.enhancing) {
            return;
        }
        this.enhancing = true;
        try {
            this.enhanceAllCards();
        }
        finally {
            this.enhancing = false;
        }
    }

    /** Organizes each newly rendered HartsyInference card. */
    enhanceAllCards() {
        if (typeof backends_loaded == 'undefined') {
            return;
        }
        for (let backend of Object.values(backends_loaded)) {
            if (backend.type != this.typeId) {
                continue;
            }
            let cardBody = document.getElementById(`backend-card-${backend.id}`)?.querySelector('.card-body');
            if (!cardBody || cardBody.classList.contains('hartsy-card-organized')) {
                continue;
            }
            cardBody.classList.add('hartsy-card-organized');
            this.keepSavedValues(cardBody, backend);
            this.organize(cardBody, backend);
            this.applyGpuList(cardBody);
            this.applyRules(cardBody);
        }
    }

    /** Relists GPU when Compute Backend changes, and rechecks the multi-GPU rules after any edit on a HartsyInference card. */
    onFieldChanged(e) {
        let name = e.target?.dataset?.name;
        let cardBody = e.target?.closest ? e.target.closest('.hartsy-card-organized') : null;
        if (!name || !cardBody) {
            return;
        }
        if (name == 'ComputeBackend') {
            // A GPU number means a different device under another compute backend, so a value the new list lacks is not kept.
            cardBody.dataset.hartsyGpuReset = 'true';
            this.applyGpuList(cardBody);
        }
        if (name == 'GPU_ID') {
            this.clearMainGpuChoices(cardBody);
        }
        this.applyRules(cardBody);
    }

    /** Clears a second-GPU choice that now names the main GPU, since the backend ignores a placement on its own main GPU. */
    clearMainGpuChoices(cardBody) {
        let main = this.valueOf(cardBody, 'GPU_ID');
        for (let name of this.secondGpuFields) {
            let select = cardBody.querySelector(`select[data-name="${name}"]`);
            if (select && main != '' && select.value == main) {
                select.value = '';
            }
        }
    }

    /**
     * Core shows a dropdown whose saved value is not one of its options with the first option selected, and the next Save writes
     * that back. Adds the saved value as its own option instead, so nothing changes unless the user picks.
     */
    keepSavedValues(cardBody, backend) {
        for (let select of cardBody.querySelectorAll('select[data-name]')) {
            let saved = backend.settings?.[select.dataset.name];
            if (saved == null || `${saved}` == '') {
                continue;
            }
            this.selectOrAdd(select, `${saved}`);
        }
    }

    /** Selects a value, adding it as "(saved value)" when the dropdown does not list it. */
    selectOrAdd(select, value) {
        let listed = false;
        for (let option of select.options) {
            if (option.value == value) {
                listed = true;
            }
        }
        if (!listed) {
            let option = document.createElement('option');
            option.value = value;
            option.innerText = `${value} (saved value)`;
            select.appendChild(option);
        }
        select.value = value;
    }

    /** Moves the fields into sections with plain labels. Each input keeps its id and data-name, so core's Edit and Save work unchanged. */
    organize(cardBody, backend) {
        let rows = {};
        for (let child of [...cardBody.children]) {
            let input = child.querySelector('[data-name]');
            if (input) {
                rows[input.dataset.name] = child;
            }
        }
        for (let section of this.sections) {
            let members = [];
            for (let name of section.fields) {
                if (rows[name]) {
                    members.push(rows[name]);
                }
            }
            if (members.length == 0) {
                continue;
            }
            let box = createDiv(null, `hartsy-card-section hartsy-card-section-${section.key}`);
            let header = createDiv(null, 'hartsy-card-section-header');
            let caret = document.createElement('span');
            caret.className = 'hartsy-card-section-caret';
            let title = document.createElement('span');
            title.className = 'hartsy-card-section-title';
            title.innerText = translate(section.title);
            header.appendChild(caret);
            header.appendChild(title);
            header.appendChild(this.makeGuideLink(section.guide, 'Guide'));
            let notice = createDiv(null, 'hartsy-card-section-notice');
            notice.style.display = 'none';
            let body = createDiv(null, 'hartsy-card-section-body');
            box.appendChild(header);
            box.appendChild(notice);
            box.appendChild(body);
            // Anchor before moving anything: a row moved into the body is no longer in the card to anchor against.
            members[0].before(box);
            for (let row of members) {
                this.relabel(row);
                body.appendChild(row);
            }
            let stateKey = `${backend.id}:${section.key}`;
            let open = this.openState[stateKey];
            if (open == null) {
                open = section.key == 'basics' || this.hasChangedField(section, backend);
            }
            this.setOpen(box, open);
            header.addEventListener('click', () => {
                let opening = body.style.display == 'none';
                this.openState[stateKey] = opening;
                this.setOpen(box, opening);
            });
        }
    }

    /** Whether any field in a section is saved at something other than its default. */
    hasChangedField(section, backend) {
        for (let name of section.fields) {
            let saved = backend.settings?.[name];
            if (saved != null && `${saved}` != this.fields[name].default) {
                return true;
            }
        }
        return false;
    }

    /** Opens or closes a section. */
    setOpen(box, open) {
        box.querySelector('.hartsy-card-section-body').style.display = open ? 'block' : 'none';
        box.querySelector('.hartsy-card-section-caret').innerText = open ? '▼' : '▶';
    }

    /** Shows a field's plain label, puts its saved name in the tooltip header, and adds a guide link under the tooltip. */
    relabel(row) {
        let name = row.querySelector('[data-name]').dataset.name;
        let info = this.fields[name];
        if (!info) {
            return;
        }
        for (let span of row.querySelectorAll('.auto-input-name .translate')) {
            if (span.dataset.pretranslated == name) {
                span.dataset.pretranslated = info.label;
                span.innerText = translate(info.label);
            }
        }
        let pop = row.querySelector('.sui-popover');
        if (!pop) {
            return;
        }
        // textContent, not innerText: a hidden popover's innerText is empty.
        let bold = pop.querySelector('b');
        if (bold && bold.textContent == name) {
            bold.textContent = info.label;
            // Core follows the name with its input type; the saved name is more use there, since the logs and guides use it.
            let after = bold.nextSibling;
            if (after && after.nodeType == Node.TEXT_NODE) {
                after.textContent = ` (${name}):`;
            }
        }
        pop.appendChild(document.createElement('br'));
        pop.appendChild(this.makeGuideLink(info.guide, 'Read more in the guide'));
    }

    /** A link to a guide that opens in a new tab, without toggling the section header it sits in. */
    makeGuideLink(page, text) {
        let link = document.createElement('a');
        link.className = 'hartsy-card-guide-link';
        link.href = `${this.guideRoot}${page}`;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.innerText = translate(text);
        link.addEventListener('click', e => e.stopPropagation());
        return link;
    }

    /** Lists the GPU choices for the card's Compute Backend, and greys out GPU when that compute backend has no GPU to pick. */
    applyGpuList(cardBody) {
        let compute = cardBody.querySelector('[data-name="ComputeBackend"]');
        let gpu = cardBody.querySelector('[data-name="GPU_ID"]');
        if (!compute || !gpu || gpu.tagName != 'SELECT') {
            return;
        }
        if (!(compute.value in this.gpuLists)) {
            this.requestGpuList(compute.value);
            return;
        }
        let list = this.gpuLists[compute.value];
        if (!list) {
            // The request failed, so the list core rendered stays.
            return;
        }
        let reset = cardBody.dataset.hartsyGpuReset == 'true';
        delete cardBody.dataset.hartsyGpuReset;
        cardBody.dataset.hartsyKind = list.kind;
        gpu.closest('.auto-input')?.classList.toggle('hartsy-card-inactive', list.kind == 'cpu');
        if (list.kind == 'cpu') {
            // The backend ignores GPU on CPU, so the choice stays as it was for when a GPU backend is picked again.
            return;
        }
        let signature = JSON.stringify([list.values, list.names]);
        if (gpu.dataset.hartsySig != signature) {
            let current = gpu.value;
            gpu.innerHTML = '';
            for (let i = 0; i < list.values.length; i++) {
                let option = document.createElement('option');
                option.value = list.values[i];
                option.innerText = list.names[i];
                gpu.appendChild(option);
            }
            gpu.dataset.hartsySig = signature;
            if (reset && !list.values.includes(current)) {
                gpu.value = list.values[0];
            }
            else {
                this.selectOrAdd(gpu, current);
            }
        }
    }

    /** Fetches the GPU choices for one Compute Backend value once, then updates every organized card. */
    requestGpuList(computeValue) {
        if (this.gpuPending[computeValue]) {
            return;
        }
        this.gpuPending[computeValue] = true;
        genericRequest('HartsyInferenceListGpus', { 'compute_backend': computeValue }, data => {
            this.gpuPending[computeValue] = false;
            this.gpuLists[computeValue] = data;
            this.refreshOrganizedCards();
        }, 0, e => {
            this.gpuPending[computeValue] = false;
            this.gpuLists[computeValue] = null;
            console.warn(`[HartsyInference] Could not list GPUs for '${computeValue}', so the card keeps its own list: ${e}`);
        });
    }

    /** Re-applies the GPU list and the multi-GPU rules on every organized card. */
    refreshOrganizedCards() {
        for (let cardBody of document.querySelectorAll('.hartsy-card-organized')) {
            this.applyGpuList(cardBody);
            this.applyRules(cardBody);
        }
    }

    /**
     * Marks the main GPU in the second-GPU lists, blocks picking Model Split GPU and CFG Parallel GPU together, and says why the
     * section is greyed out when it cannot work here. A selected option is never disabled, so a saved value still shows.
     */
    applyRules(cardBody) {
        let box = cardBody.querySelector('.hartsy-card-section-multigpu');
        if (!box) {
            return;
        }
        let kind = cardBody.dataset.hartsyKind || '';
        let main = kind == 'cuda' ? this.valueOf(cardBody, 'GPU_ID') : null;
        let split = this.valueOf(cardBody, 'DitShardGpuId');
        let cfg = this.valueOf(cardBody, 'CfgParallelGpuId');
        for (let name of this.secondGpuFields) {
            let select = cardBody.querySelector(`select[data-name="${name}"]`);
            if (!select) {
                continue;
            }
            let rival = name == 'DitShardGpuId' ? cfg : (name == 'CfgParallelGpuId' ? split : '');
            for (let option of select.options) {
                if (option.dataset.hartsyName == null) {
                    option.dataset.hartsyName = option.textContent;
                }
                let isMain = option.value != '' && option.value == main;
                let text = isMain ? `${option.dataset.hartsyName} (main GPU)` : option.dataset.hartsyName;
                if (option.textContent != text) {
                    option.textContent = text;
                }
                option.disabled = option.value != '' && option.value != select.value && (isMain || rival != '');
            }
        }
        let gpuCount = this.cudaGpuCount();
        let notes = [];
        if (split != '' && cfg != '') {
            notes.push('Model Split GPU and CFG Parallel GPU can\'t be used together, and the backend won\'t start with both. Set one of them to None.');
        }
        if (gpuCount == 0) {
            notes.push('These settings need two NVIDIA GPUs that work with CUDA, and none was found on this machine.');
        }
        else if (kind != '' && kind != 'cuda') {
            notes.push('These settings need the CUDA compute backend.');
        }
        else if (gpuCount == 1) {
            notes.push('These settings need a second NVIDIA GPU.');
        }
        if (main != null && main != '' && this.secondGpuFields.some(name => this.valueOf(cardBody, name) == main)) {
            notes.push('A GPU setting in this section names the main GPU, so the backend ignores it. Set it to None.');
        }
        let notice = box.querySelector('.hartsy-card-section-notice');
        let text = notes.join('\n');
        if (notice.textContent != text) {
            notice.textContent = text;
        }
        notice.style.display = notes.length == 0 ? 'none' : 'block';
        box.querySelector('.hartsy-card-section-body').classList.toggle('hartsy-card-inactive', (kind != '' && kind != 'cuda') || gpuCount < 2);
    }

    /** The current value of one field on a card, or an empty string. */
    valueOf(cardBody, name) {
        return cardBody.querySelector(`[data-name="${name}"]`)?.value || '';
    }

    /** How many CUDA GPUs the second-GPU lists offer, read from the backend type core already loaded. */
    cudaGpuCount() {
        let settings = typeof backend_types == 'undefined' ? null : backend_types[this.typeId]?.settings;
        for (let setting of settings || []) {
            if (setting.name == 'TextEncoderGpuId') {
                let count = 0;
                for (let value of setting.values || []) {
                    if (value != '') {
                        count++;
                    }
                }
                return count;
            }
        }
        return 0;
    }
}

let hartsyBackendCard = new HartsyBackendCard();

sessionReadyCallbacks.push(() => hartsyBackendCard.init());
