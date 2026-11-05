/* Reader preferences are local; no network calls or analytics. */
(() => {
    // mdBook fingerprints additional scripts; resolve from the theme directory.
    const script = document.currentScript || [...document.scripts].find(item => /\/theme\/site(?:-[\w]+)?\.js$/.test(item.src));
    const root = script ? new URL('../', script.src) : new URL(window.path_to_root || '.', location.href);
    const html = document.documentElement;
    const helpTitle = document.querySelector('.mdbook-help-title');
    if (helpTitle) helpTitle.textContent = 'Keyboard Shortcuts';
    const storageKey = 'torchpairwise-example-language';
    let language = 'python';
    try { language = localStorage.getItem(storageKey) === 'cpp' ? 'cpp' : 'python'; } catch (_) { /* storage can be disabled */ }
    const buttons = [];
    // Discover paired examples once. Switching only changes one root attribute,
    // including on print.html, which contains examples from the entire book.
    document.querySelectorAll('.code-pair').forEach(pair => {
        if (pair.querySelector('.code-panel[data-language="python"]') &&
            pair.querySelector('.code-panel[data-language="cpp"]')) {
            pair.dataset.codePaired = '';
        }
    });
    let savePreference;
    function select(value, persist = true) {
        if (html.dataset.codeLanguage === value) return;
        language = value;
        html.dataset.codeLanguage = value;
        buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.language === value)));
        clearTimeout(savePreference);
        savePreference = undefined;
        // Let the new selection paint before synchronous storage work; rapid
        // clicks persist only the final choice. Storage events never echo it.
        if (persist) savePreference = setTimeout(() => {
            try { localStorage.setItem(storageKey, language); } catch (_) { /* storage can be disabled */ }
            savePreference = undefined;
        }, 150);
    }
    const controls = document.createElement('div');
    controls.className = 'language-switch';
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', 'Example language, applies to all paired examples');
    for (const [value, label] of [['python', 'Python'], ['cpp', 'C++']]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.language = value;
        button.textContent = label;
        button.addEventListener('click', () => select(value));
        button.addEventListener('keydown', event => {
            if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                event.preventDefault();
                event.stopPropagation();
                const next = buttons.find(other => other !== button);
                select(next.dataset.language);
                next.focus();
            }
        });
        buttons.push(button);
        controls.append(button);
    }
    const right = document.querySelector('.menu-bar .right-buttons');
    if (right) right.prepend(controls);
    const left = document.querySelector('.menu-bar .left-buttons');
    if (left) {
        const brand = document.createElement('a');
        brand.href = new URL('index.html', root).href;
        brand.className = 'tp-brand';
        brand.setAttribute('aria-label', 'TorchPairwise documentation home');
        const logo = document.createElement('img');
        logo.alt = '';
        const label = document.createElement('span');
        label.textContent = 'TorchPairwise';
        brand.append(logo, label);
        left.append(brand);
        const updateLogo = () => {
            const dark = !html.classList.contains('light') && !html.classList.contains('rust');
            logo.src = new URL(`assets/logo${dark ? '-dark' : ''}.svg`, root).href;
        };
        updateLogo();
        new MutationObserver(updateLogo).observe(html, {attributes: true, attributeFilter: ['class']});
    }
    select(language, false);
    // Preserve a just-clicked choice even when navigating before the save timer.
    window.addEventListener('pagehide', () => {
        if (savePreference) {
            clearTimeout(savePreference);
            try { localStorage.setItem(storageKey, language); } catch (_) { /* optional */ }
        }
    });
    // Cross-tab preference synchronization is optional when storage is available.
    window.addEventListener('storage', event => {
        if (event.key === storageKey) select(event.newValue === 'cpp' ? 'cpp' : 'python', false);
    });
    const catalog = document.querySelector('[data-metric-catalog]');
    if (catalog) {
        const input = catalog.querySelector('input[type="search"]');
        const family = catalog.querySelector('select');
        const status = catalog.querySelector('[role="status"]');
        const cards = [...catalog.querySelectorAll('.metric-card')];
        const filter = () => {
            const query = input.value.toLowerCase().trim();
            let visible = 0;
            cards.forEach(card => {
                const match = (!query || card.dataset.search.includes(query)) && (!family.value || card.dataset.family === family.value);
                card.hidden = !match;
                if (match) visible++;
            });
            status.textContent = `${visible} of ${cards.length} comparisons`;
        };
        input.addEventListener('input', filter);
        family.addEventListener('change', filter);
        catalog.querySelector('.catalog-controls').hidden = false;
        filter();
    }
})();
