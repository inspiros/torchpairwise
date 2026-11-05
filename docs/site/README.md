# Documentation Website

This mdBook presents comparison equations, sample layouts, modes and streamed
operators for Python and C++/LibTorch. It needs Python 3.10+ and mdBook 0.5.4;
building the website does not import PyTorch or compile the extension.

From the repository root:

```sh
python docs/site/generate_reference.py --check
python docs/site/check_docs.py
mdbook build docs/site
python docs/site/check_docs.py --built-dir develop/site-preview/book
mdbook serve docs/site --hostname 127.0.0.1 --port 3000
```

The default output is ignored under `develop/site-preview/book`. MathJax loads
from the CDN configured by mdBook; equations require network access for rendering.
The book itself, search, code selection and branding use static local assets.

## Editing

- Edit guide pages in `src/guide/` and the overview in `src/index.md`.
- Edit the algorithm gallery in `src/examples/index.md`; its computation diagrams
  live alongside the shared branding assets in `docs/assets/`.
- Edit metric titles, equations and short domain notes in `metric-notes.json`.
  Run `python docs/site/generate_reference.py` to regenerate metric pages and
  navigation. Mode, option and layout metadata comes from the native comparison
  catalog; the generator needs no compiled library.
- Shared hero/logo assets live in `docs/assets/`. The preprocessor copies them
  into ignored `src/assets/` at build time. The favicon is the comparison logo.
- Keep Python/C++ examples in a `code-pair` container with two `code-panel`
  children, `data-language="python"` and `data-language="cpp"`, each with a
  `code-label` and a fenced code block. Follow an existing guide example.
  The global switch persists across pages; without JavaScript both panels show.
  Python-only callback examples remain visible in either language setting.
- Write equations with `\(...\)` or `\[...\]`; the preprocessor preserves
  these delimiters through Markdown rendering. Prefer shape comments and short
  deterministic numeric outputs when they help explain a result.

Use Title Case for headings, with connector words such as "and", "or", "at"
and "of" lowercase within a heading. Preserve acronym and product spelling.
Use ordinary hyphens for compound names and words for numeric ranges; omit
sentence-ending periods inside displayed equations. Smart punctuation is
disabled so the renderer preserves those choices.

The checker validates generated content, Python snippet syntax, local links and
built anchors. It does not execute metric examples or compile C++ snippets.

## GitHub Pages

In repository **Settings → Pages**, select **GitHub Actions** as the build source.
After pushing the documentation, manually run the **Documentation** workflow.
It builds the static book, checks local links and uploads the Pages artifact.
Select **Deploy the built documentation to GitHub Pages** to publish that build;
the default run only builds and checks the site. The workflow has no push,
pull-request or scheduled trigger. No native library build is part of it.

The configured project path is `/torchpairwise/`, corresponding to
`https://inspiros.github.io/torchpairwise/`. Update `site-url` in `book.toml`
if using another repository path or a custom domain.
