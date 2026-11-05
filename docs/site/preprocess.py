#!/usr/bin/env python3
"""Prepare shared branding assets and preserve TeX through Markdown rendering."""
import html
import json
from pathlib import Path
import re
import shutil
import sys


def preserve_math(source):
    pieces = re.split(r'(```.*?```|`[^`\n]*`)', source, flags=re.S)
    def replace(match):
        tex = match.group(0)
        tag = 'div' if tex.startswith(r'\[') else 'span'
        kind = 'math-display' if tag == 'div' else 'math-inline'
        # Inline HTML still passes through Markdown's backslash/emphasis parser.
        # Entities become the original TeX in the DOM before MathJax reads it.
        safe = html.escape(tex) if tag == 'div' else ''.join(f'&#{ord(char)};' for char in tex)
        return f'<{tag} class="{kind}">{safe}</{tag}>'
    for i in range(0, len(pieces), 2):
        pieces[i] = re.sub(r'\\\((.*?)\\\)|\\\[(.*?)\\\]', replace, pieces[i], flags=re.S)
    return ''.join(pieces)


def chapters(items):
    for item in items:
        chapter = item.get('Chapter')
        if chapter:
            chapter['content'] = preserve_math(chapter['content'])
            chapters(chapter.get('sub_items', []))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'supports':
        return 0 if sys.argv[-1] == 'html' else 1
    context, book = json.load(sys.stdin)
    site = Path(context['root'])
    assets = site/'src'/'assets'
    assets.mkdir(exist_ok=True)
    for name in ('hero.svg', 'logo.svg', 'logo-dark.svg',
                 'gallery-ridge.svg', 'gallery-pca.svg',
                 'gallery-lda.svg', 'gallery-svm.svg', 'gallery-lsqr.svg',
                 'gallery-pagerank.svg', 'gallery-kkt.svg', 'gallery-svd.svg'):
        source = site.parent/'assets'/name
        target = assets/name
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
    chapters(book.get('sections', book.get('items', [])))
    json.dump(book, sys.stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
