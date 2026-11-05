#!/usr/bin/env python3
"""Check the source book and, optionally, a built HTML tree (stdlib only)."""
import argparse
import ast
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from generate_reference import SITE, format_headers, generate, native_catalog


class Links(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.links = []
        self.ids = set()
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if attributes.get('id'):
            self.ids.add(attributes['id'])
        for key in ('href', 'src'):
            if attributes.get(key):
                self.links.append(attributes[key])


def check_source(errors):
    generated = generate()
    for name, content in generated.items():
        path = SITE / 'src' / name
        if not path.exists() or path.read_text() != content:
            errors.append(f'Regenerate reference: {name}')
    pages = list((SITE / 'src').rglob('*.md'))
    for page in pages:
        content = page.read_text()
        if re.search(r'\.maintainer/|\.private/|tran254|server(?:9|11)\b', content):
            errors.append(f'Private or maintainer-only content in {page.name}')
        if content.count('class="code-pair"') * 2 != content.count('class="code-panel"'):
            errors.append(f'Incomplete Python/C++ example in {page.name}')
        for number, snippet in enumerate(re.findall(r'```python\n(.*?)```', content, re.S), 1):
            try:
                ast.parse(snippet)
            except SyntaxError as error:
                errors.append(f'{page.name}, Python example {number}: {error}')
        without_code = re.sub(r'```.*?```|`[^`\n]*`', '', content, flags=re.S)
        if format_headers(content, navigation=page.name == 'SUMMARY.md') != content:
            errors.append(f'Use Title Case headings in {page.name}')
        if re.search(r'[\u2013\u2014]', without_code):
            errors.append(f'Use ordinary hyphens or punctuation in {page.name}')
        if any(tex.rstrip().endswith('.') for tex in re.findall(r'\\\[(.*?)\\\]', without_code, re.S)):
            errors.append(f'Remove equation-ending periods in {page.name}')
        for opening, closing in ((r'\[', r'\]'), (r'\(', r'\)')):
            if without_code.count(opening) != without_code.count(closing):
                errors.append(f'Unbalanced math delimiters in {page.name}')
        targets = re.findall(r'\]\(([^\s)]+)\)', content)
        targets += Links(content).links
        for target in targets:
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            relative = unquote(parts.path)
            if relative.endswith('.html'):
                relative = relative[:-5] + '.md'
            path = page.parent / relative
            if relative.startswith('assets/') or '/assets/' in relative:
                path = SITE.parent / 'assets' / Path(relative).name
            if not path.is_file():
                errors.append(f'{page.relative_to(SITE)}: missing {target}')
    for entry in native_catalog():
        text = generated['metrics/' + entry['name'] + '.md']
        for required in ('## Shapes and Modes', 'One sample', 'Collection X', 'Collection Y', '## Entry Points', 'data-language="python"', 'data-language="cpp"'):
            if required not in text:
                errors.append(f'{entry["name"]}: missing {required}')
    print(f'Checked {len(pages)} source pages, {len(native_catalog())} catalog entries and Python snippet syntax')


def check_built(root, errors):
    root = root.resolve()
    pages = {path: Links(path.read_text()) for path in root.rglob('*.html')}
    if not pages:
        errors.append(f'No HTML pages under {root}')
    for page, parsed in pages.items():
        for target in parsed.links:
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            path = unquote(parts.path)
            if path.startswith('/torchpairwise/'):
                destination = root / path.removeprefix('/torchpairwise/')
            elif path.startswith('/'):
                destination = root / path.lstrip('/')
            else:
                destination = page.parent / path if path else page
            if destination.is_dir():
                destination /= 'index.html'
            destination = destination.resolve()
            if not destination.is_file():
                errors.append(f'{page.relative_to(root)}: missing {target}')
            elif parts.fragment and destination in pages and unquote(parts.fragment) not in pages[destination].ids:
                errors.append(f'{page.relative_to(root)}: missing anchor {target}')
    print(f'Checked local links, anchors and assets in {len(pages)} built HTML pages')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built-dir', type=Path)
    args = parser.parse_args()
    errors = []
    check_source(errors)
    if args.built_dir:
        check_built(args.built_dir, errors)
    if errors:
        raise SystemExit('\n'.join(sorted(set(errors))))


if __name__ == '__main__':
    main()
