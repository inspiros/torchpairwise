"""Generate transparent light/dark comparison-tile SVG logos, with no dependencies.

Run: python examples/generate_logo.py
Teal diagonal cells suggest paired comparisons; the gold upper triangle suggests
condensed comparisons. Dots on the two axes represent the sample collections.
"""
import argparse
from pathlib import Path


def create_logo(*, dark=False):
    teal, gold = ('#69d9c5', '#f0c783') if dark else ('#1b8678', '#a9701e')
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128" role="img" aria-labelledby="title desc">
<title id="title">TorchPairwise comparison tile</title>
<desc id="desc">Two collections of sample dots frame a three-by-three comparison matrix. Diagonal cells are teal, cells above the diagonal are gold, and cells below are faint gold.</desc>''']
    for axis in range(3):
        coordinate = 45+axis*29
        parts.append(f'<circle cx="{coordinate}" cy="16" r="5" fill="{gold}"/>')
        parts.append(f'<circle cx="16" cy="{coordinate}" r="5" fill="{teal}"/>')
    for row in range(3):
        for column in range(3):
            color = teal if row == column else gold
            opacity = '0.24' if row > column else '1'
            parts.append(f'<rect x="{34+column*29}" y="{34+row*29}" width="22" height="22" rx="5" fill="{color}" opacity="{opacity}"/>')
    parts.append('</svg>\n')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1]/'docs/assets')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for dark, filename in ((False, 'logo.svg'), (True, 'logo-dark.svg')):
        target = args.directory/filename
        target.write_text(create_logo(dark=dark), encoding='utf-8')
        print('Saved', target.resolve())


if __name__ == '__main__':
    main()
