"""Generate the README's self-contained SVG overview (Python standard library only).

Run: python examples/generate_readme_hero.py
Geometry and heatmap colors illustrate Euclidean comparisons of fixed 2D points.
The streaming diagram is schematic, not a memory or performance measurement.
"""
import argparse
from html import escape
import math
from pathlib import Path


CREAM = '#f4f3e9'
MUTED = '#a9babe'
TEAL = '#69d9c5'
GOLD = '#f0c783'
BLUE = '#8fbbfa'
ROSE = '#dfa9d1'


def create_hero():
    elements = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="790" viewBox="0 0 1440 790" role="img" aria-labelledby="title desc">
<title id="title">TorchPairwise: four ways to compare, with streamed results</title>
<desc id="desc">87 comparison entries across 9 input families, including 13 kernels. All-pairs produces an N by M matrix, paired compares corresponding samples, condensed stores the strict upper triangle, and single compares two samples. Streamed operators compute tiles and accumulate reductions, selections or products without retaining the full generated comparison matrix. Schematics omit leading batch dimensions.</desc>
<defs>
  <linearGradient id="background" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#101f2b"/><stop offset=".6" stop-color="#102a30"/><stop offset="1" stop-color="#142524"/></linearGradient>
  <radialGradient id="wash"><stop stop-color="#357e79" stop-opacity=".28"/><stop offset="1" stop-color="#357e79" stop-opacity="0"/></radialGradient>
  <linearGradient id="card" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#1a3039"/><stop offset="1" stop-color="#152930"/></linearGradient>
  <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="none" stroke="#a9babe" stroke-width="1.2"/></marker>
  <clipPath id="frame"><rect width="1440" height="790" rx="28"/></clipPath>
</defs>
<g clip-path="url(#frame)">
<rect width="1440" height="790" fill="url(#background)"/>
<ellipse cx="1120" cy="190" rx="480" ry="330" fill="url(#wash)"/>
''']

    def rect(x, y, w, h, fill, radius=0, stroke=None, opacity=1):
        elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" opacity="{opacity}"'+(f' stroke="{stroke}"' if stroke else '')+'/>')

    def text(x, y, content, size=20, color=CREAM, weight='400', family='sans', **attrs):
        fonts = {'sans': 'Inter, Noto Sans, DejaVu Sans, sans-serif', 'serif': 'Georgia, Noto Serif, DejaVu Serif, serif', 'mono': 'DejaVu Sans Mono, monospace'}
        extra = ''.join(f' {key.replace("_", "-")}="{value}"' for key, value in attrs.items())
        elements.append(f'<text x="{x}" y="{y}" font-family="{fonts[family]}" font-size="{size}" font-weight="{weight}" fill="{color}"{extra}>{escape(content)}</text>')

    def line(x1, y1, x2, y2, color=MUTED, width=1, opacity=1, arrow=False):
        elements.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

    def dot(x, y, color, radius=4, opacity=1):
        elements.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" opacity="{opacity}"/>')

    # Transparent, true constant-distance contours and sample connections.
    # An affine projection makes Euclidean circles into ellipses in the banner.
    elements.append('<g transform="translate(1110 140) rotate(-18)" fill="none" stroke="#79cbbb">')
    for radius in (45, 77, 110, 145, 180, 218, 259):
        elements.append(f'<ellipse rx="{radius}" ry="{radius*.58:.2f}" stroke-width="1" opacity=".13"/>')
    elements.append('</g>')
    xs = [(925, 106), (1032, 181), (1138, 72), (1258, 144)]
    ys = [(1000, 65), (1091, 128), (1202, 210), (1312, 84), (1338, 191)]
    for x, y in xs:
        for a, b in ys:
            line(x, y, a, b, TEAL, 1, .08)
        dot(x, y, TEAL, 4, .38)
    for x, y in ys:
        dot(x, y, GOLD, 4, .38)
    line(1032, 181, 1091, 128, CREAM, 1.5, .5)
    text(1074, 177, 'd(x, y)', 16, MUTED, family='serif', opacity='.55')

    text(58, 62, 'NATIVE TENSOR COMPARISONS', 15, TEAL, '600', letter_spacing='2.5')
    text(54, 143, 'TorchPairwise', 76, CREAM, '500', 'serif', letter_spacing='-2')
    text(59, 192, 'Distances, Kernels, and Streamed Operators', 29, CREAM)
    text(60, 235, '87 comparisons', 20, TEAL, '600')
    text(242, 235, '·   13 kernels   ·   9 input families', 20, MUTED)
    rect(60, 259, 230, 31, '#213a42', 15)
    text(175, 280, 'Python + C++ / LibTorch', 14, CREAM, '500', text_anchor='middle')
    text(311, 280, 'CPU / CUDA', 15, MUTED)

    # Illustrative values come from actual Euclidean distances, not random cells.
    x_points = [(.12, .18), (.25, .78), (.69, .32), (.88, .84)]
    y_points = [(.06, .68), (.37, .12), (.5, .49), (.82, .19), (.93, .66)]
    distances = [[math.dist(x, y) for y in y_points] for x in x_points]

    def heat(value, accent=TEAL):
        base = (28, 52, 61)
        rgb = tuple(int(accent[i:i+2], 16) for i in (1, 3, 5))
        t = .17 + .8 * min(value, 1.)
        return '#'+''.join(f'{round(a+(b-a)*t):02x}' for a, b in zip(base, rgb))

    cards = [(60, 'All-Pairs', 'Every x against every y', TEAL, '[N, M]'),
             (396, 'Paired', 'Corresponding samples', BLUE, '[N]'),
             (732, 'Condensed', 'Each unordered pair once', GOLD, '[N(N−1)/2]'),
             (1068, 'Single', 'One sample against one', ROSE, 'scalar')]
    for index, (left, title, subtitle, accent, shape) in enumerate(cards):
        rect(left, 319, 312, 225, 'url(#card)', 15, '#30464c')
        line(left+22, 320, left+290, 320, accent, 2, .85)
        text(left+22, 354, title, 24, CREAM, '600')
        text(left+22, 381, subtitle, 16, MUTED)
        text(left+22, 520, shape, 20, accent, '500', 'mono')
        if index == 0:
            for row in range(4):
                dot(left+23, 405+row*20+7, TEAL, 3)
                for col in range(5):
                    rect(left+37+col*25, 405+row*20, 22, 15, heat(distances[row][col]), 3)
            for col in range(5):
                dot(left+48+col*25, 396, GOLD, 3)
            text(left+185, 448, 'N × M', 20, accent, family='mono')
        elif index in (1, 2):
            for row in range(4):
                for col in range(4):
                    active = row == col if index == 1 else row < col
                    value = math.dist(x_points[row], (y_points if index == 1 else x_points)[col])
                    rect(left+23+col*20, 400+row*20, 16, 16, heat(value, accent) if active else '#21343e', 3,
                         stroke=accent if active else '#2b4049', opacity=1 if active else .5)
            line(left+117, 439, left+145, 439, arrow=True)
            if index == 1:
                for row in range(4):
                    rect(left+160, 400+row*20, 33, 16, heat(distances[row][row], accent), 3)
                text(left+211, 446, 'N', 20, accent, family='mono')
            else:
                values = [math.dist(x_points[i], x_points[j]) for i in range(4) for j in range(i+1, 4)]
                for j, value in enumerate(values):
                    rect(left+158+j*19, 430, 16, 18, heat(value, accent), 3)
        else:
            # d((0.2,0.3), (0.9,0.6)) ≈ 0.76.
            dot(left+33, 449, accent, 7)
            dot(left+134, 410, GOLD, 7)
            line(left+42, 446, left+125, 414, accent, 2, .8)
            text(left+23, 476, 'x', 15, MUTED, family='mono')
            text(left+127, 393, 'y', 15, MUTED, family='mono')
            line(left+160, 435, left+187, 435, arrow=True)
            rect(left+202, 410, 86, 49, '#302e43', 10, '#6d536d')
            text(left+245, 442, f'{math.dist((.2,.3),(.9,.6)):.2f}', 24, accent, family='mono', text_anchor='middle')

    text(62, 570, 'Leading batches broadcast. Condensed mode requires a supported symmetric comparison.', 15, MUTED)

    # A matrix is logical. Only highlighted tiles and the terminal answer are kept.
    rect(60, 596, 1320, 153, '#112129', 16, '#34474a')
    text(83, 629, 'STREAMING OPERATORS', 14, TEAL, '600', letter_spacing='1.6')
    text(82, 668, 'Tiled Execution', 29, CREAM, '500', 'serif')
    text(84, 703, 'Reduce · select · multiply', 18, MUTED)
    for row in range(5):
        for col in range(9):
            active = 1 <= row <= 2 and 3 <= col <= 5
            rect(428+col*16, 621+row*20, 13, 16, TEAL if active else '#23373f', 2,
                 opacity=.85 if active else .65)
    text(499, 737, 'computed tiles', 13, MUTED, text_anchor='middle')
    line(591, 665, 633, 665, arrow=True)
    rect(654, 625, 361, 78, '#1c3239', 11)
    text(835, 656, 'sum · topk · quantile · K @ V', 17, CREAM, family='mono', text_anchor='middle')
    text(835, 684, '25 shared scalar consumers', 16, TEAL, text_anchor='middle')
    line(1036, 665, 1079, 665, arrow=True)
    for row, width in enumerate((102, 76, 119, 88)):
        rect(1101, 624+row*23, width, 15, heat(.3+row*.17), 3)
    text(1272, 669, 'result', 19, CREAM)
    text(720, 777, 'Distances · divergences · kernels · scores   /   Native autograd · torch.compile · torch.export', 15, MUTED, text_anchor='middle')
    elements.append('</g></svg>\n')
    return '\n'.join(elements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', type=Path, default=Path(__file__).resolve().parents[1]/'docs/assets/hero.svg')
    args = parser.parse_args()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(create_hero(), encoding='utf-8')
    print('Saved', args.save.resolve())


if __name__ == '__main__':
    main()
