#!/usr/bin/env python3
"""Generate metric-oriented chapters from native metadata and authored equations.

No PyTorch import or compiled extension is needed. --check compares generated
chapters with their checked-in versions without changing files.
"""
import argparse
import ast
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import re

SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]
GROUPS = ['Vector geometry', 'Boolean and equality', 'Probability and histograms',
          'Point sets and transport', 'Kernels', 'Gaussians and covariances',
          'Gaussian mixtures', 'Images', 'Audio', 'Sequences and tokens']
FAMILY_NAMES = {'vector':'Vectors', 'point_cloud':'Point sets', 'gaussian':'Gaussians',
                'covariance':'Covariances', 'mixture':'Mixtures', 'image':'Images',
                'audio':'Audio', 'sequence':'Sequences', 'tokens':'Tokens'}


def title_case(text):
    """Capitalize heading words, preserving connectors, code and product names."""
    minor = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'from', 'in',
             'into', 'nor', 'of', 'on', 'onto', 'or', 'over', 'per', 'the',
             'to', 'via', 'vs', 'versus', 'with', 'without'}
    tokens = list(re.finditer(r'`[^`]+`|<[^>]+>|\]\([^)]*\)|[^\W\d_][^\W_]*', text))
    words = [token for token in tokens if token.group()[0].isalpha()]
    if not words:
        return text
    ends = {words[0].start(), words[-1].start()}
    def replace(match):
        word = match.group()
        if not word[0].isalpha():
            return word
        if word.lower() in minor and match.start() not in ends:
            return word.lower()
        # Keep spelling such as PyTorch, LibTorch, mdBook, SSIM and LogDet.
        return word[0].upper() + word[1:]
    return re.sub(r'`[^`]+`|<[^>]+>|\]\([^)]*\)|[^\W\d_][^\W_]*', replace, text)


def format_headers(source, *, navigation=False):
    fence = None
    lines = []
    for line in source.splitlines(keepends=True):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            fence = marker[1][0] if fence is None else None if fence == marker[1][0] else fence
        if fence is None:
            line = re.sub(r'^(#{1,6}\s+)(.*)', lambda m: m[1] + title_case(m[2]), line)
            line = re.sub(r'(<h[1-6](?:\s[^>]*)?>)(.*?)(</h[1-6]>)',
                          lambda m: m[1] + title_case(m[2]) + m[3], line)
            if navigation:
                line = re.sub(r'\[([^]]+)\]\(', lambda m: '[' + title_case(m[1]) + '](', line)
        lines.append(line)
    return ''.join(lines)


def native_catalog():
    source = (ROOT/'torchpairwise/csrc/ops/comparison_selectors.cpp').read_text()
    source = source.split('static const std::vector<Entry> entries = {', 1)[1].split('\n  };', 1)[0]
    # This table is a literal initializer of strings, booleans and integer lists.
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|true|false|-?\d+|[{},]', source)
    encoded = []
    for token in tokens:
        if token.startswith('"') and encoded and encoded[-1].startswith('"'):
            encoded[-1] = json.dumps(json.loads(encoded[-1])+json.loads(token))
        else:
            encoded.append({'{':'[','}':']'}.get(token,token))
    rows = json.loads(re.sub(r',\s*]',']','['+''.join(encoded)+']'))
    keys = ('name','family','inputs','modes','parameters','symmetric','differentiable','aliases','role','sample_ndims')
    return [dict(zip(keys,row)) for row in rows]


def code_pair(python, cpp):
    return '\n'.join(['<div class="code-pair">','<div class="code-panel" data-language="python">',
        '<div class="code-label">Python</div>','', '```python',python.rstrip(),'```','', '</div>',
        '<div class="code-panel" data-language="cpp">','<div class="code-label">C++ / LibTorch</div>',
        '', '```cpp',cpp.rstrip(),'```','', '</div>','</div>'])


def badges(c, small=False):
    return '<div class="badges">'+''.join(f'<span class="badge mode-{mode.replace("_","-")}">{mode.replace("_","-")}</span>' for mode in c['modes'])+('</div>' if small else f'<span class="badge">{escape(c["role"])}</span><span class="badge">{escape(FAMILY_NAMES[c["family"]])}</span></div>')


def category(c):
    name=c['name']
    if c['role']=='kernel':return 'Kernels'
    if c['family'] in ('gaussian','covariance'):return 'Gaussians and covariances'
    if c['family']=='mixture':return 'Gaussian mixtures'
    if c['family']=='point_cloud':return 'Point sets and transport'
    if c['family']=='image':return 'Images'
    if c['family']=='audio':return 'Audio'
    if c['family'] in ('sequence','tokens'):return 'Sequences and tokens'
    if name in {key+'_distances' for key in ('hamming','jaccard','dice','yule','rogerstanimoto','sokalmichener','russellrao','sokalsneath','kulczynski1','kulsinski')}:return 'Boolean and equality'
    if name in {key+'_distances' for key in ('hellinger','bhattacharyya','jensenshannon','total_variation','fisher_rao','triangular_discrimination','aitchison','wasserstein_1d')} or name.endswith('_divergences'):return 'Probability and histograms'
    return 'Vector geometry'


def shapes(c):
    return {
      'vector': ('[D]', '[...,N,D]', '[...,M,D]'),
      'point_cloud': ('[P,D]', '[...,N,P,D]', '[...,M,Q,D]'),
      'image': ('[C,H,W]', '[...,N,C,H,W]', '[...,M,C,H,W]'),
      'audio': ('[C,T]', '[...,N,C,T]', '[...,M,C,T]'),
      'sequence': ('[T,D]', '[...,N,T,D]', '[...,M,S,D]'),
      'tokens': ('[T]', '[...,N,T]', '[...,M,S]'),
      'covariance': ('[D,D]', '[...,N,D,D]', '[...,M,D,D]'),
      'gaussian': ('(mean[D], covariance[D,D])', '(means[...,N,D], covariances[...,N,D,D])', '(means[...,M,D], covariances[...,M,D,D])'),
      'mixture': ('(means[K,D], covariances[K,D,D], weights[K])', '(means[...,N,K,D], covariances[...,N,K,D,D], weights[...,N,K])', '(means[...,M,L,D], covariances[...,M,L,D,D], weights[...,M,L])')
    }[c['family']]


def examples(c):
    family,name=c['family'],c['name']
    py=['import torch','import torchpairwise as tp','']
    cpp=['#include <torchpairwise.h>','','namespace tp = torchpairwise::ops;']
    if family=='vector':
        boolean=category(c)=='Boolean and equality' and name!='hamming_distances'
        py+=['X = torch.tensor([[True, False], [True, True]])  # [2,2]' if boolean else 'X = torch.rand(2, 2) + 0.2  # [N=2,D=2]',
             'Y = torch.tensor([[True, True], [False, True], [True, False]])' if boolean else 'Y = torch.rand(3, 2) + 0.2  # [M=3,D=2]']
        cpp+=['auto X = at::tensor({true, false, true, true}, at::kBool).reshape({2, 2});' if boolean else 'auto X = at::rand({2, 2}) + 0.2;  // [N=2,D=2]',
              'auto Y = at::tensor({true, true, false, true, true, false}, at::kBool).reshape({3, 2});' if boolean else 'auto Y = at::rand({3, 2}) + 0.2;  // [M=3,D=2]']
    elif family=='point_cloud':
        py+=['X = torch.rand(2, 3, 2)  # [N=2,P=3,D=2]','Y = torch.rand(3, 4, 2)  # [M=3,Q=4,D=2]']
        cpp+=['auto X = at::rand({2, 3, 2});  // [N=2,P=3,D=2]','auto Y = at::rand({3, 4, 2});  // [M=3,Q=4,D=2]']
    elif family=='image':
        py+=['X = torch.rand(2, 1, 8, 8)  # [N=2,C=1,H=8,W=8]','Y = torch.rand(3, 1, 8, 8)  # [M=3,C=1,H=8,W=8]']
        cpp+=['auto X = at::rand({2, 1, 8, 8});','auto Y = at::rand({3, 1, 8, 8});']
    elif family=='audio':
        py+=['X = torch.rand(2, 1, 24)  # [N=2,C=1,T=24]','Y = torch.rand(3, 1, 24)  # [M=3,C=1,T=24]']
        cpp+=['auto X = at::rand({2, 1, 24});','auto Y = at::rand({3, 1, 24});']
    elif family=='sequence':
        py+=['X = torch.rand(2, 4, 2)  # [N=2,T=4,D=2]','Y = torch.rand(3, 5, 2)  # [M=3,S=5,D=2]']
        cpp+=['auto X = at::rand({2, 4, 2});','auto Y = at::rand({3, 5, 2});']
    elif family=='tokens':
        py+=['X = torch.tensor([[1, 2, 3], [3, 2, 1]])  # [N=2,T=3]','Y = torch.tensor([[1, 2, 2], [3, 3, 1], [2, 2, 1]])  # [M=3,S=3]']
        cpp+=['auto X = at::tensor({1, 2, 3, 3, 2, 1}, at::kLong).reshape({2, 3});','auto Y = at::tensor({1, 2, 2, 3, 3, 1, 2, 2, 1}, at::kLong).reshape({3, 3});']
    elif family=='covariance':
        py+=['X = torch.eye(2).repeat(2, 1, 1)  # [N=2,D=2,D=2]','Y = 2 * torch.eye(2).repeat(3, 1, 1)  # [M=3,D=2,D=2]']
        cpp+=['auto X = at::eye(2).repeat({2, 1, 1});','auto Y = 2 * at::eye(2).repeat({3, 1, 1});']
    elif family=='gaussian':
        py+=['X = (torch.rand(2, 2), torch.eye(2).repeat(2, 1, 1))',
              'Y = (torch.rand(3, 2), 2 * torch.eye(2).repeat(3, 1, 1))',
              '# (means [N,D], covariances [N,D,D])']
        cpp+=['tp::ComparisonCollection X{at::rand({2, 2}), at::eye(2).repeat({2, 1, 1})};',
               'tp::ComparisonCollection Y{at::rand({3, 2}), 2 * at::eye(2).repeat({3, 1, 1})};']
    else:
        py+=['X = (torch.rand(2, 2, 2), torch.eye(2).repeat(2, 2, 1, 1), torch.ones(2, 2))',
              'Y = (torch.rand(3, 2, 2), torch.eye(2).repeat(3, 2, 1, 1), torch.ones(3, 2))',
              '# (means [N,K,D], covariances [N,K,D,D], weights [N,K]); K=D=2']
        cpp+=['tp::ComparisonCollection X{at::rand({2, 2, 2}),', '    at::eye(2).repeat({2, 2, 1, 1}), at::ones({2, 2})};',
              'tp::ComparisonCollection Y{at::rand({3, 2, 2}),', '    at::eye(2).repeat({3, 2, 1, 1}), at::ones({3, 2})};']
    opts={}
    if name=='wminkowski_distances':opts['w']=('torch.tensor([1., 2.])','at::tensor({1.f, 2.f})')
    if name=='seuclidean_distances':opts['V']=('torch.ones(2)','at::ones({2})')
    if name=='mahalanobis_distances':opts['VI']=('torch.eye(2)','at::eye(2)')
    if name=='sliced_wasserstein_distances':opts['projections']=('torch.eye(2)','at::eye(2)')
    if name=='soft_dtw_divergences':opts['gamma']=('0.7','0.7')
    if family=='image':
        opts['data_range']=('1.0','1.0')
        if name!='psnr_scores':opts['window_size']=('3','3')
        if name=='multiscale_ssim_scores':opts['weights']=('[1.0]','std::vector<double>{1.0}')
    if family=='audio' and name not in ('si_sdr_scores','signal_snr_scores'):
        if name=='multiresolution_stft_distances':opts['resolutions']=('[(8, 2, 8)]','std::vector<int64_t>{8, 2, 8}')
        else:opts['n_fft']=('8','8')
    py+=['',f'metric = "{name}"']
    tail=''
    if opts:
        py+=['options = dict('+', '.join(k+'='+v[0] for k,v in opts.items())+')']
        tail=', **options'
    cpp+=['',f'const std::string metric = "{name}";','tp::ComparisonOptions options;']
    cpp+=['options.'+key+' = '+value[1]+';' for key,value in opts.items()]
    structured=family in ('gaussian','mixture')
    py+=['all_pairs = tp.compare(X, Y, metric=metric'+tail+')  # [2,3]']
    cpp+=['auto all_pairs = tp::compare(X, Y, metric, "all_pairs", options);  // [2,3]']
    if structured:
        py+=['paired_Y = tuple(t[:2] for t in Y)']
        cpp+=['tp::ComparisonCollection paired_Y{'+', '.join(f'Y[{i}].narrow(0, 0, 2)' for i in range(2 if family=='gaussian' else 3))+'};']
    else:
        py+=['paired_Y = Y[:2]']
        cpp+=['auto paired_Y = Y.narrow(0, 0, 2);']
    py+=['paired = tp.compare(X, paired_Y, metric=metric, mode="paired"'+tail+')  # [2]']
    cpp+=['auto paired = tp::compare(X, paired_Y, metric, "paired", options);  // [2]']
    if 'condensed' in c['modes']:
        py+=['condensed = tp.compare(X, metric=metric, mode="condensed"'+tail+')  # [1]']
        cpp+=['auto condensed = tp::compare(X, c10::nullopt, metric, "condensed", options);  // [1]']
    if structured:
        py+=['x, y = tuple(t[0] for t in X), tuple(t[0] for t in Y)']
        cpp+=['tp::ComparisonCollection x{'+', '.join(f'X[{i}][0]' for i in range(2 if family=='gaussian' else 3))+'};',
              'tp::ComparisonCollection y{'+', '.join(f'Y[{i}][0]' for i in range(2 if family=='gaussian' else 3))+'};']
    else:
        py+=['x, y = X[0], Y[0]']
        cpp+=['auto x = X[0], y = Y[0];']
    py+=['single = tp.compare(x, y, metric=metric, mode="single"'+tail+')  # scalar']
    cpp+=['auto single = tp::compare(x, y, metric, "single", options);  // scalar']
    return code_pair('\n'.join(py),'\n'.join(cpp))


def page(c,note,named):
    name=c['name'];sample,x,y=shapes(c)
    content=f'''<div class="eyebrow">{escape(category(c))}</div>

# {note['title']}

<div class="metric-key">{name}</div>
{badges(c)}

\\[
{note['formula']}
\\]

{note['note']}

## Shapes and modes

| Input | Layout |
|---|---|
| One sample | `{sample}` |
| Collection X | `{x}` |
| Collection Y | `{y}` |

| Mode | Selector | Output |
|---|---|---|
| <span class="badge mode-all-pairs">all-pairs</span> | `compare(X,Y,metric=...)` | `[...,N,M]` |
| <span class="badge mode-paired">paired</span> | `compare(X,Y,metric=...,mode="paired")` | `[...,N]`; equal sample counts |
'''
    if 'condensed' in c['modes']:content+='| <span class="badge mode-condensed">condensed</span> | `compare(X,metric=...,mode="condensed")` | `[...,N*(N-1)//2]`; strict upper triangle |\n'
    content+='| <span class="badge mode-single">single</span> | `compare(x,y,metric=...,mode="single")` | `[...]`; leading batches only |\n'
    content+='\nLeading batch dimensions broadcast. '+('Condensed mode is not exposed for this directional comparison. ' if 'condensed' not in c['modes'] else '')+'See [mode and shape conventions](../guide/modes.md).\n'
    content+='\n## Entry points\n\n'
    functions=[('All-pairs',name)]
    for mode in ('paired','condensed'):
        if mode in c['modes'] and mode+'_'+name in named:functions.append((mode.capitalize(),mode+'_'+name))
    content+='| Named Python entry | Function |\n|---|---|\n'+''.join(f'| {label} | `{function}` |\n' for label,function in functions)
    content+='\nThe Python and C++ `compare` selectors cover every mode shown above. '
    if c['role']=='distance':content+='`cdist` / `pairwise_distances` and `paired_distances` select this comparison; '+('`pdist` / `condensed_distances` select its condensed mode. ' if 'condensed' in c['modes'] else 'condensed distance selectors reject it. ')
    elif c['role']=='kernel':content+='`pairwise_kernels`, `paired_kernels` and `condensed_kernels` also select it. '
    else:content+='Use `compare` for this '+c['role']+'; distance/kernel selectors have different roles. '
    content+='`single_comparison` is shorthand for single mode.\n'
    aliases=', '.join('`'+a+'`' for a in c['aliases'])
    if aliases:content+='\nCatalog aliases: '+aliases+'.\n'
    if c['parameters']:content+='\nAccepted metric options: '+', '.join('`'+a+'`' for a in c['parameters'])+'. Omitted options keep the metric defaults; unused options are rejected.\n'
    content+='\n## Example\n\nChoose Python or C++ in the top bar; the setting applies across this guide.\n\n'+examples(c)+'\n'
    content+='\n## Streamed evaluation\n\n'
    content+='`tp.lazy(x, y, name)` binds this '+('kernel, with its feature-space geometry methods,' if c['role']=='kernel' else c['role'])+' for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).\n'
    content+='\n## Numerical contract\n\n'+('Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. ' if c['differentiable'] else 'Token inputs are integer tensors with integer outputs; no input derivatives. ' if c['family']=='tokens' else 'Boolean/equality inputs do not have derivatives; feature weights may have derivatives for finite weighted formulas. ')
    content+=f'See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/{note["guide"]}) and [precision and deployment](../guide/deployment.md).\n'
    if note['related']:
        content+='\n## Related implementations\n\n'+''.join(f'- [{label}]({url}).\n' for label,url in note['related'])
        content+='\nRelated APIs can differ in reduction, normalization, units or boundary behavior; match these before comparing values.\n'
    return content


def generate():
    catalog=native_catalog();notes=json.loads((SITE/'metric-notes.json').read_text())
    assert {c['name'] for c in catalog}==set(notes),'Native catalog and authored metric notes differ'
    named=set()
    for source in (ROOT/'torchpairwise/ops').rglob('*.py'):
        named.update(node.name for node in ast.walk(ast.parse(source.read_text())) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)))
    files={}
    groups=defaultdict(list)
    for c in catalog:
        groups[category(c)].append(c)
        files['metrics/'+c['name']+'.md']=page(c,notes[c['name']],named)
    summary='''# Summary

[Overview](index.md)

# Getting started

- [Install and build](guide/install.md)
- [Quick start](guide/start.md)
- [Using torchpairwise in models](guide/ml.md)
- [Modes and shapes](guide/modes.md)
- [Comparison selection](guide/selection.md)

# Streamed operators

- [Sources and consumers](guide/streaming.md)
- [Algebra and kernel methods](guide/algebra.md)
- [Traces and matrix functions](guide/spectral.md)
- [Reductions and statistics](guide/statistics.md)
- [Feature and image axes](guide/events.md)

# Example gallery

- [Streaming ML algorithms](examples/index.md)

# Metric reference

- [Browse all comparisons](metrics/index.md)
'''
    for group in GROUPS:
        slug=group.lower().replace(' ','-')
        summary+=f'- [{group}](metrics/{slug}.md)\n'
        members=sorted(groups[group],key=lambda c:notes[c['name']]['title'])
        familypage=f'<div class="eyebrow">Metric reference</div>\n\n# {group}\n\nChoose a comparison for its equation, input shapes, modes and Python/C++ examples.\n\n'
        familypage+='| Comparison | Role | Modes |\n|---|---|---|\n'
        for c in members:
            title=notes[c['name']]['title']
            summary+=f'  - [{title}](metrics/{c["name"]}.md)\n'
            familypage+=f'| [{title}]({c["name"]}.md) | {c["role"]} | '+', '.join(c['modes'])+' |\n'
        files['metrics/'+slug+'.md']=familypage
    summary+='''
- [Custom Comparisons](metrics/custom-comparisons.md)
  - [User-Defined Metrics](metrics/custom-metrics.md)
  - [Encoded Feature Reuse](metrics/encoded-features.md)

# Integration

- [Python and C++](guide/cpp.md)
- [Extending from C++](guide/extensions.md)
- [Precision and deployment](guide/deployment.md)
- [Benchmarking](guide/benchmarks.md)
'''
    files['SUMMARY.md']=summary
    index='''<div class="eyebrow">Metric reference</div>

# Comparison catalog

87 entries across 9 input families, including 13 kernels. Entries include
mathematical synonyms; each page states its equation and conventions.
Use the language toggle for Python or C++ examples.

<div data-metric-catalog>
<div class="catalog-controls" hidden>
<label>Find a comparison<input type="search" placeholder="Name, alias, family or role…" aria-label="Filter comparisons" /></label>
<label>Input family<select aria-label="Filter input family"><option value="">All input families</option>'''
    index+=''.join(f'<option value="{key}">{value}</option>' for key,value in FAMILY_NAMES.items())
    index+='''</select></label>
</div>
<p class="catalog-status" role="status" aria-live="polite">87 comparisons</p>
<div class="metric-grid">
'''
    for c in sorted(catalog,key=lambda c:notes[c['name']]['title']):
        n=notes[c['name']];search=escape(' '.join([c['name'],n['title'],c['family'],c['role'],*c['aliases']]).lower(),quote=True)
        index+=f'<article class="metric-card" data-family="{c["family"]}" data-search="{search}"><a href="{c["name"]}.html">{escape(n["title"])}</a><p>{FAMILY_NAMES[c["family"]]} · {c["role"]}</p>{badges(c,True)}</article>\n'
    index+='</div>\n</div>\n'
    files['metrics/index.md']=index
    return {name: format_headers(content, navigation=name == 'SUMMARY.md')
            for name, content in files.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    files=generate();different=[]
    for name,source in files.items():
        target=SITE/'src'/name
        if args.check:
            if not target.exists() or target.read_text()!=source:different.append(name)
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source)
    if different:raise SystemExit('Reference pages need regeneration: '+', '.join(different))
    print(('Checked' if args.check else 'Generated'),len(files),'reference/navigation pages')


if __name__=='__main__':main()
