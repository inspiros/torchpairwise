"""Compare equivalent dense/streamed consumers, validating values and gradients.

Run from a source checkout with its native extension installed/built. For usage:
    python examples/benchmark_operators.py --help
JSON files are opt-in; disposable results belong in develop/.
"""
import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def probability(text):
    value = float(text)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('must be finite and between 0 and 1')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=('distance-sum', 'rbf-matmul', 'distance-quantile', 'kernel-regression', 'attention'), default='distance-sum')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='CUDA uses the device selected by CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    for name, default in (('n', 512), ('m', 768), ('d', 32), ('channels', 8), ('chunk-size', 128), ('feature-chunk-size', 32), ('warmup', 2), ('repeats', 5), ('threads', 1)):
        parser.add_argument('--'+name, type=positive_integer, default=default)
    parser.add_argument('--q', type=probability, default=.9)
    parser.add_argument('--backward', action='store_true', help='time forward plus grad(output.sum()) for every floating input')
    parser.add_argument('--skip-compile', action='store_true', help='omit the compiled-PyTorch baseline; otherwise compilation is warmed before timing')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--output', type=Path, help='write JSON here; environment metadata omits hostname/account/module paths')
    return parser.parse_args(argv)


def synchronize(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def execute(torch, fn, inputs, backward):
    with torch.enable_grad() if backward else torch.no_grad():
        value = fn(*inputs)
        if backward:
            gradients = torch.autograd.grad(value.sum(), inputs)
            return (value.detach(), *(g.detach() for g in gradients))
        return (value.detach(),)


def rank_parameters(torch, m, q, dtype):
    # Match native/torch.quantile rank arithmetic without timed device scalars.
    rank = float(torch.tensor(q, dtype=dtype) * (m - 1))
    return math.floor(rank), math.ceil(rank), rank - math.floor(rank)


def variants(torch, tp, args, dtype):
    def distances(x, y):
        return (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)
    lower, upper, weight = rank_parameters(torch, args.m, args.q, dtype)
    def quantile(values):
        # Stable sorting matches the operator's tie policy. Benchmark inputs
        # are finite; this is linear quantile, not a general NaN-aware adapter.
        ordered = values.sort(dim=-1, stable=True).values
        return torch.lerp(ordered[..., lower], ordered[..., upper], weight)
    gamma = 1. / args.d
    if args.case == 'distance-sum':
        def eager(x, y): return distances(x, y).sum(-1)
        methods = {
            'native-dense': lambda x, y: tp.sqeuclidean_distances(x, y).sum(-1),
            'reduce-streamed': lambda x, y: tp.lazy(x, y, metric='sqeuclidean', chunk_size=args.chunk_size).reduce('sum', dim=-1),
            'operator-streamed': lambda x, y: tp.lazy(x, y, metric='sqeuclidean', chunk_size=args.chunk_size).sum(-1),
            'event-streamed': lambda x, y: tp.residual_operator(x, y, transform='square', chunk_size=args.chunk_size, feature_chunk_size=args.feature_chunk_size).sum(-1).sum(-1),
        }
    elif args.case == 'rbf-matmul':
        def eager(x, y, v): return (-gamma * distances(x, y)).exp().matmul(v)
        methods = {
            'native-dense': lambda x, y, v: tp.rbf_kernel(x, y, gamma=gamma).matmul(v),
            'matmul-streamed': lambda x, y, v: tp.lazy(x, y, metric='rbf', gamma=gamma, chunk_size=args.chunk_size) @ v,
            'operator-streamed': lambda x, y, v: tp.lazy(x, y, metric='rbf', gamma=gamma, chunk_size=args.chunk_size).matmul(v),
        }
    elif args.case == 'distance-quantile':
        def eager(x, y): return quantile(distances(x, y))
        methods = {
            'native-dense': lambda x, y: quantile(tp.sqeuclidean_distances(x, y)),
            'operator-streamed': lambda x, y: tp.lazy(x, y, metric='sqeuclidean', chunk_size=args.chunk_size).quantile(args.q),
            'event-streamed': lambda x, y: tp.residual_operator(x, y, transform='square', chunk_size=args.chunk_size, feature_chunk_size=args.feature_chunk_size).sum(-1).quantile(args.q),
        }
    elif args.case == 'kernel-regression':
        from streaming_ml import kernel_regression, dense_kernel_regression
        def eager(x, y, v):
            kernel = (-gamma * distances(x, y)).exp()
            values = torch.cat((v, torch.ones_like(v[:, :1])), dim=-1)
            weighted = kernel @ values
            return weighted[:, :-1] / weighted[:, -1:].clamp_min(torch.finfo(v.dtype).tiny)
        methods = {
            'native-dense': lambda x, y, v: dense_kernel_regression(x, y, v, gamma=gamma),
            'operator-streamed': lambda x, y, v: kernel_regression(x, y, v, gamma=gamma, chunk_size=args.chunk_size),
        }
    else:
        from streaming_ml import attention, dense_attention
        eager = dense_attention
        methods = {
            'operator-streamed': lambda x, y, v: attention(x, y, v, chunk_size=args.chunk_size),
        }
    methods = {'torch-eager-direct': eager, **methods}
    if not args.skip_compile:
        methods['torch-compiled-direct'] = torch.compile(eager, fullgraph=True)
    return methods


def revision():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, stderr=subprocess.DEVNULL))
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def environment(torch, tp, device):
    from torchpairwise.extension import _get_extension_path
    library = Path(_get_extension_path('_C'))
    digest = hashlib.sha256()
    with library.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    result = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchpairwise': tp.__version__, 'platform': platform.system(),
        'cuda_runtime': torch.version.cuda, 'device': str(device),
        'threads': torch.get_num_threads(), 'matmul_precision': torch.get_float32_matmul_precision(),
        'extension_sha256': digest.hexdigest(), 'extension_bytes': library.stat().st_size,
        'revision': revision(),
    }
    if device.type == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(device), compute_capability=list(torch.cuda.get_device_capability(device)))
    return result


def write_report(report, output):
    payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    sys.stdout.write(payload)


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torchpairwise as tp
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; choose --device cpu or select an accessible GPU')
    dtype = getattr(torch, args.dtype)
    # A local generator makes inputs reproducible without relying on another
    # example's global RNG state. It also records the seed in the result.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    def tensor(shape):
        return torch.rand(shape, device=device, dtype=dtype, generator=generator, requires_grad=args.backward)
    inputs = (tensor((args.n, args.d)), tensor((args.m, args.d)))
    if args.case in ('rbf-matmul', 'kernel-regression', 'attention'):
        inputs += (tensor((args.m, args.channels)),)
    methods = variants(torch, tp, args, dtype)
    config = vars(args).copy(); config.pop('output')
    report = {
        'schema_version': 1, 'configuration': config, 'environment': environment(torch, tp, device),
        'methodology': {
            'timing': 'synchronized wall time; rotated/shuffled variant order each repeat',
            'work': 'forward+grad(output.sum())' if args.backward else 'forward without autograd',
            'torch_formula': ('softmax((Q @ K.T) / sqrt(D)) @ V' if args.case == 'attention' else
                              'broadcast differences, square, feature sum; RBF gamma=1/D; regression carries row mass in one value channel; quantile uses stable sort and linear interpolation'),
            'gpu_memory': 'extra max_memory_allocated above live baseline; allocator reservation excluded; CPU peak is not measured',
            'inputs': 'finite contiguous unbatched random tensors shared by all variants; operator construction/preparation included',
            'compile': 'first call reported separately; compilation/autotuning excluded from steady state',
        },
        'logical_score_bytes': args.n * args.m * torch.empty((), dtype=dtype).element_size(),
        'input_bytes': sum(t.numel() * t.element_size() for t in inputs),
        'validation': {'rtol': 5e-4 if dtype == torch.float32 else 1e-7, 'atol': 5e-5 if dtype == torch.float32 else 1e-8},
        'results': {},
    }
    try:
        reference = execute(torch, methods['torch-eager-direct'], inputs, args.backward)
        synchronize(torch, device)
    except RuntimeError as exc:
        # No variant can be checked if the dense reference itself fails.
        report['reference_error'] = str(exc)
        print('Dense validation reference failed; no timings recorded', file=sys.stderr)
        write_report(report, args.output)
        return 1
    active = {}
    failed = False
    # Validate and warm every path before measurement. Failures are explicit;
    # they never turn into timing results for an unchecked variant.
    for name, fn in methods.items():
        first_call_ms = None
        actual = None
        try:
            synchronize(torch, device); start = time.perf_counter()
            actual = execute(torch, fn, inputs, args.backward)
            synchronize(torch, device); first_call_ms = (time.perf_counter()-start)*1000
            torch.testing.assert_close(actual, reference, **report['validation'])
            del actual
            for _ in range(args.warmup):
                execute(torch, fn, inputs, args.backward)
            synchronize(torch, device)
            active[name] = fn
            report['results'][name] = {'status': 'validated', 'first_call_ms': first_call_ms, 'samples_ms': [], 'extra_peak_bytes_samples': [], 'live_baseline_bytes_samples': []}
        except (RuntimeError, AssertionError) as exc:
            synchronize(torch, device)
            report['results'][name] = {'status': 'error', 'first_call_ms': first_call_ms, 'error': str(exc)}
            print(name + ': validation/warmup failed', file=sys.stderr)
            failed = True
        finally:
            # Do not keep a failed path's outputs alive during later timings.
            actual = None
    del reference
    order = list(active)
    rng = random.Random(args.seed)
    for _ in range(args.repeats):
        rng.shuffle(order)
        for name in order:
            try:
                synchronize(torch, device)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline = torch.cuda.memory_allocated(device)
                else:
                    baseline = None
                start = time.perf_counter()
                result = execute(torch, active[name], inputs, args.backward)
                synchronize(torch, device)
                elapsed = (time.perf_counter()-start)*1000
                peak = torch.cuda.max_memory_allocated(device)-baseline if device.type == 'cuda' else None
                del result
                record = report['results'][name]
                record['samples_ms'].append(elapsed)
                record['extra_peak_bytes_samples'].append(peak)
                record['live_baseline_bytes_samples'].append(baseline)
            except RuntimeError as exc:
                report['results'][name].update(status='error', error=str(exc))
                active.pop(name)
                failed = True
        order = list(active)
    for name, record in report['results'].items():
        if record['status'] != 'validated':
            continue
        samples = record['samples_ms']
        record.update(status='ok', median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
        peaks = [v for v in record['extra_peak_bytes_samples'] if v is not None]
        record['extra_peak_bytes'] = max(peaks) if peaks else None
        memory = 'unmeasured' if not peaks else f'{max(peaks)/1024**2:.2f} MiB'
        print(f'{name:24s} {record["median_ms"]:10.3f} ms  extra GPU peak: {memory}', file=sys.stderr)
    write_report(report, args.output)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
