"""Measure complete dense and streamed kernel fits with identical inputs.

The default dense baseline uses the same iterative solver as streaming, isolating
matrix storage and evaluation. Stored sources use direct ATen products;
generated sources evaluate tiles. --dense-direct adds a separately labeled full
solve/eigh/SVD baseline. CPU tensor peak memory is not measured. Results and
commands intended for sharing contain no private host or environment paths.
"""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import gc
import math
import random
import statistics
import sys
import time

from benchmark_operators import environment, positive_integer, synchronize, write_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('ridge', 'landmark', 'pca', 'lda', 'svd'), default='ridge')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    for name, default in (('n', 1024), ('m', 768), ('queries', 128), ('d', 8), ('channels', 2),
                          ('rank', 2), ('subspace-size', 10), ('classes', 4), ('chunk-size', 128),
                          ('max-iter', 200), ('warmup', 1), ('repeats', 3), ('threads', 1)):
        parser.add_argument('--'+name, type=positive_integer, default=default)
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=1e-9)
    parser.add_argument('--regularization', type=float, default=.5,
                        help='ridge/LDA identity shift; landmark uses its square as LSQR damp^2')
    parser.add_argument('--backward', action='store_true', help='ridge/landmark only: include mean squared prediction and input VJPs')
    parser.add_argument('--dense-direct', action='store_true', help='also measure direct dense solve/full eigh/SVD')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v >= 0 for v in (args.rtol, args.atol)):
        parser.error('tolerances must be finite and nonnegative')
    if not math.isfinite(args.regularization) or args.regularization <= 0:
        parser.error('regularization must be finite and positive')
    if args.backward and args.case not in ('ridge', 'landmark'):
        parser.error('--backward supports ridge and landmark only')
    if args.case in ('pca', 'svd') and not 1 <= args.rank <= args.subspace_size <= min(args.n, args.m if args.case == 'svd' else args.n):
        parser.error('require rank <= subspace-size <= the smaller matrix dimension')
    if args.case == 'lda' and not 1 <= args.rank < args.classes <= args.n:
        parser.error('LDA requires rank < classes <= n')
    return args


def workload(torch, tp, args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    def tensor(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator,
                           requires_grad=args.backward)
    x = tensor(args.n, args.d)
    inputs = [x]
    y = tensor(args.m, args.d) if args.case in ('landmark', 'svd') else None
    if y is not None:
        inputs.append(y)
    query = tensor(args.queries, args.d) if args.case != 'svd' else None
    if query is not None:
        inputs.append(query)
    targets = tensor(args.n, args.channels) if args.case in ('ridge', 'landmark') else None
    if targets is not None:
        inputs.append(targets)
    width = args.subspace_size
    initial = torch.randn(args.m if args.case == 'svd' else args.n, width,
                          dtype=dtype, device=device, generator=generator) if args.case in ('pca', 'svd') else None
    labels = torch.arange(args.n, device=device) % args.classes if args.case == 'lda' else None
    if labels is not None:
        # Separated classes avoid a numerically negligible between-class rank.
        means = torch.randn(args.classes, args.d, dtype=dtype, device=device, generator=generator)
        x.add_(means[labels])
        membership = torch.nn.functional.one_hot(labels, args.classes).to(dtype)
        counts = membership.sum(0)
        normalized = membership / counts.sqrt()
        class_block = normalized - normalized.mean(0, keepdim=True)
    gamma, alpha = 1. / args.d, args.regularization
    def check(fit):
        if not fit.converged.all():
            raise RuntimeError(f'{args.case} did not converge; status={fit.status.tolist()}')
    def run(kind):
        dense = kind != 'streamed'
        direct = kind == 'dense-direct'
        def kernel(a, b=None):
            if dense:
                return tp.as_lazy(tp.rbf_kernel(a, b, gamma=gamma), chunk_size=args.chunk_size)
            return tp.lazy(a, b, gamma=gamma, chunk_size=args.chunk_size, metric='rbf')
        refs = y if args.case == 'landmark' else x
        source = None if direct else kernel(x, y if args.case in ('landmark', 'svd') else None)
        a = None if direct else source
        diagnostics = {}
        if args.case in ('ridge', 'landmark'):
            if direct:
                matrix = tp.rbf_kernel(x, refs, gamma=gamma)
                if args.case == 'ridge':
                    coefficients = torch.linalg.solve(matrix + alpha * torch.eye(args.n, device=device, dtype=dtype), targets)
                else:
                    # QR solve of the augmented system, avoiding a dense normal matrix.
                    augmented = torch.cat((matrix, alpha * torch.eye(args.m, device=device, dtype=dtype)), 0)
                    rhs = torch.cat((targets, targets.new_zeros(args.m, args.channels)), 0)
                    q, r = torch.linalg.qr(augmented, mode='reduced')
                    coefficients = torch.linalg.solve_triangular(r, q.T @ rhs, upper=True)
            else:
                options = dict(rtol=args.rtol, atol=args.atol, max_iter=args.max_iter, differentiable=args.backward)
                fit = (tp.linalg.solve(a + alpha * tp.eye_like(a), targets, method='cg', **options)
                       if args.case == 'ridge' else tp.linalg.lstsq(a, targets, damp=alpha, **options))
                check(fit)
                coefficients = fit.solution
                diagnostics = {'iterations': fit.iterations, 'residual_norm': fit.residual_norm}
            prediction = kernel(query, refs).matmul(coefficients)
            result = dict(coefficients=coefficients, prediction=prediction)
        elif args.case == 'pca':
            if direct:
                matrix = tp.rbf_kernel(x, gamma=gamma)
                mean = matrix.mean(0)
                centered = matrix - matrix.mean(-1, keepdim=True) - mean + mean.mean()
                values, vectors = torch.linalg.eigh((centered + centered.T) * .5)
                values, vectors = values[-args.rank:].flip(0), vectors[:, -args.rank:].flip(-1)
            else:
                mean = source.reduce('mean', dim=-2)
                fit = tp.linalg.eigh(a.centered(mean), k=args.rank, initial=initial, max_iter=args.max_iter,
                                    rtol=args.rtol, atol=args.atol)
                check(fit)
                values, vectors = fit.eigenvalues, fit.eigenvectors
                diagnostics = {'iterations': fit.iterations, 'residual_norm': fit.residual_norm}
            cutoff = torch.finfo(dtype).eps * args.n * values.abs().max()
            positive = values > cutoff
            safe = torch.where(positive, values, torch.ones_like(values))
            weights = vectors * torch.where(positive, safe.rsqrt(), torch.zeros_like(safe))
            prediction = kernel(query, x).centered(mean).matmul(weights)
            result = dict(values=values, basis=vectors, prediction=prediction)
        elif args.case == 'svd':
            if direct:
                matrix = tp.rbf_kernel(x, y, gamma=gamma)
                left, values, vh = torch.linalg.svd(matrix, full_matrices=False)
                values, left, right = values[:args.rank].clone(), left[:, :args.rank].clone(), vh[:args.rank].T.clone()
            else:
                fit = tp.linalg.svd(a, k=args.rank, initial=initial, max_iter=args.max_iter, rtol=args.rtol, atol=args.atol)
                check(fit)
                values, left, right = fit.singular_values, fit.left_vectors, fit.right_vectors
                diagnostics = {'iterations': fit.iterations, 'residual_norm': fit.residual_norm}
            result = dict(values=values, basis=left, right_basis=right)
        else:
            projection = tp.group_projection(labels, like=x)
            if direct:
                matrix = tp.rbf_kernel(x, gamma=gamma)
                residual = matrix - projection.matmul(matrix)
                within = residual.T @ residual + alpha * torch.eye(args.n, device=device, dtype=dtype)
                f = matrix @ class_block
                z = torch.linalg.solve(within, f)
            else:
                residual = a - projection @ a
                within = residual.mT @ residual
                within = within + alpha * tp.eye_like(within)
                f = source.matmul(class_block)
                fit = tp.linalg.solve(within, f, method='cg', rtol=args.rtol, atol=args.atol, max_iter=args.max_iter)
                check(fit)
                z = fit.solution
                diagnostics = {'iterations': fit.iterations, 'residual_norm': fit.residual_norm}
            small = f.T @ z
            values, u = torch.linalg.eigh((small + small.T) * .5)
            values, u = values[-args.rank:].flip(0), u[:, -args.rank:].flip(-1)
            if (values <= torch.finfo(dtype).eps * args.n * values.abs().max()).any():
                raise RuntimeError('LDA between-class numerical rank below requested rank')
            directions = z @ (u / values.sqrt())
            result = dict(values=values, basis=directions, prediction=kernel(query, x).matmul(directions))
        if args.backward:
            loss = result['prediction'].square().mean()
            gradients = torch.autograd.grad(loss, inputs)
            result = {name: value.detach() for name, value in result.items()}
            result.update({f'input_gradient_{i}': g.detach() for i, g in enumerate(gradients)})
        result['diagnostics'] = diagnostics
        return result
    common = [t for t in (initial, labels) if t is not None]
    if labels is not None:
        common.append(class_block)
    return run, inputs, common


def validate(torch, actual, expected, args):
    rtol, atol = max(args.rtol * 50, 1e-7), max(args.atol * 50, 1e-8)
    errors = {}
    for name, value in actual.items():
        if name in ('diagnostics', 'basis', 'right_basis', 'prediction') and 'basis' in actual:
            continue
        if name == 'diagnostics':
            continue
        difference = (value - expected[name]).norm().item()
        reference_norm = expected[name].norm().item()
        if not math.isfinite(difference) or difference > max(atol, rtol * reference_norm):
            raise AssertionError(f'{name} global L2 error {difference}, reference norm {reference_norm}')
        errors[name + '_relative_l2'] = difference / max(reference_norm, torch.finfo(value.dtype).tiny)
        errors[name + '_max_abs'] = (value - expected[name]).abs().max().item()
    if 'basis' in actual:
        # Subspace checks avoid arbitrary signs and repeated-direction rotations.
        for name in ('basis', 'right_basis'):
            if name not in actual:
                continue
            q = torch.linalg.qr(actual[name], mode='reduced').Q
            ref = torch.linalg.qr(expected[name], mode='reduced').Q
            error = ((args.rank - (ref.T @ q).square().sum()).clamp_min(0) / args.rank).sqrt().item()
            if error > max(1e-3, args.rtol * 100):
                raise AssertionError(f'{name} subspace error={error}')
            errors[name + '_subspace_error'] = error
        if 'prediction' in actual:
            # Directions may rotate, but the held-out map must rotate with them.
            transform = torch.linalg.lstsq(actual['basis'], expected['basis']).solution
            difference = (actual['prediction'] @ transform - expected['prediction']).norm().item()
            reference_norm = expected['prediction'].norm().item()
            if not math.isfinite(difference) or difference > max(atol*10, rtol*10*reference_norm):
                raise AssertionError(f'held-out map L2 error {difference}, reference norm {reference_norm}')
            errors['prediction_aligned_relative_l2'] = difference / max(reference_norm, torch.finfo(actual['prediction'].dtype).tiny)
    return errors


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torchpairwise as tp
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    run, inputs, common = workload(torch, tp, args, device)
    names = ['dense-iterative', 'streamed'] + (['dense-direct'] if args.dense_direct else [])
    config = vars(args).copy(); config.pop('output')
    report = dict(schema_version=1, configuration=config, environment=environment(torch,tp,device), results={},
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        logical_fit_matrix_bytes=args.n*(args.m if args.case in ('svd','landmark') else args.n)*inputs[0].element_size(),
        input_bytes=sum(t.numel()*t.element_size() for t in inputs),
        common_auxiliary_bytes=sum(t.numel()*t.element_size() for t in common),
        methodology=dict(scope='source construction, fitting, and held-out prediction except SVD; optional backward included',
            baseline='dense-iterative uses the same native solver/start/options as streamed; dense-direct is optional and uses direct solve/full decomposition',
            gpu_memory='extra peak allocated CUDA bytes above live inputs after garbage collection before each sample; reserved cache and driver/external-library allocations excluded',
            cpu_memory='not measured; null',timing='synchronized wall time after warmup; shuffled method order',
            accuracy='shared inputs and spectral start; convergence plus value/VJP or sign/rotation-invariant subspace checks before timing'))
    active=[];failed=False
    def execute(name):
        with torch.enable_grad() if args.backward else torch.no_grad():
            return run(name)
    try:
        reference=execute('dense-iterative')
        synchronize(torch,device)
    except RuntimeError as error:
        report['reference_error']=str(error);write_report(report,args.output);return 1
    for name in names:
        actual=None
        try:
            synchronize(torch,device);start=time.perf_counter()
            actual=execute(name);synchronize(torch,device)
            first=(time.perf_counter()-start)*1000
            errors=validate(torch,actual,reference,args)
            diagnostics={key:value.tolist() for key,value in actual['diagnostics'].items()}
            actual=None
            for _ in range(args.warmup):execute(name)
            synchronize(torch,device)
            active.append(name)
            report['results'][name]=dict(status='validated',first_call_ms=first,validation=errors,diagnostics=diagnostics,
                                        samples_ms=[],extra_peak_bytes_samples=[],live_baseline_bytes_samples=[])
        except (RuntimeError,AssertionError) as error:
            report['results'][name]=dict(status='error',error=str(error));failed=True
        finally:actual=None
    del reference
    gc.collect()
    rng=random.Random(args.seed)
    for _ in range(args.repeats):
        order=list(active);rng.shuffle(order)
        for name in order:
            try:
                # Operator closures can form cycles. Release prior sources before
                # establishing the baseline so collection during this sample
                # cannot subtract a previous fit from its measured peak.
                gc.collect()
                synchronize(torch,device)
                baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                start=time.perf_counter();result=execute(name);synchronize(torch,device)
                elapsed=(time.perf_counter()-start)*1000
                peak=torch.cuda.max_memory_allocated(device)-baseline if device.type=='cuda' else None
                if peak is not None and peak < 0:
                    raise RuntimeError('CUDA allocated-memory peak fell below the sample baseline')
                del result
                record=report['results'][name]
                record['samples_ms'].append(elapsed);record['extra_peak_bytes_samples'].append(peak)
                record['live_baseline_bytes_samples'].append(baseline)
            except RuntimeError as error:
                report['results'][name].update(status='error',error=str(error));active.remove(name);failed=True
    for name in active:
        record=report['results'][name]
        peaks=[v for v in record['extra_peak_bytes_samples'] if v is not None]
        record.update(status='ok',median_ms=statistics.median(record['samples_ms']),extra_peak_bytes=max(peaks) if peaks else None)
        memory=f'{max(peaks)/1024**2:.2f} MiB' if peaks else 'unmeasured'
        print(f'{name:18s} {record["median_ms"]:10.3f} ms, extra GPU peak {memory}',file=sys.stderr)
    write_report(report,args.output)
    return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
