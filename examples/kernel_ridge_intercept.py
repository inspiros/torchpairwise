"""Kernel ridge with an unpenalized intercept, using a block KKT MINRES solve.

python examples/kernel_ridge_intercept.py --device cuda --dense-reference

For symmetric PSD K and alpha>0, solve
[K+alpha I, 1; 1.T, 0] [coefficients; intercept] = [targets; 0].
The KKT map is symmetric indefinite. Its only explicit zero block is [1,1].
Held-out predictions use a rectangular streamed kernel product plus intercept.
No [N,N] Gram or [(N+1),(N+1)] KKT matrix is stored in the streamed path.
"""
import argparse

import torch
import torchpairwise as tp


@torch.no_grad()
def fit_kernel_ridge(x, targets, *, alpha=.3, gamma=.3, chunk_size=64,
                     rtol=1e-9, max_iter=300):
    """Return coefficients [N,R], unpenalized intercept [1,R], and MINRES fit."""
    if x.ndim!=2 or targets.ndim!=2 or targets.shape[0]!=x.shape[0] or x.shape[0]==0:
        raise ValueError('expected nonempty X:[N,D] and targets:[N,R]')
    if not 0<alpha<float('inf'):
        raise ValueError('alpha must be finite/positive')
    kernel=tp.lazy(x,gamma=gamma,chunk_size=chunk_size, metric='rbf')
    a=kernel + alpha * tp.eye_like(kernel)
    ones=tp.as_lazy(x.new_ones(1,x.shape[0]))
    kkt=tp.block([[a,ones.T],[ones,x.new_zeros(1,1)]])
    rhs=torch.cat((targets,targets.new_zeros(1,targets.shape[-1])),-2)
    fit=tp.linalg.minres(kkt,rhs,rtol=rtol,max_iter=max_iter)
    if not fit.converged.all():
        raise RuntimeError(f'MINRES failed: status={fit.status.tolist()}, '
                           f'residual={fit.residual_norm.tolist()}')
    return fit.solution[:-1],fit.solution[-1:],fit


@torch.no_grad()
def predict(x_test,x,coefficients,intercept,*,gamma=.3,chunk_size=64):
    return tp.lazy(x_test,x,gamma=gamma,chunk_size=chunk_size, metric='rbf').matmul(coefficients)+intercept


def targets(x):
    return torch.cat((2.+x[:,:1].sin(),-1.+.2*x[:,1:2].square()),-1)


@torch.no_grad()
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--samples',type=int,default=128)
    parser.add_argument('--test-samples',type=int,default=32)
    parser.add_argument('--chunk-size',type=int,default=32)
    parser.add_argument('--dense-reference',action='store_true')
    args=parser.parse_args(argv)
    if min(args.samples,args.test_samples)<1:
        parser.error('sample counts must be positive')
    generator=torch.Generator(device=args.device).manual_seed(47)
    x=torch.randn(args.samples,3,device=args.device,dtype=torch.float64,generator=generator)
    test=torch.randn(args.test_samples,3,device=x.device,dtype=x.dtype,generator=generator)
    y=targets(x)
    alpha,gamma=.3,.3
    coefficients,intercept,fit=fit_kernel_ridge(x,y,alpha=alpha,gamma=gamma,
                                              chunk_size=args.chunk_size,rtol=1e-10)
    scores=predict(test,x,coefficients,intercept,gamma=gamma,chunk_size=args.chunk_size)
    print('Coefficients:',list(coefficients.shape),'intercept:',intercept.tolist())
    print('Held-out predictions:',list(scores.shape),'RMSE:',(scores-targets(test)).square().mean().sqrt().item())
    print('Lanczos steps:',fit.iterations.tolist(),'KKT residual:',fit.residual_norm.tolist())
    torch.testing.assert_close(coefficients.sum(0),torch.zeros_like(coefficients.sum(0)),atol=1e-7,rtol=0)
    if args.dense_reference:
        gram=tp.rbf_kernel(x,gamma=gamma)
        shifted=gram+alpha*torch.eye(args.samples,device=x.device,dtype=x.dtype)
        ones=x.new_ones(args.samples,1)
        matrix=torch.cat((torch.cat((shifted,ones),-1),torch.cat((ones.T,x.new_zeros(1,1)),-1)),-2)
        rhs=torch.cat((y,y.new_zeros(1,y.shape[-1])),-2)
        reference=torch.linalg.solve(matrix,rhs)
        torch.testing.assert_close(coefficients,reference[:-1],atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(intercept,reference[-1:],atol=2e-8,rtol=2e-7)
        expected=tp.rbf_kernel(test,x,gamma=gamma)@reference[:-1]+reference[-1:]
        torch.testing.assert_close(scores,expected,atol=2e-7,rtol=2e-6)
        # Independent Schur-complement solution, using two SPD solves rather
        # than indefinite factorization, verifies the intercept convention.
        solved=torch.linalg.solve(shifted,torch.cat((y,ones),-1))
        bias=solved[:,:-1].sum(0,keepdim=True)/solved[:,-1:].sum()
        weights=solved[:,:-1]-solved[:,-1:]*bias
        torch.testing.assert_close(intercept,bias,atol=2e-8,rtol=2e-7)
        torch.testing.assert_close(coefficients,weights,atol=2e-7,rtol=2e-6)
        print('Dense KKT, Schur-complement intercept, coefficient constraint and held-out predictions agree')


if __name__=='__main__':
    main()
