"""Dense image maps and native reference-axis reductions over aligned images."""
import torch
from ._image import _prepare, _window_options, _positive_scalar


def _call(x, y, comparison, mode, reduction, data_range, chunk_size, window_size,
          sigma, k1, k2, keepdim):
    if comparison not in ('residual', 'absolute_error', 'squared_error', 'ssim'):
        raise ValueError('Unknown image map comparison')
    if reduction not in ('none', 'sum', 'mean', 'min', 'max'):
        raise ValueError('Image maps support sum, mean, min or max reduction')
    if type(keepdim) is not bool:
        raise TypeError('keepdim must be a boolean')
    if not torch.compiler.is_compiling():
        _prepare(x, x if y is None else y, mode, chunk_size)
        if comparison == 'ssim':
            data_range = _positive_scalar(data_range, 'data_range')
            _window_options(x, window_size, sigma, k1, k2, None)
    return torch.ops.torchpairwise.image_maps(x, y, comparison, mode, reduction,
        data_range, chunk_size, window_size, sigma, k1, k2, keepdim)


def image_maps(x1, x2=None, *, comparison='squared_error', mode='all_pairs',
               data_range=None, window_size=11, sigma=1.5, k1=.01, k2=.03, chunk_size=16):
    """Unreduced aligned image comparisons [...,N,M,C,H,W].

    residual is Y-X; absolute_error and squared_error transform that residual.
    ssim requires data_range and uses the existing valid Gaussian-window SSIM
    convention, with spatial size H-window_size+1, W-window_size+1.
    Paired and condensed modes retain C/H/W; directed residual cannot condense.
    A complete map needs its natural output storage. Use tp.image_map_operator(...).reduce(dim=-4) to
    stream directly over reference images instead. First gradients replay tiles.
    """
    return _call(x1,x2,comparison,mode,'none',data_range,chunk_size,window_size,sigma,k1,k2,False)



__all__ = ['image_maps']
