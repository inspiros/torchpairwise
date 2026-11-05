from typing import Any, Callable, Literal, Optional, Sequence, Union
import torch
from torch import Tensor, dtype
from ._pairwise_operator import _ScalarOperator

Mode = Literal['all_pairs', 'paired', 'condensed', 'single']
Form = Literal['scalar', 'broadcast']

class CustomPairwiseOperator(_ScalarOperator):
    @property
    def shape(self) -> torch.Size: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def T(self) -> _ScalarOperator: ...

class CustomComparison:
    mode: Mode
    form: Form
    parameters: tuple[Tensor, ...]
    forward_op: str
    backward_op: str
    def __call__(self, x1: Tensor, x2: Optional[Tensor] = None, *, mode: Optional[Mode] = None) -> Tensor: ...

def make_pairwise(fn: Union[torch.nn.Module, Callable[..., Tensor], str], *,
                  backward: Optional[Union[Callable[..., Sequence[Tensor]], str]] = None,
                  parameters: Sequence[Tensor] = (), mode: Mode = 'all_pairs',
                  sample_ndim: int = 1, chunk_size: Optional[int] = None,
                  compile: bool = False, symmetric: bool = False,
                  output_dtype: Optional[dtype] = None, name: Optional[str] = None,
                  form: Form = 'scalar', keops: bool = False) -> CustomComparison: ...
