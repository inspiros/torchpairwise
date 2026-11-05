from typing import NamedTuple, Optional, Tuple, Union
from torch import Tensor

Collection = Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]

class ComparisonInfo(NamedTuple):
    name: str
    family: str
    inputs: str
    modes: tuple
    parameters: tuple
    symmetric: bool
    differentiable: bool
    aliases: tuple
    role: str
    sample_ndims: tuple

def compare(x1: Collection, x2: Optional[Collection] = None,
            metric: str = 'euclidean', *, mode: str = 'all_pairs',
            **kwargs) -> Tensor: ...
def single_comparison(x1: Collection, x2: Collection, metric: str = 'euclidean', **kwargs) -> Tensor: ...
def list_comparisons(*, family: Optional[str] = None, role: Optional[str] = None) -> Tuple[ComparisonInfo, ...]: ...
