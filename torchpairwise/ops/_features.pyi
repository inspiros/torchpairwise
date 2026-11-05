from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor

_FeatureTree = Union[Tensor, Tuple['_FeatureTree', ...], List['_FeatureTree'], Dict[Any, '_FeatureTree']]
_Encoder = Callable[[Tensor], _FeatureTree]

class PreparedFeatureComparison:
    features1: _FeatureTree
    features2: Optional[_FeatureTree]
    mode: Optional[str]
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class FeatureComparison:
    query_encoder: _Encoder
    reference_encoder: _Encoder
    metric: Optional[str]
    head: Optional[Callable[..., Any]]
    mode: Optional[str]
    @property
    def encoder(self) -> Union[_Encoder, Tuple[_Encoder, _Encoder]]: ...
    def __call__(self, x1: Tensor, x2: Optional[Tensor] = None, *args: Any, **kwargs: Any) -> Any: ...
    def encode(self, x: Tensor, *, single: bool = False,
               tower: Optional[Literal['query', 'reference']] = None) -> _FeatureTree: ...
    def prepare(self, x1: Tensor, x2: Optional[Tensor] = None, *, mode: Optional[str] = None) -> PreparedFeatureComparison: ...

def make_feature_pairwise(encoder: Union[_Encoder, Tuple[_Encoder, _Encoder]],
                          metric: Optional[str] = None, *, head: Optional[Callable[..., Any]] = None,
                          mode: Optional[str] = None, sample_ndim: int = 1,
                          encode_chunk_size: Optional[int] = None,
                          **metric_kwargs: Any) -> FeatureComparison: ...
