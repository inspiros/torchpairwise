from torch import Tensor


def cdist(x1: Tensor,
          x2: Tensor,
          metric: str = 'minkowski',
          **kwargs): ...


def pdist(input: Tensor,
          metric: str = 'minkowski',
          **kwargs): ...
