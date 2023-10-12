TorchPairwise [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/inspiros/torchpairwise/build_wheels.yml)](https://github.com/inspiros/torchpairwise/actions) [![GitHub](https://img.shields.io/github/license/inspiros/torchpairwise)](LICENSE.txt)
======

This package provides highly-efficient pairwise metrics for **PyTorch**.

## Highlights

``torchpairwise`` is a collection of **general purposes** pairwise metric functions that behave similar to
``torch.cdist`` (which only implements $L_p$ distance).
Instead, we offer a lot more metrics ported from other packages such as
``scipy.spatial.distance`` and ``sklearn.metrics.pairwise``.
For task-specific metrics (e.g. for evaluation of classification, regression, clustering, ...), you should be in the
wrong place, please head to the [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) repo.

Written in ``torch``'s C++ API, the main differences are that our metrics:

- are all (_except some boolean distances_) **differentiable** with backward formulas manually derived and implemented.
- are **batched** and can exploit GPU parallelization.
- can be integrated seamlessly within **PyTorch**-based projects, all functions are ``torch.jit.script``-able.

### List of pairwise distance metrics

| ``torchpairwise`` ops            | Equivalences in other libraries                                                                                                                                              | Differentiable |
|:---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------:|
| ``euclidean_distances``          | [``sklearn.metrics.pairwise.euclidean_distances``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html)                      |       ✔️       |
| ``haversine_distances``          | [``sklearn.metrics.pairwise.haversine_distances``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html)                      |       ✔️       |
| ``manhattan_distances``          | [``sklearn.metrics.pairwise.manhattan_distances``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html)                      |       ✔️       |
| ``cosine_distances``             | [``sklearn.metrics.pairwise.cosine_distances``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html)                            |       ✔️       |
| ``l1_distances``                 | _(Alias of ``manhattan_distances``)_                                                                                                                                         |       ✔️       |
| ``l2_distances``                 | _(Alias of ``euclidean_distances``)_                                                                                                                                         |       ✔️       |
| ``lp_distances``                 | _(Alias of ``minkowski_distances``)_                                                                                                                                         |       ✔️       |
| ``directed_hausdorff_distances`` | [``scipy.spatial.distance.directed_hausdorff``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html) [^1]                    |       ✔️       |
| ``minkowski_distances``          | [``scipy.spatial.distance.minkowski``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html) [^1]                                      |       ✔️       |
| ``wminkowski_distances``         | [``scipy.spatial.distance.wminkowski``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.wminkowski.html) [^1]                                    |       ✔️       |
| ``sqeuclidean_distances``        | [``scipy.spatial.distance.sqeuclidean_distances``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean_distances.html) [^1]              |       ✔️       |
| ``correlation_distances``        | [``scipy.spatial.distance.correlation``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html) [^1]                                  |       ✔️       |
| ``hamming_distances``            | [``scipy.spatial.distance.hamming``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html) [^1]                                          |     ❌[^2]      |
| ``jaccard_distances``            | [``scipy.spatial.distance.jaccard``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html) [^1]                                          |     ❌[^2]      |
| ``kulsinski_distances``          | [``scipy.spatial.distance.kulsinski``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.kulsinski.html) [^1]                                      |     ❌[^2]      |
| ``kulczynski1_distances``        | [``scipy.spatial.distance.kulczynski1``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.kulczynski1.html) [^1]                                  |     ❌[^2]      |
| ``seuclidean_distances``         | [``scipy.spatial.distance.seuclidean``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.seuclidean.html) [^1]                                    |       ✔️       |
| ``cityblock_distances``          | [``scipy.spatial.distance.cityblock``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html) [^1] _(Alias of ``manhattan_distances``)_ |       ✔️       |
| ``mahalanobis_distances``        | [``scipy.spatial.distance.mahalanobis``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html) [^1]                                  |       ✔️       |
| ``chebyshev_distances``          | [``scipy.spatial.distance.chebyshev``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.chebyshev.html) [^1]                                      |       ✔️       |
| ``braycurtis_distances``         | [``scipy.spatial.distance.braycurtis``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html) [^1]                                    |       ✔️       |
| ``canberra_distances``           | [``scipy.spatial.distance.canberra``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.canberra.html) [^1]                                        |       ✔️       |
| ``jensenshannon_distances``      | [``scipy.spatial.distance.jensenshannon``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html) [^1]                              |       ✔️       |
| ``yule_distances``               | [``scipy.spatial.distance.yule``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.yule.html) [^1]                                                |     ❌[^2]      |
| ``dice_distances``               | [``scipy.spatial.distance.dice``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.dice.html) [^1]                                                |     ❌[^2]      |
| ``rogerstanimoto_distances``     | [``scipy.spatial.distance.rogerstanimoto``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.rogerstanimoto.html) [^1]                            |     ❌[^2]      |
| ``russellrao_distances``         | [``scipy.spatial.distance.russellrao``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.russellrao.html) [^1]                                    |     ❌[^2]      |
| ``sokalmichener_distances``      | [``scipy.spatial.distance.sokalmichener``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sokalmichener.html) [^1]                              |     ❌[^2]      |
| ``sokalsneath_distances``        | [``scipy.spatial.distance.sokalsneath``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sokalsneath.html) [^1]                                  |     ❌[^2]      |

[^1]: ``scipy.spatial.distance`` metrics are not pairwise but a pairwise form can be computed by
calling ``scipy.spatial.distance.cdist(x1, x2, metric="[metric_name]")``.

[^2]: These are boolean distances. ``hamming_distances`` can be applied for floating point inputs but involves
comparison.

### Other pairwise metrics or kernel functions

These metrics are usually used to compute kernel for machine learning algorithms.

| ``torchpairwise`` ops    | Equivalences in other libraries                                                                                                                           | Differentiable |
|:-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------:|
| ``linear_kernel``        | [``sklearn.metrics.pairwise.linear_kernel``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html)               |       ✔️       |
| ``polynomial_kernel``    | [``sklearn.metrics.pairwise.polynomial_kernel``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html)       |       ✔️       |
| ``sigmoid_kernel``       | [``sklearn.metrics.pairwise.sigmoid_kernel``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html)             |       ✔️       |
| ``rbf_kernel``           | [``sklearn.metrics.pairwise.rbf_kernel``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html)                     |       ✔️       |
| ``laplacian_kernel``     | [``sklearn.metrics.pairwise.laplacian_kernel``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html)         |       ✔️       |
| ``cosine_similarity``    | [``sklearn.metrics.pairwise.cosine_similarity``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)       |       ✔️       |
| ``additive_chi2_kernel`` | [``sklearn.metrics.pairwise.additive_chi2_kernel``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.additive_chi2_kernel.html) |       ✔️       |
| ``chi2_kernel``          | [``sklearn.metrics.pairwise.chi2_kernel``](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.chi2_kernel.html)                   |       ✔️       |

### Custom ``cdist`` and ``pdist``

Furthermore, we provide a convenient wrapper function analoguous to ``torch.cdist`` excepts that it takes a string
``metric: str = "minkowski"`` indicating the desired metric to be used as the third argument,
and extra metric-specific arguments are passed as keywords.

```python
import torch, torchpairwise

# directed_hausdorff_distances is a pairwise 2d metric
x1 = torch.rand(10, 6, 3)
x2 = torch.rand(8, 5, 3)

generator = torch.Generator().manual_seed(1)
output = torchpairwise.cdist(x1, x2,
                             metric="directed_hausdorff",
                             shuffle=True,  # kwargs exclusive to directed_hausdorff
                             generator=generator)
```

Note that pairwise metrics on the second table are currently not allowed keys for ``cdist``
because they are not _dist_.
We have a similar plan for ``pdist`` (which is equivalent to calling ``cdist(x1, x1)`` but avoid storing duplicated
positions).
However, that requires a total overhaul of existing C++/Cuda kernels and won't be available soon.

## Future Improvements

- Add more metrics:
    - (_Uncertain_) ``snr_distances``
      from [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#snrdistance).
- Add memory-efficient ``argkmin`` for retrieving pairwise neighbors' distances and indices without storing the whole
  pairwise distance matrix.
- Add an equivalence of ``torch.pdist`` with ``metric: str = "minkowski"`` argument.
- (_Unlikely_) Support sparse layouts.

## Requirements

- `torch>=2.0.0` (you may sucess with versions as old as `1.9.0` when compiling from source)

## Installation

#### From PyPI:

To install prebuilt wheels from [torchpairwise](https://pypi.org/project/torchpairwise), simply run:

```console
pip install torchpairwise
```

Note that the Linux and Windows wheels in **PyPI** are compiled with ``torch==2.1.0`` and **Cuda 12.1**.
We only do a non-strict version checking and a warning will be raised if ``torch``'s and ``torchpairwise``'s
Cuda versions do not match.

#### From Source:

Make sure your machine has a C++17 and a Cuda compiler installed, then clone the repo and run:

```console
pip install .
```

## Usage

The basic usecase is very straight-forward if you are familiar with
``sklearn.metrics.pairwise`` and ``scipy.spatial.distance``:

<table>
<tr>
<th>scikit-learn / SciPy</th>
<th>TorchPairwise</th>
</tr>

<tr>
<td>
<sub>

```python
import numpy as np
import sklearn.metrics.pairwise as sklearn_pairwise

x1 = np.random.rand(10, 5)
x2 = np.random.rand(12, 5)

output = sklearn_pairwise.cosine_similarity(x1, x2)
print(output)
```

</sub>
<td>
<sub>

```python
import torch
import torchpairwise

x1 = torch.rand(10, 5, device='cuda')
x2 = torch.rand(12, 5, device='cuda')

output = torchpairwise.cosine_similarity(x1, x2)
print(output)
```

</sub>
</td>
</tr>

<tr>
<td>
<sub>

```python
import numpy as np
import scipy.spatial.distance as distance

x1 = np.random.binomial(
    1, p=0.6, size=(10, 5)).astype(np.bool_)
x2 = np.random.binomial(
    1, p=0.7, size=(12, 5)).astype(np.bool_)

output = distance.cdist(x1, x2, metric='jaccard')
print(output)
```

</sub>
<td>
<sub>

```python
import torch
import torchpairwise

x1 = torch.bernoulli(
    torch.full((10, 5), fill_value=0.6, device='cuda')).to(torch.bool)
x2 = torch.bernoulli(
    torch.full((12, 5), fill_value=0.7, device='cuda')).to(torch.bool)

output = torchpairwise.jaccard_distances(x1, x2)
print(output)
```

</sub>
</td>
</tr>

</table>

Please check the [tests](tests) folder where we will add more examples.

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
