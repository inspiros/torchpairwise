# Distances Between Sets of Vectors

A point set is an unordered collection of vectors. A vector may contain XYZ
coordinates, a learned image descriptor, or other features. The term describes
the data being compared, not a special tensor class or a sparse storage format.
These functions take ordinary dense PyTorch tensors (`torch.strided`). PyTorch
sparse COO/CSR and nested tensors are not supported.

## Vectors versus Sets of Vectors

For vector distances such as Euclidean, one sample is one vector:

```python
import torch
import torchpairwise as tp

X = torch.rand(10, 3)
Y = torch.rand(8, 3)
D = tp.euclidean_distances(X, Y)  # [10,8]
```

For Hausdorff or Chamfer, one sample is a whole collection of vectors:

```python
X = torch.rand(10, 1000, 3)     # Ten clouds, 1000 XYZ points per cloud.
Y = torch.rand(8, 700, 3)       # Eight clouds, 700 XYZ points per cloud.
D = tp.chamfer_distances(X, Y)  # [10,8], one distance per cloud pair.
```

The axes are `[N,P,C]`: samples, points per sample, features per point. C need
not be 3. X and Y can have different sample counts and point counts, but their
feature counts must agree. To compare just two clouds stored as `[P,C]` and
`[Q,C]`, add a sample axis with `unsqueeze(0)`; the result has shape `[1,1]`.

With a batch axis, use `[B,N,P,C]` and `[B,M,Q,C]` to obtain `[B,N,M]`.
Each batch independently compares its N clouds against its M clouds. This is
different from treating N as a batch of corresponding clouds: corresponding
clouds use `paired_distances`, which returns `[B,N]` and requires the same N
on both sides. `pdist` returns the strict upper triangle for symmetric Hausdorff
or Chamfer, with shape `[B,N*(N-1)//2]`.

The point-set interpretation applies to `directed_hausdorff`, `hausdorff`, and
`chamfer`. Vector metrics retain their own contracts: Euclidean on `[N,P,C]`
treats N as a batch axis and P as vector samples, rather than returning cloud
distances. Choose a point-set metric when each sample contains multiple vectors.

The transport functions `wasserstein`, `sliced_wasserstein`, and `sinkhorn` also
use these layouts and comparison modes, with uniform mass per valid point.
They compare point distributions using transport rather than nearest-point
reductions. See the [transport guide](transport.md) for projection parameters,
regularization, and the exact solver's CPU coordination on CUDA inputs.

## Meaning of the Distances

- Directed Hausdorff takes each point in X to its nearest point in Y, then
  returns the largest such distance.
- Symmetric Hausdorff returns the larger of the two directed distances.
- Chamfer averages nearest-point distances in each direction and sums those
  two averages. `squared=True` uses squared Euclidean point distances; the
  default uses Euclidean distances.

Point ordering does not affect distance values. Ties can have different valid
witnesses/subgradients. Points are not deduplicated: repeated points can change
Chamfer's averages, although exact duplicates do not change Hausdorff values.
The directed function also returns indices of the source/target points that
attain its distance. Symmetric Hausdorff and Chamfer return only distances.

## Different Numbers of Points Within One Tensor

Ordinary dense tensors have rectangular shapes. Pad to a common point count
and supply lengths to identify each cloud's valid prefix:

```python
X = torch.rand(3, 1000, 3)
lengths = torch.tensor([1000, 600, 850], device=X.device, dtype=torch.int64)
D = tp.chamfer_distances(X, x1_lengths=lengths)  # [3,3]
```

The second cloud uses `X[1,:600]`; the other 400 entries are ignored and receive
zero gradient. Padding still occupies dense input storage. It can contain zeros
or other values, including NaNs. Zero-valued coordinates are valid points, so
the functions never infer padding from zero values. Empty clouds are currently
rejected. Lengths belong on the input device; their leading batch axes broadcast
within the existing input batch axes. Arbitrary point masks are not implemented.

For explicit X/Y comparisons, use `x1_lengths` and `x2_lengths` independently.
When Y is omitted, the right lengths default to the left lengths. Condensed
calls accept `x1_lengths` only. These are dense padded tensors with metadata,
not sparse tensors or compressed point storage.

## Image Features

`[B,N,H,W,C]` can represent H*W feature vectors per image sample:

```python
D = tp.chamfer_distances(X, Y, batched=True)  # [B,N,M]
# Channel-first inputs [B,N,C,H,W]:
D = tp.chamfer_distances(X_channels_first, Y_channels_first,
                         batched=True, feature_dim=2)
```

Use `batched=False` for `[N,H,W,C]`. An explicit batch flag distinguishes
spatial axes from the existing `[B,N,P,C]` cloud layout. When omitted, the
established `[...,N,P,C]` interpretation remains in effect.

The vectors form an unordered collection; permuting pixels leaves the distance
unchanged. Pixel locations are not automatically part of the vectors. Comparing
foreground geometry instead requires the foreground spatial coordinates.

## Limiting Intermediate Memory

`chunk_size` optionally bounds the number of cloud samples on each side of an
all-pairs tile. It does not change point counts or output shape:

```python
D = tp.chamfer_distances(X, Y, x1_lengths=x_lengths, x2_lengths=y_lengths,
                         chunk_size=8)
```

Each tile compares at most 8 clouds against 8 clouds per batch. Deterministic training uses native C++ replay to recompute tile witnesses in
backward rather than saving all point-level intermediates from forward. This trades
extra searches/launches for lower intermediate memory. Inputs and the final
dense `[B,N,M]` result still need storage. Batches and feature-layout conversions
can also allocate memory; this is not a total-memory cap.

Paired mode chunks at most `chunk_size` corresponding cloud pairs per batch at a time.
Condensed mode chunks row suffixes. `None` uses bounded default tiles for Chamfer/ragged comparisons and preserves
the optimized full-cloud Hausdorff forward search.
Chunking requires a positive integer and cannot be combined with Hausdorff
shuffling/generators. Cloud tiling and first backward execute in C++. Representative fullgraph
compile and export/save/load tests cover changing cloud counts, point counts,
length values and first gradients. General dynamic condensed exports use N>=3;
export singleton/empty boundaries separately. Shuffling/generators are outside
the compile/export contract. See the
[API contracts](api-contracts.md) for details and
[dense versus reduced outputs](reductions.md) for how point-level reductions
can avoid inner dense matrices while returning a dense cloud-distance matrix.


## Native C++ Deployment

Chamfer and directed/symmetric Hausdorff use shared C++ forward/backward,
including layouts, ragged lengths and all pairing modes. Python validation
adapters preserve existing errors. The [native API](native-api.md#nearest-point-set-comparisons)
provides typed options and a standalone LibTorch example;
[compiled_pointsets.py](../examples/compiled_pointsets.py) demonstrates dynamic
export from the usual Python API. Higher derivatives of native replay are
unsupported. Energy/MMD and transport have separate migration status.
