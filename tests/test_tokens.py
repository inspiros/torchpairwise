"""Independent scalar edit oracle, ragged masks and pair geometry."""
import itertools

import pytest
import torch

from torchpairwise.ops import _tokens as tokens


def edit_oracle(a, b, substitution):
    row = list(range(len(b) + 1))
    for i, first in enumerate(a, 1):
        new = [i]
        for j, second in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1,
                           row[j - 1] + (first != second) * substitution))
        row = new
    return row[-1]


@pytest.mark.parametrize('name,cost', [('levenshtein', 1), ('indel', 2)])
def test_token_modes_ragged_broadcast_and_oracle(name, cost, device):
    # Different padding lengths, empty sequences, and noncontiguous tokens.
    x = torch.tensor([[[9, 1, 8, 2, 7, 1], [0, 3, 0, 3, 0, 3],
                       [0, 1, 0, 2, 0, 3]]], device=device)[..., 1::2]
    y = torch.tensor([[[1, 2, 2, 1], [2, 1, 2, 1], [2, 3, 1, 0]],
                      [[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1]]], device=device)
    xl = torch.tensor([3, 0, 2], device=device)
    yl = torch.tensor([[4, 1, 2], [0, 3, 4]], device=device)
    fn = getattr(tokens, name + '_distances')
    value = fn(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=2)
    expected = torch.tensor([[[edit_oracle(x[0, i, :xl[i]].tolist(),
                                          y[b, j, :yl[b, j]].tolist(), cost)
                               for j in range(3)] for i in range(3)]
                            for b in range(2)], device=device)
    assert value.dtype == torch.int64
    torch.testing.assert_close(value, expected)
    paired = getattr(tokens, 'paired_' + name + '_distances')
    torch.testing.assert_close(paired(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=1),
                               expected.diagonal(dim1=-2, dim2=-1))
    square = fn(x, x1_lengths=xl, chunk_size=4)
    torch.testing.assert_close(square, square.transpose(-1, -2))
    assert not square.diagonal(dim1=-2, dim2=-1).any()
    upper = torch.triu_indices(3, 3, 1, device=device)
    condensed = getattr(tokens, 'condensed_' + name + '_distances')
    torch.testing.assert_close(condensed(x, x_lengths=xl, chunk_size=1),
                               square[..., upper[0], upper[1]])
    # Swapping collections exercises the shorter-row workspace branch.
    torch.testing.assert_close(fn(y, x, x1_lengths=yl, x2_lengths=xl), value.transpose(-1, -2))


@pytest.mark.parametrize('name,cost', [('levenshtein', 1), ('indel', 2)])
def test_exhaustive_short_sequences(name, cost, device):
    sequences = [list(s) for n in range(4) for s in itertools.product((0, 1), repeat=n)]
    x = torch.zeros(len(sequences), 3, device=device, dtype=torch.int16)
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    for i, sequence in enumerate(sequences):
        x[i, :len(sequence)] = torch.tensor(sequence, device=device, dtype=x.dtype)
    actual = getattr(tokens, name + '_distances')(x, x1_lengths=lengths, chunk_size=7)
    expected = torch.tensor([[edit_oracle(a, b, cost) for b in sequences] for a in sequences], device=device)
    torch.testing.assert_close(actual, expected)


def test_token_empty_sample_and_sequence_axes(device):
    x = torch.empty(2, 0, dtype=torch.int32, device=device)
    y = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=device)
    torch.testing.assert_close(tokens.levenshtein_distances(x, y),
                               torch.full((2, 1), 3, device=device, dtype=torch.int64))
    assert tokens.indel_distances(x).tolist() == [[0, 0], [0, 0]]
    assert tokens.levenshtein_distances(x[:0], y).shape == (0, 1)
    assert tokens.condensed_indel_distances(x[:1]).shape == (0,)
    assert tokens.levenshtein_distances(x[None, :0].expand(0, 0, 0), y).shape == (0, 0, 1)


def test_token_contracts(device):
    x = torch.zeros(2, 3, device=device, dtype=torch.int64)
    with pytest.raises(TypeError, match='integer'):
        tokens.levenshtein_distances(x.float())
    with pytest.raises(ValueError, match='equal sample'):
        tokens.paired_indel_distances(x, x[:1])
    with pytest.raises(ValueError, match='between zero'):
        tokens.indel_distances(x, x1_lengths=torch.tensor([4, 2], device=device))
    with pytest.raises(TypeError, match='int64'):
        tokens.indel_distances(x, x1_lengths=x[:, 0].int())
    with pytest.raises(ValueError, match='chunk_size'):
        tokens.indel_distances(x, chunk_size=True)


@pytest.mark.cuda
def test_token_cuda_nondefault_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.tensor([[1, 2, 3], [1, 3, 2]], device='cuda')
        value = tokens.levenshtein_distances(x)
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert value.tolist() == [[0, 2], [2, 0]]
