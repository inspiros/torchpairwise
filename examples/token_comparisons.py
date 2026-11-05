"""Compare padded integer token sequences without owning tokenization."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokens = torch.tensor([[1, 2, 3, 0], [1, 3, 0, 0], [4, 2, 3, 1]], device=device)
    lengths = torch.tensor([3, 2, 4], device=device)
    edits = tp.levenshtein_distances(tokens, x1_lengths=lengths)
    indels = tp.condensed_indel_distances(tokens, x_lengths=lengths)
    assert edits.tolist() == [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    print('Token edit distances on', device, ':', edits.tolist(), indels.tolist())


if __name__ == '__main__':
    main()
