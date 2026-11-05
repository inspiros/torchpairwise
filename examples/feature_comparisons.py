"""A caller-owned image encoder, scalar head and structured feature head."""
import torch
import torchpairwise as tp


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 5)

    def forward(self, images):
        # Receives [samples,C,H,W]; image preprocessing remains caller-owned.
        return self.projection(images.mean((-1, -2))).tanh()


def main():
    torch.manual_seed(21)
    encoder = ImageEncoder()
    encoder.eval()  # The application chooses the mode and gradient policy.
    images_x = torch.randn(2, 1, 3, 3, 8, 8)
    images_y = torch.randn(1, 4, 2, 3, 8, 8)
    compare = tp.make_feature_pairwise(
        encoder, 'euclidean', sample_ndim=3,
    )
    with torch.no_grad():
        print('All-pairs embeddings:', compare(images_x, images_y).shape)

    # Model parameters and explicit head parameters remain differentiable.
    weight = torch.nn.Parameter(torch.ones(5))
    head = tp.make_pairwise(lambda a, b, w: ((a - b).square() * w).sum(),
                            parameters=(weight,), mode='all_pairs')
    train_compare = tp.make_feature_pairwise(encoder, head=head, sample_ndim=3,
                                             encode_chunk_size=4)
    train_compare(images_x, images_y).sum().backward()
    print('Encoder receives gradients:', encoder.projection.weight.grad is not None)
    print('Head receives gradients:', weight.grad is not None)

    # Explicit preparation keeps encoder features for multiple head consumers.
    reusable = tp.make_feature_pairwise(
        encoder, 'sqeuclidean', sample_ndim=3)
    prepared = reusable.prepare(images_x, images_y)
    source = tp.lazy(prepared, chunk_size=2)
    print('Prepared row means:', source.mean(-1).shape)
    print('Prepared nearest references:', source.topk(1, largest=False)[0].shape)
    torch.testing.assert_close(source.mean(-1), prepared().mean(-1))
    print('Single image comparison:', reusable(images_x[0,0,0], images_y[0,0,0], mode='single').shape)

    retrieval = tp.make_feature_pairwise((encoder, ImageEncoder()), 'sqeuclidean',
                                          sample_ndim=3)
    nearest = tp.lazy(images_x, images_y, retrieval).topk(1, largest=False)
    torch.testing.assert_close(nearest[0], retrieval(images_x, images_y).topk(
        1, dim=-1, largest=False).values)
    print('Two-tower nearest references:', nearest[1].shape)

    def structured_encoder(images):
        return {'embedding': encoder(images), 'mean': images.mean((-1, -2))}

    def structured_head(a, b=None):
        b = a if b is None else b
        return tp.cdist(a['embedding'], b['embedding'], metric='euclidean')

    structured = tp.make_feature_pairwise(structured_encoder, head=structured_head, sample_ndim=3)
    print('Self comparisons reuse one feature tree:', structured(images_x).shape)


if __name__ == '__main__':
    main()
