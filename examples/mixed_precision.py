"""Autocast-producing layers with compiled distance-based training."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    layer = torch.nn.Linear(16, 8, device=device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=.01)
    scaler = torch.amp.GradScaler(device, enabled=device == 'cuda')

    def loss_function(x):
        with torch.autocast(device, dtype=dtype):
            features = layer(x)
            # Pairwise reductions promote features to float32 for accuracy.
            return tp.rbf_kernel(features).mean()

    loss_function = torch.compile(loss_function, fullgraph=True)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(torch.randn(12, 16, device=device))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f'{device}: loss={loss.item():.6f}, dtype={loss.dtype}')


if __name__ == '__main__':
    main()
