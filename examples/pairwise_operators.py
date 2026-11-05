"""Directed matrix products, contractions and row/column aggregates in tiles."""
import torch
import torchpairwise as tp


def main():
    torch.manual_seed(7)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x=(torch.rand(7,5,device=device)+.2).requires_grad_()
    y=(torch.rand(9,5,device=device)+.2).requires_grad_()
    v=torch.randn(9,3,device=device,requires_grad=True)
    u=torch.randn(7,2,device=device,requires_grad=True)
    a=tp.lazy(x,y,metric='kl_divergences',chunk_size=3)
    product=a@v
    transposed=a.T@u
    form=a.bilinear(u,v)
    rows=a.mean(-1);columns=a.amin(-2)
    (product.square().mean()+transposed.square().mean()+form.square().mean()).backward()
    assert all(value.grad.isfinite().all() for value in (x,y,u,v))
    print('operator:',tuple(a.shape),'products:',tuple(product.shape),tuple(transposed.shape))
    print('form:',tuple(form.shape),'row/column aggregates:',tuple(rows.shape),tuple(columns.shape))
    print('finite input/coefficient gradients: True')


if __name__=='__main__':main()
