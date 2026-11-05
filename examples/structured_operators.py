"""Stream products and local aggregates over Gaussian and covariance collections."""
import torch
import torchpairwise as tp


def main():
    torch.manual_seed(8)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    means_x=torch.randn(5,3,device=device,requires_grad=True)
    means_y=torch.randn(7,3,device=device,requires_grad=True)
    a=torch.randn(5,3,3,device=device);b=torch.randn(7,3,3,device=device)
    cov_x=(a@a.transpose(-1,-2)+torch.eye(3,device=device)).requires_grad_()
    cov_y=(b@b.transpose(-1,-2)+torch.eye(3,device=device)).requires_grad_()
    values=torch.randn(7,2,device=device,requires_grad=True)
    operator=tp.lazy((means_x,cov_x),(means_y,cov_y),metric='gaussian_wasserstein',squared=True,chunk_size=2)
    result=operator.matmul(values)
    nearest=operator.amin(-1)
    covariance_total=tp.lazy(cov_x,cov_y,metric='log_euclidean',chunk_size=2).sum()
    (result.square().mean()+covariance_total).backward()
    assert all(t.grad.isfinite().all() for t in (means_x,means_y,cov_x,cov_y,values))
    print('Gaussian matrix:',tuple(operator.shape),'product:',tuple(result.shape),'row minima:',tuple(nearest.shape))
    print('finite mean/covariance/coefficient gradients: True')


if __name__=='__main__':main()
