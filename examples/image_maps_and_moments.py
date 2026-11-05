"""Reference-image heatmaps and local geometric statistics in native tiles."""
import torch
import torchpairwise as tp


def main():
    torch.manual_seed(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    images=torch.rand(3,2,12,14,device=device,requires_grad=True)
    references=torch.rand(5,2,12,14,device=device,requires_grad=True)
    error=tp.image_map_operator(images,references,comparison='squared_error',chunk_size=2).reduce('mean')
    ssim=tp.image_map_operator(images,references,comparison='ssim',data_range=1.,window_size=3,chunk_size=2).reduce('max')
    x=torch.randn(7,5,device=device,requires_grad=True)
    y=torch.randn(9,5,device=device,requires_grad=True)
    covariance=tp.lazy(x,y,metric='rbf',gamma=.2,chunk_size=3).residual_outer_product(normalize=True,center=True)
    (error.mean()+ssim.mean()+covariance.square().mean()).backward()
    assert all(t.grad.isfinite().all() for t in (images,references,x,y))
    print('mean pixel error:',tuple(error.shape))
    print('best local SSIM:',tuple(ssim.shape))
    print('local population covariance:',tuple(covariance.shape))
    print('finite image/coordinate gradients: True')


if __name__=='__main__':main()
