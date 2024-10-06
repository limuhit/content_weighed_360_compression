import math

import torch
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from PCONV2_operator import MultiProject

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, mask = None):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if not mask is None:
        ssim_map = ssim_map * mask
        ret = torch.sum(ssim_map) / torch.sum(mask)
        return ret
    else:
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, channel =1, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask = None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

from compressai.registry import register_criterion

class PMSE(nn.Module):
    
    def __init__(self,size_average=True) -> None:
        super().__init__()
        self.size_average = size_average
        
    def forward(self,x,y,mask=None):
        se = torch.square(x-y)
        if self.size_average:
            if not mask is None:
                se = se * mask
                mse = torch.sum(se) / torch.sum(mask)
            else:
                mse = torch.mean(se)
        else:
            if not mask is None:
                se = se * mask
                mse = torch.sum(se,dim=(1,2,3)) / torch.sum(mask,dim=(1,2,3))
            else:
                mse = torch.mean(se,dim=(1,2,3))
        return mse
        

@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        self.metric_proto = metric
        if metric == "mse":
            self.metric = PMSE()
        elif metric == "ssim":
            self.metric = SSIM()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target, pr_src, pr_dst):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        ps = pr_src(output["x_hat"])
        pt = pr_dst(target)
        if self.metric_proto == 'ssim':
            out["ssim_loss"] = self.metric(ps, pt)
            distortion = 1 - out["ssim_loss"]
        else:
            out["mse_loss"] = self.metric(ps, pt)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
class RateDistortionLossMask(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        self.metric_proto = metric
        if metric == "mse":
            self.metric = PMSE()
        elif metric == "ssim":
            self.metric = SSIM()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type
        self.raw_mask = torch.load('./config/vp_weight.pth.tar',map_location='cpu')
        self.mask = None
        self.N = -1
    
    def setup_mask(self,x):
        N,C = x.shape[:2]
        with torch.no_grad():
            if self.mask is None:
                self.mask = self.raw_mask.repeat(N,C,1,1).to(x.device)
            elif not (N == self.N and self.mask.device == x.device):
                self.mask = self.raw_mask.repeat(N,C,1,1).to(x.device)
        self.N = N   

    def forward(self, output, target, pr_src, pr_dst):
        self.setup_mask(target)
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        ps = pr_src(output["x_hat"])
        pt = pr_dst(target)
        if self.metric_proto == 'ssim':
            out["ssim_loss"] = self.metric(ps, pt,self.mask)
            distortion = 1 - out["ssim_loss"]
        else:
            out["mse_loss"] = self.metric(ps, pt,self.mask)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
@register_criterion("RateDistortionLossPConv")
class RateDistortionLossPConv(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        self.metric_proto = metric
        if metric == "mse":
            self.metric = PMSE()
        elif metric == "ssim":
            self.metric = SSIM()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target, pr_src, pr_dst):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = ((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum() + torch.log(output["likelihoods"]["z"]).sum()) / (-math.log(2) * num_pixels)
        #print(torch.mean(output["y_mask"],dim=(1,2,3)))
        #print((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum(),torch.log(output["likelihoods"]["y"]).sum())
        out["mask_ave"] = torch.mean(output["y_mask"])
        out["entropy_y_ave"] = (torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum() / torch.sum(output["y_mask"])
        #print(torch.mean((output["x_hat"]-target)**2,dim=(1,2,3)))
        ps = pr_src(output["x_hat"])
        pt = pr_dst(target)
        if self.metric_proto == 'ssim':
            out["ssim_loss"] = self.metric(ps, pt)
            distortion = 1 - out["ssim_loss"]
        else:
            out["mse_loss"] = self.metric(ps, pt)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion  + out["bpp_loss"] #/ 0.82
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
@register_criterion("RateDistortionLossPConv")
class RateDistortionLossPConv2(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        self.metric_proto = metric
        if metric == "mse":
            self.metric = PMSE()
        elif metric == "ssim":
            self.metric = SSIM()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target, pr_src, pr_dst):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = ((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum() + 
                           (torch.log(output["likelihoods"]["z"])*output["z_mask"]).sum()) / (-math.log(2) * num_pixels)
        #print(torch.mean(output["y_mask"]))
        #print((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum(),torch.log(output["likelihoods"]["y"]).sum())
        ps = pr_src(output["x_hat"])
        pt = pr_dst(target)
        if self.metric_proto == 'ssim':
            out["ssim_loss"] = self.metric(ps, pt)
            distortion = 1 - out["ssim_loss"]
        else:
            out["mse_loss"] = self.metric(ps, pt)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
class RateDistortionLossPConvMask(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        self.metric_proto = metric
        if metric == "mse":
            self.metric = PMSE()
        elif metric == "ssim":
            self.metric = SSIM()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type
        self.raw_mask = torch.load('./config/vp_weight.pth.tar',map_location='cpu')
        self.mask = None
        self.N = -1
    
    def setup_mask(self,x):
        N,C = x.shape[:2]
        with torch.no_grad():
            if self.mask is None:
                self.mask = self.raw_mask.repeat(N,C,1,1).to(x.device)
            elif not (N == self.N and self.mask.device == x.device):
                self.mask = self.raw_mask.repeat(N,C,1,1).to(x.device)
        self.N = N           
        
    def forward(self, output, target, pr_src, pr_dst):
        self.setup_mask(target)
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = ((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum() + torch.log(output["likelihoods"]["z"]).sum()) / (-math.log(2) * num_pixels)
        #print(torch.mean(output["y_mask"]))
        #print((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum(),torch.log(output["likelihoods"]["y"]).sum())
        ps = pr_src(output["x_hat"])
        pt = pr_dst(target)

        if self.metric_proto == 'ssim':
            out["ssim_loss"] = self.metric(ps, pt, self.mask)
            distortion = 1 - out["ssim_loss"]
        else:
            out["mse_loss"] = self.metric(ps, pt, self.mask)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion  + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
class RateDistortionLossPConvTest(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, return_type="all"):
        super().__init__()
        self.metric_mse = PMSE()
        self.metric_ssim = SSIM()
        self.return_type = return_type

    def forward(self, output, target, pr_src, pr_dst):
        
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = ((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum() + torch.log(output["likelihoods"]["z"]).sum()) / (-math.log(2) * num_pixels)
        
        ps = pr_src(output["x_hat"])
        pt = pr_dst(target)
        
        out["ssim_loss"] = self.metric_ssim(ps, pt)
        out["mse_loss"] = self.metric_mse(ps, pt)
        
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
        
class RateDistortionLossTest(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, return_type="all"):
        super().__init__()
        self.metric_mse = PMSE()
        self.metric_ssim = SSIM()
        self.return_type = return_type

    def forward(self, output, target, pr_src, pr_dst):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        ps = pr_src(output["x_hat"])
        pt = pr_dst(target)
        
        out["ssim_loss"] = self.metric_ssim(ps, pt)
        out["mse_loss"] = self.metric_mse(ps, pt)
        
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
class RateDistortionLossProj(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", viewport_size=171, gpu_id=0, return_type="all"):
        super().__init__()
        self.metric_proto = metric
        if metric == "mse":
            self.metric = PMSE()
        elif metric == "ssim":
            self.metric = SSIM()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type
        self.device = f'cuda:{gpu_id}'
        self.pr_src = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, gpu_id).to(self.device)
        self.pr_dst = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, gpu_id).to(self.device)

    def forward(self, output, target):
        
        if not target.device == self.device:
            self.device = target.device
            self.pr_src = self.pr_src.to(self.device)
            self.pr_dst = self.pr_dst.to(self.device)
            
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        
        out["bpp_loss"] = ((torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum() + torch.log(output["likelihoods"]["z"]).sum()) / (-math.log(2) * num_pixels)
        out["mask_ave"] = torch.mean(output["y_mask"])
        out["entropy_y_ave"] = (torch.log(output["likelihoods"]["y"])*output["y_mask"]).sum() / torch.sum(output["y_mask"])
        
        ps = self.pr_src(output["x_hat"])
        pt = self.pr_dst(target)
        
        if self.metric_proto == 'ssim':
            out["ssim_loss"] = self.metric(ps, pt)
            distortion = 1 - out["ssim_loss"]
        else:
            out["mse_loss"] = self.metric(ps, pt)
            distortion = 255**2 * out["mse_loss"]
            
        out["loss"] = self.lmbda * distortion  + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
class RateDistortionLossProjS(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", viewport_size=171, npart=16, gpu_id=0, return_type="all"):
        super().__init__()
        self.npart = npart
        self.metric_proto = metric
        if metric == "mse":
            self.metric = PMSE(size_average=False)
        elif metric == "ssim":
            self.metric = SSIM(size_average=False)
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type
        self.device = f'cuda:{gpu_id}'
        self.pr_src = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, gpu_id).to(self.device)
        self.pr_dst = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, gpu_id).to(self.device)

    def forward(self, output, target):
        
        if not target.device == self.device:
            self.device = target.device
            self.pr_src = self.pr_src.to(self.device)
            self.pr_dst = self.pr_dst.to(self.device)
            
        N, _, H, W = target.size()
        out = {}
        num_pixels =  H * W
        py = torch.log(output["likelihoods"]["y"])*output["y_mask"]
        py = torch.sum(py.reshape(N,-1),dim=1)
        pz = torch.log(output["likelihoods"]["z"])
        pz = torch.sum(pz.reshape(N,-1),dim=1)
        out["bpp_loss"] = (py + pz) / (-math.log(2) * num_pixels)
        out["mask_ave"] = torch.mean(output["y_mask"])
        ps = self.pr_src(output["x_hat"])
        pt = self.pr_dst(target)
        
        if self.metric_proto == 'ssim':
            pssim = self.metric(ps, pt)
            out["ssim_loss"] = torch.mean(pssim.view(N,-1),dim=1)
            distortion = 1 - out["ssim_loss"]
        else:
            pmse = self.metric(ps, pt)
            out["mse_loss"] = torch.mean(pmse.view(N,-1),dim=1)
            distortion = 255**2 * out["mse_loss"]
            
        out["loss"] = self.lmbda * distortion  + out["bpp_loss"]
        self.out = out
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]