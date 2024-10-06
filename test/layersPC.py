from typing import Any

import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Function
from PCONV2_operator import PseudoContextV2,  PseudoPadV2, PseudoFillV2

import torch
import torch.nn as nn
import torch.nn.functional as F


from compressai.ops.parametrizers import NonNegativeParametrizer



class GDN_PC(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        npart,
        ctx:PseudoContextV2,
        device=0,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)
        self.trim = PseudoFillV2(0,npart,ctx,device=device)
        self.mask = None
        
    def setup_mask(self,x):
        self.mask = torch.ones_like(x).detach().contiguous()
        self.mask = self.trim(self.mask)
        #print(torch.mean(self.mask))

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()
        self.setup_mask(x)
        x = x * self.mask
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)
        norm = norm * self.mask + 1 - self.mask
        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class conv3x3_PCONV(nn.Module):
    
    def __init__(self, npart:int, ctx:PseudoContextV2, in_ch: int, out_ch: int, stride: int = 1, device_id: int = 0) -> None:
        super().__init__()
        self.pad = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)
        
    def forward(self,x):
        px = self.pad(x)
        return self.trim(self.conv(px))


class subpel_conv3x3_PCONV(nn.Module):
    
    def __init__(self, npart:int, ctx:PseudoContextV2, in_ch: int, out_ch: int, r: int = 1, device_id: int = 0) -> None:
        super().__init__()
        self.pad = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv = nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3)
        self.d2w = nn.PixelShuffle(r)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)
        
    def forward(self,x):
        px = self.pad(x)
        px = self.conv(px)
        return self.trim(self.d2w(px))




def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride_PCONV(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, npart, ctx:PseudoContextV2, in_ch: int, out_ch: int, stride: int = 2, device_id: int = 0):
        super().__init__()
        self.pad1 = PseudoPadV2(1,npart, ctx, device=device_id)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 2)
        self.leaky_relu = nn.LeakyReLU()
        self.pad2 = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1)
        self.gdn = GDN_PC(out_ch, npart, ctx, device_id)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 2) 
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)


    def forward(self, x: Tensor) -> Tensor:
        t = self.skip(x)
        y = self.pad1(x)
        y = self.leaky_relu(self.conv1(y))
        y = self.pad2(y)
        y = self.gdn(self.conv2(y))
        return self.trim(t + y)

class ResidualBlockWithStride_PCONVT(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, npart, ctx:PseudoContextV2, in_ch: int, out_ch: int, stride: int = 2, device_id: int = 0):
        super().__init__()
        self.pad1 = PseudoPadV2(1,npart, ctx, device=device_id)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 2)
        self.leaky_relu = nn.LeakyReLU()
        self.pad2 = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1)
        self.gdn = GDN_PC(out_ch, npart, ctx, device_id)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)


    def forward(self, x: Tensor) -> Tensor:
        y = self.pad1(x)
        y = self.leaky_relu(self.conv1(y))
        y = self.pad2(y)
        y = self.gdn(self.conv2(y))
        return y


    
def subpel_conv3x3_raw(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3), nn.PixelShuffle(r)
    )

class ResidualBlockUpsample_PCONV(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, npart, ctx:PseudoContextV2, in_ch: int, out_ch: int, upsample: int = 2, device_id: int = 0):
        super().__init__()
        self.pad1 = PseudoPadV2(1,npart,ctx,device=device_id)
        self.subpel_conv = subpel_conv3x3_raw(in_ch,out_ch,upsample)
        self.leaky_relu = nn.LeakyReLU()
        self.pad2 = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, 1)
        self.igdn = GDN_PC(out_ch, npart, ctx, device_id, inverse = True)
        self.upsample = subpel_conv3x3_raw(in_ch,out_ch,upsample)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self, x: Tensor) -> Tensor:
        px = self.pad1(x)
        br1 = self.subpel_conv(px)
        br1 = self.leaky_relu(br1)
        br1 = self.pad2(br1)
        br1 = self.igdn(self.conv(br1))
        br2 = self.upsample(px)
        return self.trim(br1 + br2)


class ResidualBlock_PCONV(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, npart, ctx:PseudoContextV2, in_ch: int, out_ch: int, device_id: int = 0):
        super().__init__()
        self.pad = PseudoPadV2(2,npart,ctx,device=device_id)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1)
        self.leaky_relu =  nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1)
        #self.relu2 = nn.LeakyReLU()
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)
        
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        tx = self.pad(x)
        y = self.leaky_relu(self.conv1(tx))
        out = self.leaky_relu(self.conv2(y))

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return self.trim(out)


class AttentionBlock_PCONV(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int, npart, ctx:PseudoContextV2, device_id=0):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self, ):
                super().__init__()
                self.pad = PseudoPadV2(1,npart,ctx,device=device_id)
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(),
                    nn.Conv2d(N // 2, N // 2, 3, 1),
                    nn.ReLU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU()
                self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                px = self.pad(x)
                out = self.conv(px)
                out += identity
                out = self.relu(out)
                return self.trim(out)

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return self.trim(out)