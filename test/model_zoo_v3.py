import torch
import torch.nn as nn
from compressai.layers import MaskedConv2d
from model_zoo_v2 import Cheng2020Anchor
from PCONV2_operator import SphereSlice,SphereUslice,PseudoContextV2,PseudoFillV2,PseudoEntropyContext,PseudoEntropyPad
from PCONV2_operator import GaussianTable
from layers import ResidualBlockWithStride, ResidualBlock
from PCONV2_operator.BaseOpModule import BaseOpModule
import PCONV2
import numpy as np
import coder
import os

from layersPC import (
    AttentionBlock_PCONV,
    ResidualBlock_PCONV,
    ResidualBlockUpsample_PCONV,
    ResidualBlockWithStride_PCONV,
    ResidualBlockWithStride_PCONVT,
    conv3x3_PCONV,
    subpel_conv3x3_PCONV,
)

class ParamGrad_AF(torch.autograd.Function):
    val = None
    @staticmethod
    def forward(ctx, param, y, ymask, gamma, eta):
        ctx.gamma = gamma
        ctx.ymask = ymask
        ctx.eta = eta
        return y
        
    @staticmethod
    def backward(ctx, grad_output):
        
        ymask,gamma = ctx.ymask, ctx.gamma
        grad_sum = torch.sum(torch.abs(grad_output)*ymask, dim=(1,2,3))
        ParamGrad_AF.val = torch.sum(grad_sum).item() #* ctx.eta
        #print(ctx.eta)
        #print('here',ParamGrad_AF.val)
        count = torch.sum(ymask,dim=(1,2,3))
        ygrad_ave = grad_sum / count 
        #ygrad_ave = ygrad_ave.view(-1,16)
        #print('here',count,grad_sum,ygrad_ave)
        grad = (gamma-ygrad_ave) * ctx.eta
        #grad_norm = torch.norm(grad)
        #grad = grad / grad_norm * ctx.eta
        #print(grad,flush=True)
        return grad, grad_output, None, None, None

class ParamLoss(nn.Module):
    
    def __init__(self, gamma, eta) -> None:
        super().__init__()
        self.gamma = gamma
        self.eta = eta
        self.func = ParamGrad_AF
    
    def forward(self, param, y, ymask):
        y = ParamGrad_AF.apply(param, y, ymask,self.gamma,self.eta)
        return y    

class ParamGrad2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, param, y, ymask, gamma, eta, target_mask_ave, op):
        ctx.gamma = gamma
        ctx.ymask = ymask
        ctx.param = param
        ctx.eta = eta
        ctx.target_mask_ave = target_mask_ave
        gid = y.device.index
        ctx.op = op[gid]
        return y
        
    @staticmethod
    def backward(ctx, grad_output):
        ymask,gamma,param,eta = ctx.ymask, ctx.gamma, ctx.param, ctx.eta
        grad_sum = torch.sum(torch.abs(grad_output)*ymask, dim=(1,2))
        #count = torch.sum(ymask,dim=(1,2)) + 1
        #grad_sum /= count
        grad_sum = grad_sum.contiguous() 
        #print(grad_sum,flush=True)
        mask_ave = torch.mean(param) / 64.
        decay_ratio = 0.1 if mask_ave.item() < ctx.target_mask_ave else 1
        acc_grad = ctx.op.apply(grad_sum,gamma*decay_ratio)[0]
        #print(acc_grad[:16,:64],flush=True)
        idxs = torch.argmin(acc_grad,1) + 1
        #print(idxs[:16],flush=True)
        current = param / 64. * grad_sum.shape[1]
        #print(current[:16])
        grad = torch.zeros_like(param) - eta
        grad[idxs<current] = eta
        #print(grad[:16])
        return grad, grad_output, None, None, None, None, None

class ParamLoss2(BaseOpModule):
    
    def __init__(self, gamma, eta=1e-3, target_mask_ave=0.85, device=0) -> None:
        super(ParamLoss2, self).__init__(device)
        self.gamma = gamma
        self.eta = eta
        self.target_mask_ave = target_mask_ave
        self.op = { gid : PCONV2.AccGradOp(gid, False) for gid in self.device_list}
    
    def forward(self, param, y, ymask):
        y = ParamGrad2_AF.apply(param, y, ymask,self.gamma, self.eta, self.target_mask_ave, self.op)
        return y

class ClipData_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        y = x.clone().detach()
        f1 = x<1
        f2 = x>64
        y[f1] = 1
        y[f2] = 64
        ctx.save_for_backward(f1,f2)
        return y
        
    @staticmethod
    def backward(ctx, grad_output):
        y = grad_output.clone().detach()
        f1,f2 = ctx.saved_tensors
        y1 = grad_output[f1]
        fy1 = y1 > 0
        y1[fy1] = 0
        y2 = grad_output[f2]
        fy2 = y2 < 0
        y2[fy2] = 0
        y[f1] = y1
        y[f2] = y2
        #print(y,flush=True)
        return y

class ClipData(nn.Module):

    def __init__(self):
        super(ClipData,self).__init__()

    def forward(self,x):
        return ClipData_AF.apply(x)


class HeadResidualBlock(nn.Module):
    
    def __init__(self,nft) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(nft,nft,3,padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(nft,nft,3,padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
    def forward(self,x):
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu2(x+y)
    
class HeadResidualBlock2(nn.Module):
    
    def __init__(self,nft) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(nft,nft,3,padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(nft,1,3,padding=1)
        self.connect = nn.Conv1d(nft,1,1)
    def forward(self,x):
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        tx = self.connect(x)
        return tx+y

class ParamHead(nn.Module):
    def __init__(self,nft,npart) -> None:
        super().__init__()
        self.npart = npart
        self.head = nn.Linear(nft,1)
        self.pool = nn.AdaptiveAvgPool1d(nft)
        dt = np.array([17.0, 32.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 32.0, 17.0],dtype=np.float32)
        #dt = np.array([64.0000, 64.0000, 64.0000, 64.0000, 64.0000, 64.0000, 64.0000, 64.0000,64.0000, 64.0000, 64.0000, 64.0000, 64.0000, 64.0000, 64.0000,  9.3646],dtype=np.float32)
        self.raw = torch.from_numpy(dt)
        self.base = None
        self.clip = ClipData()
    def cast_raw(self,fp):
        with torch.no_grad():
            if self.base is None:
                self.base = self.raw.to(fp.device).repeat(fp.shape[0]).contiguous()
                return 
            if self.base.device == fp.device and self.base.shape[0] == fp.shape[0]*16 :
                return
            
            self.base = self.raw.to(fp.device).repeat(fp.shape[0]).contiguous()
    
    def forward(self,fp):
        self.cast_raw(fp)
        n,c,h,w = fp.size()
        fp = fp.view(n,c,self.npart,h//self.npart,w).permute(0,2,1,3,4).contiguous().view(n*self.npart,1,-1)
        tp = self.pool(fp).view(n*self.npart,-1)
        #param = self.head(tp).view(n*self.npart) + self.base
        #param = self.clip(param)
        param = torch.sigmoid(self.head(tp).view(n*self.npart))*64 + 0.5
        param = torch.clip(param,1,64)
        return param.contiguous()
        
class ParamHead2(nn.Module):
    def __init__(self,nft,npart) -> None:
        super().__init__()
        self.npart = npart
        self.head = nn.Sequential(
            HeadResidualBlock(nft),
            HeadResidualBlock(nft),
            HeadResidualBlock2(nft)
        )
        self.pool = nn.AdaptiveAvgPool1d(nft)
        self.clip = ClipData()
    
    def forward(self,fp):
        n,c,h,w = fp.size()
        fp = fp.view(n,c,self.npart,h//self.npart,w).permute(0,2,1,3,4).contiguous().view(n,self.npart,-1)
        tp = self.pool(fp)
        tp = tp.view(n,self.npart,-1).permute(0,2,1)
        pout = self.head(tp).view(n*self.npart)
        param = torch.sigmoid(pout)*64 + 1
        param = torch.clip(param,1,64)
        return param.contiguous()

class Cheng2020AttentionPConvV2(Cheng2020Anchor):

    def __init__(self, N=192, npart = 16, gamma=1e-5, eta = 1, device_id = 0, **kwargs):
        super().__init__(N=N, **kwargs)
        self.npart = npart
        self.slice = SphereSlice(npart,pad=0,device=device_id)
        self.uslice = SphereUslice(npart,pad=0,device=device_id)
        self.ctx = PseudoContextV2(npart,device=device_id)
        self.ctx_ent = PseudoEntropyContext(npart,2,device=device_id)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride_PCONV(npart, self.ctx, 3, N, stride=2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockWithStride_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockWithStride_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            conv3x3_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
        )
        self.g_s = nn.Sequential(
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            subpel_conv3x3_PCONV(npart, self.ctx, N, 3, 2, device_id=device_id),
        )
        self.param_net = nn.Sequential(
            nn.Conv2d(3,128,5,2,2),
            nn.LeakyReLU(inplace=True),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ParamHead(128,npart),
        )
        self.slice_code = SphereSlice(npart,device=device_id)
        self.uslice_code = SphereUslice(npart,device=device_id)
        self.code_mask = None
        self.trim = PseudoFillV2(0,npart,self.ctx,device=device_id)
        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=0, stride=1
        )
        self.context_pad = PseudoEntropyPad(2,npart,self.ctx_ent,device=device_id)
        self.param_loss = ParamLoss(gamma,eta)
        self.coding_bias,self.coding_range = 76,160
        self.wt_table = torch.Tensor([0, 1, 2, 22, 165, 349, 799, 1188, 1659, 2171, 2764, 3214, 3664, 
                                      4216, 4707, 5239, 5771, 6201, 6672, 7163, 7470, 7859, 8268, 8677, 
                                      9045, 9475, 9946, 10212, 10621, 10846, 11214, 11582, 11889, 12114, 
                                      12359, 12768, 13034, 13300, 13525, 13873, 14118, 14404, 14547, 14751, 
                                      14996, 15221, 15364, 15671, 15916, 16161, 16365, 16672, 16958, 17162,
                                      17448, 17693, 17959, 18327, 18490, 18940, 19267, 19615, 20086, 20823, 65510]).type(torch.int32)
        self.print_const_table()
        self.device = None

    def print_const_table(self):
        ncount = 512
        total = 65536
        part = total // ncount
        ptable = [i*part for i in range(ncount+1)]
        table = torch.Tensor(ptable).type(torch.int32).contiguous()
        self.const_table = table.view(1,-1).repeat(2,1)
        
    def forward(self,x):
        n,_,_,w = x.shape
        pw = self.param_net(x)
        self.ctx.setup_context(n,w,pw)
        self.ctx_ent.setup_context(n,w,pw)
        tx = self.slice(x,pw)
        raw_y = self.g_a(tx)
        code_mask = torch.ones_like(raw_y).detach()
        self.trim(code_mask) 
        y = self.param_loss(pw,raw_y,code_mask)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        
        py_hat = self.context_pad(y_hat)
        ctx_params = self.context_prediction(py_hat)
        
        ty = self.uslice_code(y.contiguous(),pw)
        z = self.h_a(ty)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat).contiguous()
        tparams = self.slice_code(params,pw)
        gaussian_params = self.entropy_parameters(
            torch.cat((tparams, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        #y_hat2, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        tx_hat = self.uslice(x_hat,pw)
        
        return {
            "x_hat": tx_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_mask": code_mask,
            'weight':pw,
            'data':tx,
            'tx':x_hat
        }
    
    
    @torch.no_grad()
    def encoding(self,x,code_file='data'):
        self.device = x.device
        mcoder = coder.coder(code_file)
        mcoder.start_encoder()
        n,_,_,w = x.shape
        pw = self.param_net(x)
        pw = torch.round(pw).type(torch.float32)
        tlabel = (pw - 1).contiguous().type(torch.int32).to('cpu')
        pred = self.wt_table.view(1,-1).repeat(pw.shape[0],1).contiguous()
        mcoder.encodes(pred,64,tlabel,pw.shape[0])
        self.ctx.setup_context(n,w,pw)
        self.ctx_ent.setup_context(n,w,pw)
        tx = self.slice(x,pw)
        raw_y = self.g_a(tx)
        code_mask = torch.ones_like(raw_y).detach()
        self.trim(code_mask) 
        y = self.param_loss(pw,raw_y,code_mask)
        ty = self.uslice_code(y.contiguous(),pw)
        z = self.h_a(ty)
        
        ########### determine coding range for y and coding it ###########
        ry_hat = self.gaussian_conditional.quantize(y, "dequantize")
        rpy_hat = self.context_pad(ry_hat)
        ctx_params = self.context_prediction(rpy_hat)
        rty = self.uslice_code(y.contiguous(),pw)
        rz = self.h_a(rty)
        rz_hat, _ = self.entropy_bottleneck(z)
        rparams = self.h_s(rz_hat).contiguous()
        rtparams = self.slice_code(rparams,pw)
        gaussian_params = self.entropy_parameters(
            torch.cat((rtparams, ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        idxs = torch.round(y-means_hat).int()
        idxs_min, idxs_max = idxs.min().item()-10,idxs.max().item()+10
        coding_bias,coding_range = -idxs_min, idxs_max-idxs_min+1
        tlabel = torch.Tensor([coding_bias,coding_range]).type(torch.int32).contiguous()
        mcoder.encodes(self.const_table,512,tlabel,2)
        
        
        self.entropy_bottleneck.update()
        z_hat = torch.zeros_like(z)
        medians = self.entropy_bottleneck._get_medians().detach()
        _,zc,zh,zw = z.shape
        for tc in range(zc):
            rag = self.entropy_bottleneck.cdf_length[tc]
            tb = self.entropy_bottleneck.quantized_cdf[tc][:rag].view(1,-1).repeat(zh*zw,1).contiguous()
            idx =  torch.round(z[0,tc]-medians[tc])
            idx = idx - self.entropy_bottleneck.offset[tc]
            pred,tlabel = tb.contiguous().type(torch.int32).to('cpu'), idx.view(-1).contiguous().type(torch.int32).to('cpu')
            mcoder.encodes(pred,rag-1,tlabel,zh*zw)
            z_hhat = idx + self.entropy_bottleneck.offset[tc] + medians[tc]
            z_hat[0,tc] += z_hhat
        params = self.h_s(z_hat).contiguous()
        tparams = self.slice_code(params,pw)
        gt = GaussianTable(coding_range,y.shape[1],coding_bias,65536.,1e-6)
        fake_y_hat = torch.zeros_like(y)
        padded = self.context_pad(fake_y_hat)
        fake_py = torch.zeros_like(padded)
        tn,tc,th,tw = y.shape
        ept = torch.zeros((1,1,1,tc),dtype=torch.float32).to(self.device)
        for pg in range(tn):
            if pg>0:
                padded = self.context_pad(fake_y_hat)
                fake_py[pg:pg+1] = padded[pg:pg+1]
            for ph in range(th):
                for pww in range(tw):
                    if code_mask[pg,0,ph,pww] < 0.5:break
                    tmp = self.context_prediction(fake_py[pg:pg+1,:,ph:ph+5,pww:pww+5])
                    combined_ft = torch.cat((tparams[pg:pg+1,:,ph:ph+1,pww:pww+1], tmp), dim=1)
                    gparam = self.entropy_parameters(combined_ft)
                    scales_hat, means_hat = gparam[0,:tc,0,0], gparam[0,tc:,0,0]
                    y_hhat = torch.round(y[pg,:,ph,pww]-means_hat)# + means_hat
                    idxs = y_hhat.int() + coding_bias
                    ept = torch.zeros_like(means_hat).view(1,1,1,tc)
                    tbs = gt(scales_hat.view(1,1,1,tc), ept)
                    pred,tlabel = tbs.contiguous().type(torch.int32).to('cpu'), idxs.contiguous().type(torch.int32).to('cpu')
                    mcoder.encodes(pred,coding_range,tlabel,tc)
                    y_hhat = idxs - coding_bias + means_hat
                    fake_py[pg,:,ph+2,pww+2] = y_hhat
                if pww==tw-1: pww+=1
                fake_py[pg:pg+1,:,ph+2,pww+2:pww+4] = fake_py[pg:pg+1,:,ph+2,2:4]
            fake_y_hat[pg:pg+1,:] = fake_py[pg:pg+1,:,2:2+th,2:2+tw]
        nbits = os.path.getsize(code_file)*8.
        #print(nbits/512/1024)
        mcoder.end_encoder()
        return nbits
    
    @torch.no_grad()
    def decoding(self, w=1024, code_channels=192, n=16, code_file='data'):
        assert not self.device is None, "Initialize the cuda device for the codec before decoding"
        mcoder = coder.coder(code_file)
        mcoder.start_decoder()
        pred = self.wt_table.view(1,-1).repeat(n,1).contiguous()
        pw = mcoder.decodes(pred,64,n).to(self.device) + 1
        labels = mcoder.decodes(self.const_table,512,2)
        coding_bias,coding_range = labels[0],labels[1]
        self.ctx.setup_context(n,w,pw)
        self.ctx_ent.setup_context(n,w,pw)
        code_mask = torch.ones((n,code_channels,w//(32*n),w//16),dtype=torch.float32).to(self.device)
        self.trim(code_mask) 
        self.entropy_bottleneck.update()
        z_hat = torch.zeros((1,code_channels,w//128,w//64),dtype=torch.float32).to(self.device)
        medians = self.entropy_bottleneck._get_medians().detach()
        zh,zw = z_hat.shape[2:]
        for tc in range(code_channels):
            rag = self.entropy_bottleneck.cdf_length[tc]
            tb = self.entropy_bottleneck.quantized_cdf[tc][:rag].view(1,-1).repeat(zh*zw,1).contiguous()
            pred = tb.contiguous().type(torch.int32).to('cpu')
            tlabel = mcoder.decodes(pred,rag-1,zh*zw).to(self.device).contiguous().int().view(zh,zw)
            z_hhat = tlabel + self.entropy_bottleneck.offset[tc] + medians[tc]
            z_hat[0,tc] += z_hhat
        params = self.h_s(z_hat).contiguous()
        tparams = self.slice_code(params,pw)
        gt = GaussianTable(coding_range,code_channels,coding_bias,65536.,1e-6)
        fake_y_hat = torch.zeros_like(code_mask)
        padded = self.context_pad(fake_y_hat)
        fake_py = torch.zeros_like(padded)
        tn,tc,th,tw = code_mask.shape
        ept = torch.zeros((1,1,1,code_channels),dtype=torch.float32).to(self.device)
        for pg in range(tn):
            if pg>0:
                padded = self.context_pad(fake_y_hat)
                fake_py[pg:pg+1] = padded[pg:pg+1]
            for ph in range(th):
                for pww in range(tw):
                    if code_mask[pg,0,ph,pww] < 0.5:break
                    tmp = self.context_prediction(fake_py[pg:pg+1,:,ph:ph+5,pww:pww+5])
                    combined_ft = torch.cat((tparams[pg:pg+1,:,ph:ph+1,pww:pww+1], tmp), dim=1)
                    gparam = self.entropy_parameters(combined_ft)
                    scales_hat, means_hat = gparam[0,:tc,0,0], gparam[0,tc:,0,0]
                    tbs = gt(scales_hat.view(1,1,1,code_channels), ept)
                    pred = tbs.contiguous().type(torch.int32).to('cpu')
                    tlabel = mcoder.decodes(pred,coding_range,code_channels).to(self.device).contiguous().int()
                    y_hhat = tlabel - coding_bias + means_hat
                    fake_py[pg,:,ph+2,pww+2] = y_hhat#y_hat[pg:pg+1,:,ph,pww]
                if pww==tw-1: pww+=1
                fake_py[pg:pg+1,:,ph+2,pww+2:pww+4] =  fake_py[pg:pg+1,:,ph+2,2:4]
            fake_y_hat[pg:pg+1,:] = fake_py[pg:pg+1,:,2:2+th,2:2+tw]
        
        x_hat = self.g_s(fake_y_hat.contiguous())
        tx_hat = self.uslice(x_hat,pw)
        
        return tx_hat

class Cheng2020AttentionPConvV2P(nn.Module):
    
    def __init__(self, npart):
        super().__init__()
        self.npart = npart
        self.param_net = nn.Sequential(
            nn.Conv2d(3,128,5,2,2),
            nn.LeakyReLU(inplace=True),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ParamHead(128,npart),
        )
        
    def forward(self,x):
        pw = self.param_net(x)
        return pw

class Cheng2020AttentionPConvV2L2(Cheng2020Anchor):
    
    def __init__(self, N=192, npart = 16, gamma=1e-5, eta = 1, mask_ave=0.85, device_id = 0, **kwargs):
        super().__init__(N=N, **kwargs)
        self.npart = npart
        self.slice = SphereSlice(npart,pad=0,device=device_id)
        self.uslice = SphereUslice(npart,pad=0,device=device_id)
        self.ctx = PseudoContextV2(npart,device=device_id)
        self.ctx_ent = PseudoEntropyContext(npart,2,device=device_id)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride_PCONV(npart, self.ctx, 3, N, stride=2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockWithStride_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockWithStride_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            conv3x3_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
        )
        self.g_s = nn.Sequential(
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            subpel_conv3x3_PCONV(npart, self.ctx, N, 3, 2, device_id=device_id),
        )
        self.param_net = nn.Sequential(
            nn.Conv2d(3,128,5,2,2),
            nn.LeakyReLU(inplace=True),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ResidualBlockWithStride(128,128,2),
            ResidualBlock(128,128),
            ParamHead(128,npart),
        )
        self.slice_code = SphereSlice(npart,device=device_id)
        self.uslice_code = SphereUslice(npart,device=device_id)
        self.code_mask = None
        self.trim = PseudoFillV2(0,npart,self.ctx,device=device_id)
        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=0, stride=1
        )
        self.context_pad = PseudoEntropyPad(2,npart,self.ctx_ent,device=device_id)
        self.param_loss = ParamLoss2(gamma,eta,target_mask_ave=mask_ave,device=device_id)
        

        
    def forward(self,x):
        n,_,_,w = x.shape
        pw = self.param_net(x)
        
        self.ctx.setup_context(n,w,pw)
        self.ctx_ent.setup_context(n,w,pw)
        tx = self.slice(x,pw)
        raw_y = self.g_a(tx)
        code_mask = torch.ones_like(raw_y).detach()
        self.trim(code_mask) 
        y = self.param_loss(pw,raw_y,code_mask)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        
        py_hat = self.context_pad(y_hat)
        ctx_params = self.context_prediction(py_hat)
        ty = self.uslice_code(y.contiguous(),pw)
        z = self.h_a(ty)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat).contiguous()
        tparams = self.slice_code(params,pw)
        gaussian_params = self.entropy_parameters(
            torch.cat((tparams, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        tx_hat = self.uslice(x_hat,pw)
        
        return {
            "x_hat": tx_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_mask": code_mask,
            'weight':pw,
            'data':tx,
            'tx':x_hat
        }

class Cheng2020AttentionPConvV2T(Cheng2020Anchor):
    
    def __init__(self, N=192, npart = 16, gamma=1e-5, eta = 1, device_id = 0, **kwargs):
        super().__init__(N=N, **kwargs)
        self.npart = npart
        self.slice = SphereSlice(npart,pad=0,device=device_id)
        self.uslice = SphereUslice(npart,pad=0,device=device_id)
        self.ctx = PseudoContextV2(npart,device=device_id)
        self.ctx_ent = PseudoEntropyContext(npart,2,device=device_id)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride_PCONV(npart, self.ctx, 3, N, stride=2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockWithStride_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockWithStride_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            conv3x3_PCONV(npart, self.ctx, N, N, stride=2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
        )
        self.g_s = nn.Sequential(
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            AttentionBlock_PCONV(N, npart, self.ctx, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            ResidualBlockUpsample_PCONV(npart, self.ctx, N, N, 2, device_id=device_id),
            ResidualBlock_PCONV(npart, self.ctx, N, N, device_id=device_id),
            subpel_conv3x3_PCONV(npart, self.ctx, N, 3, 2, device_id=device_id),
        )
        self.slice_code = SphereSlice(npart,device=device_id)
        self.uslice_code = SphereUslice(npart,device=device_id)
        self.code_mask = None
        self.trim = PseudoFillV2(0,npart,self.ctx,device=device_id)
        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=0, stride=1
        )
        self.context_pad = PseudoEntropyPad(2,npart,self.ctx_ent,device=device_id)
        self.param_loss = ParamLoss(gamma,eta)
        

        
    def forward(self,x,pw):
        n,c,h,w = x.shape
        self.ctx.setup_context(n,w,pw)
        self.ctx_ent.setup_context(n,w,pw)
        tx = self.slice(x,pw)
        raw_y = self.g_a(tx)
        code_mask = torch.ones_like(raw_y).detach()
        self.trim(code_mask) 
        y = self.param_loss(pw,raw_y,code_mask)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        
        py_hat = self.context_pad(y_hat)
        ctx_params = self.context_prediction(py_hat)
        ty = self.uslice_code(y.contiguous(),pw)
        z = self.h_a(ty)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat).contiguous()
        tparams = self.slice_code(params,pw)
        gaussian_params = self.entropy_parameters(
            torch.cat((tparams, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        tx_hat = self.uslice(x_hat,pw)
        
        return {
            "x_hat": tx_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_mask": code_mask,
            'weight':pw,
            "data":tx,
            "xr":x_hat
        }
    
def lookup(mdict,prex,post):
    for pkey in mdict.keys():
        if pkey.startswith(prex) and pkey.find(post) > 0:
            return pkey
    
def cast_param_v2(state_dict):
    ndict = {}
    for pkey in state_dict.keys():
        ndict[pkey.replace('module.','')] = state_dict[pkey]
    key_map = {"g_a.7.conv.weight":"g_a.7.weight","g_a.7.conv.bias":"g_a.7.bias","g_s.9.conv.weight":"g_s.9.0.weight",
               "g_s.9.conv.bias":"g_s.9.0.bias", }
    for idx in [0,2,4,6,8]:
        key_map[f"h_a.{idx}.conv.weight"] = f"h_a.{idx}.weight"
        key_map[f"h_a.{idx}.conv.bias"] = f"h_a.{idx}.bias"
        key_map[f"h_s.{idx}.conv.weight"] = lookup(ndict,f"h_s.{idx}",'weight')
        key_map[f"h_s.{idx}.conv.bias"] = lookup(ndict,f"h_s.{idx}",'bias')
    for pk in key_map.keys():
        ndict[pk] = ndict.pop(key_map[pk])
    return ndict

def cast_param(state_dict):
    ndict = {}
    for pkey in state_dict.keys():
        ndict[pkey.replace('module.','')] = state_dict[pkey]
    key_map = {"g_a.7.conv.weight":"g_a.7.weight","g_a.7.conv.bias":"g_a.7.bias","g_s.9.conv.weight":"g_s.9.0.weight",
               "g_s.9.conv.bias":"g_s.9.0.bias", }
    for pk in key_map.keys():
        ndict[pk] = ndict.pop(key_map[pk])
    return ndict

def test_model():
    from loss import RateDistortionLossPConv
    from PCONV2_operator import  MultiProject
    data = torch.rand((1,3,512,1024),dtype=torch.float32,device=self.device)
    model = Cheng2020AttentionPConvV2(device_id=0).to('cuda:0')
    viewport_size = 171
    pr1 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, 0).to('cuda:0')
    pr2 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, 0).to('cuda:0') 
    #param = torch.load('./save_models/checkpoint_best_loss.pth.tar',map_location='cuda:0')['state_dict']
    #model.load_state_dict(cast_param(param))
    crt = RateDistortionLossPConv()
    y = model(data)
    loss = crt(y,data,pr1,pr2)
    loss["loss"].backward()
    pass

def test_param_loss():
    
    data = torch.rand((16,1,2,64),device='cuda:0',dtype=torch.float32)
    y = torch.autograd.Variable(data,requires_grad=True)
    y.retain_grad()
    y_mask = torch.ones_like(y)
    pdata = torch.rand((16),device='cuda:0',dtype=torch.float32)*10 - 5
    pm = torch.autograd.Variable(pdata.contiguous(),requires_grad=True)
    pm.retain_grad()
    pl = ParamLoss(1e-3,1e-4)
    
    z = pl(pm,y,y_mask)
    z.retain_grad()
    loss = torch.sum(z**2/2)
    loss.backward()
    print(ParamGrad_AF.val)
    print(torch.sum(y,dim=(1,2,3)))
    print(torch.sum(torch.abs(z.grad-y.grad)))
    print(pm.grad)
    
def test_param_loss2():
    dt = np.array([17.0, 32.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 32.0, 17.0],dtype=np.float32)
    raw = torch.from_numpy(dt).to('cuda:0')
    data = torch.rand((16,1,2,64),device='cuda:0',dtype=torch.float32)
    y = torch.autograd.Variable(data,requires_grad=True)
    y.retain_grad()
    y_mask = torch.zeros_like(y)
    with torch.no_grad():
        for i in range(16):
            wt = int(dt[i])
            y_mask[i,:,:,:wt] = 1
            data[i] = 0.0001 * i
    pm = torch.autograd.Variable(raw.contiguous(),requires_grad=True)
    pm.retain_grad()
    pl = ParamLoss2(8e-4)
    z = pl(pm,y,y_mask)
    z.retain_grad()
    loss = torch.sum(z**2/2)
    loss.backward()
    print(torch.sum(torch.abs(z.grad-y.grad)))
    pass
    

def test_one_pass(data,model,crt,pr_src,pr_dst,pw):
    y = model(data,pw)
    out_criterion = crt(y, data, pr_src, pr_dst)
    print(out_criterion['mse_loss'])
    return {"data":y['data'],"xr":y['xr']}
    #return y['xr'].clone().detach()

def init_with_pconv(model,pdict):
    ndict = model.state_dict()
    for pkey in ndict.keys():
        tpk = f'module.{pkey}'
        if tpk in pdict.keys():
            ndict[pkey] = pdict[tpk]
    model.load_state_dict(ndict)

    

if __name__ == '__main__':
    #test_model()
    test_param_loss()
    #test_net_mult()    
    #test_param_loss2()
    