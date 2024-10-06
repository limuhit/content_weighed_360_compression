import torch
import PCONV2
from torch._C import device, dtype
from PCONV2_operator.BaseOpModule import BaseOpModule
from torch import nn
import math
from PCONV2_operator.base import set_weight



class PseudoContextV2(BaseOpModule):

    def __init__(self, npart, rt=20, device=0, time_it=False):
        super(PseudoContextV2,self).__init__(device)
        self.op = { gid : PCONV2.PseudoContextOp(npart, rt, gid, time_it) for gid in self.device_list}
  
    def setup_context(self,num,w,weight):
        for gid in self.op.keys():
            self.op[gid].start_context(num,w,weight)

    def get_addr(self,gid):
        return self.op[gid].addr()

    def produce_fill_param(self, gid, n, h, w):
        return self.op[gid].produce_fill_param(n,h,w)
    
    def produce_pad_param(self,gid,n,c,h,w,pad):
        return self.op[gid].produce_pad_param(n,c,h,w,pad)
    def forward(self,x):
        pass

class PseudoEntropyContext(BaseOpModule):
    
    def __init__(self, npart, context_version=1, rt=20, device=0, time_it=False):
        super(PseudoEntropyContext,self).__init__(device)
        self.op = { gid : PCONV2.PseudoEntropyContextOp(npart, rt, context_version, gid, time_it) for gid in self.device_list}
  
    def setup_context(self,n,w,weight):
        for gid in self.op.keys():
            self.op[gid].start_context(n,w,weight)

    def get_addr(self,gid):
        return self.op[gid].addr()
    
    def forward(self,x):
        pass

class PseudoEntropyPad_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoEntropyPad(BaseOpModule):
    
    def __init__(self, pad, npart, ctx:PseudoEntropyContext, device = 0, time_it = False):
        super(PseudoEntropyPad, self).__init__(device)
        self.op = { gid : PCONV2.PseudoEntropyPadOp(pad,npart,ctx.get_addr(gid), gid, time_it) for gid in self.device_list}

    def forward(self, x):
        res = PseudoEntropyPad_AF.apply(x, self.op)
        return res


class PseudoPadV2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoPadV2(BaseOpModule):
    
    def __init__(self, pad, npart, ctx:PseudoContextV2, device = 0, time_it = False):
        super(PseudoPadV2, self).__init__(device)
        self.op = { gid : PCONV2.PseudoPadOp(pad,npart,ctx.get_addr(gid), gid, time_it) for gid in self.device_list}
        self.pad = pad
        
    def forward(self, x):
        if self.pad > 0:
            res = PseudoPadV2_AF.apply(x, self.op)
            return res
        else:
            return x

class PseudoFillV2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoFillV2(BaseOpModule):
    #int pad, int npart, int fvalue, int trim, std::string addr, int context_version, int device = 0, bool timeit=false
    def __init__(self, pad, npart, ctx, fvalue=0, trim=0, device = 0, time_it = False):
        super(PseudoFillV2, self).__init__(device)
        self.ctx = ctx
        if isinstance(ctx,PseudoContextV2):
            self.op = { gid : PCONV2.PseudoFillOp(pad, npart, fvalue, trim, ctx.get_addr(gid), 0, gid, time_it) for gid in self.device_list}
        elif isinstance(ctx,PseudoEntropyContext):
            self.op = { gid : PCONV2.PseudoFillOp(pad, npart, fvalue, trim, ctx.get_addr(gid), 1, gid, time_it) for gid in self.device_list}
        else:
            self.op = { gid : PCONV2.PseudoFillOp(pad, npart, fvalue, trim, ctx.get_addr(gid), 2, gid, time_it) for gid in self.device_list}

    def forward(self, x):
        #n,c,h,w = x.shape
        #param = self.ctx.produce_fill_param(0,n,h,w)
        #print(param[0,0].item())
        res = PseudoFillV2_AF.apply(x, self.op)
        return res

class PseudoMerge_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoMerge(BaseOpModule):
    #int pad, int npart, int fvalue, int trim, std::string addr, int context_version, int device = 0, bool timeit=false
    def __init__(self, npart, ctx, device = 0, time_it = False):
        super(PseudoMerge, self).__init__(device)
        if isinstance(ctx,PseudoContextV2):
            self.op = { gid : PCONV2.PseudoMergeOp(npart, ctx.get_addr(gid), 0, gid, time_it) for gid in self.device_list}
        elif isinstance(ctx,PseudoEntropyContext):
            self.op = { gid : PCONV2.PseudoMergeOp(npart, ctx.get_addr(gid), 1, gid, time_it) for gid in self.device_list}
        else:
            self.op = { gid : PCONV2.PseudoMergeOp(npart, ctx.get_addr(gid), 2, gid, time_it) for gid in self.device_list}

    def forward(self, x):
        res = PseudoMerge_AF.apply(x, self.op)
        return res


class PseudoSplit_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoSplit(BaseOpModule):
    #int pad, int npart, int fvalue, int trim, std::string addr, int context_version, int device = 0, bool timeit=false
    def __init__(self, npart, ctx, device = 0, time_it = False):
        super(PseudoSplit, self).__init__(device)
        if isinstance(ctx,PseudoContextV2):
            self.op = { gid : PCONV2.PseudoSplitOp(npart, ctx.get_addr(gid), 0, gid, time_it) for gid in self.device_list}
        elif isinstance(ctx,PseudoEntropyContext):
            self.op = { gid : PCONV2.PseudoSplitOp(npart, ctx.get_addr(gid), 1, gid, time_it) for gid in self.device_list}
        else:
            self.op = { gid : PCONV2.PseudoSplitOp(npart, ctx.get_addr(gid), 2, gid, time_it) for gid in self.device_list}

    def forward(self, x):
        res = PseudoSplit_AF.apply(x, self.op)
        return res

if __name__ == '__main__':
    import cv2
    import numpy as np
    import os
    from PCONV2_operator import SphereSlice
    raw = cv2.imread('/data2/imgs/360_512/47075159692_ccfa639898_o.png')
    #raw[:,1020:] = 255
    img = raw.transpose(2,0,1).astype(np.float32)
    data = torch.from_numpy(img).contiguous().unsqueeze(0)
    data = torch.autograd.Variable(data.to('cuda:0'),requires_grad=True)
    data.retain_grad()
    slice = SphereSlice(16,pad=0,opt=True,device=0)
    ctx = PseudoContextV2(16,True,device=0)
    merge = PseudoMerge(16,ctx,0)
    split = PseudoSplit(16,ctx,0)
    pd = slice(data)
    pd.retain_grad()
    px = merge(pd)
    px.retain_grad()
    pz = split(px)
    loss = torch.sum(pz**2)/2
    loss.backward()
    print(torch.sum(torch.abs(pd-pz)))
    print(torch.sum(torch.abs(px.grad-px.data)))
    print(torch.sum(torch.abs(pd.grad-pd.data)))
    px = torch.clip(px,0,255)
    py = px.detach().to('cpu').numpy()
    ty = py.transpose(0,2,3,1).astype(np.uint8)
    #cv2.imshow('img',img)
    #cv2.imshow('pimg',ty)
    #cv2.waitKey()
    os.makedirs('./tmp',exist_ok=True)
    cv2.imwrite('./tmp/src.png',raw)
    cv2.imwrite('./tmp/dst.png',ty[0])
    