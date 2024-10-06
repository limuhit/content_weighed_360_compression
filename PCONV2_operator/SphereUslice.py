import torch
import torch.nn as nn
import PCONV2
from PCONV2_operator.BaseOpModule import BaseOpModule
from PCONV2_operator.base import set_weight

class SphereUslice_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, op):
        if not x.is_contiguous(): x = x.contiguous()
        gid = x.device.index
        outputs = op[gid].forward(x,weight)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None, None
    

class SphereUslice(BaseOpModule):
    
    def __init__(self, npart, interp_type=0, pad=0, device = 0, time_it = False):
        super(SphereUslice, self).__init__(device)
        #weight = set_weight(npart,opt)
        self.op = { gid : PCONV2.SphereUsliceOp(npart,interp_type, pad, gid, time_it) for gid in self.device_list}

    def forward(self, x, weight):
        res = SphereUslice_AF.apply(x, weight, self.op)
        return res
