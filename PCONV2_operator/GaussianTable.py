import torch
import torch.nn as nn
import PCONV2
from PCONV2_operator.BaseOpModule import BaseOpModule

class GaussianTable_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, delta, mean, op):
        gid = mean.device.index
        if not delta.is_contiguous(): delta = delta.contiguous()
        if not mean.is_contiguous(): mean = mean.contiguous()
        outputs = op[gid].forward(delta, mean)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None

class GaussianTable(BaseOpModule):
    
    def __init__(self, nstep, nelement, bias, total_region, lower_bound=1e-9, device = 0, time_it = False):
        super(GaussianTable, self).__init__(device)
        self.op = { gid : PCONV2.GaussianTableOp(nstep,nelement,bias,total_region,lower_bound, gid, time_it) for gid in self.device_list}
        

    def forward(self,  delta, mean):
        res = GaussianTable_AF.apply(delta, mean, self.op)
        return res


if __name__ == '__main__':
    import math
    data = torch.arange(-3,4).type(torch.float32).to('cuda:0')
    mean = torch.rand(3).type(torch.float32).view(1,1,1,3).to('cuda:0')
    delta = torch.rand(3).type(torch.float32).view(1,1,1,3).to('cuda:0')+0.5
    out = 0.5 * (1 + torch.erf((data - 0.5- mean[0,0,0,1]) / delta[0,0,0,1] / math.sqrt(2)))*65536
    print(out)
    gt = GaussianTable(7,3,3.,65536.)
    out2 = gt(delta,mean)
    print(out2)
    
    
    