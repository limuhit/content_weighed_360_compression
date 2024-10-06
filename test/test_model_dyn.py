import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import sys
import torch
from SphereDataset import load_test_dataset
from loss import RateDistortionLossPConvTest
from PCONV2_operator import  MultiProject
from model_zoo_v3 import Cheng2020AttentionPConvV2, Cheng2020AttentionPConvV2L2
import numpy as np
import cv2

root = 'G:/Jobs/test'


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def test_epoch(epoch, test_dataloader, model, criterion,pr_scr,pr_dst,log):
    model.eval()
    device = next(model.parameters()).device

    ssim_loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d, pr_scr, pr_dst)
            bpp_loss.update(out_criterion["bpp_loss"])
            ssim_loss.update(out_criterion["ssim_loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(f"{mse_loss.avg:.8f},{ssim_loss.avg:.5f},{bpp_loss.avg:.5f}\n", file=log, flush=True)
    
    return None

def save_img(output):
    timg = torch.clip(output,0,1)*255
    timg = timg[0].to('cpu').detach().numpy().transpose(1,2,0).astype(np.uint8)
    timg = timg[:,:,::-1]
    return timg

def test_epoch_detail(epoch, test_dataloader, model, criterion,pr_scr,pr_dst,log,logw,args):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for tidx,d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d, pr_scr, pr_dst)
            print(f'{out_criterion["mse_loss"]:.8f},{out_criterion["ssim_loss"]:.5f},{out_criterion["bpp_loss"]:.5f}', file=log, flush=True)
            weight = out_net['weight']
            res_list = []
            for idx in range(16):
                res_list.append(f'{int(weight[idx].item()+0.5)}')
            print(','.join(res_list), file=logw, flush=True)
            img = save_img(out_net['x_hat'])
            img_name = f'{args.img_dir}/{tidx+1}_{int(out_criterion["bpp_loss"]*1000)}_ourp.png'
            print(img_name)
            cv2.imwrite(img_name,img)

    #print(f"{mse_loss.avg:.8f},{ssim_loss.avg:.5f},{bpp_loss.avg:.5f}\n", file=log, flush=True)
    
    return None

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument('--gpu-id', type=int, default=0, metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument("--checkpoint", default="dyn_mse/dyn_v3_4_best.pth.tar",type=str, help="Path to a checkpoint")
    parser.add_argument("--log", default="test.log",type=str, help="Path to a output file")
    parser.add_argument('--viewport_size', type=int, default = 171, metavar='viewport', 
                        help='viewport size for 360 projection.')
    args = parser.parse_args(argv)
    return args

def modify_param(old,new):
    ndict = {}
    for pkey in new.keys():
        ndict[pkey.replace('module.','')] = new[pkey]
    for pk in old.keys():
        if old[pk].size() == ndict[pk].size():
            old[pk] = ndict[pk]
    return old

def compare(test_dataloader, model, criterion, model2, criterion2, pr_scr,pr_dst,log):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            #out_criterion = criterion(out_net, d, pr_scr, pr_dst)
            out_net2 = model2(d)
            #out_criterion2 = criterion2(out_net, d, pr_scr, pr_dst)
            print(torch.mean(torch.abs(out_net["y"]-out_net2["y"])))
    return 


def load_pconv(args,device,cid):

    checkpoint = torch.load(f'{root}/save_models/{args.checkpoint}',map_location=device)
    N = checkpoint["state_dict"]["module.g_a.0.conv1.weight"].size(0)#pdict["g_a.0.conv1.weight"].size(0)#
    model = Cheng2020AttentionPConvV2(N=N, gamma=1, eta=1, device_id=cid) # 1.838: v5, 1.840: v3, 
    pdict = modify_param(model.state_dict(),checkpoint["state_dict"])
    model.load_state_dict(pdict)
    model = model.to(device)
    criterion = RateDistortionLossPConvTest()
    return model,criterion

def Job(args):
    cid = args.gpu_id
    device = torch.device("cuda:%d"%cid)
    test_dataloader = load_test_dataset(args.test_batch_size)
    viewport_size = args.viewport_size
    pr1 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    pr2 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    model_pconv, creiterion_pconv = load_pconv(args,device,cid)
    log = open(f'{root}/save_models/{args.log}','w')
    logw = open(f'{root}/save_models/{args.log}.wt','w')
    args.img_dir = f'{root}/save_models/imgs'
    os.makedirs(args.img_dir,exist_ok=True)
    test_epoch_detail(0, test_dataloader, model_pconv, creiterion_pconv,pr1,pr2,log,logw,args)
    
def main(argv):
    args = parse_args(argv)
    Job(args)
    
if __name__ == "__main__":
    main(sys.argv[1:])