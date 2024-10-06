import os
import argparse
import random
import shutil
import sys
import torch

from PCONV2_operator import  MultiProject
from SphereDataset import load_train_test_distribute
from loss import RateDistortionLossPConv
from net_aux import net_aux_optimizer
from model_zoo_v3 import Cheng2020AttentionPConvV2, Cheng2020AttentionPConvV2L2

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pretrained import load_pretrained

root = '/home/csmuli/PCONV2'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size)

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



def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
        "param": {"type": "Adam", "lr": args.learning_rate}
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"], optimizer["param"]

def get_params(model, param_flag=False):
    if not param_flag:
        param = (
                param 
                for name, param in model.module.named_parameters()
                if param.requires_grad and not name.endswith(".quantiles") and name.find('param_net') < 0
        )
    else:
        param = (
                param 
                for name, param in model.module.named_parameters()
                if param.requires_grad and  name.find('param_net') >= 0
        )
    return param

def zero_grad(model):
    for name, param in model.module.named_parameters():
        if param.requires_grad and not (param.grad is None):
            param.grad.fill_(0)

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,
    pr_scr,pr_dst,log, args, param_flag = False
):
    model.train()
    device = next(model.parameters()).device
    train_dataloader.sampler.set_epoch(epoch)
    dist_proto = 'mse_loss'#'ssim_loss'

        
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}",file=log,flush=True)
    print(f"gamma: {args.gamma}, lambda: {criterion.lmbda}",file=log,flush=True)
    
    for i, d in enumerate(train_dataloader):        
        d = d.to(device)
        if d.shape[0] < args.batch_size: break
        
        zero_grad(model)
        
        out_net = model(d)
        out_criterion = criterion(out_net, d, pr_scr, pr_dst)
        out_criterion["loss"].backward()
        
        aux_loss = model.module.aux_loss()
        aux_loss.backward()
        
        param = get_params(model, param_flag)
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(param, clip_max_norm)
        
        #old_p = model.module.g_s[4].conv.weight.clone()
            
        optimizer.step()
        aux_optimizer.step()

        if i % 1 == 0:
            dist = out_criterion[dist_proto]
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |',
                f'\tProxy:{model.module.param_loss.func.val:.3f} |',
                f'\t{dist_proto}: {dist.item():.7f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f"\tAux loss: {aux_loss.item():.2f} |"
                f'\tMask Ave: {out_criterion["mask_ave"].item():.3f} |',
                file=log, flush=True
            )
            #'''
            if i>100:
                #torch.save({'data':d,'tx':out_criterion},'tmp.pt')
                break
            #'''


def test_epoch(epoch, test_dataloader, model, criterion,pr_scr,pr_dst,log):
    model.eval()
    device = next(model.parameters()).device
    dist_proto = 'mse_loss'#'ssim_loss'
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    dist_loss = AverageMeter()
    aux_loss = AverageMeter()
    mask_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d, pr_scr, pr_dst)
            aux_loss.update(model.module.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            dist_loss.update(out_criterion[dist_proto])
            mask_loss.update(out_criterion['mask_ave'])
            #print(out_criterion['mask_ave'].item(),flush=True)
            #print(out_criterion[dist_proto].item(),file=log,flush=True)

    print(
        f"\nTest epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\t{dist_proto}: {dist_loss.avg:.7f} |"
        f"\tBpp loss: {bpp_loss.avg:.3f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tMask loss: {mask_loss.avg:.3f}\n",
        file=log, flush=True
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint_pconv.pth.tar"):
    os.makedirs(f'{root}/save_models',exist_ok=True)
    torch.save(state, f'{root}/save_models/{filename}')
    tmp = filename.split('.')
    tmp[0] += '_best'
    best = '.'.join(tmp)
    print(f'{best} pt')
    if is_best:
        shutil.copyfile(f'{root}/save_models/{filename}', f"{root}/save_models/{best}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument('--gpu-ids', nargs='*', default=[0], metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--acc-batch', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument('--init', type=str, #default='mse_pconv/pc_v3_6_best.pth.tar', 
                        help="Path to a intialization model") 
    
    parser.add_argument("--checkpoint", type=str, default='dyn_mse/dyn_v3_2_best.pth.tar',
                        help="Path to a checkpoint")
    
    parser.add_argument('--save-name', type=str, default='test', help="Path to a intialization model") 
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.03,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )

    parser.add_argument(
        "--gamma",
        dest="gamma",
        type=float,
        default=1e-4,
        help="gamma"
    )
    
    parser.add_argument(
        "--eta",
        dest="eta",
        type=float,
        default=1000,
        help="eta"
    )
    
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=False, help="Save model to disk"
    )
    parser.add_argument(
        "--freeze", action="store_true", default=False,
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    
    parser.add_argument(
        "--clip_max_norm",
        default=1,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    
    parser.add_argument('--viewport_size', type=int, default = 171, metavar='viewport', 
                        help='viewport size for 360 projection.')
    parser.add_argument("--N", type=int, default=192, help="network params")
    args = parser.parse_args(argv)
    return args

def modify_param(old,new):
    for pk in old.keys():
        if old[pk].size() == new[pk].size():
            old[pk] = new[pk]
    return old

def load_checkpoint(model,checkpoint,device):
    checkpoint = torch.load(checkpoint, map_location=device)
    pdict = modify_param(model.state_dict(),checkpoint["state_dict"])
    model.load_state_dict(pdict)
    
def init_with_pconv(model,pdict):
    ndict = model.state_dict()
    for pkey in ndict.keys():
        tpk = f'module.{pkey}'
        if tpk in pdict.keys():
            ndict[pkey] = pdict[tpk]
    model.load_state_dict(ndict)

def Job(rank, world_size, args):
    cid = args.gpu_ids[rank]
    args.gpu_id = cid
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    setup(rank,world_size)
    device = torch.device("cuda:%d"%cid)
    train_dataloader,test_dataloader = load_train_test_distribute(world_size,rank, args.batch_size, args.test_batch_size, mean = 1.5, acc_batch=args.acc_batch)
    viewport_size = args.viewport_size
    pr1 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    pr2 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    model = Cheng2020AttentionPConvV2(N=args.N, gamma=args.gamma, eta=args.eta, device_id=cid) # 1.838: v5, 1.840: v3, 
    if args.init:
        pdict = torch.load(f'{root}/save_models/{args.init}',map_location=device)['state_dict']
        param_dict = torch.load(f'{root}/save_models/param_ssim_best.pth.tar',map_location=device)['state_dict']
        pdict = {**pdict,**param_dict}
        init_with_pconv(model,pdict)
        print(f'init with {args.init}')
    model = model.to(device)
    model = DDP(model,[cid])
    optimizer, aux_optimizer, param_optimizer = configure_optimizers(model, args)
    criterion = RateDistortionLossPConv(lmbda=args.lmbda,metric='mse')
    prex = args.save_name.split('.')[0]
    log = open(f'{root}/save_models/{prex}_{rank}_log.txt','a')
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        load_checkpoint(model,f'{root}/save_models/{args.checkpoint}',device)
    
    if args.init:
        test_epoch(0, test_dataloader, model, criterion,pr1,pr2,log)    
    
    if args.checkpoint:
        best_loss = test_epoch(0, test_dataloader, model, criterion,pr1,pr2,log)
    else:
        best_loss = float("inf")
    
    for epoch in range(args.epochs):
        
        current_optimizer = param_optimizer if  epoch % 2 == 0 else optimizer
        param_flag = True if  epoch % 2 == 0 else False
        if args.freeze:
            current_optimizer, param_flag = optimizer, False
        train_one_epoch(
            model,
            criterion,
            train_dataloader,
            current_optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            pr1,pr2,log,args,
            param_flag
        )
        loss = test_epoch(epoch, test_dataloader, model, criterion,pr1,pr2,log)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save and rank == 0:
            save_checkpoint({ "state_dict": model.state_dict()}, is_best,filename=args.save_name)
            
    log.close()
    
def main(argv):
    args = parse_args(argv)
    args.gpu_ids = [int(pt) for pt in args.gpu_ids]
    world_size = len(args.gpu_ids)
    mp.spawn(Job,
             args=(world_size,args,),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    main(sys.argv[1:])