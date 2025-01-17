import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import sys
import torch
from loss import CodingLoss
from PCONV2_operator import  MultiProject
from model_zoo_v3 import Cheng2020AttentionPConvV2
import numpy as np
import cv2
import math
root = '.'
psnr_f = lambda xa: 10*math.log10(1./xa)
model_ssim_list = [f'dyn_ssim_v3_{midx}_best.pth.tar' for midx in range(1,3)]+['dyn_ssim_v3_25_best.pth.tar']+[f'dyn_ssim_v3_{midx}_best.pth.tar' for midx in range(3,7)]
model_mse_list = [f'dyn_v3_{midx}_best.pth.tar' for midx in range(2,7)]
mse_model_dir = 'E:/VSProjects/contentPCONV/save_models/dyn_mse/'
ssim_model_dir = 'E:/VSProjects/contentPCONV/save_models/dyn_ssim/'

def modify_param(old,new):
    ndict = {}
    for pkey in new.keys():
        ndict[pkey.replace('module.','')] = new[pkey]
    for pk in old.keys():
        if old[pk].size() == ndict[pk].size():
            old[pk] = ndict[pk]
    return old

def load_pconv(checkpoint_path,device,cid):
    checkpoint = torch.load(checkpoint_path,map_location=device)
    N = checkpoint["state_dict"]["module.g_a.0.conv1.weight"].size(0)
    model = Cheng2020AttentionPConvV2(N=N, gamma=1, eta=1, device_id=cid)
    pdict = modify_param(model.state_dict(),checkpoint["state_dict"])
    model.load_state_dict(pdict)
    model = model.to(device)
    criterion = CodingLoss()
    return model,criterion,N

def img2tensor(img,device):
    img = img[:,:,::-1]
    ts = torch.from_numpy(img.transpose(2,0,1).astype(np.float32))/255.
    return torch.unsqueeze(ts,0).to(device).contiguous()

def tensor2img(data):
    timg = torch.clip(data,0,1)*255
    timg = timg[0].to('cpu').detach().numpy().transpose(1,2,0).astype(np.uint8)
    timg = timg[:,:,::-1]
    return timg

def check_img(img):
    h,w = img.shape[:2]
    if not(h==512 and w==1024):
        return cv2.resize(img,(1024,512),interp=cv2.INTER_CUBIC)
    else:
        return img

def encoding(img_list, out_list, model_idx=0, mse=True, device_id = 0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    checkpoint_path = model_dir + prex
    cuda = 'cuda:{}'.format(device_id)
    model_pconv, _,_ = load_pconv(checkpoint_path,cuda,device_id)
    with torch.no_grad():
        for fn, fo in zip(img_list,out_list):
            img = check_img(cv2.imread(fn))
            data = img2tensor(img,cuda)
            model_pconv.encoding(data,fo)
            print('Encoding {}, bitrate: {:.3f}bpp'.format(fn,os.path.getsize(fo)*8/1024./512.))

def decoding(code_list, decoded_img_list, model_idx=0,mse=True, device_id=0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    checkpoint_path = model_dir + prex
    cuda = 'cuda:{}'.format(device_id)
    model_pconv, creiterion_pconv,N = load_pconv(checkpoint_path,cuda,device_id)
    model_pconv.device = cuda
    with torch.no_grad():
        for fc,fo in zip(code_list,decoded_img_list):
            rdata = model_pconv.decoding(code_channels=N,code_file=fc)
            img = tensor2img(rdata)
            cv2.imwrite(fo,img)
            print('Decoding {}, output to {}'.format(fc,fo))
    

def decoding_and_test(code_list, img_list, model_idx=0,mse=True,device_id=0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    checkpoint_path = model_dir + prex
    cuda = 'cuda:{}'.format(device_id)
    model_pconv, criterion,N = load_pconv(checkpoint_path,cuda,device_id)
    model_pconv.device = cuda
    viewport_size = 172
    pr_scr = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, device_id).to(cuda)
    pr_dst = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, device_id).to(cuda)
    rt_list, pr_list, ssim_list = [], [], []
    with torch.no_grad():
        for fc, fn in zip(code_list,img_list):
            rdata = model_pconv.decoding(code_channels=N,code_file=fc)
            simg = tensor2img(rdata)
            cv2.imwrite(fc+'.png',simg)
            img = check_img(cv2.imread(fn))
            data = img2tensor(img,cuda)
            bits = os.path.getsize(fc)*8
            out_criterion = criterion(rdata, data, bits, pr_scr,pr_dst)
            pr,vssim,rt = psnr_f(out_criterion["mse_loss"]),out_criterion["ssim_loss"],out_criterion["bpp_loss"]
            rt_list.append(rt)
            pr_list.append(pr)
            ssim_list.append(vssim)
            print('Decoding {}, compare it to {} \n Bitrate:{:.3f}bpp, PSNR:{:.2f}dB, SSIM:{:.4f}'.format(fc, fn, rt, pr, vssim))
    print('-----------------------------------------------------\nAverage Performance\n-----------------------------------------------------')
    rt,pr,vssim = np.average(np.array(rt_list)), np.average(np.array(pr_list)), np.average(np.array(ssim_list))
    print('Bitrate:{:.3f}bpp, PSNR:{:.2f}dB, SSIM:{:.4f}'.format(rt, pr, vssim))

def test_pseudo_image_coding():
    img_dir = 'e:/360_dataset/360_512'
    code_dir = '../tmp'
    with open('e:/360_dataset/test.txt') as f:
        test_list = [pt[:-1] for pt in f.readlines()]
    img_list = ['{}/{}'.format(img_dir,fn) for fn in test_list[:10]]
    code_list = ['{}/{}'.format(code_dir,idx) for idx in range(10)]
    model_idx = 2
    encoding(img_list,code_list,model_idx)
    decoding_and_test(code_list,img_list,model_idx)
    #decoding(code_list,[f'../tmp/{fidx}.png' for fidx in range(0,10)],model_idx)

def read_list(fname):
    with open(fname) as f:
        return [line.rstrip('\n') for line in f.readlines()]

def check_models():
    assert(os.path.exists('{}/{}'.format(mse_model_dir,model_mse_list[0]))),'Please make sure the pretrained models for VMSE exists in the mse_model_dir'
    assert(os.path.exists('{}/{}'.format(ssim_model_dir,model_ssim_list[0]))),'Please make sure the pretrained models for VSSIM exists in the ssim_model_dir'

if __name__ == '__main__':
    #test_pseudo_image_coding()
    parser = argparse.ArgumentParser(description='Pseudo Convolution for 360 Image Compression')
    parser.add_argument('--img-list', nargs='*', help='The image list contains the input images for encoding and testing')
    parser.add_argument('--code-list', nargs='*', help='The code file list for codes')
    parser.add_argument('--out-list', nargs='*', help='The out list for saving decoded images.')
    parser.add_argument('--img-file', help='The file contains the input images for encoding and testing')
    parser.add_argument('--code-file', help='The file contains the list for codes')
    parser.add_argument('--out-file', help='The file  contains the names of decoded images.')
    parser.add_argument('--model-idx', type=int, default=0, help='Model index (0-4) for VMSE, (0-6) for VSSIM')
    parser.add_argument('--enc', action='store_true', default=False, help='Encoding flag, set for encoding phase.')
    parser.add_argument('--dec', action='store_true', default=False, help='Decoding flag, set for decoding phase.')
    parser.add_argument('--test', action='store_true', default=False, help='Testing flag, set for decoding and evalating the performance.')
    parser.add_argument('--ssim', action='store_true', default=False, help='Default with models optimized for VMSE, \
        set this flag for choosing the models optimized for VSSIM')
    parser.add_argument('--gpu-id', type=int, default=0, help='The graphic card id for encoding and decoding.')
    args = parser.parse_args()
    check_models()
    midx = args.model_idx
    if args.ssim:
        assert(midx<9 and midx>=0),'(0-6) for VSSIM'
    else:
        assert(midx<10 and midx>=0),'(0-4) for VMSE'
    assert(args.enc or args.dec or args.test),'Should set one flag, (--enc) for encoding, (--dec) for decoding, (--test) for testing.'
    img_lnone, img_fnone = args.img_list is not None, args.img_file is not None
    code_lnone, code_fnone = args.code_list is not None, args.code_file is not None
    out_lnone, out_fnone = args.out_list is not None, args.out_file is not None
    if args.enc:
        assert(img_fnone or img_lnone), 'No input images for encoding'
        assert(code_lnone or code_fnone), 'No code files for saving the codes'
        img_list = args.img_list if img_lnone else read_list(args.img_file)
        code_list = args.code_list if code_lnone else read_list(args.code_file)
        assert(len(img_list)==len(code_list)), 'The number of images and codes should be the same'
        encoding(img_list,code_list,midx,not args.ssim,args.gpu_id)
    else:
        assert(code_lnone or code_fnone), 'No code files for decoding'
        code_list = args.code_list if code_lnone else read_list(args.code_file)
        if args.dec:
            assert(out_lnone or out_fnone), 'No out files for saving the decoded images'
            out_list = args.out_list if out_lnone else read_list(args.out_file)
            assert(len(code_list)==len(out_list)), 'The number of codes and reconstructed images should be the same'
            decoding(code_list,out_list,midx,not args.ssim,args.gpu_id)
        else:
            assert(img_fnone or img_lnone), 'No source images for evaluation.'
            img_list = args.img_list if img_lnone else read_list(args.img_file)
            assert(len(code_list)==len(img_list)), 'The number of codes and corresponding source images should be the same'
            decoding_and_test(code_list,img_list,midx,not args.ssim,args.gpu_id)
