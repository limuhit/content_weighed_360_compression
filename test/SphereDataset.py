import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from PIL import Image
from torchvision import transforms
import numpy as np

class SphereDataSet(Dataset):

    def __init__(self, train = True, root = '/data1/home/csmuli/360_dataset'):
        self.img_path = f'{root}/360_512'
        #self.img_list = os.listdir(img_dir)
        self.convert_tensor = transforms.ToTensor()
        if not train:
            f = open(f'{root}/test.txt')
            self.img_list = [pt[:-1] for pt in f.readlines()]
            f.close()
        else:
            f = open(f'{root}/train.txt')
            self.img_list = [pt[:-1] for pt in f.readlines()]
            #self.img_list = self.img_list[:1000]
            f.close()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_path, self.img_list[idx])
        #img = cv2.imread(img_name)
        img = self.convert_tensor(Image.open(img_name).convert("RGB")).type(torch.float32)
        #img = torch.from_numpy(img.transpose(2,0,1)).type(torch.float32) / 255.0
        img = img.contiguous()
        return img

class MyDistributeSampler(torch.utils.data.distributed.DistributedSampler):
    
    def __init__(self,dataset,num_replicas,rank,batch_size, shuffle=True,seed=0,mean=1.4,acc_batch=1,root='/data1/home/csmuli/360_dataset'):
        super(MyDistributeSampler,self).__init__(dataset,num_replicas,rank,shuffle,seed)
        with open(f'{root}/train_val.dic','rb') as f: self.vdict = pickle.load(f)
        self.flist = self.dataset.img_list
        if self.flist[0].find('npy')>=0:
            self.flist = [pt.replace('npy','png') for pt in self.flist]
        self.ws = batch_size*num_replicas*acc_batch
        self.thr = mean*self.ws
        self.seed_ext = 0

    def ws_fn(self,idxs,pid):
        midx,mv,ts = -1,1e9,0
        fn = lambda xi: self.vdict[self.flist[idxs[xi]]]
        for j in range(self.ws):
            a = fn(pid*self.ws+j)
            if a<mv:
                midx = j
                mv = a
            ts += a
        return midx,mv,ts

    @staticmethod
    def find_best_idx(vlist,val,sf,thr):
        tidx = sorted(range(len(vlist)),key=lambda k:vlist[k],reverse=True)
        for idx in tidx:
            if sf-vlist[idx]+val > thr:
                return idx
        return -1

    def check_modify(self, idxs, aidx, bidx):
        l = len(self.flist)
        ln = l//self.ws
        sf = [0 for _ in range(ln)]
        mvf = [0 for _ in range(ln)]
        mif = [-1 for _ in range(ln)]
        fn = lambda xi: self.vdict[self.flist[idxs[xi]]]
        sidx = 0
        last_k = -1
        for i in range(ln):  mif[i],mvf[i],sf[i]=self.ws_fn(idxs,i)
        for i in range(ln):
            while sf[i] < self.thr:
                while sidx<ln and sf[sidx] < self.thr + 0.618 and i > 0: sidx += 1 
                if sidx >= ln: return False
                for k in range(sidx,ln):
                    if sf[k] > self.thr :
                        kidx = self.find_best_idx([fn(k*self.ws+ki) for ki in range(self.ws)],mvf[i],sf[k],self.thr)
                        if kidx >= 0: break 
                if last_k == k: sidx+=1
                nk = k*self.ws+kidx
                ni = i*self.ws+mif[i]
                a = idxs[ni]
                idxs[ni] = idxs[nk]
                idxs[nk] = a
                mif[i],mvf[i],sf[i]=self.ws_fn(idxs,i)
                mif[k],mvf[k],sf[k]=self.ws_fn(idxs,k)
                last_k = k
        return True 

    def __iter__(self):
        ln = len(self.flist)//self.ws
        ln_half = ln // 2
        flag = False
        while not flag:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch + self.seed_ext)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  
            else:
                indices = list(range(len(self.dataset))) 

            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size
            aidx = torch.randperm(ln_half, generator=g).tolist()
            bidx = torch.randperm(ln-ln_half, generator=g).tolist()
            flag = self.check_modify(indices,aidx,bidx)
            if not flag: self.seed_ext += 1
        

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        print(self.rank,indices[:10])
        return iter(indices)


def load_train_test_distribute(world_size,rank, batch_size, test_batch_size, shuffle=True, seed = 0, mean=1.4,acc_batch=1):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    root = '/data2/imgs'
    train_data = SphereDataSet(True,root)
    train_sampler = MyDistributeSampler(train_data,num_replicas=world_size,rank=rank, batch_size=batch_size, shuffle=shuffle,seed=seed, 
                                        mean=mean,acc_batch=acc_batch,root=root)
    train_loader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler,
        batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        SphereDataSet(False,root),
        batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

class SphereDataSetVal(Dataset):
    
    def __init__(self, root = '/data1/home/csmuli/360_dataset'):
        self.img_path = f'{root}/360_512'
        self.convert_tensor = transforms.ToTensor()
        f = open(f'{root}/test.txt')
        self.img_list = [pt[:-1] for pt in f.readlines()]
        #print(len(self.img_list))
        f.close()
       

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_path, self.img_list[idx])
        #img = cv2.imread(img_name)
        img = self.convert_tensor(Image.open(img_name).convert("RGB"))
        #img = torch.from_numpy(img.transpose(2,0,1)).type(torch.float32) / 255.0
        img = img.contiguous()
        return img

class SphereDataSetValLarge(Dataset):
    
    def __init__(self, root = '/data2/imgs'):
        self.img_path = f'{root}/360_1024'
        self.convert_tensor = transforms.ToTensor()
        self.img_list = [f'{pid}.png' for pid in range(1,101)]
       

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_path, self.img_list[idx])
        img = self.convert_tensor(Image.open(img_name).convert("RGB"))
        img = img.contiguous()
        return img

def load_test_dataset(test_batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    root = 'E:/360_dataset/'
    test_loader = torch.utils.data.DataLoader(
        SphereDataSetVal(root),
        batch_size=test_batch_size, shuffle=False, **kwargs)
    return test_loader

def load_test_dataset_large(test_batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    root = '/data2/imgs'
    test_loader = torch.utils.data.DataLoader(
        SphereDataSetValLarge(root),
        batch_size=test_batch_size, shuffle=False, **kwargs)
    return test_loader

if __name__ == '__main__':
    train_loader, test_loader = load_train_test_distribute(4,0,2,2)
    for dt in test_loader:
        pass