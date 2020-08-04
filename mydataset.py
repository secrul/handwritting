import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ToTensor(object):

    def __call__(self, sample):
        data = torch.Tensor(sample['traindata']).unsqueeze(0)
        label = torch.Tensor([sample['trainlabel']])
        label = torch.tensor(label, dtype=torch.int64)
        return {'data':data,
                'label':label}


class MyDataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.size = 0
        self.data = []
        self.label = []
        # 每10帧一组
        files = os.listdir(path)
        self.size = len(files)
        for file in files:
            tag = file.split('_')[0]
            self.label.append(int(tag))
            txt = []
            with open(path + '/' + file,'r') as f:
                lines = f.readlines()
                for line in lines:
                    tt = []
                    for s in line.strip():
                        tt.append(int(s))
                    txt.append(tt)
            self.data.append(np.array(txt))


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = {'traindata':self.data[idx],
                  'trainlabel':self.label[idx]
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample
