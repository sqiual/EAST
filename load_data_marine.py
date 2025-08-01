from os import listdir
from os.path import join
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import torch
import time
import pickle
import pandas as pd

    
class MyDataset(Dataset):
    def __init__(self, args, mode, percent):
        super(MyDataset, self).__init__()
        self.percent = percent
        print('loading data ... ')
        if mode in ['train', 'val']:
            self.data_train = torch.load("{}{}_{}.pt".format(args.root_path, args.data, 'train'))
            self.data_val = torch.load("{}{}_{}.pt".format(args.root_path, args.data, 'val'))
            print("{}{}_{}.pt".format(args.root_path, args.data, 'train'))
            print("{}{}_{}.pt".format(args.root_path, args.data, 'val'))
            self.x, self.y = self.data_train.tensors
            # self.x_train, self.y_train = self.data_train.tensors
            # self.x_val, self.y_val = self.data_val.tensors
            # self.x = torch.cat([self.x_train, self.x_val], dim = 0) #[46028, 8, 8, 12, 12]
            # self.y = torch.cat([self.y_train, self.y_val], dim = 0)
            
        else:
            self.data_test = torch.load("{}{}_{}.pt".format(args.root_path, args.data, 'test'))
            print("{}{}_{}.pt".format(args.root_path, args.data, 'test'))
            self.x_test, self.y_test = self.data_test.tensors
            self.x = self.x_test
            self.y = self.y_test
            
        # ========================
        # ========================
        # # 2. Normalize only feature 2 (DENSITY), using training stats only
        # if mode == 'train':
        #     feat3 = self.x[:, :, :, :, 2]
        #     self.mean_feat3 = feat3.mean(dim=(0, 3), keepdim=True)  # [1, 1, H, W]
        #     self.std_feat3 = feat3.std(dim=(0, 3), keepdim=True) + 1e-8
        #     torch.save({'mean': self.mean_feat3, 'std': self.std_feat3}, 'density_scaler.pt')
        # else:
        #     # load mean/std from train set
        #     scaler = torch.load('density_scaler.pt')
        #     self.mean_feat3 = scaler['mean']
        #     self.std_feat3 = scaler['std']
        
        # # 3. Normalize x and y on feature 2
        # feat3_x = self.x[:, :, :, :, 2]
        # feat3_y = self.y[:, :, :, :, 2]

        # x_norm = self.x.clone()
        # y_norm = self.y.clone()

        # x_norm[:, :, :, :, 2] = (feat3_x - self.mean_feat3) / self.std_feat3
        # y_norm[:, :, :, :, 2] = (feat3_y - self.mean_feat3) / self.std_feat3

        # self.x_normalized = x_norm
        # self.y_normalized = y_norm
        # ========================
        # ========================
        
        #['LAT_BIN','LON_BIN','DENSITY','MAIN_DIRECTION','DIRECTION_STD','timestep','year','month','day','doy','LAT','LON']
        print('data shape: ', self.x.shape)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return max(int(len(self.x)*self.percent),1)


        