import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import numpy as np


class Datasat(Dataset):
    def __init__(self, mode, size, IN_CH_HSI=31, IN_CH_MSI=3):

        super(Datasat, self).__init__()
        self.band = IN_CH_HSI
        self.size = int(size)
        self.img_path1 = []
        self.img_path2 = []
        self.img_path3 = []
        self.upSample = torch.nn.Upsample(scale_factor=4, mode='bicubic')

        path = '../data/Houston/'
        N = 246
        if mode == 'train':
            self.GTHS_path = path+'/train/gtHS'
            self.img_path1 = os.listdir(self.GTHS_path)[0:N]
            sorted(self.img_path1, key=lambda x:int(x.split('.')[0]))
            self.HRMS_path = path+'/train/hrMS'
            self.img_path2 = os.listdir(self.HRMS_path)[0:N]
            sorted(self.img_path2, key=lambda x: int(x.split('.')[0])) ##_Elastic1000
            self.LRHS_path = path+'/train/LRHS_Elastic600'
            self.img_path3 = os.listdir(self.LRHS_path)[0:N]
            sorted(self.img_path3, key=lambda x: int(x.split('.')[0]))
            print('训练数据初始化')
        if mode == 'test':
            self.GTHS_path = path+'/test/gtHS'
            self.img_path1 = os.listdir(self.GTHS_path)[0:N]
            self.img_path1.sort(key=lambda x: int(x.split(".")[0]))
            sorted(self.img_path1, key=lambda x: int(x.split('.')[0]))
            self.HRMS_path = path+'/test/hrMS'
            self.img_path2 = os.listdir(self.HRMS_path)[0:N]
            self.img_path2.sort(key=lambda x: int(x.split(".")[0]))
            sorted(self.img_path2, key=lambda x: int(x.split('.')[0]))
            self.LRHS_path = path+'/test/LRHS_Elastic1000'
            self.img_path3 = os.listdir(self.LRHS_path)[0:N]
            self.img_path3.sort(key=lambda x: int(x.split(".")[0]))
            sorted(self.img_path3, key=lambda x: int(x.split('.')[0]))
            print('测试数据初始化')

        self.gtHS = []
        self.hrMS = []
        self.lrHS = []

        for i in range(len(self.img_path1)):

            self.real_GTHS_path = os.path.join(self.GTHS_path, self.img_path1[i])
            gtHS_temp = loadmat(self.real_GTHS_path)['gtHS'].reshape(self.band, self.size, self.size)
            self.gtHS.append(gtHS_temp)
            self.real_HRMS_path = os.path.join(self.HRMS_path, self.img_path2[i])
            hrMS_temp = loadmat(self.real_HRMS_path)['hrMS'].reshape(IN_CH_MSI, self.size, self.size)
            self.hrMS.append(hrMS_temp)
            self.real_LRHS_path = os.path.join(self.LRHS_path, self.img_path3[i])
            lrhs_temp_org = loadmat(self.real_LRHS_path)['LRHS'].reshape(self.band, self.size // 4, self.size // 4)

            self.lrHS.append(lrhs_temp_org)


        print('数据初始化完成')

    def __getitem__(self, item):

        gtHS = self.gtHS[item]
        hrMS = self.hrMS[item]
        lrHS = self.lrHS[item]
        gtHS = torch.from_numpy(self.gtHS[item]).float().clamp(-1.0, 1.0)
        hrMS = torch.from_numpy(self.hrMS[item]).float().clamp(-1.0, 1.0)
        lrHS = torch.from_numpy(self.lrHS[item]).float().clamp(-1.0, 1.0)
        def print_stats(name, x):
            print(
                f"{name}: "
                f"min={x.min().item():.6f}, "
                f"max={x.max().item():.6f}, "
                f"mean={x.mean().item():.6f}"
            )

        print_stats("gtHS", gtHS)
        print_stats("hrMS", hrMS)
        print_stats("lrHS", lrHS)

        return gtHS, hrMS, lrHS


    def __len__(self):
        return len(self.img_path1)

if __name__ == '__main__':
    pass

