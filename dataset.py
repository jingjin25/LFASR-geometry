import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
import cv2
from scipy import misc
from math import ceil
import random

class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()
        
        hf = h5py.File(opt.dataset_path)
        self.LFI = hf.get('LFI')  # [N,ah,aw,h,w]
        self.LFI = self.LFI[:, :opt.angular_out, :opt.angular_out, :, :]
   
        self.psize = opt.patch_size
        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in
    
    def __getitem__(self, index):
                        
        # get one item
        lfi = self.LFI[index]  # [ah,aw,h,w]

        # crop to patch
        H = lfi.shape[2]
        W = lfi.shape[3]

        # print('H: ', H)
        # print('W: ', W)
        x = random.randrange(0, H-self.psize)    
        y = random.randrange(0, W-self.psize) 
        lfi = lfi[:, :, x:x+self.psize, y:y+self.psize] # [ah,aw,ph,pw]   
        
        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            lfi = np.flip(np.flip(lfi,0),2)          
        if np.random.rand(1)>0.5:
            lfi = np.flip(np.flip(lfi,1),3)            
        # rotate
        r_ang = np.random.randint(1,5)
        lfi = np.rot90(lfi,r_ang,(2,3))
        lfi = np.rot90(lfi,r_ang,(0,1))
            
        
        ##### get input index ######         
        ind_all = np.arange(self.ang_out*self.ang_out).reshape(self.ang_out, self.ang_out)
        delt = (self.ang_out-1) // (self.ang_in-1)
        ind_source = ind_all[0:self.ang_out:delt, 0:self.ang_out:delt]
        ind_source = ind_source.reshape(-1)
        # print(ind_source)
            
        ##### get input and label ######    
        lfi = lfi.reshape(-1, self.psize, self.psize) #[ah*aw,ph,pw]
        input = lfi[ind_source, :, :]  #[num_source,ph,pw]
                     
        # to tensor   
        input = torch.from_numpy(input.astype(np.float32)/255.0) #[num_source,h,w]
        label = torch.from_numpy(lfi.astype(np.float32)/255.0) #[an2,h,w]
        
        return ind_source, input, label

            
    def __len__(self):
        return self.LFI.shape[0]