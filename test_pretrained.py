
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os.path import join

import math
import copy
import pandas as pd
 
import h5py
import matplotlib
matplotlib.use('Agg')
from scipy import misc
from skimage.measure import compare_ssim

from model_for_test import Net_view, Net_refine 
#------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------------------------------------------------------------------------#
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)
        
# Test settings
parser = argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR -- test pretrained model")

parser.add_argument("--model_path", type=str, default="./pretrained_model/HCI_2x2-7x7.pth", help="pretrained model path")

parser.add_argument("--layer_num", type=int, default=4, help="layer_num of SAS")
parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field [AngOut x AngOut]")
parser.add_argument("--angular_in", type=int, default=2, help="angular number of the sparse light field [AngIn x AngIn]")

parser.add_argument("--test_dataset", type=str, default="HCI", help="dataset for testing")
parser.add_argument("--data_path", type=str, default="./LFData/test_HCI.h5",help="file path contained the dataset for testing")

parser.add_argument("--save_img", type=int, default=0,help="save image or not")
parser.add_argument("--crop", type=int, default=0,help="crop the image into patches when out of memory")
opt = parser.parse_args()
print(opt)
#-----------------------------------------------------------------------------------#   
class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()
        
        hf = h5py.File(opt.data_path)
        self.LFI_ycbcr = hf.get('LFI_ycbcr') # [N,ah,aw,h,w,3]        

        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in

    def __getitem__(self, index):
        
        H, W = self.LFI_ycbcr.shape[3:5]
        
        lfi_ycbcr = self.LFI_ycbcr[index]  #[ah,aw,h,w,3] 
        lfi_ycbcr = lfi_ycbcr[:opt.angular_out, :opt.angular_out, :].reshape(-1, H, W, 3) #[ah*aw,h,w,3]
        
        ### input
        ind_all = np.arange(self.ang_out*self.ang_out).reshape(self.ang_out, self.ang_out)
        delt = (self.ang_out-1) // (self.ang_in-1)
        ind_source = ind_all[0:self.ang_out:delt, 0:self.ang_out:delt]
        ind_source = ind_source.reshape(-1)
        input = lfi_ycbcr[ind_source, :, :, 0]  # [num_source,H,W]
   
        ### target
        target_y = lfi_ycbcr[:, :, :, 0] #[ah*aw,h,w]
                           
        input = torch.from_numpy(input.astype(np.float32)/255.0)
        target_y = torch.from_numpy(target_y.astype(np.float32)/255.0)
        
        # keep cbcr for RGB reconstruction (Using Ground truth just for visual results)
        lfi_ycbcr = torch.from_numpy(lfi_ycbcr.astype(np.float32)/255.0) 
        
        return ind_source, input, target_y, lfi_ycbcr
        
    def __len__(self):
        return self.LFI_ycbcr.shape[0]
#-----------------------------------------------------------------------------------#         

#-------------------------------------------------------------------------------#

if not os.path.exists(opt.model_path):
    print('model folder is not found ')
    
if not os.path.exists('quan_results'):
    os.makedirs('quan_results')    
    
if opt.save_img:
    save_dir = 'saveImg/resIm_{}'.format(opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
#------------------------------------------------------------------------#
# Data loader
print('===> Loading test datasets')
# data_path = join('LFData', 'test_{}.h5'.format(opt.test_dataset))
test_set = DatasetFromHdf5(opt)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
print('loaded {} LFIs from {}'.format(len(test_loader), opt.data_path))
#-------------------------------------------------------------------------#
# Build model
print("building net")
opt.num_source = opt.angular_in * opt.angular_in
model_view = Net_view(opt).to(device)
model_refine = Net_refine(opt).to(device)
#------------------------------------------------------------------------#

#-------------------------------------------------------------------------#    
# test  
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16. / 255.
    rgb[:,1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)
    
def CropPatches_w(image,len,crop):
    #image [1,an2,4,ph,pw]
    #left [1,an2,4,h,lw]
    #middles[n,an2,4,h,mw]
    #right [1,an2,4,h,rw]
    an,f,h,w = image.shape[1:5]
    left = image[:,:,:,:,0:len+crop]
    num = math.floor((w-len-crop)/len)
    middles = torch.Tensor(num,an,f,h,len+crop*2).to(image.device)
    for i in range(num):
        middles[i] = image[0,:,:,:,(i+1)*len-crop:(i+2)*len+crop]      
    right = image[:,:,:,:,-(len+crop):]
    return left,middles,right

def MergePatches_w(left,middles,right,h,w,len,crop):
    #[N,4,h,w]
    n,a = left.shape[0:2]
    out = torch.Tensor(n,a,h,w).to(left.device)
    out[:,:,:,:len] = left[:,:,:,:-crop]
    for i in range(middles.shape[0]): 
        out[:,:,:,len*(i+1):len*(i+2)] = middles[i:i+1,:,:,crop:-crop]        
    out[:,:,:,-len:]=right[:,:,:,crop:]
    return out

           
def compt_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def save_img(pred_y, pred_ycbcr, lfi_no):
    if opt.save_img:
        for i in range(opt.angular_out * opt.angular_out):
            img_ycbcr = pred_ycbcr[0, i]
            img_ycbcr[:, :, 0] = pred_y[0, i]  # [h,w,3]
            img_name = '{}/SynLFI{}_view{}.png'.format(save_dir, lfi_no, i)
            img_rgb = ycbcr2rgb(img_ycbcr)
            img = (img_rgb.clip(0, 1) * 255.0).astype(np.uint8)
            misc.toimage(img, cmin=0, cmax=255).save(img_name)

def compute_quan(pred_y, target_y, ind_source, csv_name, lfi_no):
    view_list = []
    view_psnr_y = []
    view_ssim_y = []

    for i in range(opt.angular_out * opt.angular_out):
        if i not in ind_source:
            cur_target_y = target_y[0, i]
            cur_pred_y = pred_y[0, i]

            cur_psnr_y = compt_psnr(cur_target_y, cur_pred_y)
            cur_ssim_y = compare_ssim((cur_target_y * 255.0).astype(np.uint8), (cur_pred_y * 255.0).astype(np.uint8),
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

            view_list.append(i)
            view_psnr_y.append(cur_psnr_y)
            view_ssim_y.append(cur_ssim_y)

    dataframe_lfi = pd.DataFrame(
        {'targetView_LFI{}'.format(lfi_no): view_list, 'psnr Y': view_psnr_y, 'ssim Y': view_ssim_y})
    dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')

    return np.mean(view_psnr_y), np.mean(view_ssim_y)


def test():
    
    lf_list = []                
    lf_psnr_list = []          
    lf_ssim_list = []

    csv_name = 'quan_results/res_{}.csv'.format(opt.test_dataset)

    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            print('testing LF {}'.format(k))
            #------------------ SR ------------------------#
            ind_source, input, target_y, lfi_ycbcr = batch[0], batch[1], batch[2].numpy(), batch[3].numpy()
        
            ## view synthesis
            input = input.to(device)
            inter_lf = model_view(ind_source, input, opt)

            ## refine            
            length = 120
            crop = 20

            input_l, input_m, input_r = CropPatches_w(inter_lf, length, crop)
            
            ################### left ###################
            pred_l = model_refine(input_l, opt)
            ################### middles ###################
            pred_m = torch.Tensor(input_m.shape[0], opt.angular_out*opt.angular_out, input_m.shape[3], input_m.shape[4])
            for i in range(input_m.shape[0]):
                cur_input_m = input_m[i:i+1]
                pred_m[i:i+1] = model_refine(cur_input_m, opt)
             
            ################### right ###################
            pred_r = model_refine(input_r, opt)
                            
            pred_y = MergePatches_w(pred_l, pred_m, pred_r, input.shape[2], input.shape[3], length, crop)  #[N,an2,hs,ws]
            pred_y = pred_y.cpu().numpy()
         

            ## shave boundary
            bd = 22
            pred_y = pred_y[:,:,bd:-bd,bd:-bd]
            target_y = target_y[:,:,bd:-bd,bd:-bd]            
            pred_ycbcr = lfi_ycbcr[:,:,bd:-bd,bd:-bd,:]

            #------------------------------save imgs -----------------------------------#                       
            save_img(pred_y, pred_ycbcr, k)

            #---------------compute PSNR/SSIM for each view ------------------------#
            lf_psnr, lf_ssim = compute_quan(pred_y, target_y, ind_source, csv_name, k)
            print('psnr: {:.2f}, ssim: {:.3f}'.format(lf_psnr, lf_ssim))

            #---------------compute PSNR/SSIM for each LFI ------------------------#
            lf_list.append(k)
            lf_psnr_list.append(lf_psnr)
            lf_ssim_list.append(lf_ssim)

        dataframe_lfi = pd.DataFrame({'LFI': lf_list, 'psnr Y':lf_psnr_list, 'ssim Y':lf_ssim_list})
        dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')

#------------------------------------------------------------------------#

# for epoch in test_epochs: 
print('===> test')
checkpoint = torch.load(opt.model_path)
ckp_dict = checkpoint['model']

model_view_dict = model_view.state_dict()
ckp_dict_view = {k: v for k, v in ckp_dict.items() if k in model_view_dict}
model_view_dict.update(ckp_dict_view)
model_view.load_state_dict(model_view_dict)

model_refine_dict = model_refine.state_dict()
ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_refine_dict}
model_refine_dict.update(ckp_dict_refine)
model_refine.load_state_dict(model_refine_dict)
 
print('loaded model {}'.format(opt.model_path))
model_view.eval()
model_refine.eval()
test()
                  

