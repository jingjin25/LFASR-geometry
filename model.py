
import torch
import torch.nn as nn
import torch.nn.functional as functional
from model_utility import *


class Net(nn.Module):    
    def __init__(self, opt):        
        
        super(Net, self).__init__()

        an2 = opt.angular_out * opt.angular_out
        
        # disparity
        self.disp_estimator = nn.Sequential(
            nn.Conv2d(opt.num_source,16,kernel_size=7,stride=1,dilation=2,padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,kernel_size=7,stride=1,dilation=2,padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,an2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2,an2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2,an2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2,an2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2,an2,kernel_size=3,stride=1,padding=1),
            )
        
        # LF     
        self.lf_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=opt.num_source, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lf_altblock = make_Altlayer(layer_num=opt.layer_num, an=opt.angular_out, ch=64)
        if opt.angular_out == 9:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5,3,3), stride=(4,1,1), padding=(0,1,1)),#81->20
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5,3,3), stride=(3,1,1), padding=(0,1,1)), #20->6
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=81, kernel_size=(6,3,3), stride=(1,1,1), padding=(0,1,1)), #6-->1
            )              
 

        if opt.angular_out == 8:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4,3,3), stride=(4,1,1), padding=(0,1,1)),#64-->16
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4,3,3), stride=(4,1,1), padding=(0,1,1)), #16-->4
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4,3,3), stride=(1,1,1), padding=(0,1,1)),#4-->1
            )

        if opt.angular_out == 7:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5,3,3), stride=(4,1,1), padding=(0,1,1)),#49-->12
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4,3,3), stride=(4,1,1), padding=(0,1,1)), #12-->3
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=49, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1)),#3-->1
            )      
        
        
    def forward(self, ind_source, img_source, opt):
        
        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out
        
        # ind_source 
        N,num_source,h,w = img_source.shape   #[N,num_source,h,w]
        ind_source = torch.squeeze(ind_source) #[num_source]                  
        
        #################### disparity estimation ##############################
        disp_target = self.disp_estimator(img_source)     #[N,an2,h,w]

        #################### intermediate LF ##############################
        warp_img_input = img_source.view(N*num_source,1,h,w).repeat(an2,1,1,1) #[N*an2*4,1,h,w]
        
        grid = []
        for k_t in range(0,an2):
            for k_s in range(0,num_source):
                ind_s = ind_source[k_s].type_as(img_source)
                ind_t = torch.arange(an2)[k_t].type_as(img_source)
                ind_s_h = torch.floor(ind_s/an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t/an)
                ind_t_w = ind_t % an   
                disp = disp_target[:,k_t,:,:]
                
                XX = torch.arange(0,w).view(1,1,w).expand(N,h,w).type_as(img_source) #[N,h,w]
                YY = torch.arange(0,h).view(1,h,1).expand(N,h,w).type_as(img_source)                 
                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)
                grid_w_t_norm = 2.0 * grid_w_t / (w-1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h-1) - 1.0                
                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm),dim=3) #[N,h,w,2]    
                grid.append(grid_t)        
        grid = torch.cat(grid,0) #[N*an2*4,h,w,2]
        
        warped_img = functional.grid_sample(warp_img_input,grid).view(N,an2,num_source,h,w)

        ################# refine LF ###########################
        feat = self.lf_conv0(warped_img.view(N*an2,num_source,h,w)) #[N*an2,64,h,w]
        feat = self.lf_altblock(feat) #[N*an2,64,h,w]
        feat = torch.transpose(feat.view(N,an2,64,h,w),1,2) #[N,64,an2,h,w]
        res = self.lf_res_conv(feat) #[N,an2,1,h,w]
        
        lf = warped_img[:,:,0,:,:] + torch.squeeze(res,2) #[N,an2,h,w] 
        
        return warped_img, disp_target, lf
