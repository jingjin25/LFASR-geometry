
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import math
import os
from os.path import join
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import DatasetFromHdf5
from model import Net
#--------------------------------------------------------------------------#
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR")

#training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=500, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=96, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_cp", type=int, default=25, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")

parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss")
parser.add_argument("--epi", type=float, default=1.0, help="epi loss")

parser.add_argument("--dataset", type=str, default="HCI", help="Dataset for training")
parser.add_argument("--dataset_path", type=str, default="./LFData/train_HCI.h5", help="H5 file containing the dataset for training")
#model 
parser.add_argument("--layer_num", type=int, default=4, help="layer_num of SAS")
parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field")
parser.add_argument("--angular_in", type=int, default=2, help="angular number of the sparse light field [AngIn x AngIn]")

opt = parser.parse_args()
print(opt)
#--------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#--------------------------------------------------------------------------#
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

opt.num_source = opt.angular_in * opt.angular_in
model_dir = 'model_{}_S{}_epi{}_lr{}_step{}x{}'.format(opt.dataset, opt.num_source, opt.epi, opt.lr, opt.step,opt.reduce)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
#--------------------------------------------------------------------------#
# Data loader
print('===> Loading datasets')
# dataset_path = join('LFData', 'train_{}.h5'.format(opt.dataset))
train_set = DatasetFromHdf5(opt )
train_loader = DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=True)
print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))
#--------------------------------------------------------------------------#
# Build model
print("building net")
model = Net(opt).to(device)
#-------------------------------------------------------------------------#
# optimizer and loss logger
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
losslogger = defaultdict(list)
#------------------------------------------------------------------------#    
# optionally resume from a checkpoint
if opt.resume_epoch:
    resume_path = join(model_dir,'model_epoch_{}.pth'.format(opt.resume_epoch))
    if os.path.isfile(resume_path):
        print("==>loading checkpoint 'epoch{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losslogger = checkpoint['losslogger']
    else:
        print("==> no model found at 'epoch{}'".format(opt.resume_epoch))
#------------------------------------------------------------------------#
# loss
def reconstruction_loss(X,Y):
# L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss

def gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def smooth_loss(pred_map):
    #[N,an2,h,w]   
    loss = 0
    weight = 1.   
    dx, dy = gradient(pred_map)
    dx2,dxdy = gradient(dx)
    dydx,dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight 
    return loss
    
def epi_loss(pred,label):
# epi loss

    def lf2epi(lf):
        N,an2,h,w = lf.shape
        an = int(math.sqrt(an2))
        # [N,an2,h,w] -> [N*ah*h,aw,w]  &  [N*aw*w,ah,h]
        # print(an)
        # print(lf.view(N,an,an,h,w).permute(0,1,3,2,4).view(-1,an,w).shape)
        epi_h = lf.view(N,an,an,h,w).permute(0,1,3,2,4).contiguous().view(-1,1,an,w)
        epi_v = lf.view(N,an,an,h,w).permute(0,2,4,1,3).contiguous().view(-1,1,an,h)
        return epi_h, epi_v
    
    epi_h_pred, epi_v_pred = lf2epi(pred)
    dx_h_pred, dy_h_pred = gradient(epi_h_pred)
    dx_v_pred, dy_v_pred = gradient(epi_v_pred)
    
    epi_h_label, epi_v_label = lf2epi(label)
    dx_h_label, dy_h_label = gradient(epi_h_label)
    dx_v_label, dy_v_label = gradient(epi_v_label)
    
    return reconstruction_loss(dx_h_pred, dx_h_label) + reconstruction_loss(dy_h_pred, dy_h_label) + reconstruction_loss(dx_v_pred, dx_v_label) + reconstruction_loss(dy_v_pred, dy_v_label)
    
#-----------------------------------------------------------------------#  
 
def train(epoch):

    model.train()
    scheduler.step()    
    loss_count = 0.   
    for k in range(10):  
        for i, batch in enumerate(train_loader, 1):
            # print(i)
            ind_source, input, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # forward pass
            pred_views, disp, pred_lf = model(ind_source, input, opt)   
            
            # loss
            loss = reconstruction_loss(pred_lf, label) + opt.smooth * smooth_loss(disp)
            for i in range(pred_views.shape[2]):
                loss += reconstruction_loss(pred_views[:, :, i, :, :], label)
            loss += opt.epi * epi_loss(pred_lf, label)
            loss_count += loss.item()

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count/len(train_loader))       

#-------------------------------------------------------------------------#
print('==>training')
for epoch in range(opt.resume_epoch+1, 3000):
   
    train(epoch)

#     checkpoint
    if epoch % opt.num_cp == 0:        
        model_save_path = join(model_dir,"model_epoch_{}.pth".format(epoch))        
        state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'losslogger': losslogger,}
        torch.save(state,model_save_path)
        print("checkpoint saved to {}".format(model_save_path))     

    if epoch % opt.num_snapshot == 0:   
        plt.figure()
        plt.title('loss')
        plt.plot(losslogger['epoch'],losslogger['loss'])
        plt.savefig(model_dir+".jpg")
        plt.close()
        

