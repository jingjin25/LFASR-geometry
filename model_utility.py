
import torch
import torch.nn as nn
import torch.nn.functional as functional

##########
def warping(disp, ind_source, ind_target, img_source, an):
    '''warping one source image/map to the target'''
    # an angular number 
    # disparity int / [N,h,w]
    # ind_souce 0-3  -->[0,an-1,an(an-1),an2-1]
    # ind_target 0-an2-1
    # img_source [N,h,w]
   
    # ==> out [N,1,h,w]
    
    an2 = an*an
    N,h,w = img_source.shape
    ind_source = ind_source.type_as(disp)
    ind_target = ind_target.type_as(disp)
    #print(img_source.shape)
    # coordinate for source and target
    # ind_souce = torch.tensor([0,an-1,an2-an,an2-1])[ind_source]
    ind_h_source = torch.floor(ind_source / an )
    ind_w_source = ind_source % an
    
    ind_h_target = torch.floor(ind_target / an)
    ind_w_target = ind_target % an

    # generate grid
    XX = torch.arange(0,w).view(1,1,w).expand(N,h,w).type_as(img_source) #[N,h,w]
    YY = torch.arange(0,h).view(1,h,1).expand(N,h,w).type_as(img_source)
    
    grid_w = XX + disp * (ind_w_target - ind_w_source)
    grid_h = YY + disp * (ind_h_target - ind_h_source)
    

    grid_w_norm = 2.0 * grid_w / (w-1) -1.0
    grid_h_norm = 2.0 * grid_h / (h-1) -1.0
            
    grid = torch.stack((grid_w_norm, grid_h_norm),dim=3) #[N,h,w,2]

    # inverse warp
    img_source = torch.unsqueeze(img_source,0)
    img_target = functional.grid_sample(img_source,grid) # [N,1,h,w]
    img_target = torch.squeeze(img_target,1) #[N,h,w]
    return img_target

#############################

class AltFilter(nn.Module):
    def __init__(self, an, ch):
        super(AltFilter, self).__init__()
                
        self.relu = nn.ReLU(inplace=True)
        
        self.spaconv = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = (3,3), stride = 1, padding = 2, dilation=2)
        self.angconv = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = (3,3), stride = 1, padding = (1,1))
        
        self.an = an
        self.an2 = an*an
    
    def forward(self,x):

        N,c,h,w = x.shape #[N*81,c,h,w]
        N = N // self.an2
        
        out = self.relu(self.spaconv(x)) #[N*81,c,h,w]
        
        out = out.view(N,self.an2,c,h*w)
        out = torch.transpose(out,1,3) #[N,h*w,c,81]
        out = out.view(N*h*w,c,self.an,self.an)  #[N*h*w,c,9,9]

        out = self.relu(self.angconv(out)) #[N*h*w,c,9,9]
        
        out = out.view(N,h*w,c,self.an2)
        out = torch.transpose(out,1,3)
        out = out.view(N*self.an2,c,h,w)   #[N*81,c,h,w]

        return out

def make_Altlayer(layer_num, an, ch):
    layers = []
    for i in range( layer_num ):
        layers.append( AltFilter(an, ch) )
    return nn.Sequential(*layers)  
