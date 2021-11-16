%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data from HCI old dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; close all;

dataset = 'HCI_old';
%% path
savepath = sprintf('test_%s.h5',dataset);
folder = './HCI_old/synthetic'; 

listname = sprintf('Test_%s.txt',dataset);
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params
H = 768;
W = 768;

ah = 9;
aw = 9;

%% initialization
LFI_ycbcr = zeros(H, W, 3, ah, aw, 1, 'uint8');
count = 0;

%% generate data
for k = 1:length(list)
    lfname = list{k};
    lf_path = fullfile(folder,lfname,'lf.h5');
    disp(lf_path);
    
    lf = h5read(lf_path,'/LF'); %[3,w,h,aw,ah]
    lf = permute(lf,[3,2,1,5,4]); %[h,w,3,ah,aw]
    lf = flip(lf,5);
    
    lf_ycbcr = lf;  
    for v = 1 : ah
        for u = 1 : aw
            img_rgb = lf(:,:,:,v,u);
            img_ycbcr = rgb2ycbcr(img_rgb);
            lf_ycbcr(:,:,:,v,u) = img_ycbcr;
        end
    end

    count = count+1;    
    LFI_ycbcr(:, :, :, :, :, count) = lf_ycbcr;
      
end  
%% generate dat
LFI_ycbcr = permute(LFI_ycbcr,[3,2,1,5,4,6]);%[h,w,3,ah,aw,n]--->[3,w,h,aw,ah,N]
%% save data
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath,'/LFI_ycbcr',size(LFI_ycbcr),'Datatype','uint8');
h5write(savepath, '/LFI_ycbcr', LFI_ycbcr);

h5disp(savepath);