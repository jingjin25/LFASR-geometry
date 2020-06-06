%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data from HCI dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===> test_SIG.h5 (in python)
% uint8 0-255
%  ['LFI']            [N,ah,aw,h,w]
%  ['LFI_rgb']        [N,ah,aw,h,w,3]
%  ['LFI_ycbcr_up']   [N,ah,aw,h,w,3]
% ===> in matlab (inverse)
%  [3,w,h,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; close all;

dataset = 'HCI';
%% path
savepath = sprintf('test_%s.h5',dataset);
folder = './HCI'; 

listname = sprintf('Test_%s.txt',dataset);
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params
H = 512;
W = 512;

ah = 9;
aw = 9;

%% initialization
LFI_ycbcr = zeros(H, W, 3, ah, aw, 1, 'uint8');

count = 0;

%% generate data
for k = 1:length(list)
    lfname = list{k};
    lf_path = fullfile(folder,lfname);
    disp(lf_path);
        
    img_ycbcr = zeros(H,W,3,ah,aw,'uint8');   
    for v = 1 : ah
        for u = 1 : aw
            ind = (v-1)*9+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub = imread(fullfile(lf_path,imgname));
            sub = rgb2ycbcr(sub);
            img_ycbcr(:,:,:,v,u) = sub(1:H,1:W,:);
        end
    end

    count = count+1;    
    LFI_ycbcr(:, :, :, :, :, count) = img_ycbcr;
      
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