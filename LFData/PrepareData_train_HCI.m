%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data from HCI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===> train_HCI.h5 (in python)
% uint8 0-255
%  ['LFI']   [N,ah,aw,h,w]

% ===> in matlab (inverse)
%  [w,h,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% path
savepath = 'train_HCI.h5';
folder = './HCI';

listname = 'Train_HCI.txt';
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
LFI = zeros(H, W, ah, aw, 1, 'uint8');

count = 0;

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = fullfile(folder,lfname);
    disp(lf_path);
    
    img = zeros(H,W,ah,aw,'uint8');    
    for v = 1 : ah
        for u = 1 : aw
            ind = (v-1)*9+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub = imread(fullfile(lf_path,imgname));
            sub = rgb2ycbcr(sub);
            img(:,:,v,u) = sub(:,:,1);
        end
    end
      
    % generate patches
    count = count+1;    
    LFI(:, :, :, :, count) = img;
      
end  
 
%% generate data
order = randperm(count);
LFI = permute(LFI(:, :, :, :, order),[2,1,4,3,5]); %[h,w,ah,aw,N] -> [w,h,aw,ah,N]  

%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath, '/LFI', size(LFI), 'Datatype', 'uint8'); % width, height, channels, number 
h5write(savepath, '/LFI', LFI);

h5disp(savepath);
