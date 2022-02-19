# LFASR-geometry
PyTorch implementation of **AAAI 2020** paper: **Learning Light Field Angular Super-Resolution via a Geometry-Aware Network**.

[[paper]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-JinJ.8502.pdf)

## Requirements
- Python 3.6
- PyTorch 1.1
- Matlab (for training/test data generation)

## Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in `LFData`.


## Demo 
To produce the results in the paper, run:

```
python test_pretrained.py --model_path ./pretrained_model/HCI_2x2-7x7.pth   --test_dataset HCI --data_path ./LFData/test_HCI.h5 --angular_out 7 --angular_in 2 --crop 1 --save_img 1
```

## Training
To re-train the model, run:

```
python train.py --lr 1e-4 --step 500 --epi 1.0 --patch_size 96 --num_cp 10   --layer_num 4  --angular_out 7 --angular_in 2 --dataset HCI --dataset_path ./LFData/train_HCI.h5
```
