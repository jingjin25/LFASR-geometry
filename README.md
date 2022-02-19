# LFASR-geometry
PyTorch implementation of **AAAI 2020** paper: **Learning Light Field Angular Super-Resolution via a Geometry-Aware Network**.

[[paper]](https://dl.acm.org/doi/10.1145/3394171.3413585)

## Requirements
- Python 3.6
- PyTorch 1.3
- Matlab (for training/test data generation)

## Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in `LFData`.


## Training
To re-train the model, run:

```
python train_fusionNet.py --dataset HCI --angular_num 9 --scale 4 --num_cp 10 --patch_size 64 --lr 1e-4  --step 500
python train_fusionNet.py --dataset Lytro --angular_num 8 --scale 4 --num_cp 10 --patch_size 64 --lr 1e-4  --step 200
python train_fusionNet.py --dataset HCI --angular_num 9 --scale 8 --num_cp 10 --patch_size 64 --lr 1e-5  --step 1000
python train_fusionNet.py --dataset Lytro --angular_num 8 --scale 8 --num_cp 10 --patch_size 64 --lr 1e-5  --step 200
```
