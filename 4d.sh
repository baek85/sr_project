CUDA_VISIBLE_DEVICES=1 python3 main.py --lr 1e-5 --gamma 0.25 --model EDSR --scale 4 --loss_type L1 --model_path edsr_4D_x4 --ext bin --epochs 1000 --n_colors 4

#CUDA_VISIBLE_DEVICES=1 python3 main.py --model EDSR --scale 4 --patch_size 96 --model_path edsr_4D_x4 --ext bin --epochs 1000 --n_colors 4 --gamma 0.25
#CUDA_VISIBLE_DEVICES=1 python3 main.py --data_train cityscapes/4Dbin --data_test cityscapes/4Dbin --model EDSR --scale 4 --patch_size 96 --model_path edsr_4D_x4 --ext bin --epochs 1000 --n_colors 4
