#CUDA_VISIBILE_DEVICES=1 python main.py

#CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --model_path edsr_baseline_x2.pth

CUDA_VISIBILE_DEVICES=1 python main.py --data_test DIV2K --data_range 801-900 --pretrain edsr_baseline_x2.pth? --test_only