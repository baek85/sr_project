#CUDA_VISIBILE_DEVICES=1 python main.py

#CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --model_path edsr_baseline_x2.pth --ext sep_reset
#CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --model_path edsr_baseline_x22.pth --ext sep
CUDA_VISIBILE_DEVICES=1 python main.py --data_test CUB200 --data_range 801-900 --pretrain edsr_baseline_x22.pth --test_only