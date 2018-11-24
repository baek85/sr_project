#CUDA_VISIBILE_DEVICES=1 python main.py

#CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --model_path edsr_baseline_x2.pth --ext sep_reset
## Training
CUDA_VISIBLE_DEVICES=1 python main.py --data_train CUB200 --model EDSR --scale 2 --patch_size 96 --model_path edsr_baseline_x22.pth --ext sep

## Test
CUDA_VISIBILE_DEVICES=1 python main.py --test_only --data_test CUB200 --pretrain edsr_baseline_x22.pth --model EDSR
