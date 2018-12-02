#CUDA_VISIBILE_DEVICES=1 python main.py

#CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --model_path edsr_baseline_x2.pth --ext sep_reset
CUDA_VISIBLE_DEVICES=1 python main.py --data_train cityscapes/leftImg8bit --data_test cityscapes/leftImg8bit --model EDSR --scale 4 --patch_size 96 --model_path edsr_baseline_x4.pth --ext sep --epochs 1

CUDA_VISIBLE_DEVICES=1 python main.py --data_train cityscapes/leftImg8bit --data_test cityscapes/leftImg8bit --model EDSR --scale 2 --patch_size 96 --model_path edsr_baseline_x22.pth --ext sep --epochs 1

CUDA_VISIBLE_DEVICES=1 python main.py --data_train cityscapes/leftImg8bit --data_test cityscapes/leftImg8bit --model EDSR --scale 3 --patch_size 96 --model_path edsr_baseline_x22.pth --ext sep --epochs 1

