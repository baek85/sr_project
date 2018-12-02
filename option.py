import argparse


parser = argparse.ArgumentParser(description='PyTorch HDR reconstruction Example')

# Hardware
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed to use. Default=123')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')                    
# Data
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--dir_data', type=str, default='../../../dataset',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution size')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--pretrain', type=str, default= '', 
                    help='pretrained model path')
parser.add_argument('--n_resblocks', type=int, default=16, 
                    help='number of conv layers')
parser.add_argument('--n_feats', type=int, default=64, 
                    help='number of conv layers')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')

# Training
parser.add_argument('--test_every', type=int, default=5, 
                    help='test images every n iters')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')   

# Optimization
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
                    
# Loss
parser.add_argument('--loss_type', type=str, default='L1', help='loss type   L1 or GAN')

# Log
parser.add_argument('--model_path', type=str, default="model/EDSRx2.pth", help='model output path')
parser.add_argument('--image_path', type=str, default='Result', help='output path')
parser.add_argument('--print_every', type=int, default=10000,
                    help='how many batches to wait before logging training status')


args = parser.parse_args()


args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

