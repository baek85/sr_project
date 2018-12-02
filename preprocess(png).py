import os
import pickle
from PIL import Image
from os import listdir
from option import args
from tqdm import tqdm
import numpy as np
import cv2
def load_img(filepath):
    img = Image.open(filepath)
    #img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img

class MakeDataset():
    def __init__(self, args, name='leftImg8bit' ,train=True):
        super(MakeDataset, self).__init__()
        self.args = args
        self.train = train
        self.scale = args.scale[0]
        self.LR_filenames, self.HR_filenames = self._get_filenames(name)

        self.basenames = [os.path.basename(x) for x in self.HR_filenames]
        root_dir = os.path.join(args.dir_data, name) 
        if self.train:
            self.HR2 = [os.path.join(root_dir,'train_HR/' + x) for x in self.basenames]
            self.LR_filenames = [os.path.join(root_dir,'train_LR_bicubic/X'+str(self.scale) +'/' + x) for x in self.basenames]
        else:
            self.HR2 = [os.path.join(root_dir,'val_HR/' + x) for x in self.basenames]
            self.LR_filenames = [os.path.join(root_dir,'val_LR_bicubic/X'+str(self.scale) +'/' + x) for x in self.basenames]
        self.LR_dir = os.path.dirname(self.LR_filenames[0])
        if not os.path.exists(self.LR_dir):
            os.makedirs(self.LR_dir)
        self.HR_dir = os.path.dirname(self.HR2[0])
        if not os.path.exists(self.HR_dir):
            os.makedirs(self.HR_dir)
        self._make_LR(name)

    def _make_LR(self, name):
        for index, HR_filename in enumerate(tqdm(self.HR_filenames, ncols=80)):
            HR = load_img(HR_filename)
            HR.save(self.HR2[index])
            HR = np.asarray(HR)
            size = np.shape(HR)
            h, w = size[0], size[1]
            h2, w2 = (h//self.scale)*self.scale, (w//self.scale)*self.scale
            if len(size) == 3:
                HR = HR[0:h2, 0:w2, :]
            else:
                HR = HR[0:h2, 0:w2]
            LR = cv2.resize(HR, None, fx = 1 / self.scale, fy = 1 / self.scale,interpolation = cv2.INTER_CUBIC)
            im = Image.fromarray(LR)     
            im.save(self.LR_filenames[index])
            #print('Save LR image X' + str(self.scale), 'to', self.LR_filenames[index])

    def _get_filenames(self, name):
        
        if name == 'DIV2K':
            root_dir = os.path.join(self.args.dir_data, name)
            LR_dir = os.path.join(root_dir, 'DIV2K_train_LR_bicubic')
            LR_dir = os.path.join(LR_dir, 'X'+ str(self.args.scale[0]))

            HR_dir = os.path.join(root_dir, 'DIV2K_train_HR')
            r = self.args.data_range.split('/')
            if self.train:
                data_range = r[0].split('-')
            elif self.args.test_only:
                data_range = r[0].split('-')
            else:    
                data_range = r[1].split('-')

            HR_names = sorted(listdir(HR_dir))
            HR_names = HR_names[int(data_range[0])-1:int(data_range[1])]
            LR_names = sorted(listdir(LR_dir))
            LR_names = LR_names[int(data_range[0])-1:int(data_range[1])]
            LR_filenames = [os.path.join(LR_dir, x) for x in LR_names]
            HR_filenames = [os.path.join(HR_dir, x) for x in HR_names]
        elif name.find('leftImg8bit') >= 0:
            root_dir = os.path.join(self.args.dir_data, name)
            if self.train:
                HR_dir = os.path.join(root_dir, 'train')
            else:
                HR_dir = os.path.join(root_dir, 'val')
            HR_dir2 = [os.path.join(HR_dir, x) for x in listdir(HR_dir)]
            HR_dir2 = sorted(HR_dir2)
            HR_filenames = []
            for idx in range(len(HR_dir2)):
                HR_filenames += sorted([os.path.join(HR_dir2[idx], x) for x in listdir(HR_dir2[idx])])
            
            LR_filenames = None
        elif name.find('gtFine') >=0:
            root_dir = os.path.join(self.args.dir_data, name)
            if self.train:
                HR_dir = os.path.join(root_dir, 'train')
            else:
                HR_dir = os.path.join(root_dir, 'val')
            HR_dir2 = [os.path.join(HR_dir, x) for x in listdir(HR_dir)]
            HR_dir2 = sorted(HR_dir2)
            HR_filenames = []
            for idx in range(len(HR_dir2)):
                HR_filenames += sorted([os.path.join(HR_dir2[idx], x) for x in listdir(HR_dir2[idx])])
            HR_filenames = [x for x in HR_filenames if os.path.basename(x).find('labelIds')>=0]
            LR_filenames = None
        elif name == 'CUB200':
            image_dir = os.path.join(root_dir, 'images', 'images')
            mid_dir = [os.path.join(image_dir, x) for x in listdir(image_dir)]
            class_dir = sorted(mid_dir)
            class_dir = class_dir[200:400]

            HR_filenames = []
            classes = []
            for idx in range(len(class_dir)):
                image_names = listdir(class_dir[idx])
                image_names = sorted(image_names)
                image_names = image_names[len(image_names)//2: len(image_names)]

                HR_filenames += [os.path.join(class_dir[idx], x) for x in image_names]
                
                classes += [idx for x in range(len(image_names))]

            self.classes = classes
            LR_filenames = None
        
        else:
            LR_filenames = None 
            HR_filenames = None

        return LR_filenames, HR_filenames

def make_train_set(args):
    train_set = MakeDataset(args, 'cityscapes/gtFine', train=True)
    train_set = MakeDataset(args, args.data_train[0], train=True)
    return train_set

def make_val_set(args):
    
    val_set = MakeDataset(args, 'cityscapes/gtFine', train=False)
    val_set = MakeDataset(args, args.data_test[0], train=False)
    return val_set