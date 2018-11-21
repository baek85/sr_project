# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:07:09 2018

@author: Baek
"""
from option import args
import os

import data
#from mydata import get_training_set, get_val_set
#from torch.utils import data
from trainer import Trainer


if not os.path.exists(args.image_path):
    os.makedirs(args.image_path)
"""
train_set = get_training_set(args)
train_loader = data.DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
                            
val_set = get_val_set(args)
val_loader = data.DataLoader(dataset=val_set,
                              batch_size=1,
                              shuffle=False)
"""

loader = data.Data(args)
train_loader = loader.loader_train
val_loader = loader.loader_test
model = Trainer(args, train_loader, val_loader)

model.run()
