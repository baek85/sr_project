# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:07:09 2018

@author: Baek
"""
from option import args
import os

import data
from mydata import get_train_loader, get_val_loader
from preprocess import make_train_set, make_val_set
from trainer import Trainer


if not os.path.exists(args.image_path):
    os.makedirs(args.image_path)
    
#valset = make_val_set(args)
#trainset = make_train_set(args)

#print('Make Dataset format same with DIV2K')

print('Make DataLoader ')
#loader = data.Data(args)
#train_loader = loader.loader_train
#val_loader = loader.loader_test
train_loader = get_train_loader(args)
val_loader = get_val_loader(args)
model = Trainer(args, train_loader, val_loader)

model.run()
