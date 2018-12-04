# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:38:36 2018

@author: Baek
"""

from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model import edsr

"""
"""
from torchvision.utils import save_image
import os


import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytorch_ssim
import pytorch_msssim
from tqdm import tqdm
from utility import timer
from loss.loss import Adversarial
from loss.vgg import Perceptual

def print_save(line, text_path):
    print(line)
    f = open(text_path, 'a')
    f.write(line + '\n')
    f.close()
class Trainer(object):
    def __init__(self, args, training_loader, testing_loader):
        super(Trainer, self).__init__()
        self.args = args
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.scale = args.scale
        self.batch_size = args.batch_size
        self.model = None
        self.down = None
        self.lr = args.lr
        self.epochs = args.epochs
        self.L1 = None
        self.optimizer = None
        self.scheduler = None
        self.seed = args.seed
        self.data_timer = None
        self.train_timer = None
        self.test_timer = None
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.model_path = os.path.join('experiment',args.model_path)
        self.image_path = os.path.join(self.model_path, 'image')
        self.loss_type = args.loss_type
        self.text_path = os.path.join(self.model_path,"log/lr_{}_bat_{}.txt".format(self.lr, self.batch_size))
        if not os.path.exists(self.model_path):
             os.makedirs(self.model_path)
        if not os.path.exists(self.image_path):
             os.makedirs(self.image_path)
        if not os.path.exists(os.path.dirname(self.text_path)):
             os.makedirs(os.path.dirname(self.text_path))
                 
    def build_model(self):
        if self.args.pretrain:
            print('Loading  ', self.args.pretrain)
            self.model = torch.load(self.args.pretrain, map_location=lambda storage, loc: storage).to(self.device)
        else:
            self.model = edsr.EDSR(self.args).to(self.device)

        self.data_timer = timer(self.args)
        self.train_timer = timer(self.args)
        self.test_timer = timer(self.args)

        self.L1 = nn.L1Loss()
        self.L2= nn.MSELoss()
        self.ssim_loss = pytorch_ssim.SSIM(window_size = 11)
        self.msssim_loss = pytorch_msssim.MSSSIM()
        self.gan_loss = Adversarial(self.args, 'GAN')
        self.p_loss = Perceptual()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.L1.cuda()
            self.L2.cuda()
            self.ssim_loss.cuda()
            self.msssim_loss.cuda()
            self.gan_loss.cuda()
            self.p_loss.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        end = int(self.epochs/self.args.lr_decay) * self.args.lr_decay
        #milestones = np.linspace(self.args.lr_decay, end, num = int(self.epochs/self.args.lr_decay), dtype = int)
        milestones = [100, 300, 500]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=self.args.gamma)  # lr decay



    def train(self):
        self.model.train()
        train_loss = 0
        train_length = 0
        avg_psnr = 0
        avg_ssim = 0
        self.data_timer.start()
        for batch_num, (LR, HR, filename) in enumerate(tqdm(self.training_loader, ncols=80)):
            
            if self.args.n_colors == 4:
                LR, LRlabel = LR[0].to(self.device), LR[1].to(self.device)
                HR, HRlabel = HR[0].to(self.device), HR[1].to(self.device)
                LRlabel, HRlabel = LRlabel.unsqueeze(dim=1)*self.args.rgb_range/33, HRlabel.unsqueeze(dim=1)*self.args.rgb_range/33
                input, output = torch.cat((LR,LRlabel),1), torch.cat((HR, HRlabel),1)
            else:
                input, output = LR.to(self.device), HR.to(self.device)
            HR = output[:,0:3,:,:]
            batch_size = LR.size(0)
            #print(LR.size(), LRlabel.size())
            self.data_timer.stop()
            if batch_num == 0:
                self.train_timer.start()
            else:
                self.train_timer.go()

            SR = self.model(input)
                

            #save_image(LR/self.args.rgb_range, os.path.join(self.image_path,'{}_LR.png'.format(batch_num)))
            #save_image(HR/self.args.rgb_range,os.path.join(self.image_path,'{}_HR.png'.format(batch_num)))
            #save_image(SR/self.args.rgb_range, os.path.join(self.image_path,'{}_SR.png'.format(batch_num)))
            
            ## MSE Loss

            loss = 0
            if self.loss_type.find('L1') >=0:
                L1_loss = self.L1(SR, HR)
                loss += L1_loss
            ## GAN Loss
            if self.loss_type.find('GAN') >=0:
                gan_loss = self.gan_loss(SR,HR)
                loss += 1e-3*gan_loss
                #print(gan_loss)
            if self.loss_type.find('VGG') >=0:
                vgg_loss = self.p_loss(SR,HR, self.args.rgb_range)
                loss += vgg_loss
                #print(vgg_loss)

            train_loss += loss.item() * batch_size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_timer.stop()
            with torch.no_grad():
                
                for idx in range(batch_size):
                    sr = SR[idx, :, :, :].unsqueeze(dim=0)
                    hr = HR[idx, :, :, :].unsqueeze(dim=0)
                    #print(sr.size(), hr.size())
                    psnr = calc_psnr(sr, hr, int(self.args.scale[0]), self.args.rgb_range)
                    ssim = self.ssim_loss(sr/self.args.rgb_range, hr/self.args.rgb_range)
                    avg_psnr += psnr
                    avg_ssim += ssim

            train_length += batch_size
            if(batch_num+1) % self.args.print_every == 0:
                print_save("\n[{:4d}/{:4d}]    Average Loss: {:.4f}    overall time: {:.4f} + {:.4f}"
                .format(batch_num+1, len(self.training_loader),(train_loss / train_length), 
                self.data_timer.overall, self.train_timer.overall), self.text_path)
                self.data_timer.start()
                self.train_timer.start()
            else:
                self.data_timer.go()

        self.loss.append(train_loss / train_length)
        print_save("\n[{:4d}/{:4d}]    Average Loss: {:.4f}    overall time: {:.4f} + {:.4f}"
                .format(batch_num+1, len(self.training_loader),(train_loss / train_length), 
                self.data_timer.overall, self.train_timer.overall), self.text_path)
        self.psnr.append(avg_psnr/train_length)
        self.ssim.append(avg_ssim/train_length)
        print_save("Average PSNR: {:.4f}".format(avg_psnr/ train_length), self.text_path)
        print_save("Average SSIM: {:.4f}".format(avg_ssim/ train_length), self.text_path)

    def test(self):
        val_length = 0
        avg_psnr = 0
        avg_ssim = 0
        avg_msssim = 0
        self.model.eval()
        with torch.no_grad():
            for batch_num, (LR, HR, filename) in enumerate(tqdm(self.testing_loader, ncols=80)):
                
                if self.args.n_colors == 4:
                    LR, LRlabel = LR[0].to(self.device), LR[1].to(self.device)
                    HR, HRlabel = HR[0].to(self.device), HR[1].to(self.device)
                    LRlabel, HRlabel = LRlabel.unsqueeze(dim=1)*self.args.rgb_range/33, HRlabel.unsqueeze(dim=1)*self.args.rgb_range/33
                    input, output = torch.cat((LR,LRlabel),1), torch.cat((HR, HRlabel),1)
                else:
                    input, output = LR.to(self.device), HR.to(self.device)
                HR = output[:,0:3,:,:]
                batch_size = input.size(0)

                SR = self.model(input)
                
                val_length +=1
                mse = self.L2(SR/self.args.rgb_range,HR/self.args.rgb_range)
                #psnr = -10 * math.log10( mse)
                psnr = calc_psnr(SR, HR, int(self.args.scale[0]), self.args.rgb_range)
                ssim = self.ssim_loss(SR/self.args.rgb_range, HR/self.args.rgb_range)
                msssim = self.msssim_loss(SR/self.args.rgb_range, HR/self.args.rgb_range)
                avg_psnr += psnr
                avg_ssim += ssim
                avg_msssim += msssim

                
                if self.args.test_only:
                    basename = os.path.basename(filename[0])
                    HR_name = os.path.join(self.image_path, '{}_HR_'.format(batch_num)+basename.replace('pt', 'png'))
                    LR_name = HR_name.replace("HR", "LR")
                    res_name = HR_name.replace("HR", "SR")
                    
                    HR_dir = os.path.dirname(HR_name)
                    if not os.path.exists(HR_dir):
                        os.makedirs(HR_dir)
                    save_image(LR/self.args.rgb_range, LR_name)
                    save_image(HR/self.args.rgb_range, HR_name)
                    save_image(SR/self.args.rgb_range, res_name)
                    #print(res_name, 'is completed')          
            self.psnr.append(avg_psnr/val_length)
            self.ssim.append(avg_ssim/val_length)
            print_save("Average PSNR: {:.4f}".format(avg_psnr/ val_length), self.text_path)
            print_save("Average SSIM: {:.4f}".format(avg_ssim/ val_length), self.text_path)
            print_save("Average MSSSIM: {:.4f}".format(avg_msssim/ val_length), self.text_path)

    def save(self):
        #model_out_path = str(self.epoch) + self.model_out_path
        model_out_path = os.path.join(self.model_path, 'model.pth') 
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def plot(self, epoch):
        X = np.linspace(0, epoch+1, epoch+1)
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('PSNR(dB)')
        ax2 = fig.add_subplot(2,1,2)
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        ax1.plot(X, self.psnr)
        ax2.plot(X, self.ssim)
        plt.savefig(os.path.join(self.image_path,'loss_psnr_ssim.png'))

    def run(self):
        self.build_model()
        f = open(self.text_path, 'a')
        line = "lr : {}\n".format(self.lr)
        f.write(line)
        f.close()
        self.loss = []
        self.psnr =[]
        self.ssim = []
        plt.switch_backend('agg')

        print(self.args.test_only)
        for epoch in range(0, self.epochs):
            print_save("\n===> Epoch {} starts:".format(epoch), self.text_path)
            self.epoch = epoch
            if not self.args.test_only:
                self.train()
                self.scheduler.step(epoch)
                self.gan_loss.scheduler.step(epoch)
                self.plot(epoch)
                #if (epoch) % self.args.test_every == 0:
                    #self.test()
                    
                self.save() 
                
            if self.args.test_only:
                print('Test start')
                self.test()
                break
            
            
def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)