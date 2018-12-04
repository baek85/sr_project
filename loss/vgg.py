import torchvision.models as models
import torch.nn as nn
import torch 
import torchvision.transforms as transforms

class Perceptual(nn.Module):
    def __init__(self, count=4):
        super(Perceptual, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        layers = []
        count = 0
        for m in vgg19:
            if isinstance(m, nn.MaxPool2d):
                count +=1
                if count ==4:
                    break
            layers.append(m)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg19 = nn.Sequential(*layers).to(self.device)
        self.vgg19.eval()
        self.L2= nn.MSELoss()
        if torch.cuda.is_available():
            self.L2.cuda()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, sr, hr, rgb_range):
        sr, hr = sr/rgb_range, hr/rgb_range
        for idx in range(sr.size(0)):
            sr[idx,:,:,:], hr[idx,:,:,:] = self.normalize(sr[idx,:,:,:]), self.normalize(hr[idx,:,:,:])
        sr_feat, hr_feat = self.vgg19(sr), self.vgg19(hr)

        ploss = self.L2(sr_feat, hr_feat.detach())
        return ploss