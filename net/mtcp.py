import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):

        super(CSRNet, self).__init__()

        # feature extractor
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256]
        # self.backend_feat  = [512, 512, 512, 256,128,64]

        self.frontend = make_layers(self.frontend_feat, in_channels=3, batch_norm=False, dilation=False)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        # density regressor 1
        self.backend_den_feat_1 = [128, 64]
        self.backend_den_1 = make_layers(self.backend_den_feat_1, in_channels=256, batch_norm=False, dilation=True)
        self.output_den_1 = nn.Conv2d(64, 1, kernel_size=1)
        
        # mask segmentation
        self.backend_seg_feat1 = [128, 64]
        self.backend_seg_1 = make_layers(self.backend_seg_feat1, in_channels=256, batch_norm=False, dilation=True)
        self.output_seg_1 = nn.Conv2d(64, 2, kernel_size=1)


        self.uncertainty = ConfidentNet()

        # self.uncertainty1_1 = nn.Conv2d(64, 400, 3, 1, 1)
        # self.uncertainty2_1 = nn.Conv2d(400, 120, 3, 1, 1)
        # self.uncertainty3_1 = nn.Conv2d(120, 64, 3, 1, 1)
        # self.uncertainty4_1 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.uncertainty5_1 = nn.Conv2d(64, 1, 3, 1, 1)

        if not load_weights:
            mod = models.vgg16(pretrained = False)
            mod.load_state_dict(torch.load('./vgg16-397923af.pth'))
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self,x):

        x = self.frontend(x)
        x = self.backend(x)

        # density map
        d = self.backend_den_1(x)
        d = self.output_den_1(d)
        d = F.relu(d)

        # segmetation mask
        s = self.backend_seg_1(x)
        u = s
        s = self.output_seg_1(s)

        # uncer
        u = u.detach_()
        u = self.uncertainty(u)
        # u = F.relu(self.uncertainty1_1(u))
        # u = F.relu(self.uncertainty2_1(u))
        # u = F.relu(self.uncertainty3_1(u))
        # u = F.relu(self.uncertainty4_1(u))
        # u = self.uncertainty5_1(u)

        # d=nn.functional.interpolate(d,size=(d.shape[2]*8,d.shape[3]*8),mode='bilinear',align_corners=False)
        # s=nn.functional.interpolate(s,size=(s.shape[2]*8,s.shape[3]*8),mode='bilinear',align_corners=False)
        # u=nn.functional.interpolate(u,size=(u.shape[2]*8,u.shape[3]*8),mode='bilinear',align_corners=False)

        return d, s, u

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3, batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)     

class ConfidentNet(nn.Module):
    def __init__(self):
        super(ConfidentNet, self).__init__()
        self.uncertainty1_1 = nn.Conv2d(64, 400, 3, 1, 1)
        self.uncertainty2_1 = nn.Conv2d(400, 120, 3, 1, 1)
        self.uncertainty3_1 = nn.Conv2d(120, 64, 3, 1, 1)
        self.uncertainty4_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.uncertainty5_1 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, u):
        u = F.relu(self.uncertainty1_1(u))
        u = F.relu(self.uncertainty2_1(u))
        u = F.relu(self.uncertainty3_1(u))
        u = F.relu(self.uncertainty4_1(u))
        u = self.uncertainty5_1(u)
        return u


