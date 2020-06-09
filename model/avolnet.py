import torch
import torch.nn as nn
import torch.nn.functional as F

from subnet import ImageConvNet, AudioConvNet


class AVOLNet(nn.Module):
    def __init__(self):
        super(AVOLNet, self).__init__()

        # image subnetwork
        self.icn = ImageConvNet()
        self.im_conv5 = nn.Conv2d(512, 128, 1)
        self.im_conv6 = nn.Conv2d(128, 128, 1)
        
        # audio subnetwork
        self.acn = AudioConvNet()
        self.aud_pool = nn.AdaptiveMaxPool2d(1)
        self.aud_fc1 = nn.Linear(512, 128)
        self.aud_fc2 = nn.Linear(128, 128)

        # fusion network
        self.fus_conv7 = nn.Conv2d(1, 1, 1)
        self.fus_sig = nn.Sigmoid()
        self.fus_pool = nn.AdaptiveMaxPool2d(1)
        #self.fus_fc = nn.Linear(1, 2)
        # self.fus_fc.weight.data[0] = -0.7
        # self.fus_fc.weight.data[1] = 0.7
        # self.fus_fc.bias.data[0] = 1.2
        # self.fus_fc.bias.data[1] = -1.2
        
    def forward(self, img, aud):
        # image subnetwork
        img = self.icn(img)
        img = self.im_conv5(img)
        img = self.im_conv6(img)
        
        # audio subnetwork
        aud = self.acn(aud)
        aud = self.aud_pool(aud)
        aud = aud.squeeze(2).squeeze(2)
        aud = F.relu(self.aud_fc1(aud))
        aud = self.aud_fc2(aud)
        
        # fusion network
        img = img.view(img.size(0), 128, -1)
        aud = aud.view(aud.size(0), 1, -1)
        scalar_prod = torch.bmm(aud, img).view(aud.size(0), 1, 14, 14)
        loc = self.fus_conv7(scalar_prod)
        loc = self.fus_sig(loc)
        out = self.fus_pool(loc).squeeze()
        #out = self.fus_fc(out)
        
        return out, loc


# forward propagation test
if __name__ == "__main__":
    img = torch.rand((16, 3, 224, 224))
    aud = torch.rand((16, 1, 257, 200))
    avolnet = AVOLNet()
    out, loc = avolnet(img, aud)
    print(out.shape, loc.shape)