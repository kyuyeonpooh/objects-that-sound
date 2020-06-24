import torch
import torch.nn as nn
import torch.nn.functional as F

from subnet import ImageConvNet, AudioConvNet


class L3Net(nn.Module):
    def __init__(self):
        super(L3Net, self).__init__()

        # image subnetwork
        self.icn = ImageConvNet()
        self.img_pool = nn.AdaptiveMaxPool2d(1)

        # audio subnetwork
        self.acn = AudioConvNet()
        self.aud_pool = nn.AdaptiveMaxPool2d(1)

        # fusion network
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, img, aud):
        # image subnetwork
        img = self.icn(img)
        img = self.img_pool(img)
        img_emb = img.squeeze(2).squeeze(2)  # [N, 512, 1, 1] to [N, 512]

        # audio subnetwork
        aud = self.acn(aud)
        aud = self.aud_pool(aud)
        aud_emb = aud.squeeze(2).squeeze(2)  # [N, 512, 1, 1] to [N, 512]

        # fusion network
        concat = torch.cat((img_emb, aud_emb), dim=1)
        out = F.relu(self.fc1(concat))
        out = self.fc2(out)
        return out, img_emb, aud_emb


# forward propagation test
if __name__ == "__main__":
    img = torch.rand((16, 3, 224, 224))
    aud = torch.rand((16, 1, 257, 199))
    l3net = L3Net()
    print(l3net(img, aud)[0].shape)
