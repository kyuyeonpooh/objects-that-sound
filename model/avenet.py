import torch
import torch.nn as nn
import torch.nn.functional as F

from subnet import ImageConvNet, AudioConvNet


class AVENet(nn.Module):
    def __init__(self):
        super(AVENet, self).__init__()

        # image subnetwork
        self.icn = ImageConvNet()
        self.img_pool = nn.AdaptiveMaxPool2d(1)
        self.img_fc1 = nn.Linear(512, 128)
        self.img_fc2 = nn.Linear(128, 128)

        # audio subnetwork
        self.acn = AudioConvNet()
        self.aud_pool = nn.AdaptiveMaxPool2d(1)
        self.aud_fc1 = nn.Linear(512, 128)
        self.aud_fc2 = nn.Linear(128, 128)

        # fusion network
        self.fc3 = nn.Linear(1, 2)

    def forward(self, img, aud):
        # image subnetwork
        img = self.icn(img)
        img = self.img_pool(img)
        img = img.squeeze(2).squeeze(2)  # [N, 512, 1, 1] to [N, 512]
        img = F.relu(self.img_fc1(img))
        img_emb = F.normalize(self.img_fc2(img), p=2, dim=1)  # L2 normalization

        # audio subnetwork
        aud = self.acn(aud)
        aud = self.aud_pool(aud)
        aud = aud.squeeze(2).squeeze(2)  # [N, 512, 1, 1] to [N, 512]
        aud = F.relu(self.aud_fc1(aud))
        aud_emb = F.normalize(self.aud_fc2(aud), p=2, dim=1)  # L2 normalization

        # fusion network
        euc_dist = ((img_emb - aud_emb) ** 2).sum(dim=1, keepdim=True).sqrt()  # Euclidean distance
        out = self.fc3(euc_dist)
        return out


# forward propagation test
if __name__ == "__main__":
    img = torch.rand((16, 3, 224, 224))
    aud = torch.rand((16, 1, 257, 200))
    avenet = AVENet()
    print(avenet(img, aud).shape)
